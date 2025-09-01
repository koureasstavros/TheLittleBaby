#########################
# Gated Grouped Linear (GGL)
# Author: Koureas Stavros
#########################

from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import sigmoid, sigmoid_prime

class GGL(Module):
    """
    Gated Grouped Linear:
    - Split embedding into groups
    - Each group has its own linear + gate
    - Merge back to full embedding
    """
    def __init__(self, mp, n_ctx, n_emb, p_dropout, n_groups):
        super().__init__()
        assert n_emb % n_groups == 0
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.n_groups = n_groups
        self.group_dim = n_emb // n_groups
        self.p_dropout = p_dropout

        # Linear layers for each group
        self.linears = [Linear(mp, self.group_dim, self.group_dim, bias=False) for _ in range(n_groups)]

        # Gating layers for each group
        self.gates = [Linear(mp, self.group_dim, self.group_dim, bias=False) for _ in range(n_groups)]

        # Dropout layer
        self.dropout = Dropout(mp, p_dropout)

    def parameters(self):
        params = []
        for l, g in zip(self.linears, self.gates):
            params += l.parameters() + g.parameters()
        return params

    def flops(self, batch_size, training):
        """
        Estimate FLOPs for the GGL forward pass.
        Multiply-adds are counted as 2 FLOPs.
        training: if True, include backward/update cost (~3x forward)
        """
        def linear_flops(in_f, out_f):
            return 2 * batch_size * self.n_ctx * in_f * out_f

        flops = 0

        # Per group: linear projection + gate projection + sigmoid + elementwise multiply
        for _ in range(self.n_groups):
            # Linear projection
            flops += linear_flops(self.group_dim, self.group_dim)
            # Gate projection
            flops += linear_flops(self.group_dim, self.group_dim)
            # Sigmoid activation (~4 FLOPs per element)
            flops += 4 * batch_size * self.n_ctx * self.group_dim
            # Elementwise multiply
            flops += batch_size * self.n_ctx * self.group_dim

        if training:
            flops *= 3  # forward + backward + update

        return flops
    
    def forward(self, x):
        B, T, D = x.shape
        groups = self.mp.split(x, self.n_groups, axis=2)
        outputs = []
        gate_pre_acts = []  # store raw gate outputs before sigmoid
        for i in range(self.n_groups):
            h = self.linears[i].forward(groups[i])
            gate_raw = self.gates[i].forward(groups[i])  # pre-activation
            gate = sigmoid(self.mp, gate_raw)
            outputs.append(h * gate)
            gate_pre_acts.append(gate_raw)
        out = self.mp.concatenate(outputs, axis=2)
        out = self.dropout.forward(out)
        if self.setting:
            self._cache = (x, groups, outputs, gate_pre_acts)
        return out

    def backward(self, grad_output):
        grad_out, _ = self.dropout.backward(grad_output)
        grad_groups = self.mp.split(grad_out, self.n_groups, axis=2)
        grad_x_parts = []
        grads_all = []
        _, _, _, gate_pre_acts = self._cache
        for i in range(self.n_groups):
            grad_h = grad_groups[i]
            grad_gate = grad_groups[i]
            grad_h_in = grad_h * sigmoid(self.mp, gate_pre_acts[i])
            grad_gate_in = grad_gate * sigmoid_prime(self.mp, gate_pre_acts[i])
            grad_x_l, l_grads = self.linears[i].backward(grad_h_in)
            grad_x_g, g_grads = self.gates[i].backward(grad_gate_in)
            grad_x_parts.append(grad_x_l + grad_x_g)
            grads_all += l_grads + g_grads
        grad_x = self.mp.concatenate(grad_x_parts, axis=2)
        return grad_x, grads_all

    def from_dict(self, weights_dict, i):
        for group_idx in range(self.n_groups):
            self.linears[group_idx].weight = weights_dict[f'block_{i}_ggl_linear_{group_idx}_weight']            
            self.gates[group_idx].weight = weights_dict[f'block_{i}_ggl_gate_{group_idx}_weight']

            self.linears[group_idx].synchronize()
            self.gates[group_idx].synchronize()

    def to_dict(self, weights_dict, i):
        for group_idx in range(self.n_groups):
            weights_dict[f'block_{i}_ggl_linear_{group_idx}_weight'] = self.linears[group_idx].weight
            weights_dict[f'block_{i}_ggl_gate_{group_idx}_weight'] = self.gates[group_idx].weight