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
    Gated Grouped Linear (LIN)
    """
    def __init__(self, mp, d_type, n_ctx, n_emb, r_dropout, n_groups):
        super().__init__()
        assert n_emb % n_groups == 0
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.r_dropout = r_dropout
        self.n_groups = n_groups
        self.group_dim = n_emb // n_groups

        # Linear layers for each group
        self.linears = [Linear(mp, d_type, self.group_dim, self.group_dim, bias=False) for _ in range(n_groups)]

        # Gating layers for each group
        self.gates = [Linear(mp, d_type, self.group_dim, self.group_dim, bias=False) for _ in range(n_groups)]

        # Dropout layer
        self.dropout = Dropout(mp, r_dropout)

    def set(self, mode=True):
        for linear in self.linears:
            linear.set(mode)
        for gate in self.gates:
            gate.set(mode)
        self.dropout.set(mode)

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
        """
        Forward pass for GGL.
        """

        # 1. Split input into groups:
        B, T, D = x.shape
        groups = self.mp.split(x, self.n_groups, axis=2)

        # 2. Process each group
        outputs = []
        gates = []          # store sigmoid outputs
        gate_pre_acts = []  # store raw gate outputs before sigmoid
        for i in range(self.n_groups):
            h = self.linears[i].forward(groups[i])
            gate_raw = self.gates[i].forward(groups[i])  # pre-activation
            gate = sigmoid(self.mp, gate_raw)
            outputs.append(h * gate)
            gate_pre_acts.append(gate_raw)
            gates.append(gate)

        # 3. Merge groups and apply dropout
        out = self.mp.concatenate(outputs, axis=2)

        # 4. Apply dropout
        out = self.dropout.forward(out)

        # 5. Cache intermediate results if needed
        if self.setting:
            self._cache = (x, groups, outputs, gate_pre_acts, gates)

        return out

    def backward(self, grad_output):
        """
        Backward pass for GGL.
        """

        # 1. Unpack cached values
        _, _, _, gate_pre_acts, gates = self._cache

        # 2. Backward through dropout
        grad_out, _ = self.dropout.backward(grad_output)

        # 3. Backward through split gradient into groups
        grad_groups = self.mp.split(grad_out, self.n_groups, axis=2)

        # 4. Backward through each group
        grad_x_parts = []
        param_grads = []
        for i in range(self.n_groups):
            grad_h = grad_groups[i]
            grad_gate = grad_groups[i]
            grad_h_in = grad_h * gates[i]
            grad_gate_in = grad_gate * sigmoid_prime(self.mp, gates[i])
            grad_x_l, l_grads = self.linears[i].backward(grad_h_in)
            grad_x_g, g_grads = self.gates[i].backward(grad_gate_in)
            grad_x_parts.append(grad_x_l + grad_x_g)
            param_grads += l_grads + g_grads
            
        # 5. Merge gradients from all groups
        grad_x = self.mp.concatenate(grad_x_parts, axis=2)

        return grad_x, param_grads

    def from_dict(self, weights_dict, i):
        for group_idx in range(self.n_groups):
            self.linears[group_idx].weight = weights_dict[f'block_{i}_ggl_linear_{group_idx}_weight']            
            self.gates[group_idx].weight = weights_dict[f'block_{i}_ggl_gate_{group_idx}_weight']

            self.linears[group_idx].synchronize()
            self.gates[group_idx].synchronize()

    def towa_dict(self, weights_dict, i):
        for group_idx in range(self.n_groups):
            weights_dict[f'block_{i}_ggl_linear_{group_idx}_weight'] = self.linears[group_idx].weight
            weights_dict[f'block_{i}_ggl_gate_{group_idx}_weight'] = self.gates[group_idx].weight