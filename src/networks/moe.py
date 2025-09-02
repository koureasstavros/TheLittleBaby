#########################
# Mixture of Experts (MOE)
# Author: Koureas Stavros
#########################

from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import softmax, gelu, gelu_prime

class MOE(Module):
    """
    Dense (non-sparse) Mixture-of-Experts feed-forward:
    y = sum_e softmax(gate(x))_e * Expert_e(x)
    Each Expert_e: Linear_up -> GELU -> Linear_down -> Dropout
    """
    def __init__(self, mp, n_ctx, n_emb, p_dropout, n_expansion, n_experts):
        super().__init__()
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.n_expansion = n_expansion
        self.n_experts = n_experts

        # Gating linear (no bias for simplicity)
        self.g_proj = Linear(mp, n_emb, n_experts, bias=False)

        # Experts: separate parameter sets
        self.c_proj_up = [Linear(mp, n_emb, n_expansion * n_emb, bias=True) for _ in range(n_experts)]
        self.c_proj_dn = [Linear(mp, n_expansion * n_emb, n_emb, bias=True) for _ in range(n_experts)]

        # Dropout layer
        self.p_dropout = Dropout(mp, p_dropout)

    def parameters(self):
        params = self.g_proj.parameters()
        for u, d in zip(self.c_proj_up, self.c_proj_dn):
            params += u.parameters()
            params += d.parameters()
        return params

    def flops(self, batch_size, training):
        """
        Estimate FLOPs for the dense MOE forward pass.
        Multiply-adds are counted as 2 FLOPs.
        batch_size: number of sequences in the batch
        training: if True, include backward/update cost (~3x forward)
        """
        def linear_flops(in_f, out_f):
            return 2 * batch_size * self.n_ctx * in_f * out_f

        flops = 0

        # 1. Gating projection: (B, T, n_emb) -> (B, T, n_experts)
        flops += linear_flops(self.n_emb, self.n_experts)

        # 2. Experts (dense: compute all experts)
        hidden_size = self.n_expansion * self.n_emb
        for _ in range(self.n_experts):
            # Up projection
            flops += linear_flops(self.n_emb, hidden_size)
            # GELU activation (~4 FLOPs per element)
            flops += 4 * batch_size * self.n_ctx * hidden_size
            # Down projection
            flops += linear_flops(hidden_size, self.n_emb)

        if training:
            flops *= 3  # forward + backward + update

        return flops

    def set(self, mode=True):
        super().set(mode)
        self.g_proj.set(mode)
        for u, d in zip(self.c_proj_up, self.c_proj_dn):
            u.set(mode); d.set(mode)
        self.p_dropout.set(mode)

    def forward(self, x):
        """
        x: (B, T, n_emb)
        returns: (B, T, n_emb)
        """
        B, T, D = x.shape

        # 1. Gating
        gate_logits = self.g_proj.forward(x)                    # (B,T,E)

        # 2. Compute gate probabilities
        gate_probs = softmax(self.mp, gate_logits, axis=-1)     # (B,T,E)

        # 3. Experts forward (dense: compute all)
        expert_fc = []
        expert_gelu = []
        expert_out = []
        for e in range(self.n_experts):
            fc = self.c_proj_up[e].forward(x)                   # (B,T,exp*D)
            g  = gelu(self.mp, fc)
            o  = self.c_proj_dn[e].forward(g)                   # (B,T,D)
            expert_fc.append(fc); expert_gelu.append(g); expert_out.append(o)

        # 4. Stack: (E,B,T,D) -> (B,T,E,D)
        expert_out_stacked = self.mp.stack(expert_out, axis=0).transpose(1,2,0,3)

        # 5. Weighted sum over experts
        y = self.mp.sum(gate_probs[..., None] * expert_out_stacked, axis=2)  # (B,T,D)

        # 6. Apply dropout
        y = self.p_dropout.forward(y)

        # 7. Cache for backward
        self._cache = (x, gate_logits, gate_probs,
                       expert_fc, expert_gelu, expert_out, expert_out_stacked)
        
        return y

    def backward(self, grad_output):
        """
        grad_output: (B,T,D)
        returns: (grad_x, param_grads_list)
        """

        # 1. Unpack cached values
        x, gate_logits, gate_probs, expert_fc, expert_gelu, expert_out, expert_out_stacked = self._cache
        B, T, D = x.shape
        E = self.n_experts

        # 2. Dropout backward
        grad_y, _ = self.p_dropout.backward(grad_output)  # (B,T,D)

        # 3. Grad wrt expert outputs (before weighting): y = sum_e p_e * o_e
        # For each expert e: contribution scaled by p_e
        grad_expert_out = []  # list of (B,T,D)
        for e in range(E):
            grad_expert_out.append(grad_y * gate_probs[..., e:e+1])

        # 4. Grad wrt gate probs: dL/dp_e = dot(grad_y, o_e) over hidden dim
        # expert_out[e]: (B,T,D)
        gate_upstream = []
        for e in range(E):
            gate_upstream.append(self.mp.sum(grad_y * expert_out[e], axis=-1))  # (B,T)
        gate_upstream = self.mp.stack(gate_upstream, axis=-1)  # (B,T,E)

        # 5. Softmax backward: p = softmax(z)
        # dL/dz = p * (dL/dp - sum_e dL/dp_e * p_e)
        sum_term = self.mp.sum(gate_upstream * gate_probs, axis=-1, keepdims=True)
        grad_g_proj_logits = gate_probs * (gate_upstream - sum_term)  # (B,T,E)

        # 6. Backward gate linear
        grad_x_g_proj, gate_param_grads = self.g_proj.backward(grad_g_proj_logits)  # grad_x_g_proj: (B,T,D)

        # 7. Backward each expert (down then gelu then up)
        grad_x = grad_x_g_proj.copy()
        expert_param_grads_flat = []
        for e in range(E):
            # Down projection backward
            grad_gelu, down_grads = self.c_proj_dn[e].backward(grad_expert_out[e])

            # GELU backward
            grad_fc = grad_gelu * gelu_prime(self.mp, expert_fc[e])

            # Up projection backward
            grad_x_e, up_grads = self.c_proj_up[e].backward(grad_fc)

            grad_x += grad_x_e
            # Maintain ordering: up, down per expert
            expert_param_grads_flat.extend(up_grads)
            expert_param_grads_flat.extend(down_grads)

        # Assemble grads (must match parameters() order):
        # gate params first, then each expert's up then down
        param_grads = gate_param_grads + expert_param_grads_flat

        return grad_x, param_grads
    
    def from_dict(self, weights_dict, i):
        self.g_proj.weight = weights_dict[f'block_{i}_moe_g_weight']        
        self.n_expansion = int(weights_dict[f'block_{i}_moe_n_expansion'])
        self.n_experts = int(weights_dict[f'block_{i}_moe_n_experts'])
        for expert_idx in range(self.n_experts):
            self.c_proj_up[expert_idx].weight = weights_dict[f'block_{i}_moe_expert_{expert_idx}_up_weight']
            self.c_proj_up[expert_idx].bias = weights_dict[f'block_{i}_moe_expert_{expert_idx}_up_bias']
            self.c_proj_dn[expert_idx].weight = weights_dict[f'block_{i}_moe_expert_{expert_idx}_dn_weight']
            self.c_proj_dn[expert_idx].bias = weights_dict[f'block_{i}_moe_expert_{expert_idx}_dn_bias']

        self.g_proj.synchronize()        
        for expert_idx in range(self.n_experts):
            self.c_proj_up[expert_idx].synchronize()
            self.c_proj_dn[expert_idx].synchronize()

    def to_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_moe_g_weight'] = self.g_proj.weight        
        weights_dict[f'block_{i}_moe_n_expansion'] = self.n_expansion
        weights_dict[f'block_{i}_moe_n_experts'] = self.n_experts
        for expert_idx in range(self.n_experts):
            weights_dict[f'block_{i}_moe_expert_{expert_idx}_up_weight'] = self.c_proj_up[expert_idx].weight
            weights_dict[f'block_{i}_moe_expert_{expert_idx}_up_bias'] = self.c_proj_up[expert_idx].bias
            weights_dict[f'block_{i}_moe_expert_{expert_idx}_dn_weight'] = self.c_proj_dn[expert_idx].weight
            weights_dict[f'block_{i}_moe_expert_{expert_idx}_dn_bias'] = self.c_proj_dn[expert_idx].bias