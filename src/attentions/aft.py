#########################
# Attention Free Transformer (AFT)
# Author: Koureas Stavros
#########################

from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout

class AFT(Module):
    """
    Attention Free Transformer (AFT-simple, causal) in O(T·D).
    Option A: No Q (gate=1), keep K/V projections.
    - Training: s = cumsum(exp(K) * V) / cumsum(exp(K))
    - Inference: running sums S, SV with sliding window up to n_ctx
    Params order (unchanged): q_proj, k_proj, v_proj, c_proj
    """
    def __init__(self, mp, n_ctx, n_emb, p_dropout):
        super().__init__()
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb

        # Projections
        self.q_proj = Linear(mp, n_emb, n_emb, bias=False)  # kept for parameter order, unused in forward
        self.k_proj = Linear(mp, n_emb, n_emb, bias=False)
        self.v_proj = Linear(mp, n_emb, n_emb, bias=False)
        self.c_proj = Linear(mp, n_emb, n_emb, bias=True)

        # Dropout layers
        self.attn_dropout = Dropout(mp, p_dropout)
        self.resid_dropout = Dropout(mp, p_dropout)

        # KV cache for inference
        self.kv_cache = None
        
    def clear_cache(self):
        self.kv_cache = None

    def parameters(self):
        return (self.q_proj.parameters() +
                self.k_proj.parameters() +
                self.v_proj.parameters() +
                self.c_proj.parameters())

    def flops(self, batch_size, training):
        """
        Estimate FLOPs for this AFT layer.
        Since AFT avoids QK^T, complexity is O(B·T·D).
        Includes K/V projections, elementwise exp/mul/div, and output projection.
        batch_size: number of sequences in the batch
        training: if True, include backward/update cost (~3x forward)
        """
        flops = 0

        # Q projection (kept for parameter order, unused in forward)
        flops += batch_size * self.n_ctx * self.n_emb * self.n_emb * 2

        # K and V projections: (B, T, D) x (D, D)
        flops += 2 * batch_size * self.n_ctx * self.n_emb * self.n_emb * 2

        # Elementwise exp on K
        flops += batch_size * self.n_ctx * self.n_emb

        # Elementwise multiply E * V
        flops += batch_size * self.n_ctx * self.n_emb

        # Cumulative sums S and SV
        flops += 2 * batch_size * self.n_ctx * self.n_emb

        # Elementwise division SV / S
        flops += batch_size * self.n_ctx * self.n_emb

        # Output projection: (B, T, D) x (D, D)
        flops += batch_size * self.n_ctx * self.n_emb * self.n_emb * 2

        # Bias add for output projection
        if self.c_proj.bias is not None:
            flops += batch_size * self.n_ctx * self.n_emb

        # Dropout (approximate)
        flops += batch_size * self.n_ctx * self.n_emb

        if training:
            flops *= 3  # forward + backward + update

        return flops
    
    def set(self, mode=True):
        super().set(mode)
        for m in (self.q_proj, self.k_proj, self.v_proj, self.c_proj,
                  self.attn_dropout, self.resid_dropout):
            m.set(mode)
        if mode:
            self.clear_cache()

    def _exp_clip(self, x):
        return self.mp.exp(self.mp.clip(x, -20.0, 20.0))

    def forward(self, x, use_cache):
        """
        x: (B,T,D)
        returns: (B,T_q,D)
        """
        B, T, D = x.shape

        # Only K and V projections are used
        k_lin = self.k_proj.forward(x)  # (B,T,D)
        v_lin = self.v_proj.forward(x)  # (B,T,D)

        if use_cache and not self.setting:
            # Inference with running sums and sliding window
            if self.kv_cache is None:
                S = self.mp.zeros((B, D), dtype=k_lin.dtype)
                SV = self.mp.zeros((B, D), dtype=k_lin.dtype)
                E_fifo = []
                EV_fifo = []
            else:
                S = self.kv_cache['S']
                SV = self.kv_cache['SV']
                E_fifo = self.kv_cache['E_fifo']
                EV_fifo = self.kv_cache['EV_fifo']

            eps = 1e-9
            y_list = []
            for t in range(T):
                k_t = k_lin[:, t, :]
                v_t = v_lin[:, t, :]
                e_t = self._exp_clip(k_t)
                ev_t = e_t * v_t

                # Sliding window of length n_ctx
                if self.n_ctx is not None and len(E_fifo) >= self.n_ctx:
                    oldest_e = E_fifo.pop(0)
                    oldest_ev = EV_fifo.pop(0)
                    S -= oldest_e
                    SV -= oldest_ev

                E_fifo.append(e_t)
                EV_fifo.append(ev_t)
                S = S + e_t
                SV = SV + ev_t

                s_t = SV / (S + eps)
                y_t = s_t  # gate=1
                y_list.append(y_t[:, None, :])

            y = self.mp.concatenate(y_list, axis=1)      # (B,T_q,D)
            out = self.c_proj.forward(y)
            out = self.resid_dropout.forward(out)

            self.kv_cache = {'S': S, 'SV': SV, 'E_fifo': E_fifo, 'EV_fifo': EV_fifo}
            return out

        # Training: prefix sums
        eps = 1e-9
        E = self._exp_clip(k_lin)            # (B,T,D)
        S = self.mp.cumsum(E, axis=1)             # (B,T,D)
        EV = E * v_lin                       # (B,T,D)
        SV = self.mp.cumsum(EV, axis=1)           # (B,T,D)
        s = SV / (S + eps)                   # (B,T,D)

        s_d = self.attn_dropout.forward(s)
        c_in = s_d                           # gate=1

        out = self.c_proj.forward(c_in)
        out = self.resid_dropout.forward(out)

        # Cache for backward
        self._cache = (x, k_lin, v_lin, E, S, EV, SV, s, s_d)
        return out

    def backward(self, grad_output):
        """
        grad_output: (B,T,D)
        returns: grad_x, [param_grads in order q,k,v,c]
        """
        (x, k_lin, v_lin, E, S, EV, SV, s, s_d) = self._cache
        B, T, D = x.shape

        # 1) Residual dropout backward
        grad_c_out, _ = self.resid_dropout.backward(grad_output)

        # 2) c_proj backward
        grad_c_in, c_grads = self.c_proj.backward(grad_c_out)  # (B,T,D)

        # 3) Since c_in = s_d (gate=1), grad_s_d = grad_c_in
        grad_s_d = grad_c_in

        # 4) Dropout backward on s
        grad_s, _ = self.attn_dropout.backward(grad_s_d)

        # 5) s = SV / (S + eps)
        eps = 1e-9
        S_eps = S + eps
        grad_SV = grad_s / S_eps
        grad_S = -grad_s * (s / S_eps)

        # 6) Backprop through cumulative sums (reverse cumsum)
        def reverse_cumsum(g):
            return self.mp.cumsum(g[:, ::-1, :], axis=1)[:, ::-1, :]

        G_SV = reverse_cumsum(grad_SV)   # grads wrt EV
        G_S = reverse_cumsum(grad_S)     # grads wrt E

        # 7) Split EV and E
        grad_V = E * G_SV
        grad_E_fromSV = G_SV * v_lin
        grad_E = grad_E_fromSV + G_S

        # 8) E = exp(k_lin) -> grad_k_lin = grad_E * E
        grad_k_lin = grad_E * E

        # 9) Back through k_proj, v_proj (no q_proj path)
        grad_x_k, k_grads = self.k_proj.backward(grad_k_lin)
        grad_x_v, v_grads = self.v_proj.backward(grad_V)

        grad_x = grad_x_k + grad_x_v

        # 10) Assemble param grads in order: q, k, v, c
        # q grads are zeros (q is unused in forward)
        q_weight = self.q_proj.weight
        q_grads = [self.mp.zeros_like(q_weight)]

        param_grads = []
        param_grads.extend(q_grads)
        param_grads.extend(k_grads)
        param_grads.extend(v_grads)
        param_grads.extend(c_grads)

        return grad_x, param_grads
    
    def from_dict(self, weights_dict, i):
        self.q_proj.weight = weights_dict[f'block_{i}_aft_q_weight']
        self.k_proj.weight = weights_dict[f'block_{i}_aft_k_weight']
        self.v_proj.weight = weights_dict[f'block_{i}_aft_v_weight']
        self.c_proj.weight = weights_dict[f'block_{i}_aft_c_weight']
        self.c_proj.bias = weights_dict[f'block_{i}_aft_c_bias']

        self.q_proj.synchronize()
        self.k_proj.synchronize()
        self.v_proj.synchronize()
        self.c_proj.synchronize()

    def to_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_aft_q_weight'] = self.q_proj.weight
        weights_dict[f'block_{i}_aft_k_weight'] = self.k_proj.weight
        weights_dict[f'block_{i}_aft_v_weight'] = self.v_proj.weight
        weights_dict[f'block_{i}_aft_c_weight'] = self.c_proj.weight
        weights_dict[f'block_{i}_aft_c_bias'] = self.c_proj.bias