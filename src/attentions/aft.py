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
    def __init__(self, mp, d_type, n_ctx, n_emb, r_dropout, r_temp, r_clip):
        super().__init__()
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.r_dropout = r_dropout
        self.r_temp = r_temp
        self.r_clip = r_clip

        # Projections
        self.q_proj = Linear(mp, d_type, n_emb, n_emb, bias=False)  # kept for parameter order, unused in forward
        self.k_proj = Linear(mp, d_type, n_emb, n_emb, bias=False)
        self.v_proj = Linear(mp, d_type, n_emb, n_emb, bias=False)
        self.c_proj = Linear(mp, d_type, n_emb, n_emb, bias=True)

        # Dropout layers
        self.attn_dropout = Dropout(mp, r_dropout)
        self.resid_dropout = Dropout(mp, r_dropout)

        # KV cache for inference
        self.kv_cache = None
    
    def set(self, mode=True):
        super().set(mode)
        self.q_proj.set(mode)
        self.k_proj.set(mode)
        self.v_proj.set(mode)
        self.c_proj.set(mode)
        self.attn_dropout.set(mode)
        self.resid_dropout.set(mode)
        
        if mode:
            self.clear_cache()

    def parameters(self):
        return (self.q_proj.parameters() +
                self.k_proj.parameters() +
                self.v_proj.parameters() +
                self.c_proj.parameters())
        
    def clear_cache(self):
        self.kv_cache = None

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

    def forward_exp_clip(self, x):
        return self.mp.exp(self.mp.clip(x, -self.r_clip, self.r_clip))

    def backward_exp_clip(self, grad, x):
        # Gradient is grad_E * exp(k_lin) within the clipping range, 0 otherwise
        grad_x = grad * self.mp.exp(self.mp.clip(x, -self.r_clip, self.r_clip))
        grad_x *= (x >= -self.r_clip) * (x <= self.r_clip)  # Zero out gradients outside the range
        return grad_x

    def forward_cumsum(self, x):
        return self.mp.cumsum(x, axis=1)
    
    def backward_cumsum(self, g):
        return self.mp.cumsum(g[:, ::-1, :], axis=1)[:, ::-1, :]
    
    def forward(self, x, use_cache):
        """
        x: (B,T,D)
        returns: (B,T_q,D)
        """
        B, T, D = x.shape

        # 1. Project input to K, V
        # Only K and V projections are used
        k_lin = self.k_proj.forward(x)  # (B,T,D)
        v_lin = self.v_proj.forward(x)  # (B,T,D)

        # Apply temperature scaling to k_lin before exp/clip
        k_lin_scaled = k_lin / self.r_temp

        # Handle KV cache for inference
        if use_cache and not self.setting:
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
                e_t = self.forward_exp_clip(k_t)
                ev_t = e_t * v_t

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

            # Concatenate results and apply output projection
            y = self.mp.concatenate(y_list, axis=1)      # (B,T_q,D)

            # Apply output projection and residual dropout
            out = self.c_proj.forward(y)

            # Apply residual dropout
            out = self.resid_dropout.forward(out)

            # Update KV cache
            self.kv_cache = {'S': S, 'SV': SV, 'E_fifo': E_fifo, 'EV_fifo': EV_fifo}
            return out

        # 3. Compute E = exp(K) and prefix sums S
        eps = 1e-9
        E = self.forward_exp_clip(k_lin_scaled)     # (B,T,D)

        # 4. Compute cumulative sums
        S = self.forward_cumsum(E)           # (B,T,D)

        # 5. Compute E * V
        EV = E * v_lin                       # (B,T,D)
        SV = self.forward_cumsum(EV)         # (B,T,D)

        # 6. Compute prefix sums for V
        s = SV / (S + eps)                   # (B,T,D)

        # 7. Dropout on s
        s_d = self.attn_dropout.forward(s)

        # 8. Residual connection
        c_in = s_d                           # gate=1

        # 9. Final linear projection
        c_proj_out = self.c_proj.forward(c_in)

        # 10. Residual dropout
        dropped_out = self.resid_dropout.forward(c_proj_out)

        # 11. Cache intermediate values for backward pass
        if self.setting:
            self._cache = (x, k_lin, v_lin, E, S, EV, SV, s, s_d)
        
        return dropped_out

    def backward(self, grad_output):
        """
        grad_output: (B,T,D)
        returns: grad_x, [param_grads in order q,k,v,c]
        """

        # 1. Unpack cached values
        (x, k_lin, v_lin, E, S, EV, SV, s, s_d) = self._cache
        B, T, D = x.shape

        # 2. Backward through residual dropout
        grad_c_out, _ = self.resid_dropout.backward(grad_output)

        # 3. Backward through final linear projection
        grad_c_in, c_grads = self.c_proj.backward(grad_c_out)  # (B,T,D)

        # 4. Backward through c_in = s_d (gate=1), grad_s_d = grad_c_in
        grad_s_d = grad_c_in

        # 5. Backward through dropout on s
        grad_s, _ = self.attn_dropout.backward(grad_s_d)

        # 6. Backward through s = SV / (S + eps)
        eps = 1e-9
        S_eps = S + eps

        # 7. Backward through compute gradients
        grad_SV = grad_s / S_eps
        grad_S = -grad_s * (s / S_eps)

        # 8. Backward through cumulative sums (reverse cumsum)
        G_SV = self.backward_cumsum(grad_SV)   # grads wrt EV
        grad_V = E * G_SV
        grad_E_fromSV = G_SV * v_lin

        # 9. Backward through cumulative sums (reverse cumsum)
        G_S = self.backward_cumsum(grad_S)     # grads wrt E
        grad_E = grad_E_fromSV + G_S

        # 10. Backward through E = exp(k_lin) -> grad_k_lin = grad_E * E
        grad_k_lin = self.backward_exp_clip(grad_E, k_lin)

        # 11. Backward through k_proj, v_proj (no q_proj path)
        grad_x_k, k_grads = self.k_proj.backward(grad_k_lin)
        grad_x_v, v_grads = self.v_proj.backward(grad_V)

        # Combine gradients
        grad_x = grad_x_k + grad_x_v

        # Assemble param grads in order: q, k, v, c
        # q grads are zeros (q is unused in forward)
        q_weight = self.q_proj.weight
        q_grads = [self.mp.zeros_like(q_weight)]

        # Assemble grads in declared parameter order
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

    def towa_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_aft_q_weight'] = self.q_proj.weight
        weights_dict[f'block_{i}_aft_k_weight'] = self.k_proj.weight
        weights_dict[f'block_{i}_aft_v_weight'] = self.v_proj.weight
        weights_dict[f'block_{i}_aft_c_weight'] = self.c_proj.weight
        weights_dict[f'block_{i}_aft_c_bias'] = self.c_proj.bias