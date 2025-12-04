#########################
# Network Free Transformer (NFT)
# Author: Koureas Stavros
#########################

from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import sigmoid, sigmoid_prime

class NFT(Module):
    """
    Network Free Transformer (NFT)
    """
    def __init__(self, mp, d_type, n_ctx, n_emb, r_dropout, use_gate, clip):
        super().__init__()
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.r_dropout = r_dropout
        self.use_gate = bool(use_gate)
        self.clip = float(clip)

        # Projections
        self.q_proj = Linear(mp, d_type, n_emb, n_emb, bias=False) # used only if use_gate=True
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
        for m in (self.q_proj, self.k_proj, self.v_proj, self.c_proj,
                  self.attn_dropout, self.resid_dropout):
            m.set(mode)
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
        Estimate FLOPs for the NFT forward pass.
        Multiply-adds are counted as 2 FLOPs.
        batch_size: number of sequences in the batch
        training: if True, include backward/update cost (~3x forward)
        """
        def linear_flops(in_f, out_f):
            return 2 * batch_size * self.n_ctx * in_f * out_f

        flops = 0

        # q_proj only if gating is used
        if self.use_gate:
            flops += linear_flops(self.n_emb, self.n_emb)

        # k_proj and v_proj
        flops += linear_flops(self.n_emb, self.n_emb)  # k_proj
        flops += linear_flops(self.n_emb, self.n_emb)  # v_proj

        # exp/clip and elementwise mul for E and EV (~2 FLOPs per element each)
        flops += 2 * batch_size * self.n_ctx * self.n_emb  # exp/clip
        flops += 2 * batch_size * self.n_ctx * self.n_emb  # E * V

        # cumsum for S and SV (~1 FLOP per element each)
        flops += batch_size * self.n_ctx * self.n_emb * 2

        # division for s = SV / (S + eps)
        flops += batch_size * self.n_ctx * self.n_emb

        # gating multiply if used
        if self.use_gate:
            flops += batch_size * self.n_ctx * self.n_emb

        # c_proj
        flops += linear_flops(self.n_emb, self.n_emb)

        if training:
            flops *= 3  # forward + backward + update

        return flops

    def forward_exp_clip(self, x):
        return self.mp.exp(self.mp.clip(x, -self.clip, self.clip))
    
    def backward_exp_clip(self, grad, x):
        grad_x = grad * self.mp.exp(self.mp.clip(x, -self.clip, self.clip))
        grad_x *= (x >= -self.clip) * (x <= self.clip)  # Zero out gradients outside the range
        return grad_x

    def forward_cumsum(self, x):
        return self.mp.cumsum(x, axis=1)
    
    def backward_cumsum(self,g):
        return self.mp.cumsum(g[:, ::-1, :], axis=1)[:, ::-1, :]
    
    def forward(self, x, use_cache):
        """
        x: (B,T,D)
        returns: (B,T_q,D)
        """
        B, T, D = x.shape

        # 1. Projections
        if self.use_gate:
            q_lin = self.q_proj.forward(x)          # (B,T,D)
            gate = sigmoid(self.mp, q_lin)          # (B,T,D)
        else:
            q_lin = None
            gate = 1.0

        k_lin = self.k_proj.forward(x)      # (B,T,D)
        v_lin = self.v_proj.forward(x)      # (B,T,D)

        eps = 1e-9

        # Handle KV cache for inference
        if use_cache and not self.setting:
            # Streaming inference with running sums, sliding window
            from collections import deque
            if self.kv_cache is None:
                S = self.mp.zeros((B, D), dtype=self.accum_dtype)
                SV = self.mp.zeros((B, D), dtype=self.accum_dtype)
                E_fifo = deque()
                EV_fifo = deque()
            else:
                S = self.kv_cache['S']
                SV = self.kv_cache['SV']
                E_fifo = self.kv_cache['E_fifo']
                EV_fifo = self.kv_cache['EV_fifo']

            y_list = []
            for t in range(T):
                k_t = k_lin[:, t, :]  # (B,D)
                v_t = v_lin[:, t, :]
                e_t = self.forward_exp_clip(k_t)           # (B,D)
                ev_t = e_t * v_t                    # (B,D)

                # Maintain sliding window up to n_ctx
                if self.n_ctx is not None and len(E_fifo) >= self.n_ctx:
                    oldest_e = E_fifo.popleft()
                    oldest_ev = EV_fifo.popleft()
                    S -= oldest_e
                    SV -= oldest_ev

                E_fifo.append(e_t)
                EV_fifo.append(ev_t)
                S += e_t
                SV += ev_t

                s_t = SV / (S + eps)                    # (B,D)
                if self.use_gate:
                    g_t = gate[:, t, :]
                    y_t = g_t * s_t
                else:
                    y_t = s_t

                y_list.append(y_t[:, None, :])

            y = self.mp.concatenate(y_list, axis=1)     # (B,T,D)
            out = self.c_proj.forward(y)
            out = self.resid_dropout.forward(out)

            self.kv_cache = {'S': S, 'SV': SV, 'E_fifo': E_fifo, 'EV_fifo': EV_fifo}
            return out

        # 2. Training: fully vectorized prefix sums
        E = self.forward_exp_clip(k_lin)                # (B,T,D)

        # 3. Cumulative Sums
        EV = (E * v_lin)                                # (B,T,D)
        S = self.forward_cumsum(E)                      # (B,T,D)
        SV = self.forward_cumsum(EV)                    # (B,T,D)

        # 4. Compute attention scores
        s = (SV / (S + eps))                            # (B,T,D)

        # 5. Dropout on s
        s_d = self.attn_dropout.forward(s)

        # 6. Apply gating
        if self.use_gate:
            y = gate * s_d
        else:
            y = s_d

        # 7. Final Projection
        out = self.c_proj.forward(y)

        # 8. Residual Dropout
        out = self.resid_dropout.forward(out)

        # 9. Cache for backward
        self._cache = (x, q_lin, k_lin, v_lin, E, S, EV, SV, s, s_d, gate)

        return out

    def backward(self, grad_output):
        """
        grad_output: (B,T,D)
        returns: grad_x, [param_grads in order q,k,v,c]
        """

        # 1. Unpack cached values
        (x, q_lin, k_lin, v_lin, E, S, EV, SV, s, s_d, gate) = self._cache
        B, T, D = x.shape

        # 2. Backward through residual dropout
        grad_c_in, _ = self.resid_dropout.backward(grad_output)  # (B,T,D)

        # 3. Backward through final projection
        grad_y, c_grads = self.c_proj.backward(grad_c_in)        # (B,T,D)

        # 4. Split path: y = gate * s_d (gate=1 if disabled)
        if self.use_gate:
            grad_gate = grad_y * s_d                      # (B,T,D)
            grad_s_d = grad_y * gate                      # (B,T,D)
            sig = gate
            grad_q_lin = grad_gate * sigmoid_prime(self.mp, sig)    # (B,T,D)
        else:
            grad_s_d = grad_y
            grad_q_lin = self.mp.zeros_like(grad_y)

        # 5. Backward through dropout on s
        grad_s, _ = self.attn_dropout.backward(grad_s_d)  # (B,T,D)

        # 6. s = SV / (S + eps)
        eps = 1e-9
        S_eps = S + eps
        grad_SV = grad_s / S_eps                          # (B,T,D)
        grad_S = -grad_s * (s / S_eps)                    # (B,T,D)

        # 7. Backward through reverse cumsum to distribute gradients to EV and E
        G_SV = self.backward_cumsum(grad_SV)   # grads wrt EV
        G_S = self.backward_cumsum(grad_S)     # grads wrt E

        grad_V_lin = E * G_SV                                # (B,T,D)
        grad_E_fromSV = G_SV * v_lin                         # (B,T,D)
        grad_E = grad_E_fromSV + G_S                         # (B,T,D)

        # 8. Backward through E = exp(clip(k_lin)) -> use backward_exp_clip
        grad_k_lin = self.backward_exp_clip(grad_E, k_lin)   # (B,T,D)

        # 9 Back through projections
        grad_x_k, k_grads = self.k_proj.backward(grad_k_lin)
        grad_x_v, v_grads = self.v_proj.backward(grad_V_lin)

        if self.use_gate:
            grad_x_q, q_grads = self.q_proj.backward(grad_q_lin)
        else:
            # Keep parameter order; produce zero grads for q
            q_grads = [self.mp.zeros_like(self.q_proj.weight)]
            grad_x_q = self.mp.zeros_like(grad_x_k)

        # Assemble grad
        grad_x = grad_x_k + grad_x_v + grad_x_q

        # Assemble grads in order: q, k, v, c
        param_grads = []
        param_grads.extend(q_grads)
        param_grads.extend(k_grads)
        param_grads.extend(v_grads)
        param_grads.extend(c_grads)

        return grad_x, param_grads
    
    def from_dict(self, weights_dict, i):
        self.q_proj.weight = weights_dict[f'block_{i}_nft_q_weight']
        self.k_proj.weight = weights_dict[f'block_{i}_nft_k_weight']
        self.v_proj.weight = weights_dict[f'block_{i}_nft_v_weight']
        self.c_proj.weight = weights_dict[f'block_{i}_nft_c_weight']
        self.c_proj.bias = weights_dict[f'block_{i}_nft_c_bias']

        self.q_proj.synchronize()
        self.k_proj.synchronize()
        self.v_proj.synchronize()
        self.c_proj.synchronize()

    def towa_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_nft_q_weight'] = self.q_proj.weight
        weights_dict[f'block_{i}_nft_k_weight'] = self.k_proj.weight
        weights_dict[f'block_{i}_nft_v_weight'] = self.v_proj.weight
        weights_dict[f'block_{i}_nft_c_weight'] = self.c_proj.weight
        weights_dict[f'block_{i}_nft_c_bias'] = self.c_proj.bias