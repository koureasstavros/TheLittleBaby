#########################
# Network Free Transformer (NFT)
# Author: Koureas Stavros
#########################

from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout

class NFT(Module):
    """
    Network Free Transformer (NFT-simple, causal) used as the feed-forward network.
    Training (vectorized):
        E  = exp(clip(K))
        S  = cumsum(E)
        SV = cumsum(E * V)
        s  = SV / (S + eps)
        y  = gate(x) * dropout(s)    # gate(x) = sigmoid(q_proj(x)) if use_gate else 1
        out = resid_dropout(c_proj(y))

    Inference (streaming):
        Maintain running S, SV and a FIFO of last n_ctx terms to support a sliding window.

    Params order preserved: q_proj, k_proj, v_proj, c_proj
    """
    def __init__(self, mp, n_emb, n_ctx, p_dropout, use_gate=True, clip=20.0):
        super().__init__()
        self.mp = mp
        self.n_emb = n_emb
        self.n_ctx = n_ctx
        self.use_gate = bool(use_gate)
        self.clip = float(clip)

        # Projections
        self.q_proj = Linear(mp, n_emb, n_emb, bias=False) # used only if use_gate=True
        self.k_proj = Linear(mp, n_emb, n_emb, bias=False)
        self.v_proj = Linear(mp, n_emb, n_emb, bias=False)
        self.c_proj = Linear(mp, n_emb, n_emb)

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

    def set(self, mode=True):
        super().set(mode)
        for m in (self.q_proj, self.k_proj, self.v_proj, self.c_proj,
                  self.attn_dropout, self.resid_dropout):
            m.set(mode)
        if mode:
            self.clear_cache()

    def _exp_clip(self, x):
        return self.mp.exp(self.mp.clip(x, -self.clip, self.clip))

    def _sigmoid(self, x):
        return 1.0 / (1.0 + self.mp.exp(-self.mp.clip(x, -40.0, 40.0)))

    def forward(self, x, use_cache):
        """
        x: (B,T,D)
        returns: (B,T_q,D)
        """
        B, T, D = x.shape

        # Projections
        if self.use_gate:
            q_lin = self.q_proj.forward(x)  # (B,T,D)
            gate = self._sigmoid(q_lin)     # (B,T,D)
        else:
            q_lin = None
            gate = 1.0

        k_lin = self.k_proj.forward(x)      # (B,T,D)
        v_lin = self.v_proj.forward(x)      # (B,T,D)

        eps = 1e-9

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
                e_t = self._exp_clip(k_t)           # (B,D)
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

        # Training: fully vectorized prefix sums
        E = self._exp_clip(k_lin)                       # (B,T,D)
        EV = (E * v_lin)                                # (B,T,D)
        S = self.mp.cumsum(E, axis=1)                   # (B,T,D)
        SV = self.mp.cumsum(EV, axis=1)                 # (B,T,D)
        s = (SV / (S + eps))                            # (B,T,D)

        s_d = self.attn_dropout.forward(s)
        if self.use_gate:
            y = gate * s_d
        else:
            y = s_d

        out = self.c_proj.forward(y)
        out = self.resid_dropout.forward(out)

        # Cache for backward
        self._cache = (x, q_lin, k_lin, v_lin, E, S, EV, SV, s, s_d, gate)
        return out

    def backward(self, grad_output):
        """
        grad_output: (B,T,D)
        returns: grad_x, [param_grads in order q,k,v,c]
        """
        (x, q_lin, k_lin, v_lin, E, S, EV, SV, s, s_d, gate) = self._cache
        B, T, D = x.shape

        # 1) Residual dropout backward
        grad_c_in, _ = self.resid_dropout.backward(grad_output)  # (B,T,D)

        # 2) c_proj backward
        grad_y, c_grads = self.c_proj.backward(grad_c_in)        # (B,T,D)

        # 3) Split path: y = gate * s_d (gate=1 if disabled)
        if self.use_gate:
            grad_gate = grad_y * s_d                      # (B,T,D)
            grad_s_d = grad_y * gate                      # (B,T,D)
            # gate = sigmoid(q_lin)
            sig = gate
            grad_q_lin = grad_gate * sig * (1.0 - sig)    # (B,T,D)
        else:
            grad_s_d = grad_y
            grad_q_lin = self.mp.zeros_like(grad_y)

        # 4) Dropout backward on s
        grad_s, _ = self.attn_dropout.backward(grad_s_d)  # (B,T,D)

        # 5) s = SV / (S + eps)
        eps = 1e-9
        S_eps = S + eps
        grad_SV = grad_s / S_eps                          # (B,T,D)
        grad_S = -grad_s * (s / S_eps)                    # (B,T,D)

        # 6) Reverse cumsum to distribute gradients to EV and E
        def reverse_cumsum(g):
            return self.mp.cumsum(g[:, ::-1, :], axis=1)[:, ::-1, :]

        G_SV = reverse_cumsum(grad_SV)   # grads wrt EV
        G_S = reverse_cumsum(grad_S)     # grads wrt E

        # 7) Split EV and E
        grad_V_lin = E * G_SV                                # (B,T,D)
        grad_E_fromSV = G_SV * v_lin                         # (B,T,D)
        grad_E = grad_E_fromSV + G_S                         # (B,T,D)

        # 8) E = exp(k_lin) -> grad_k_lin = grad_E * E
        grad_k_lin = grad_E * E                              # (B,T,D)

        # 9) Back through projections
        grad_x_k, k_grads = self.k_proj.backward(grad_k_lin)
        grad_x_v, v_grads = self.v_proj.backward(grad_V_lin)

        if self.use_gate:
            grad_x_q, q_grads = self.q_proj.backward(grad_q_lin)
        else:
            # Keep parameter order; produce zero grads for q
            q_grads = [self.mp.zeros_like(self.q_proj.weight)]
            grad_x_q = self.mp.zeros_like(grad_x_k)

        grad_x = grad_x_k + grad_x_v + grad_x_q

        # Assemble grads in order: q, k, v, c
        param_grads = []
        param_grads.extend(q_grads)
        param_grads.extend(k_grads)
        param_grads.extend(v_grads)
        param_grads.extend(c_grads)
        return grad_x, param_grads