#########################
# Mixture Head Attention (MOH)
# Author: Koureas Stavros
#########################

import math as mt
from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import softmax

class MOH(Module):
    """
    Multi-Head Attention where head outputs are mixed by a learned
    token-wise softmax gate (treat heads as MoE experts).
    Replaces concat+linear with weighted sum across heads + projection.
    Params order: q_proj, k_proj, v_proj, g_proj, m_proj
    """
    def __init__(self, mp, n_emb, n_ctx, p_dropout, head_size, n_heads):
        super().__init__()
        assert head_size % n_heads == 0, "head_size must be divisible by n_heads"
        self.mp = mp
        self.n_emb = n_emb
        self.n_ctx = n_ctx
        self.head_size = head_size
        self.n_heads = n_heads
        self.p_dropout = p_dropout

        d_k = head_size // n_heads
        self.d_k = d_k

        # Standard Q,K,V
        self.q_proj = Linear(mp, n_emb, head_size, bias=False)
        self.k_proj = Linear(mp, n_emb, head_size, bias=False)
        self.v_proj = Linear(mp, n_emb, head_size, bias=False)

        # Gating: produce logits over heads per token
        self.g_proj = Linear(mp, n_emb, n_heads, bias=False)

        # Projection after mixture (d_k -> n_emb)
        self.m_proj = Linear(mp, self.d_k, self.n_emb)

        # Dropout layers
        self.attn_dropout = Dropout(mp, p_dropout)
        self.resid_dropout = Dropout(mp, p_dropout)

        # Causal mask to prevent looking ahead in sequence (for decoder-only models)
        causal_mask = mp.triu(mp.ones((n_ctx, n_ctx)) * -1e9, k=1)
        self.causal_mask = causal_mask

        # KV cache for inference
        self.kv_cache = None  # (K, V) with shapes (B,H,T_total,d_k)

    def clear_cache(self):
        self.kv_cache = None

    def parameters(self):
        return (self.q_proj.parameters() +
                self.k_proj.parameters() +
                self.v_proj.parameters() +
                self.g_proj.parameters() +
                self.m_proj.parameters())

    def set(self, mode=True):
        super().set(mode)
        for m in (self.q_proj, self.k_proj, self.v_proj,
                  self.g_proj, self.m_proj,
                  self.attn_dropout, self.resid_dropout):
            m.set(mode)
        if mode:
            self.clear_cache()

    def forward(self, x, use_cache):
        """
        x: (B,T,n_emb)
        returns: (B,T,n_emb)
        """
        B, T, _ = x.shape

        # Projections
        Q_lin = self.q_proj.forward(x)  # (B,T,head_size)
        K_lin = self.k_proj.forward(x)
        V_lin = self.v_proj.forward(x)

        def split(z):
            return z.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)  # (B,H,T,d_k)

        Q = split(Q_lin)
        K_new = split(K_lin)
        V_new = split(V_lin)

        if use_cache and not self.setting:
            if self.kv_cache is not None:
                K_cached, V_cached = self.kv_cache
                K = self.mp.concatenate([K_cached, K_new], axis=2)
                V = self.mp.concatenate([V_cached, V_new], axis=2)
            else:
                K = K_new
                V = V_new

            # Trim KV cache to last n_ctx tokens to keep masks valid and bounded
            if K.shape[2] > self.n_ctx:
                K = K[:, :, -self.n_ctx:, :]
                V = V[:, :, -self.n_ctx:, :]
            self.kv_cache = (K, V)

            actual_seq_len = K.shape[2]
            T_q = Q.shape[2]
        else:
            K = K_new
            V = V_new
            actual_seq_len = T
            T_q = T

        # Scores with cached keys
        scores = self.mp.matmul(Q, K.transpose(0,1,3,2)) / mt.sqrt(self.d_k)  # (B,H,T_q,total_len)

        # Apply causal mask (prevents attending to future tokens)
        if use_cache and T_q == 1 and actual_seq_len > 1:
            mask = self.mp.zeros((1, actual_seq_len))
        else:
            mask = self.causal_mask[:T_q, :actual_seq_len]
            

        masked_scores = scores + mask  # broadcast (T or T_q, actual_seq_len)
        attn_weights = softmax(self.mp, masked_scores, axis=-1)  # (B,H,T_q,actual_seq_len)
        attn_weights_dropped = self.attn_dropout.forward(attn_weights)

        # Head outputs
        o = self.mp.matmul(attn_weights_dropped, V)  # (B,H,T_q,d_k)

        # Gate over heads (token-wise). Use corresponding x slice (last tokens if cache)
        x_g_proj = x if (not use_cache or self.setting or T == o.shape[2]) else x[:, -o.shape[2]:, :]
        gate_logits = self.g_proj.forward(x_g_proj)  # (B,T_q,H)
        gate_probs = softmax(self.mp, gate_logits, axis=-1)    # (B,T_q,H)

        # Rearrange heads for mixture
        o_perm = o.transpose(0,2,1,3)  # (B,T_q,H,d_k)
        # Weighted sum across heads
        y = self.mp.sum(gate_probs[..., None] * o_perm, axis=2)  # (B,T_q,d_k)

        # Project to n_emb
        m_proj_out = self.m_proj.forward(y)  # (B,T_q,n_emb)
        out = self.resid_dropout.forward(m_proj_out)

        if self.setting:
            self._cache = (x, Q_lin, K_lin, V_lin, Q, K_new, V_new,
                           scores, attn_weights, attn_weights_dropped, o,
                           gate_logits, gate_probs, y)
        return out

    def backward(self, grad_output):
        """
        grad_output: (B,T_q,n_emb)
        returns: grad_x, param_grads
        """
        (x, Q_lin, K_lin, V_lin, Q, K_new, V_new,
         scores, attn_weights, attn_weights_d, o,
         gate_logits, gate_probs, y) = self._cache

        B = x.shape[0]
        T_q = grad_output.shape[1]

        grads_all = []

        # 1. Resid dropout
        grad_m_proj_out, _ = self.resid_dropout.backward(grad_output)

        # 2. m_proj backward
        grad_y, m_proj_grads = self.m_proj.backward(grad_m_proj_out)

        # 3. Mixture y = sum_h p_h * o_h
        # Shapes: gate_probs (B,T_q,H), o (B,H,T_q,d_k) -> o_perm (B,T_q,H,d_k)
        H = self.n_heads
        d_k = self.d_k
        o_perm = o.transpose(0,2,1,3)  # (B,T_q,H,d_k)

        # grad_y: (B,T_q,d_k)
        grad_g_proj_probs = self.mp.zeros((B,T_q,H))
        grad_o_perm = self.mp.zeros_like(o_perm)

        # For each head: grad_o_h += grad_y * p_h; grad_p_h += dot(grad_y, o_h)
        # Vectorized:
        grad_o_perm = grad_y[:, :, None, :] * gate_probs[..., None]  # (B,T_q,H,d_k)
        grad_g_proj_probs = self.mp.sum(grad_y[:, :, None, :] * o_perm, axis=-1)  # (B,T_q,H)

        # 4. Softmax backward on gate logits
        sum_gp = self.mp.sum(grad_g_proj_probs * gate_probs, axis=-1, keepdims=True)  # (B,T_q,1)
        grad_g_proj_logits = gate_probs * (grad_g_proj_probs - sum_gp)  # (B,T_q,H)

        # 5. Backward g_proj
        grad_x_g_proj, g_proj_grads = self.g_proj.backward(grad_g_proj_logits)

        # 6. Backprop to o (undo transpose)
        grad_o = grad_o_perm.transpose(0,2,1,3)  # (B,H,T_q,d_k)

        # 7. Backward attention matmul: o = A @ V (A=attn_weights_d)
        grad_attn_weights_d = self.mp.matmul(grad_o, V_new.transpose(0,1,3,2))
        grad_V_new = self.mp.matmul(attn_weights_d.transpose(0,1,3,2), grad_o)

        # 8. Dropout on attn weights
        grad_attn_weights, _ = self.attn_dropout.backward(grad_attn_weights_d)

        # 9. Softmax over scores
        sum_aw = self.mp.sum(grad_attn_weights * attn_weights, axis=-1, keepdims=True)
        grad_scores = attn_weights * (grad_attn_weights - sum_aw)

        # 10. scores = (Q K^T)/sqrt(d_k)
        scale = 1.0 / mt.sqrt(d_k)
        # grad_Q: (B,H,T_q,d_k)
        # grad_K: (B,H,T_total,d_k) but in no-cache T_total = T_q
        grad_Q = self.mp.matmul(grad_scores, K_new) * scale
        grad_K_new = self.mp.matmul(grad_scores.transpose(0,1,3,2), Q) * scale

        # 11. Merge head grads back to linear input shapes
        def un_split(z, original_shape):
            # original_shape is a tuple like (B,T,head_size)
            return z.transpose(0,2,1,3).reshape(original_shape)

        grad_Q_lin = un_split(grad_Q, Q_lin.shape)
        grad_K_lin = un_split(grad_K_new, K_lin.shape)
        grad_V_lin = un_split(grad_V_new, V_lin.shape)

        # 12. Backward q,k,v projections
        grad_x_q, q_grads = self.q_proj.backward(grad_Q_lin)
        grad_x_k, k_grads = self.k_proj.backward(grad_K_lin)
        grad_x_v, v_grads = self.v_proj.backward(grad_V_lin)

        # 13. Sum gradients into x (add gate path)
        grad_x = grad_x_q + grad_x_k + grad_x_v + grad_x_g_proj

        # Assemble grads in declared parameter order
        grads_all.extend(q_grads)
        grads_all.extend(k_grads)
        grads_all.extend(v_grads)
        grads_all.extend(g_proj_grads)
        grads_all.extend(m_proj_grads)

        return grad_x, grads_all