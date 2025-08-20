#########################
# Grouped Query Attention (GQA)
# Author: Koureas Stavros
#########################

import math as mt
from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import softmax

class GQA(Module):
    """
    Grouped-Query Attention:
    - Q uses n_heads heads (head_size = n_heads * d_k)
    - K,V use n_kv_heads heads (n_kv_heads <= n_heads)
    - Each KV head is shared across group_size = n_heads // n_kv_heads query heads
    Params order: q_proj, k_proj, v_proj, c_proj
    """
    def __init__(self, mp, n_emb, n_ctx, p_dropout, head_size, n_heads, n_kv_heads=None):
        super().__init__()
        assert head_size % n_heads == 0, "head_size must be divisible by n_heads"
        self.mp = mp
        self.n_emb = n_emb
        self.n_ctx = n_ctx
        self.head_size = head_size
        self.n_heads = n_heads

        d_k = head_size // n_heads
        self.d_k = d_k

        # Default KV heads (e.g., Hq/4), at least 1
        if n_kv_heads is None:
            n_kv_heads = max(1, n_heads // 4)

        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_kv_heads = n_kv_heads       # KV heads (Hkv)
        group_size = n_heads // n_kv_heads
        self.group_size = group_size

        # Projections: Q -> (Hq*d_k), K,V -> (Hkv*d_k)
        self.q_proj = Linear(mp, n_emb, head_size, bias=False)
        self.k_proj = Linear(mp, n_emb, n_kv_heads * d_k, bias=False)
        self.v_proj = Linear(mp, n_emb, n_kv_heads * d_k, bias=False)

        # Output projection back to n_emb (from concatenated Hq heads)
        self.c_proj = Linear(mp, head_size, n_emb)

        # Dropout layers
        self.attn_dropout = Dropout(mp, p_dropout)
        self.resid_dropout = Dropout(mp, p_dropout)

        # Causal mask
        causal_mask = mp.triu(mp.ones((n_ctx, n_ctx)) * -1e9, k=1)
        self.causal_mask = causal_mask

        # KV cache for inference (stores Hkv keys/values)
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
        if mode:  # training
            self.clear_cache()

    def _split_heads_q(self, z, B, T):
        # (B,T, Hq*d_k) -> (B,Hq,T,d_k)
        return z.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _split_heads_kv(self, z, B, T):
        # Re-derive Hkv from shape to be robust after load()
        Hkv = z.shape[-1] // self.d_k
        if Hkv != self.n_kv_heads:
            # Update internal state to match loaded weights
            assert self.n_heads % Hkv == 0, "n_heads must be divisible by inferred n_kv_heads from weights"
            self.n_kv_heads = Hkv
            self.group_size = self.n_heads // self.n_kv_heads
        # (B,T, Hkv*d_k) -> (B,Hkv,T,d_k)
        return z.reshape(B, T, self.n_kv_heads, self.d_k).transpose(0, 2, 1, 3)

    def forward(self, x, use_cache):
        """
        x: (B,T,n_emb)
        returns: (B,T,n_emb)
        """
        B, T, _ = x.shape

        # Projections
        Q_lin = self.q_proj.forward(x)              # (B,T,Hq*d_k)
        K_lin = self.k_proj.forward(x)              # (B,T,Hkv*d_k)
        V_lin = self.v_proj.forward(x)

        Q = self._split_heads_q(Q_lin, B, T)        # (B,Hq,T,d_k)
        K_new = self._split_heads_kv(K_lin, B, T)   # (B,Hkv,T,d_k)
        V_new = self._split_heads_kv(V_lin, B, T)   # (B,Hkv,T,d_k)

        # KV cache (inference only)
        if use_cache and not self.setting:
            if self.kv_cache is not None:
                K_cached, V_cached = self.kv_cache
                K = self.mp.concatenate([K_cached, K_new], axis=2)  # concat along sequence
                V = self.mp.concatenate([V_cached, V_new], axis=2)
            else:
                K = K_new
                V = V_new

             # Trim KV cache to last n_ctx tokens to keep window bounded
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

        # Repeat KV heads to align with Q heads
        # (B,Hkv,*,d_k) -> (B,Hq,*,d_k)
        K_rep = self.mp.repeat(K, repeats=self.group_size, axis=1)
        V_rep = self.mp.repeat(V, repeats=self.group_size, axis=1)

        # Scaled dot-product attention
        scores = self.mp.matmul(Q, K_rep.transpose(0, 1, 3, 2)) / mt.sqrt(self.d_k)  # (B,Hq,T_q,actual_seq_len)

        # Apply causal mask (prevents attending to future tokens)
        if use_cache and T_q == 1 and actual_seq_len > 1:
            mask = self.mp.zeros((1, actual_seq_len))
        else:
            mask = self.causal_mask[:T_q, :actual_seq_len]

        masked_scores = scores + mask

        attn_weights = softmax(self.mp, masked_scores, axis=-1)                      # (B,Hq,T_q,actual_seq_len)
        attn_weights_dropped = self.attn_dropout.forward(attn_weights)
        o = self.mp.matmul(attn_weights_dropped, V_rep)                                # (B,Hq,T_q,d_k)

        # Recombine heads
        o_combined = o.transpose(0, 2, 1, 3).reshape(B, T_q, self.head_size)  # (B,T_q,Hq*d_k)
        out = self.c_proj.forward(o_combined)
        out = self.resid_dropout.forward(out)

        # Cache only for training/backward
        if self.setting:
            self._cache = (x, Q_lin, K_lin, V_lin, Q, K_new, V_new, self.group_size,
                           scores, masked_scores, attn_weights, attn_weights_dropped, o, o_combined)
        return out

    def backward(self, grad_output):
        """
        grad_output: (B,T_q,n_emb)
        returns: grad_x, [param_grads]
        """
        (x, Q_lin, K_lin, V_lin, Q, K_new, V_new, group_size,
         scores, masked_scores, attn_weights, attn_weights_d, o, o_combined) = self._cache

        grads_all = []

        # 1. resid dropout
        grad_c_proj_in, _ = self.resid_dropout.backward(grad_output)

        # 2. c_proj backward
        grad_o_combined, c_proj_grads = self.c_proj.backward(grad_c_proj_in)

        # 3. Uncombine heads
        B, T_q, _ = grad_o_combined.shape
        grad_o = grad_o_combined.reshape(B, T_q, self.n_heads, self.d_k).transpose(0, 2, 1, 3)  # (B,Hq,T_q,d_k)

        # Recreate repeated V for backward
        V_rep = self.mp.repeat(V_new, repeats=group_size, axis=1)  # (B,Hq,*,d_k)

        # 4. o = attn_weights_d @ V_rep
        grad_attn_weights_d = self.mp.matmul(grad_o, V_rep.transpose(0, 1, 3, 2))      # (B,Hq,T_q,actual_seq_len)
        grad_V_rep = self.mp.matmul(attn_weights_d.transpose(0, 1, 3, 2), grad_o)      # (B,Hq,actual_seq_len,d_k)

        # Collapse repeats back to Hkv for V
        # (B,Hq,*,d_k) -> (B,Hkv,*,d_k) by summing groups
        Hkv = K_new.shape[1]
        grad_V_rep_grouped = grad_V_rep.reshape(B, Hkv, group_size, grad_V_rep.shape[2], self.d_k).sum(axis=2)
        grad_V_new = grad_V_rep_grouped  # (B,Hkv,*,d_k)

        # 5. Dropout on attn weights
        grad_attn_weights, _ = self.attn_dropout.backward(grad_attn_weights_d)

        # 6. softmax backward
        sum_term = self.mp.sum(grad_attn_weights * attn_weights, axis=-1, keepdims=True)
        grad_scores = attn_weights * (grad_attn_weights - sum_term)

        # 7. scores = (Q @ K_rep^T) / sqrt(d_k)
        K_rep = self.mp.repeat(K_new, repeats=group_size, axis=1)
        scale = 1.0 / mt.sqrt(self.d_k)
        grad_Q = self.mp.matmul(grad_scores, K_rep) * scale                              # (B,Hq,T_q,d_k)
        grad_K_rep = self.mp.matmul(grad_scores.transpose(0, 1, 3, 2), Q) * scale       # (B,Hq,*,d_k)

        # Collapse repeats back to Hkv for K
        grad_K_rep_grouped = grad_K_rep.reshape(B, Hkv, group_size, grad_K_rep.shape[2], self.d_k).sum(axis=2)
        grad_K_new = grad_K_rep_grouped  # (B,Hkv,*,d_k)

        # 8. Merge head grads back to linear input shapes
        def un_split_q(z_grad, original_shape):
            # (B,Hq,T_q,d_k) -> (B,T_q,Hq*d_k)
            return z_grad.transpose(0, 2, 1, 3).reshape(original_shape)

        def un_split_kv(z_grad, original_shape):
            # (B,Hkv,T_kv,d_k) -> (B,T_kv,Hkv*d_k)
            return z_grad.transpose(0, 2, 1, 3).reshape(original_shape)

        grad_Q_lin = un_split_q(grad_Q, Q_lin.shape)
        grad_K_lin = un_split_kv(grad_K_new, K_lin.shape)
        grad_V_lin = un_split_kv(grad_V_new, V_lin.shape)

        # 9. Backward q,k,v projections
        grad_x_q, q_proj_grads = self.q_proj.backward(grad_Q_lin)
        grad_x_k, k_proj_grads = self.k_proj.backward(grad_K_lin)
        grad_x_v, v_proj_grads = self.v_proj.backward(grad_V_lin)

        # 10. Sum gradients for input x
        grad_x = grad_x_q + grad_x_k + grad_x_v

        # Assemble grads in declared parameter order
        grads_all.extend(q_proj_grads)
        grads_all.extend(k_proj_grads)
        grads_all.extend(v_proj_grads)
        grads_all.extend(c_proj_grads)

        return grad_x, grads_all