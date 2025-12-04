#########################
# Grouped Query Attention (GQA)
# Author: Koureas Stavros
#########################

import math as mt
from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import split_heads, merge_heads, softmax, softmax_prime

class GQA(Module):
    """
    Grouped-Query Attention (GQA)
    """
    def __init__(self, mp, d_type, n_ctx, n_emb, r_dropout, r_temp, s_head, n_heads, n_kv_heads):
        super().__init__()
        assert s_head % n_heads == 0, "head_size must be divisible by n_heads"
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.r_dropout = r_dropout
        self.r_temp = r_temp
        self.s_head = s_head
        self.n_heads = n_heads

        d_k = s_head // n_heads
        self.d_k = d_k

        # Default KV heads (e.g., Hq/4), at least 1
        if n_kv_heads is None:
            n_kv_heads = max(1, n_heads // 4)

        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_kv_heads = n_kv_heads       # KV heads (Hkv)
        group_size = n_heads // n_kv_heads
        self.group_size = group_size

        # Projections: Q -> (Hq*d_k), K,V -> (Hkv*d_k)
        self.q_proj = Linear(mp, d_type, n_emb, s_head, bias=False)
        self.k_proj = Linear(mp, d_type, n_emb, n_kv_heads * d_k, bias=False)
        self.v_proj = Linear(mp, d_type, n_emb, n_kv_heads * d_k, bias=False)

        # Output projection back to n_emb (from concatenated Hq heads)
        self.c_proj = Linear(mp, d_type, s_head, n_emb, bias=True)

        # Dropout layers
        self.attn_dropout = Dropout(mp, r_dropout)
        self.resid_dropout = Dropout(mp, r_dropout)

        # Causal mask
        causal_mask = mp.triu(mp.ones((n_ctx, n_ctx)) * -1e9, k=1)
        self.causal_mask = causal_mask

        # KV cache for inference (stores Hkv keys/values)
        self.kv_cache = None

    def set(self, mode=True):
        super().set(mode)
        self.q_proj.set(mode)
        self.k_proj.set(mode)
        self.v_proj.set(mode)
        self.c_proj.set(mode)
        self.attn_dropout.set(mode)
        self.resid_dropout.set(mode)
        
        if mode:  # training
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
        Estimate FLOPs for this GQA layer.
        Accounts for reduced K/V heads (n_kv_heads) and group repetition.
        batch_size: number of sequences in the batch
        training: if True, include backward/update cost (~3x forward)
        """
        flops = 0

        # Q projection: (B, T, n_emb) x (n_emb, head_size)
        flops += batch_size * self.n_ctx * self.n_emb * self.s_head * 2

        # K projection: (B, T, n_emb) x (n_emb, n_kv_heads*d_k)
        flops += batch_size * self.n_ctx * self.n_emb * (self.n_kv_heads * self.n_emb) * 2

        # V projection: same as K
        flops += batch_size * self.n_ctx * self.n_emb * (self.n_kv_heads * self.n_emb) * 2

        # Attention score computation: Q @ K^T
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * self.n_emb * 2

        # Masking
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx

        # Softmax over attention scores
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * 5

        # Weighted sum: Attn @ V
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * self.n_emb * 2

        # Output projection: (B, T, head_size) x (head_size, n_emb)
        flops += batch_size * self.n_ctx * self.s_head * self.n_emb * 2

        # Bias add for output projection
        if self.c_proj.bias is not None:
            flops += batch_size * self.n_ctx * self.n_emb

        # Dropout (approximate)
        flops += batch_size * self.n_ctx * self.n_emb

        if training:
            flops *= 3  # forward + backward + update

        return flops

    def forward_split_heads_q(self, z, B, T):
        # (B,T, Hq*d_k) -> (B,Hq,T,d_k)
        return z.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def forward_split_heads_kv(self, z, B, T):
        # Re-derive Hkv from shape to be robust after load()
        Hkv = z.shape[-1] // self.d_k
        if Hkv != self.n_kv_heads:
            # Update internal state to match loaded weights
            assert self.n_heads % Hkv == 0, "n_heads must be divisible by inferred n_kv_heads from weights"
            self.n_kv_heads = Hkv
            self.group_size = self.n_heads // self.n_kv_heads
        # (B,T, Hkv*d_k) -> (B,Hkv,T,d_k)
        return z.reshape(B, T, self.n_kv_heads, self.d_k).transpose(0, 2, 1, 3)
    
    def backward_merge_q(self, z_grad, original_shape):
        # (B,Hq,T_q,d_k) -> (B,T_q,Hq*d_k)
        return z_grad.transpose(0, 2, 1, 3).reshape(original_shape)

    def backward_merge_kv(self, z_grad, original_shape):
        # (B,Hkv,T_kv,d_k) -> (B,T_kv,Hkv*d_k)
        return z_grad.transpose(0, 2, 1, 3).reshape(original_shape)

    def forward(self, x, use_cache):
        """
        x: (B,T,n_emb)
        returns: (B,T,n_emb)
        """
        B, T, _ = x.shape

        # 1. Projections
        Q_lin = self.q_proj.forward(x)              # (B,T,Hq*d_k)
        K_lin = self.k_proj.forward(x)              # (B,T,Hkv*d_k)
        V_lin = self.v_proj.forward(x)

        # 2. Split heads for Q, K, and V
        Q = self.forward_split_heads_q(Q_lin, B, T)        # (B,Hq,T,d_k)
        K_new = self.forward_split_heads_kv(K_lin, B, T)   # (B,Hkv,T,d_k)
        V_new = self.forward_split_heads_kv(V_lin, B, T)   # (B,Hkv,T,d_k)

        # Handle KV cache for inference
        if use_cache and not self.setting:
            if self.kv_cache is not None:
                K_cached, V_cached = self.kv_cache
                K = self.mp.concatenate([K_cached, K_new], axis=2)  # concat along sequence
                V = self.mp.concatenate([V_cached, V_new], axis=2)  # concat along sequence
            else:
                K = K_new
                V = V_new

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

        # 3. Repeat KV heads to align with Q heads
        # (B,Hkv,*,d_k) -> (B,Hq,*,d_k)
        K_rep = self.mp.repeat(K, repeats=self.group_size, axis=1)
        V_rep = self.mp.repeat(V, repeats=self.group_size, axis=1)

        # 4. Scaled dot-product attention
        scores = self.mp.matmul(Q, K_rep.transpose(0, 1, 3, 2)) / (mt.sqrt(self.d_k) * self.r_temp)  # (B,Hq,T_q,actual_seq_len)

        # 5. Apply causal mask (prevents attending to future tokens)
        if use_cache and T_q == 1 and actual_seq_len > 1:
            mask = self.mp.zeros((1, actual_seq_len))
        else:
            mask = self.causal_mask[:T_q, :actual_seq_len]

        masked_scores = scores + mask

        # 6. Softmax over attention scores
        attn_weights = softmax(self.mp, masked_scores, axis=-1)    

        # 7. Apply dropout
        attn_weights_dropped = self.attn_dropout.forward(attn_weights)

        # 8. Calculate weighted sum: Attn @ V
        out = self.mp.matmul(attn_weights_dropped, V_rep)                                # (B,Hq,T_q,d_k)

        # 9. Recombine heads
        out_combined = merge_heads(self, out)  # (B,T_q,Hq*d_k)

        # 10. Final linear projection
        out_proj = self.c_proj.forward(out_combined)

        # 11. Residual dropout
        out_dropped = self.resid_dropout.forward(out_proj)

        # 12. Cache intermediate values for backward pass
        if self.setting:
            self._cache = (x, Q_lin, K_lin, V_lin, Q, K_new, V_new, self.group_size,
                           scores, masked_scores, attn_weights, attn_weights_dropped, out, out_combined)

        return out_dropped

    def backward(self, grad_output):
        """
        grad_output: (B,T_q,n_emb)
        returns: grad_x, [param_grads]
        """

        # 1. Unpack cached values
        (x, Q_lin, K_lin, V_lin, Q, K_new, V_new, group_size,
         scores, masked_scores, attn_weights, attn_weights_dropped, out, out_combined) = self._cache

        # 2. Backward through residual dropout
        grad_c_proj_in, _ = self.resid_dropout.backward(grad_output)

        # 3. Backward through final linear projection
        grad_o_combined, c_proj_grads = self.c_proj.backward(grad_c_proj_in)

        # 4. Backward through uncombine heads
        B, T_q, _ = grad_o_combined.shape
        grad_o = split_heads(self, grad_o_combined)

        # 5. Backward through recreate repeated V for backward
        V_rep = self.mp.repeat(V_new, repeats=group_size, axis=1)  # (B,Hq,*,d_k)
        grad_attn_weights_d = self.mp.matmul(grad_o, V_rep.transpose(0, 1, 3, 2))      # (B,Hq,T_q,actual_seq_len)
        grad_V_rep = self.mp.matmul(attn_weights_dropped.transpose(0, 1, 3, 2), grad_o)      # (B,Hq,actual_seq_len,d_k)

        # Collapse repeats back to Hkv for V
        # (B,Hq,*,d_k) -> (B,Hkv,*,d_k) by summing groups
        Hkv = K_new.shape[1]
        grad_V_rep_grouped = grad_V_rep.reshape(B, Hkv, group_size, grad_V_rep.shape[2], self.d_k).sum(axis=2)
        grad_V_new = grad_V_rep_grouped  # (B,Hkv,*,d_k)

        # 6. Backward through dropout on attn weights
        grad_attn_weights, _ = self.attn_dropout.backward(grad_attn_weights_d)

        # 7. Backward through softmax
        grad_masked_scores = softmax_prime(self.mp, grad_attn_weights, attn_weights)

        # 8. Backward through causal_mask (causal_mask is constant, so its gradient is 0)
        grad_scores = grad_masked_scores

        # 9. Backward through gradients of attention
        K_rep = self.mp.repeat(K_new, repeats=group_size, axis=1)
        scale = 1.0 / mt.sqrt(self.d_k)
        grad_Q = self.mp.matmul(grad_scores, K_rep) * scale                              # (B,Hq,T_q,d_k)
        grad_K_rep = self.mp.matmul(grad_scores.transpose(0, 1, 3, 2), Q) * scale        # (B,Hq,*,d_k)

        # 10. Backward through collapse repeats back to Hkv for K
        grad_K_rep_grouped = grad_K_rep.reshape(B, Hkv, group_size, grad_K_rep.shape[2], self.d_k).sum(axis=2)
        grad_K_new = grad_K_rep_grouped  # (B,Hkv,*,d_k)

        # 11. Backward through merge head grads back to linear input shapes
        grad_Q_lin = self.backward_merge_q(grad_Q, Q_lin.shape)
        grad_K_lin = self.backward_merge_kv(grad_K_new, K_lin.shape)
        grad_V_lin = self.backward_merge_kv(grad_V_new, V_lin.shape)

        # 12. Backward through q,k,v projections
        grad_x_q, q_proj_grads = self.q_proj.backward(grad_Q_lin)
        grad_x_k, k_proj_grads = self.k_proj.backward(grad_K_lin)
        grad_x_v, v_proj_grads = self.v_proj.backward(grad_V_lin)

        # Sum gradients for input x
        grad_x = grad_x_q + grad_x_k + grad_x_v

        # Assemble grads in declared parameter order
        param_grads = []
        param_grads.extend(q_proj_grads)
        param_grads.extend(k_proj_grads)
        param_grads.extend(v_proj_grads)
        param_grads.extend(c_proj_grads)

        return grad_x, param_grads
    
    def from_dict(self, weights_dict, i):
        self.q_proj.weight = weights_dict[f'block_{i}_gqa_q_weight']
        self.k_proj.weight = weights_dict[f'block_{i}_gqa_k_weight']
        self.v_proj.weight = weights_dict[f'block_{i}_gqa_v_weight']
        self.c_proj.weight = weights_dict[f'block_{i}_gqa_c_weight']
        self.c_proj.bias = weights_dict[f'block_{i}_gqa_c_bias']

        self.q_proj.synchronize()
        self.k_proj.synchronize()
        self.v_proj.synchronize()
        self.c_proj.synchronize()

    def towa_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_gqa_q_weight'] = self.q_proj.weight
        weights_dict[f'block_{i}_gqa_k_weight'] = self.k_proj.weight
        weights_dict[f'block_{i}_gqa_v_weight'] = self.v_proj.weight
        weights_dict[f'block_{i}_gqa_c_weight'] = self.c_proj.weight
        weights_dict[f'block_{i}_gqa_c_bias'] = self.c_proj.bias