#########################
# Mixture Head Attention (MOH)
# Author: Koureas Stavros
#########################

import math as mt
from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import split_heads, merge_heads, softmax, softmax_prime

class MOH(Module):
    """
    Multi-Head Attention where head outputs are mixed by a learned
    token-wise softmax gate (treat heads as MoE experts).
    Replaces concat+linear with weighted sum across heads + projection.
    Params order: q_proj, k_proj, v_proj, g_proj, m_proj
    """
    def __init__(self, mp, d_type, n_ctx, n_emb, r_dropout, r_temp, s_head, n_heads):
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

        # Standard Q,K,V
        self.q_proj = Linear(mp, d_type, n_emb, s_head, bias=False)  #W^Q
        self.k_proj = Linear(mp, d_type, n_emb, s_head, bias=False)  #W^K
        self.v_proj = Linear(mp, d_type, n_emb, s_head, bias=False)  #W^V

        # Gating: produce logits over heads per token
        self.g_proj = Linear(mp, d_type, n_emb, n_heads, bias=False)

        # Projection after mixture (d_k -> n_emb)
        self.m_proj = Linear(mp, d_type, self.d_k, self.n_emb, bias=True)

        # Dropout layers
        self.attn_dropout = Dropout(mp, r_dropout)
        self.resid_dropout = Dropout(mp, r_dropout)

        # Causal mask to prevent looking ahead in sequence (for decoder-only models)
        causal_mask = mp.triu(mp.ones((n_ctx, n_ctx)) * -1e9, k=1)
        self.causal_mask = causal_mask

        # KV cache for inference
        self.kv_cache = None  # (K, V) with shapes (B,H,T_total,d_k)

    def set(self, mode=True):
        super().set(mode)
        self.q_proj.set(mode)
        self.k_proj.set(mode)
        self.v_proj.set(mode)
        self.g_proj.set(mode)
        self.m_proj.set(mode)
        self.attn_dropout.set(mode)
        self.resid_dropout.set(mode)
    
        if mode:
            self.clear_cache()

    def parameters(self):
        return (self.q_proj.parameters() +
                self.k_proj.parameters() +
                self.v_proj.parameters() +
                self.g_proj.parameters() +
                self.m_proj.parameters())

    def clear_cache(self):
        self.kv_cache = None

    def flops(self, batch_size, training):
        """
        Estimate FLOPs for this MOH layer.
        batch_size: number of sequences in the batch
        training: if True, include backward/update cost (~3x forward)
        """
        flops = 0

        # Q, K, V projections: (B, T, n_emb) x (n_emb, head_size)
        flops += 3 * batch_size * self.n_ctx * self.n_emb * self.s_head * 2

        # Attention score computation: Q @ K^T
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * self.d_k * 2

        # Softmax over attention scores
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * 5  # exp + sum + div approx

        # Weighted sum: Attn @ V
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * self.d_k * 2

        # Gating projection: (B, T, n_emb) x (n_emb, n_heads)
        flops += batch_size * self.n_ctx * self.n_emb * self.n_heads * 2

        # Softmax over gating logits
        flops += batch_size * self.n_ctx * self.n_heads * 5

        # Mixture weighted sum across heads: (B, T, H, d_k)
        flops += batch_size * self.n_ctx * self.n_heads * self.d_k * 2

        # Final projection after mixture: (B, T, d_k) x (d_k, n_emb)
        flops += batch_size * self.n_ctx * self.d_k * self.n_emb * 2

        # Bias add for final projection
        if self.m_proj.bias is not None:
            flops += batch_size * self.n_ctx * self.n_emb

        if training:
            flops *= 3  # forward + backward + update
        
        return flops
    
    def forward(self, x, use_cache):
        """
        x: (B,T,n_emb)
        returns: (B,T,n_emb)
        """
        
        B, T, _ = x.shape

        # 1. Projections
        Q_lin = self.q_proj.forward(x)  #Q = X * W^Q (B,T,head_size)
        K_lin = self.k_proj.forward(x)  #K = X * W^K (B,T,head_size)
        V_lin = self.v_proj.forward(x)  #V = X * W^V (B,T,head_size)

        # 2. Split heads and reshape
        Q = split_heads(self, Q_lin)
        K_new = split_heads(self, K_lin)
        V_new = split_heads(self, V_lin)

        # Handle KV cache for inference
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

        # 3. Compute attention scores
        scores = self.mp.matmul(Q, K.transpose(0,1,3,2)) / (mt.sqrt(self.d_k) * self.r_temp)  # (B,H,T_q,total_len)

        # 4. Apply causal mask (prevents attending to future tokens)
        if use_cache and T_q == 1 and actual_seq_len > 1:
            mask = self.mp.zeros((1, actual_seq_len))
        else:
            mask = self.causal_mask[:T_q, :actual_seq_len]
    
        masked_scores = scores + mask  # broadcast (T or T_q, actual_seq_len)

        # 5. Compute attention scores
        attn_weights = softmax(self.mp, masked_scores, axis=-1)  # (B,H,T_q,actual_seq_len)

        # 6. Apply dropout
        attn_weights_dropped = self.attn_dropout.forward(attn_weights)

        # 7. Compute weighted sum of values
        out = self.mp.matmul(attn_weights_dropped, V)  # (B,H,T_q,d_k)

        # 8. Compute gating logits and probabilities
        x_g_proj = x if (not use_cache or self.setting or T == out.shape[2]) else x[:, -out.shape[2]:, :]
        gate_logits = self.g_proj.forward(x_g_proj) / self.r_temp  # (B,T_q,H)

        # 9. Softmax over gating logits
        gate_probs = softmax(self.mp, gate_logits, axis=-1)    # (B,T_q,H)

        # 10. Compute mixture of head outputs
        o_perm = out.transpose(0,2,1,3)  # (B,T_q,H,d_k)
        y = self.mp.sum(gate_probs[..., None] * o_perm, axis=2)  # (B,T_q,d_k)

        # 11. Final projection
        m_proj_out = self.m_proj.forward(y)  # (B,T_q,n_emb)

        # 12. Dropout residual connection
        dropped_out = self.resid_dropout.forward(m_proj_out)

        # 13. Cache intermediate values for backward pass
        if self.setting:
            self._cache = (x, Q_lin, K_lin, V_lin, Q, K_new, V_new, scores, attn_weights, attn_weights_dropped, out, gate_logits, gate_probs, y)
        
        return dropped_out

    def backward(self, grad_output):
        """
        grad_output: (B,T_q,n_emb)
        returns: grad_x, param_grads
        """

        # 1. Unpack cached values
        (x, Q_lin, K_lin, V_lin, Q, K_new, V_new, scores, attn_weights, attn_weights_d, o, gate_logits, gate_probs, y) = self._cache

        B = x.shape[0]
        T_q = grad_output.shape[1]

        # 2. Backward through residual dropout
        grad_m_proj_out, _ = self.resid_dropout.backward(grad_output)

        # 3. Backward through final projection
        grad_y, m_proj_grads = self.m_proj.backward(grad_m_proj_out)

        # 4. Backward through mixture of head outputs
        # Shapes: gate_probs (B,T_q,H), o (B,H,T_q,d_k) -> o_perm (B,T_q,H,d_k)
        H = self.n_heads
        d_k = self.d_k
        o_perm = o.transpose(0,2,1,3)  # (B,T_q,H,d_k)
        grad_g_proj_probs = self.mp.zeros((B,T_q,H))
        grad_o_perm = self.mp.zeros_like(o_perm)

        # 5. Backward through softmax on gating logits and probabilities
        # For each head: grad_o_h += grad_y * p_h; grad_p_h += dot(grad_y, o_h)
        grad_o_perm = grad_y[:, :, None, :] * gate_probs[..., None]  # (B,T_q,H,d_k)
        grad_g_proj_probs = self.mp.sum(grad_y[:, :, None, :] * o_perm, axis=-1)  # (B,T_q,H)
        sum_gp = self.mp.sum(grad_g_proj_probs * gate_probs, axis=-1, keepdims=True)  # (B,T_q,1)
        grad_g_proj_logits = gate_probs * (grad_g_proj_probs - sum_gp)  # (B,T_q,H)

        # 6. Backward through gating projection
        grad_x_g_proj, g_proj_grads = self.g_proj.backward(grad_g_proj_logits)

        # 7. Backward through attention outputs
        grad_o = grad_o_perm.transpose(0,2,1,3)  # (B,H,T_q,d_k)        
        grad_attn_weights_d = self.mp.matmul(grad_o, V_new.transpose(0,1,3,2))
        grad_V_new = self.mp.matmul(attn_weights_d.transpose(0,1,3,2), grad_o)

        # 8. Backward through attention weights dropout
        grad_attn_weights, _ = self.attn_dropout.backward(grad_attn_weights_d)

        # 9. Backward through softmax on attention scores
        grad_masked_scores = softmax_prime(self.mp, grad_attn_weights, attn_weights)

        # 10. Backward through causal_mask (causal_mask is constant, so its gradient is 0)
        grad_scores = grad_masked_scores

        # 11. Backward through scaled dot-product attention
        scale = 1.0 / mt.sqrt(d_k)
        # grad_Q: (B,H,T_q,d_k)
        # grad_K: (B,H,T_total,d_k) but in no-cache T_total = T_q
        grad_Q = self.mp.matmul(grad_scores, K_new) * scale
        grad_K_new = self.mp.matmul(grad_scores.transpose(0,1,3,2), Q) * scale

        # 12. Merge head gradients back to linear input shapes
        grad_Q_lin = merge_heads(self, grad_Q)
        grad_K_lin = merge_heads(self, grad_K_new)
        grad_V_lin = merge_heads(self, grad_V_new)

        # 13. Backward through Q, K, V projections
        grad_x_q, q_grads = self.q_proj.backward(grad_Q_lin)
        grad_x_k, k_grads = self.k_proj.backward(grad_K_lin)
        grad_x_v, v_grads = self.v_proj.backward(grad_V_lin)

        # Sum gradients into x (add gate path)
        grad_x = grad_x_q + grad_x_k + grad_x_v + grad_x_g_proj

        # Assemble grads in declared parameter order
        param_grads = []
        param_grads.extend(q_grads)
        param_grads.extend(k_grads)
        param_grads.extend(v_grads)
        param_grads.extend(g_proj_grads)
        param_grads.extend(m_proj_grads)

        return grad_x, param_grads
    
    def from_dict(self, weights_dict, i):
        self.q_proj.weight = weights_dict[f'block_{i}_moh_q_weight']
        self.k_proj.weight = weights_dict[f'block_{i}_moh_k_weight']
        self.v_proj.weight = weights_dict[f'block_{i}_moh_v_weight']
        self.g_proj.weight = weights_dict[f'block_{i}_moh_g_weight']
        self.m_proj.weight = weights_dict[f'block_{i}_moh_m_weight']
        self.m_proj.bias = weights_dict[f'block_{i}_moh_m_bias']

        self.q_proj.synchronize()
        self.k_proj.synchronize()
        self.v_proj.synchronize()
        self.g_proj.synchronize()
        self.m_proj.synchronize()

    def towa_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_moh_q_weight'] = self.q_proj.weight
        weights_dict[f'block_{i}_moh_k_weight'] = self.k_proj.weight
        weights_dict[f'block_{i}_moh_v_weight'] = self.v_proj.weight
        weights_dict[f'block_{i}_moh_g_weight'] = self.g_proj.weight
        weights_dict[f'block_{i}_moh_m_weight'] = self.m_proj.weight
        weights_dict[f'block_{i}_moh_m_bias'] = self.m_proj.bias