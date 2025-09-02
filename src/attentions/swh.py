#########################
# Switch Head Attention (SWH)
# Author: Koureas Stavros
#########################

import math as mt
from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import softmax, onehot_argmax

class SWH(Module):
    """
    Switch Head Attention: token-wise top-1 head routing.
    - Compute standard multi-head attention outputs o_h (per head)
    - Gating g_proj(x)->logits over heads; forward uses one-hot(top-1) to select a single head
    - Backward uses straight-through estimator: gradients flow as if softmax (stable training)
    Params order: q_proj, k_proj, v_proj, g_proj, m_proj
    """
    def __init__(self, mp, n_ctx, n_emb, r_dropout, head_size, n_heads, temperature=1.0):
        super().__init__()
        assert head_size % n_heads == 0, "head_size must be divisible by n_heads"
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.r_dropout = r_dropout
        self.head_size = head_size
        self.n_heads = n_heads

        d_k = head_size // n_heads
        self.d_k = d_k

        temperature = max(1e-6, float(temperature))
        self.temperature = temperature

        # Projections
        self.q_proj = Linear(mp, n_emb, head_size, bias=False)
        self.k_proj = Linear(mp, n_emb, head_size, bias=False)
        self.v_proj = Linear(mp, n_emb, head_size, bias=False)

        # Gating over heads
        self.g_proj = Linear(mp, n_emb, n_heads, bias=False)

        # Project selected head d_k -> n_emb
        self.m_proj = Linear(mp, self.d_k, n_emb, bias=True)

        # Dropout layers
        self.attn_dropout = Dropout(mp, r_dropout)
        self.resid_dropout = Dropout(mp, r_dropout)

        # Causal mask
        causal_mask = mp.triu(mp.ones((n_ctx, n_ctx)) * -1e9, k=1)
        self.causal_mask = causal_mask

        # KV cache for inference
        self.kv_cache = None

    def clear_cache(self):
        self.kv_cache = None

    def parameters(self):
        return (self.q_proj.parameters() +
                self.k_proj.parameters() +
                self.v_proj.parameters() +
                self.g_proj.parameters() +
                self.m_proj.parameters())

    def flops(self, batch_size, training):
        """
        Estimate FLOPs for this SWH layer.
        Includes Q/K/V projections, attention, gating, and m_proj.
        batch_size: number of sequences in the batch
        training: if True, include backward/update cost (~3x forward)
        """
        flops = 0

        # Q, K, V projections
        flops += 3 * batch_size * self.n_ctx * self.n_emb * self.head_size * 2

        # Attention score computation: Q @ K^T (local window)
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * self.d_k * 2

        # Masking
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx

        # Softmax over local window
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * 5

        # Weighted sum: Attn @ V
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * self.d_k * 2

        # Output projection
        flops += batch_size * self.n_ctx * self.head_size * self.n_emb * 2

        # Bias add for output projection
        if self.m_proj.bias is not None:
            flops += batch_size * self.n_ctx * self.n_emb

        # Dropout (approximate)
        flops += batch_size * self.n_ctx * self.n_emb

        if training:
            flops *= 3  # forward + backward + update

        return flops

    def set(self, mode=True):
        super().set(mode)
        for m in (self.q_proj, self.k_proj, self.v_proj,
                  self.g_proj, self.m_proj,
                  self.attn_dropout, self.resid_dropout):
            m.set(mode)
        if mode:
            self.clear_cache()

    def forward_split_heads(self, z, B, T):
        return z.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)  # (B,H,T,d_k)

    def backward_unsplit_heads(self, z, original_shape):
        return z.transpose(0, 2, 1, 3).reshape(original_shape)
        
    def forward(self, x, use_cache):
        """
        x: (B,T,n_emb)
        returns: (B,T_q,n_emb)
        """
        B, T, _ = x.shape

        # 1. Linear projections for Q, K, V
        Q_lin = self.q_proj.forward(x)  # (B,T,head_size)
        K_lin = self.k_proj.forward(x)
        V_lin = self.v_proj.forward(x)

        # 2. Split heads
        Q = self.forward_split_heads(Q_lin, B, T)
        K_new = self.forward_split_heads(K_lin, B, T)
        V_new = self.forward_split_heads(V_lin, B, T)

        # Handle KV cache for inference
        if use_cache and not self.setting:  # Only use cache during inference
            if self.kv_cache is not None:
                # Concatenate with cached K, V
                K_cached, V_cached = self.kv_cache
                K = self.mp.concatenate([K_cached, K_new], axis=2)  # Concat along sequence dimension
                V = self.mp.concatenate([V_cached, V_new], axis=2)
            else:
                K = K_new
                V = V_new

            # Trim KV cache to last n_ctx tokens to keep masks valid and bounded
            if K.shape[2] > self.n_ctx:
                K = K[:, :, -self.n_ctx:, :]
                V = V[:, :, -self.n_ctx:, :]
            self.kv_cache = (K, V)
            
            # Update cache with new K, V
            self.kv_cache = (K, V)
        else:
            # Training mode or cache disabled
            K = K_new
            V = V_new

        # Get actual sequence length for attention computation
        actual_seq_len = K.shape[2]
        
        # 3. Compute scaled dot-product attention scores
        # (B, n_heads, T, d_k) @ (B, n_heads, d_k, actual_seq_len) -> (B, n_heads, T, actual_seq_len)
        scores = self.mp.matmul(Q, K.transpose(0, 1, 3, 2)) / mt.sqrt(self.d_k)

        # 4. Apply causal mask (prevents attending to future tokens)
        # Adjust mask for the actual sequence lengths
        if use_cache and T == 1 and actual_seq_len > 1:
            # For single token generation, create a mask for the last position
            # attending to all previous positions (including itself)
            # No masking needed for single token attending to past
            mask = self.mp.zeros((1, actual_seq_len))
        else:
            # Normal case: create causal mask for current sequence length
            mask = self.causal_mask[:T, :actual_seq_len]

        masked_scores = scores + mask

        # 5. Compute attention weights
        attn_weights = softmax(self.mp, masked_scores, axis=-1)             # (B,H,T_q,S)

        # 6. Apply dropout to attention weights
        attn_weights_dropped = self.attn_dropout.forward(attn_weights)

        # 7. Head outputs
        o = self.mp.matmul(attn_weights_dropped, V)  # (B,H,T_q,d_k)

        # 8. Gating over heads for each token (use last T_q tokens if cache)
        x_g_proj = x if (not use_cache or self.setting or T == o.shape[2]) else x[:, -o.shape[2]:, :]
        gate_logits = self.g_proj.forward(x_g_proj) / self.temperature  # (B,T_q,H)

        # 9. Straight-through gating:
        # - Forward uses hard one-hot of argmax
        # - Backward uses softmax probabilities for stable gradients
        gate_probs_soft = softmax(self.mp, gate_logits, axis=-1)        # (B,T_q,H)
        gate_probs_hard, gate_idx = onehot_argmax(self.mp, gate_logits, self.n_heads)   # (B,T_q,H)

        # 10. Straight-through gating
        # Rearrange heads for mixture: (B,H,T_q,d_k) -> (B,T_q,H,d_k)
        o_perm = o.transpose(0, 2, 1, 3)
        # Weighted sum across heads (hard selection)
        y = self.mp.sum(gate_probs_hard[..., None] * o_perm, axis=2)  # (B,T_q,d_k)

        # 11. Project to n_emb
        m_proj_out = self.m_proj.forward(y)                           # (B,T_q,n_emb)

        # 12. Apply residual dropout
        out = self.resid_dropout.forward(m_proj_out)

        # 13. Cache intermediate values for backward pass
        if self.setting:
            self._cache = (x, Q_lin, K_lin, V_lin, Q, K_new, V_new,
                           scores, attn_weights, attn_weights_dropped, o,
                           gate_logits, gate_probs_soft, gate_probs_hard, o_perm)
        
        return out

    def backward(self, grad_output):
        """
        grad_output: (B,T_q,n_emb)
        returns: grad_x, [param_grads]
        """

        # 1. Unpack cached values
        (x, Q_lin, K_lin, V_lin, Q, K_new, V_new,
         scores, attn_weights, attn_weights_d, o,
         gate_logits, gate_probs_soft, gate_probs_hard, o_perm) = self._cache

        # 2. Residual dropout
        grad_m_proj_in, _ = self.resid_dropout.backward(grad_output)

        # 3. m_proj
        grad_y, m_proj_grads = self.m_proj.backward(grad_m_proj_in)  # (B,T_q,d_k)

        # 4. Mixture y = sum_h p_hard * o_h
        # grad wrt o_perm uses hard probs (only selected head gets grad)
        grad_o_perm = grad_y[:, :, None, :] * gate_probs_hard[..., None]  # (B,T_q,H,d_k)
        # grad wrt probs: dL/dp_h = dot(grad_y, o_h)
        grad_g_proj_probs = self.mp.sum(grad_y[:, :, None, :] * o_perm, axis=-1)  # (B,T_q,H)

        # 5. Softmax backward on gate logits (straight-through: use soft probs in Jacobian)
        sum_gp = self.mp.sum(grad_g_proj_probs * gate_probs_soft, axis=-1, keepdims=True)
        grad_g_proj_logits = gate_probs_soft * (grad_g_proj_probs - sum_gp)  # (B,T_q,H)

        # 6. Backward g_proj
        grad_x_g_proj, g_proj_grads = self.g_proj.backward(grad_g_proj_logits)

        # 7. Backprop to o (undo transpose)
        grad_o = grad_o_perm.transpose(0, 2, 1, 3)  # (B,H,T_q,d_k)

        # 8. Attention matmul
        grad_attn_weights_d = self.mp.matmul(grad_o, V_new.transpose(0, 1, 3, 2))   # (B,H,T_q,S)
        grad_V_new = self.mp.matmul(attn_weights_d.transpose(0, 1, 3, 2), grad_o)   # (B,H,S,d_k)

        # 9. Dropout on attn weights
        grad_attn_weights, _ = self.attn_dropout.backward(grad_attn_weights_d)

        # 10. Softmax backward over scores
        sum_aw = self.mp.sum(grad_attn_weights * attn_weights, axis=-1, keepdims=True)
        grad_scores = attn_weights * (grad_attn_weights - sum_aw)

        # 11. scores = (Q K^T)/sqrt(d_k)
        scale = 1.0 / mt.sqrt(self.d_k)
        grad_Q = self.mp.matmul(grad_scores, K_new) * scale
        grad_K_new = self.mp.matmul(grad_scores.transpose(0, 1, 3, 2), Q) * scale

        # 12. Merge heads back to linear input shapes
        grad_Q_lin = self.backward_unsplit_heads(grad_Q, Q_lin.shape)
        grad_K_lin = self.backward_unsplit_heads(grad_K_new, K_lin.shape)
        grad_V_lin = self.backward_unsplit_heads(grad_V_new, V_lin.shape)

        # 13. Backward q,k,v projections
        grad_x_q, q_grads = self.q_proj.backward(grad_Q_lin)
        grad_x_k, k_grads = self.k_proj.backward(grad_K_lin)
        grad_x_v, v_grads = self.v_proj.backward(grad_V_lin)

        # Sum gradients for input x (add gate path)
        grad_x = grad_x_q + grad_x_k + grad_x_v + grad_x_g_proj

        # Assemble grads in order: q, k, v, g, m
        param_grads = []
        param_grads.extend(q_grads)
        param_grads.extend(k_grads)
        param_grads.extend(v_grads)
        param_grads.extend(g_proj_grads)
        param_grads.extend(m_proj_grads)

        return grad_x, param_grads
    
    def from_dict(self, weights_dict, i):
        self.q_proj.weight = weights_dict[f'block_{i}_swh_q_weight']
        self.k_proj.weight = weights_dict[f'block_{i}_swh_k_weight']
        self.v_proj.weight = weights_dict[f'block_{i}_swh_v_weight']
        self.g_proj.weight = weights_dict[f'block_{i}_swh_g_weight']
        self.m_proj.weight = weights_dict[f'block_{i}_swh_m_weight']
        self.m_proj.bias = weights_dict[f'block_{i}_swh_m_bias']

        self.q_proj.synchronize()
        self.k_proj.synchronize()
        self.v_proj.synchronize()
        self.g_proj.synchronize()
        self.m_proj.synchronize()

    def to_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_swh_q_weight'] = self.q_proj.weight
        weights_dict[f'block_{i}_swh_k_weight'] = self.k_proj.weight
        weights_dict[f'block_{i}_swh_v_weight'] = self.v_proj.weight
        weights_dict[f'block_{i}_swh_g_weight'] = self.g_proj.weight
        weights_dict[f'block_{i}_swh_m_weight'] = self.m_proj.weight
        weights_dict[f'block_{i}_swh_m_bias'] = self.m_proj.bias