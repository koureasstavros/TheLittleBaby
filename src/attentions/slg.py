#########################
# Sparse Local + Global Attention (SGL)
# Author: Koureas Stavros
#########################

import math as mt
from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import split_heads, merge_heads, softmax, softmax_prime

class SparseLocalGlobalAttention(Module):
    """
    Sparse Local + Global Attention (SGL)
    """
    def __init__(self, mp, d_type, n_ctx, n_emb, r_dropout, r_temp, s_head, n_heads, s_window, global_indices):
        super().__init__()
        assert s_head % n_heads == 0, "head_size must be divisible by n_heads"
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.r_dropout = r_dropout
        self.r_temp = r_temp
        self.s_head = s_head
        self.n_heads = n_heads
        self.s_window = s_window

        self.d_k = s_head // n_heads
        self.global_indices = global_indices if global_indices is not None else []

        # Linear projections for Query, Key, Value
        self.q_proj = Linear(mp, d_type, n_emb, s_head, bias=False)
        self.k_proj = Linear(mp, d_type, n_emb, s_head, bias=False)
        self.v_proj = Linear(mp, d_type, n_emb, s_head, bias=False)
        self.c_proj = Linear(mp, d_type, s_head, n_emb, bias=True)

        # Dropout layers
        self.attn_dropout = Dropout(mp, r_dropout)
        self.resid_dropout = Dropout(mp, r_dropout)

        # KV cache for inference
        self.kv_cache = None

    def set(self, mode=True):
        """Sets the attention module and its sub-modules to training/eval mode."""
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
        """Returns all parameters of the attention module."""
        return (self.q_proj.parameters() +
                self.k_proj.parameters() +
                self.v_proj.parameters() +
                self.c_proj.parameters())

    def clear_cache(self):
        """Clear the KV cache."""
        self.kv_cache = None

    def build_mask(self, T):
        """
        Builds a sparse attention mask with local window and global tokens.
        Returns: mask of shape (T, T)
        """
        mask = self.mp.full((T, T), -1e9)
        for i in range(T):
            start = max(0, i - self.s_window)
            end = min(T, i + self.s_window + 1)
            mask[i, start:end] = 0
        # Global tokens: allow full attention
        for idx in self.global_indices:
            mask[idx, :] = 0
            mask[:, idx] = 0
        return mask

    def flops(self, batch_size, training):
        """
        Estimate FLOPs for this Sparse Attention layer.
        batch_size: number of sequences in the batch
        training: if True, include backward/update cost (~3x forward)
        """
        flops = 0
        # Q, K, V projections
        flops += 3 * batch_size * self.n_ctx * self.n_emb * self.s_head * 2
        # Attention score computation
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * self.d_k * 2
        # Softmax
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * 5
        # Weighted sum: Attn @ V
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * self.d_k * 2
        # Output projection
        flops += batch_size * self.n_ctx * self.s_head * self.n_emb * 2
        # Bias add for output projection
        if self.c_proj.bias is not None:
            flops += batch_size * self.n_ctx * self.n_emb
        if training:
            flops *= 3
        return flops

    def forward(self, x, use_cache):
        """
        x: input tensor, shape (B, T, n_emb)
        Returns: output tensor, shape (B, T, n_emb)
        Supports KV cache for inference.
        """
        B, T, _ = x.shape

        # 1. Project input to Q, K, V
        Q_orig = self.q_proj.forward(x)
        K_orig = self.k_proj.forward(x)
        V_orig = self.v_proj.forward(x)

        # 2. Split heads
        Q = split_heads(self, Q_orig)
        K_new = split_heads(self, K_orig)
        V_new = split_heads(self, V_orig)

        # 3. Handle KV cache for inference
        if use_cache and not self.setting:
            if self.kv_cache is not None:
                K_cached, V_cached = self.kv_cache
                K = self.mp.concatenate([K_cached, K_new], axis=2)
                V = self.mp.concatenate([V_cached, V_new], axis=2)
            else:
                K = K_new
                V = V_new
            # Trim KV cache to last n_ctx tokens
            if K.shape[2] > self.n_ctx:
                K = K[:, :, -self.n_ctx:, :]
                V = V[:, :, -self.n_ctx:, :]
            self.kv_cache = (K, V)
        else:
            K = K_new
            V = V_new

        actual_seq_len = K.shape[2]

        # 4. Compute scaled dot-product attention scores
        scores = self.mp.matmul(Q, K.transpose(0, 1, 3, 2)) / (mt.sqrt(self.d_k) * self.r_temp)

        # 5. Build sparse mask (local + global)
        mask = self.build_mask(actual_seq_len)
        masked_scores = scores + mask

        # 6. Apply softmax to get attention weights
        attn_weights = softmax(self.mp, masked_scores, axis=-1)

        # 7. Apply dropout to attention weights
        attn_weights_dropped = self.attn_dropout.forward(attn_weights)

        # 8. Compute weighted sum of values
        out = self.mp.matmul(attn_weights_dropped, V)

        # 9. Recombine heads
        out_combined = merge_heads(self, out)

        # 10. Final linear projection
        c_proj_out = self.c_proj.forward(out_combined)

        # 11. Apply residual dropout
        dropped_out = self.resid_dropout.forward(c_proj_out)

        # 12. Cache intermediate values for backward pass
        if self.setting:
            self._cache = (x, Q_orig, K_orig, V_orig, Q, K_new, V_new, scores, masked_scores, attn_weights, attn_weights_dropped, out, out_combined)

        return dropped_out

    def backward(self, grad_output):
        """
        grad_output: gradient from the subsequent layer, shape (B, T, n_emb)
        Returns: (grad_input, list_of_param_grads)
        Validates gradient shapes.
        """
        # 1. Unpack cached values
        (x, Q_orig, K_orig, V_orig, Q, K, V, scores, masked_scores, attn_weights, attn_weights_dropped, out, out_combined) = self._cache

        # 2. Backward through residual dropout
        grad_out_dropped, _ = self.resid_dropout.backward(grad_output)

        # 3. Backward through final linear projection
        grad_out_combined, c_proj_grads = self.c_proj.backward(grad_out_dropped)

        # 4. Backward through merge_heads
        B, T, H = grad_out_combined.shape
        grad_out = split_heads(self, grad_out_combined)

        # 5. Backward through matmul(attn_weights_dropped, V)
        grad_attn_weights_dropped = self.mp.matmul(grad_out, V.transpose(0, 1, 3, 2))

        # 6. Backward through compute weighted sum of value
        grad_V = self.mp.matmul(attn_weights_dropped.transpose(0, 1, 3, 2), grad_out)

        # 7. Backward through attn_dropout
        grad_attn_weights, _ = self.attn_dropout.backward(grad_attn_weights_dropped)

        # 8. Backward through softmax
        grad_masked_scores = softmax_prime(self.mp, grad_attn_weights, attn_weights)

        # 9. Backward through mask (mask is constant, gradient is 0)
        grad_scores = grad_masked_scores

        # 10. Backward through scaled dot-product
        grad_Q = self.mp.matmul(grad_scores, K) / mt.sqrt(self.d_k)
        grad_K = self.mp.matmul(grad_scores.transpose(0, 1, 3, 2), Q) / mt.sqrt(self.d_k)

        # 11. Undo split_heads for Q, K, V
        grad_Q_orig = merge_heads(self, grad_Q)
        grad_K_orig = merge_heads(self, grad_K)
        grad_V_orig = merge_heads(self, grad_V)

        # 12. Backward through q_proj, k_proj, v_proj
        grad_x_q, q_proj_grads = self.q_proj.backward(grad_Q_orig)
        grad_x_k, k_proj_grads = self.k_proj.backward(grad_K_orig)
        grad_x_v, v_proj_grads = self.v_proj.backward(grad_V_orig)

        # Sum gradients for the input 'x'
        grad_x = grad_x_q + grad_x_k + grad_x_v

        # Assemble gradients in the correct order
        param_grads = []
        param_grads.extend(q_proj_grads)
        param_grads.extend(k_proj_grads)
        param_grads.extend(v_proj_grads)
        param_grads.extend(c_proj_grads)

        return grad_x, param_grads

    def from_dict(self, weights_dict, i):
        self.q_proj.weight = weights_dict[f'block_{i}_slg_q_weight']
        self.k_proj.weight = weights_dict[f'block_{i}_slg_k_weight']
        self.v_proj.weight = weights_dict[f'block_{i}_slg_v_weight']
        self.c_proj.weight = weights_dict[f'block_{i}_slg_c_weight']
        self.c_proj.bias = weights_dict[f'block_{i}_slg_c_bias']

        self.q_proj.synchronize()
        self.k_proj.synchronize()
        self.v_proj.synchronize()
        self.c_proj.synchronize()

    def towa_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_slg_q_weight'] = self.q_proj.weight
        weights_dict[f'block_{i}_slg_k_weight'] = self.k_proj.weight
        weights_dict[f'block_{i}_slg_v_weight'] = self.v_proj.weight
        weights_dict[f'block_{i}_slg_c_weight'] = self.c_proj.weight
        weights_dict[f'block_{i}_slg_c_bias'] = self.c_proj.bias