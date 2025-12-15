#########################
# Hybrid Differential Attention (HDA)
# Author: Koureas Stavros
#########################

import math as mt
from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import split_heads, merge_heads, softmax, softmax_prime


class HDA(Module):
    """
    Hybrid Differential Attention (HDA)
    """

    def __init__(self, mp, d_type, n_ctx, n_emb, r_dropout, r_temp, s_head, n_heads,
                 r_delta=0.1, n_refine_steps=2, r_momentum=0.9):
        super().__init__()
        assert s_head % n_heads == 0, "head_size must be divisible by n_heads"

        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.r_dropout = r_dropout
        self.r_temp = r_temp
        self.s_head = s_head
        self.n_heads = n_heads
        self.r_delta = r_delta
        self.n_refine_steps = n_refine_steps
        self.r_momentum = r_momentum

        d_k = s_head // n_heads
        self.d_k = d_k

        # Linear projections for Query, Key, Value
        self.q_proj = Linear(mp, d_type, n_emb, s_head, bias=False)
        self.k_proj = Linear(mp, d_type, n_emb, s_head, bias=False)
        self.v_proj = Linear(mp, d_type, n_emb, s_head, bias=False)

        # Output projection
        self.c_proj = Linear(mp, d_type, s_head, n_emb, bias=True)

        # Learnable refinement scale (Deep Memory delta rule)
        self.delta_scale = mp.ones((1, n_heads, 1, 1)) * r_delta

        # Dropout layers
        self.attn_dropout = Dropout(mp, r_dropout)
        self.resid_dropout = Dropout(mp, r_dropout)

        # Causal mask
        causal_mask = mp.triu(mp.ones((n_ctx, n_ctx)) * -1e9, k=1)
        self.causal_mask = causal_mask

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
        params = (self.q_proj.parameters() +
                  self.k_proj.parameters() +
                  self.v_proj.parameters() +
                  self.c_proj.parameters())
        params.append(self.delta_scale)
        return params

    def clear_cache(self):
        """Clear the KV cache."""
        self.kv_cache = None

    def flops(self, batch_size, training):
        """Estimate FLOPs for this HDA layer."""
        flops = 0

        # Q, K, V projections
        flops += 3 * batch_size * self.n_ctx * self.n_emb * self.s_head * 2

        # Attention score computation
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * self.d_k * 2

        # Softmax
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * 5

        # Differential refinement steps
        refinement_flops = self.n_refine_steps * (
            batch_size * self.n_heads * self.n_ctx * self.n_ctx * 10
        )
        flops += refinement_flops

        # Weighted sum
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * self.d_k * 2

        # Output projection
        flops += batch_size * self.n_ctx * self.s_head * self.n_emb * 2

        if self.c_proj.bias is not None:
            flops += batch_size * self.n_ctx * self.n_emb

        if training:
            flops *= 3

        return flops

    def forward(self, x, use_cache):
        """
        Forward pass with differential attention refinement.

        Implements nested optimization using delta rule:
        M_new = M + delta_scale * (target - M @ k) @ k^T
        """
        B, T, _ = x.shape

        # 1. Project input to Q, K, V
        Q_orig = self.q_proj.forward(x)
        K_orig = self.k_proj.forward(x)
        V_orig = self.v_proj.forward(x)

        # 2. Split into multiple heads
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

        # 5. Apply causal mask
        if use_cache and T == 1 and actual_seq_len > 1:
            mask = self.mp.zeros((1, actual_seq_len))
        else:
            mask = self.causal_mask[:T, :actual_seq_len]

        # 6. Differential refinement with momentum (Deep Memory concept)
        # Implements delta rule: score refinement as nested optimization
        momentum = self.mp.zeros_like(scores)

        for _ in range(self.n_refine_steps):
            masked_scores = scores + mask
            attn_weights = softmax(self.mp, masked_scores, axis=-1)

            # Compute energy gradient (sharpness signal)
            log_attn = self.mp.log(attn_weights + 1e-9)
            energy_grad = softmax_prime(self.mp, 1.0 + log_attn, attn_weights)

            # Momentum as associative memory (Nested Learning delta rule)
            momentum = self.r_momentum * momentum + (1 - self.r_momentum) * energy_grad
            scores = scores - self.delta_scale * momentum

        # 7. Final attention computation
        masked_scores = scores + mask
        attn_weights = softmax(self.mp, masked_scores, axis=-1)

        # 8. Apply dropout
        attn_weights_dropped = self.attn_dropout.forward(attn_weights)

        # 9. Compute weighted sum of values
        out = self.mp.matmul(attn_weights_dropped, V)

        # 10. Merge heads and project
        out_combined = merge_heads(self, out)
        c_proj_out = self.c_proj.forward(out_combined)
        dropped_out = self.resid_dropout.forward(c_proj_out)

        # 11. Cache for backward pass
        if self.setting:
            self._cache = (x, Q_orig, K_orig, V_orig, Q, K_new, V_new, K, V,
                           scores, masked_scores, attn_weights, attn_weights_dropped,
                           out, out_combined)

        return dropped_out

    def backward(self, grad_output):
        """
        Backward pass for HDA.

        Args:
            grad_output: Gradient from subsequent layer, shape (B, T, n_emb)

        Returns:
            (grad_input, list_of_param_grads)
        """
        # 1. Unpack cached values
        (x, Q_orig, K_orig, V_orig, Q, K_new, V_new, K, V,
         scores, masked_scores, attn_weights, attn_weights_dropped,
         out, out_combined) = self._cache

        # 2. Backward through residual dropout
        grad_out_dropped, _ = self.resid_dropout.backward(grad_output)

        # 3. Backward through final linear projection
        grad_out_combined, c_proj_grads = self.c_proj.backward(grad_out_dropped)

        # 4. Backward through head merge
        grad_out = split_heads(self, grad_out_combined)

        # 5. Backward through matmul(attn_weights_dropped, V)
        grad_attn_weights_dropped = self.mp.matmul(grad_out, V.transpose(0, 1, 3, 2))
        grad_V = self.mp.matmul(attn_weights_dropped.transpose(0, 1, 3, 2), grad_out)

        # 6. Backward through attention dropout
        grad_attn_weights, _ = self.attn_dropout.backward(grad_attn_weights_dropped)

        # 7. Backward through softmax
        grad_masked_scores = softmax_prime(self.mp, grad_attn_weights, attn_weights)

        # 8. Gradient for delta_scale (simplified)
        grad_delta_scale = self.mp.zeros_like(self.delta_scale)

        # 9. Backward through scaled dot-product
        grad_scores = grad_masked_scores
        scale = mt.sqrt(self.d_k) * self.r_temp
        grad_Q = self.mp.matmul(grad_scores, K) / scale
        grad_K = self.mp.matmul(grad_scores.transpose(0, 1, 3, 2), Q) / scale

        # 10. Merge head gradients
        grad_Q_orig = merge_heads(self, grad_Q)
        grad_K_orig = merge_heads(self, grad_K)
        grad_V_orig = merge_heads(self, grad_V)

        # 11. Backward through Q, K, V projections
        grad_x_q, q_proj_grads = self.q_proj.backward(grad_Q_orig)
        grad_x_k, k_proj_grads = self.k_proj.backward(grad_K_orig)
        grad_x_v, v_proj_grads = self.v_proj.backward(grad_V_orig)

        grad_x = grad_x_q + grad_x_k + grad_x_v

        # 12. Assemble parameter gradients
        param_grads = []
        param_grads.extend(q_proj_grads)
        param_grads.extend(k_proj_grads)
        param_grads.extend(v_proj_grads)
        param_grads.extend(c_proj_grads)
        param_grads.append(grad_delta_scale)

        return grad_x, param_grads

    def from_dict(self, weights_dict, i):
        """Load weights from dictionary."""
        self.q_proj.weight = weights_dict[f'block_{i}_hda_q_weight']
        self.k_proj.weight = weights_dict[f'block_{i}_hda_k_weight']
        self.v_proj.weight = weights_dict[f'block_{i}_hda_v_weight']
        self.c_proj.weight = weights_dict[f'block_{i}_hda_c_weight']
        self.c_proj.bias = weights_dict[f'block_{i}_hda_c_bias']

        if f'block_{i}_hda_delta_scale' in weights_dict:
            self.delta_scale = weights_dict[f'block_{i}_hda_delta_scale']

        self.q_proj.synchronize()
        self.k_proj.synchronize()
        self.v_proj.synchronize()
        self.c_proj.synchronize()

    def towa_dict(self, weights_dict, i):
        """Save weights to dictionary."""
        weights_dict[f'block_{i}_hda_q_weight'] = self.q_proj.weight
        weights_dict[f'block_{i}_hda_k_weight'] = self.k_proj.weight
        weights_dict[f'block_{i}_hda_v_weight'] = self.v_proj.weight
        weights_dict[f'block_{i}_hda_c_weight'] = self.c_proj.weight
        weights_dict[f'block_{i}_hda_c_bias'] = self.c_proj.bias
        weights_dict[f'block_{i}_hda_delta_scale'] = self.delta_scale
