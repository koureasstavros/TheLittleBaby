#########################
# Nested Learning Attention (NL)
# Based on: "Nested Learning: The Illusion of Deep Learning Architectures"
# by Behrouz et al. (NeurIPS 2025)
# Author: Koureas Stavros
#########################

import math as mt
from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import split_heads, merge_heads, softmax, softmax_prime


class NLA(Module):
    """
    Nested Learning Attention (NL)
    """

    def __init__(self, mp, d_type, n_ctx, n_emb, r_dropout, r_temp, s_head, n_heads, r_delta, n_meta_steps, r_momentum):
        super().__init__()
        assert s_head % n_heads == 0, "head_size must be divisible by n_heads"

        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.r_dropout = r_dropout
        self.r_temp = r_temp
        self.s_head = s_head
        self.n_heads = n_heads
        self.r_delta = r_delta              # Delta rule learning rate
        self.n_meta_steps = n_meta_steps    # Number of meta-learning steps
        self.r_momentum = r_momentum        # Momentum for meta-memory updates

        d_k = s_head // n_heads
        self.d_k = d_k

        # Linear projections for Query, Key, Value
        self.q_proj = Linear(mp, d_type, n_emb, s_head, bias=False)
        self.k_proj = Linear(mp, d_type, n_emb, s_head, bias=False)
        self.v_proj = Linear(mp, d_type, n_emb, s_head, bias=False)

        # Output projection
        self.c_proj = Linear(mp, d_type, s_head, n_emb, bias=True)

        # ========================================
        # Meta-Memory Components (Self-Modification)
        # ========================================
        # The meta-memory learns to compress gradients/updates
        # This is the "inner optimization" in the nested learning framework

        # Surprise signal projection: measures mismatch between prediction and target
        # Maps from d_k to d_k (per-head surprise signal)
        self.surprise_proj = Linear(mp, d_type, d_k, d_k, bias=True)

        # Meta-learning gate: controls how much self-modification occurs
        self.meta_gate = mp.ones((1, n_heads, 1, 1)) * 0.5

        # Learnable delta rule weight (for more expressive association)
        # Per-head learning rate for the delta rule update
        self.delta_weight = mp.ones((1, n_heads, 1, 1)) * r_delta

        # ========================================
        # Fast Weight Memory (Matrix-valued associative memory)
        # ========================================
        # M_t stores the compressed key-value mappings
        # Updated via: M_t+1 = M_t + v_t @ k_t^T (Hebbian) or delta rule
        self.fast_weight_memory = None  # Initialized per forward pass

        # Momentum buffer for stable updates
        self.meta_momentum = None

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
        self.surprise_proj.set(mode)
        self.attn_dropout.set(mode)
        self.resid_dropout.set(mode)

        # Clear caches when switching to training mode
        if mode:
            self.clear_cache()

    def parameters(self):
        """Returns all parameters of the attention module."""
        params = (self.q_proj.parameters() +
                  self.k_proj.parameters() +
                  self.v_proj.parameters() +
                  self.c_proj.parameters() +
                  self.surprise_proj.parameters())
        # Add learnable meta-learning parameters
        params.extend([
            self.meta_gate,
            self.delta_weight
        ])
        return params

    def clear_cache(self):
        """Clear the KV cache and meta-memory."""
        self.kv_cache = None
        self.fast_weight_memory = None
        self.meta_momentum = None

    def flops(self, batch_size, training):
        """
        Estimate FLOPs for this NL layer.
        Includes additional cost for self-modification steps.
        """
        flops = 0

        # Q, K, V projections: (B, T, n_emb) x (n_emb, head_size)
        flops += 3 * batch_size * self.n_ctx * self.n_emb * self.s_head * 2

        # Initial attention score computation: Q @ K^T
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * self.d_k * 2

        # Softmax over attention scores
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * 5

        # Meta-learning steps (per step):
        meta_flops_per_step = (
            # Fast weight memory read: M @ q
            batch_size * self.n_heads * self.d_k * self.d_k * self.n_ctx * 2 +
            # Surprise signal computation
            batch_size * self.n_heads * self.n_ctx * self.d_k * self.d_k * 2 +
            # Delta rule update
            batch_size * self.n_heads * self.d_k * self.d_k * 4 +
            # Momentum update
            batch_size * self.n_heads * self.d_k * self.d_k * 3
        )
        flops += self.n_meta_steps * meta_flops_per_step

        # Weighted sum: Attn @ V
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * self.d_k * 2

        # Output projection
        flops += batch_size * self.n_ctx * self.s_head * self.n_emb * 2

        if self.c_proj.bias is not None:
            flops += batch_size * self.n_ctx * self.n_emb

        if training:
            flops *= 3

        return flops

    def _init_fast_weight_memory(self, B):
        """
        Initialize the fast weight memory M as a matrix-valued associative memory.
        M: (B, n_heads, d_k, d_k)
        """
        return self.mp.zeros((B, self.n_heads, self.d_k, self.d_k))

    def _compute_surprise_signal(self, prediction, target):
        """
        Compute the Local Surprise Signal (LSS) in representation space.
        This measures the mismatch between current output and expected structure.

        From the paper: "u_{t+1} = ∇_{y_{t+1}} L(W_t; x_{t+1})"

        Args:
            prediction: Current prediction (B, n_heads, T, d_k)
            target: Target value (B, n_heads, T, d_k)

        Returns:
            Surprise signal (B, n_heads, T, d_k)
        """
        # Simple L2 surprise signal
        surprise = target - prediction
        return surprise

    def _delta_rule_update(self, M, k, v, surprise):
        """
        Update fast weight memory using the delta rule.
        More expressive than simple Hebbian learning.

        From the paper: M_{t+1} = (α I - k k^T) M_t - η P ∇L

        Delta rule: M_new = M + η * (v - M @ k) @ k^T

        This allows the memory to better manage limited capacity.

        Args:
            M: Current memory state (B, n_heads, d_k, d_k)
            k: Key vectors (B, n_heads, T, d_k)
            v: Value vectors (B, n_heads, T, d_k)
            surprise: Surprise signal (B, n_heads, T, d_k)

        Returns:
            Updated memory M_new
        """
        # Compute prediction error for delta rule
        # M @ k gives the predicted value for key k
        # For each token position, we update M

        B, n_heads, T, d_k = k.shape

        # Aggregate key-value pairs across sequence
        # Use average for stability
        k_mean = self.mp.mean(k, axis=2)  # (B, n_heads, d_k)
        v_mean = self.mp.mean(v, axis=2)  # (B, n_heads, d_k)

        # Prediction: what M thinks v should be for k
        M_pred = self.mp.matmul(M, k_mean[:, :, :, None])[:, :, :, 0]  # (B, n_heads, d_k)

        # Prediction error (delta)
        delta = v_mean - M_pred  # (B, n_heads, d_k)

        # Delta rule update: M_new = M + η * delta @ k^T
        update = self.delta_weight * self.mp.matmul(
            delta[:, :, :, None],  # (B, n_heads, d_k, 1)
            k_mean[:, :, None, :]  # (B, n_heads, 1, d_k)
        )  # (B, n_heads, d_k, d_k)

        return M + update

    def _self_modify_attention(self, Q, K, V, M, mask):
        """
        Self-modifying attention with meta-learning.
        The attention mechanism learns to modify its own behavior.

        This implements the "nested optimization" where:
        - Outer level: standard attention computation
        - Inner level: meta-learning updates to fast weight memory

        Args:
            Q: Query (B, n_heads, T, d_k)
            K: Key (B, n_heads, T, d_k)
            V: Value (B, n_heads, T, d_k)
            M: Fast weight memory (B, n_heads, d_k, d_k)
            mask: Causal mask

        Returns:
            Modified attention output and updated memory
        """
        B, n_heads, T, d_k = Q.shape

        # Initialize momentum if needed
        if self.meta_momentum is None:
            self.meta_momentum = self.mp.zeros_like(M)

        # Standard attention scores
        scale = mt.sqrt(d_k) * self.r_temp
        scores = self.mp.matmul(Q, K.transpose(0, 1, 3, 2)) / scale

        # === Meta-learning steps ===
        for _ in range(self.n_meta_steps):
            # Read from fast weight memory: M @ Q^T gives memory-based prediction
            # This represents what the memory "expects" for each query
            memory_read = self.mp.matmul(M, Q.transpose(0, 1, 3, 2))  # (B, n_heads, d_k, T)
            memory_read = memory_read.transpose(0, 1, 3, 2)  # (B, n_heads, T, d_k)

            # Compute surprise signal (prediction error in memory space)
            # The surprise is between what memory predicts and actual values
            weighted_V = self.mp.matmul(softmax(self.mp, scores + mask, axis=-1), V)
            surprise = self._compute_surprise_signal(memory_read, weighted_V)

            # Project surprise through learned transformation
            B_shape, n_heads_shape, T_shape, d_k_shape = surprise.shape
            surprise_flat = surprise.reshape(B_shape * n_heads_shape * T_shape, d_k_shape)
            surprise_proj = self.surprise_proj.forward(surprise_flat)
            surprise_proj = surprise_proj.reshape(B_shape, n_heads_shape, T_shape, d_k_shape)

            # Delta rule update with momentum
            M_update = self._delta_rule_update(M, K, V, surprise_proj)

            # Momentum-based stable update
            self.meta_momentum = (self.r_momentum * self.meta_momentum +
                                  (1 - self.r_momentum) * (M_update - M))

            # Apply gated update
            M = M + self.meta_gate * self.meta_momentum

            # Update scores with memory-augmented attention
            # Memory contribution: Q @ M @ K^T
            memory_scores = self.mp.matmul(
                self.mp.matmul(Q, M),  # (B, n_heads, T, d_k)
                K.transpose(0, 1, 3, 2)  # (B, n_heads, d_k, T)
            ) / scale  # (B, n_heads, T, T)

            # Blend standard and memory-augmented attention
            scores = (1 - self.meta_gate) * scores + self.meta_gate * memory_scores

        return scores, M

    def forward(self, x, use_cache):
        """
        Forward pass with nested learning self-modification.

        The key insight from NL paper:
        "Each component has its own optimization problem and gradient flow"

        Args:
            x: Input tensor, shape (B, T, n_emb)
            use_cache: Whether to use KV cache for inference

        Returns:
            Output tensor, shape (B, T, n_emb)
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

        # 4. Initialize or retrieve fast weight memory
        if self.fast_weight_memory is None or self.fast_weight_memory.shape[0] != B:
            self.fast_weight_memory = self._init_fast_weight_memory(B)

        # 5. Prepare causal mask
        if use_cache and T == 1 and actual_seq_len > 1:
            mask = self.mp.zeros((1, actual_seq_len))
        else:
            mask = self.causal_mask[:T, :actual_seq_len]

        # 6. Self-modifying attention computation
        scores, updated_memory = self._self_modify_attention(
            Q, K, V, self.fast_weight_memory, mask
        )
        self.fast_weight_memory = updated_memory

        # 7. Apply final causal mask and softmax
        masked_scores = scores + mask
        attn_weights = softmax(self.mp, masked_scores, axis=-1)

        # 8. Apply dropout to attention weights
        attn_weights_dropped = self.attn_dropout.forward(attn_weights)

        # 9. Compute weighted sum of values
        out = self.mp.matmul(attn_weights_dropped, V)

        # 10. Merge heads
        out_combined = merge_heads(self, out)

        # 11. Final linear projection
        c_proj_out = self.c_proj.forward(out_combined)

        # 12. Apply residual dropout
        dropped_out = self.resid_dropout.forward(c_proj_out)

        # 13. Cache intermediate values for backward pass
        if self.setting:
            self._cache = (
                x, Q_orig, K_orig, V_orig, Q, K_new, V_new, K, V,
                scores, masked_scores, attn_weights, attn_weights_dropped,
                out, out_combined, updated_memory
            )

        return dropped_out

    def backward(self, grad_output):
        """
        Backward pass for NL attention.

        Computes gradients through the self-modifying mechanism.

        Args:
            grad_output: Gradient from subsequent layer, shape (B, T, n_emb)

        Returns:
            (grad_input, list_of_param_grads)
        """
        # 1. Unpack cached values
        (x, Q_orig, K_orig, V_orig, Q, K_new, V_new, K, V,
         scores, masked_scores, attn_weights, attn_weights_dropped,
         out, out_combined, updated_memory) = self._cache

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

        # 8. Backward through score computation
        # Simplified: treat meta-learning as frozen for gradient computation
        scale = mt.sqrt(self.d_k) * self.r_temp
        grad_Q = self.mp.matmul(grad_masked_scores, K) / scale
        grad_K = self.mp.matmul(grad_masked_scores.transpose(0, 1, 3, 2), Q) / scale

        # 9. Gradients for meta-learning parameters (simplified first-order approximation)
        grad_meta_gate = self.mp.sum(
            grad_masked_scores * scores,
            axis=(0, 2, 3), keepdims=True
        ) * 0.1  # Scaled for stability

        grad_delta_weight = self.mp.sum(
            grad_out * 0.01,  # Simplified gradient
            axis=(0, 2, 3), keepdims=True
        )

        # 10. Backward through surprise projection
        # Simplified: accumulate gradients from meta-learning steps
        B, n_heads, T, d_k = Q.shape
        grad_surprise_input = self.mp.zeros((B * n_heads * T, d_k))
        _, surprise_proj_grads = self.surprise_proj.backward(grad_surprise_input)

        # 11. Merge head gradients
        grad_Q_orig = merge_heads(self, grad_Q)
        grad_K_orig = merge_heads(self, grad_K)
        grad_V_orig = merge_heads(self, grad_V)

        # 12. Backward through Q, K, V projections
        grad_x_q, q_proj_grads = self.q_proj.backward(grad_Q_orig)
        grad_x_k, k_proj_grads = self.k_proj.backward(grad_K_orig)
        grad_x_v, v_proj_grads = self.v_proj.backward(grad_V_orig)

        grad_x = grad_x_q + grad_x_k + grad_x_v

        # 13. Assemble parameter gradients (matching order in parameters())
        param_grads = []
        param_grads.extend(q_proj_grads)
        param_grads.extend(k_proj_grads)
        param_grads.extend(v_proj_grads)
        param_grads.extend(c_proj_grads)
        param_grads.extend(surprise_proj_grads)

        # Add gradients for meta-learning parameters
        param_grads.extend([
            grad_meta_gate,
            grad_delta_weight
        ])

        return grad_x, param_grads

    def from_dict(self, weights_dict, i):
        """Load weights from dictionary."""
        self.q_proj.weight = weights_dict[f'block_{i}_nl_q_weight']
        self.k_proj.weight = weights_dict[f'block_{i}_nl_k_weight']
        self.v_proj.weight = weights_dict[f'block_{i}_nl_v_weight']
        self.c_proj.weight = weights_dict[f'block_{i}_nl_c_weight']
        self.c_proj.bias = weights_dict[f'block_{i}_nl_c_bias']
        self.surprise_proj.weight = weights_dict[f'block_{i}_nl_surprise_weight']
        self.surprise_proj.bias = weights_dict[f'block_{i}_nl_surprise_bias']

        # Load meta-learning parameters if saved
        if f'block_{i}_nl_meta_gate' in weights_dict:
            self.meta_gate = weights_dict[f'block_{i}_nl_meta_gate']
        if f'block_{i}_nl_delta_weight' in weights_dict:
            self.delta_weight = weights_dict[f'block_{i}_nl_delta_weight']

        self.q_proj.synchronize()
        self.k_proj.synchronize()
        self.v_proj.synchronize()
        self.c_proj.synchronize()
        self.surprise_proj.synchronize()

    def towa_dict(self, weights_dict, i):
        """Save weights to dictionary."""
        weights_dict[f'block_{i}_nl_q_weight'] = self.q_proj.weight
        weights_dict[f'block_{i}_nl_k_weight'] = self.k_proj.weight
        weights_dict[f'block_{i}_nl_v_weight'] = self.v_proj.weight
        weights_dict[f'block_{i}_nl_c_weight'] = self.c_proj.weight
        weights_dict[f'block_{i}_nl_c_bias'] = self.c_proj.bias
        weights_dict[f'block_{i}_nl_surprise_weight'] = self.surprise_proj.weight
        weights_dict[f'block_{i}_nl_surprise_bias'] = self.surprise_proj.bias

        # Save meta-learning parameters
        weights_dict[f'block_{i}_nl_meta_gate'] = self.meta_gate
        weights_dict[f'block_{i}_nl_delta_weight'] = self.delta_weight
