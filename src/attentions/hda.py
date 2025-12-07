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
    Hybrid Differential Attention (HDA) v2

    Enhanced attention mechanism with improved semantic capture through:
    1. Multi-scale semantic refinement (local + global pathways)
    2. Value-space semantic coherence guidance
    3. Learnable relative position bias for context awareness
    4. Per-head adaptive refinement with momentum
    5. Semantic neighborhood preservation
    """

    def __init__(self, mp, d_type, n_ctx, n_emb, r_dropout, r_temp, s_head, n_heads,
                 r_refine=0.1, n_refine_steps=2, r_momentum=0.9, s_local_window=8):
        super().__init__()
        assert s_head % n_heads == 0, "head_size must be divisible by n_heads"

        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.r_dropout = r_dropout
        self.r_temp = r_temp
        self.s_head = s_head
        self.n_heads = n_heads
        self.r_refine = r_refine
        self.n_refine_steps = n_refine_steps
        self.r_momentum = r_momentum          # Momentum for stable refinement
        self.s_local_window = s_local_window  # Local context window size

        d_k = s_head // n_heads
        self.d_k = d_k

        # Linear projections for Query, Key, Value
        self.q_proj = Linear(mp, d_type, n_emb, s_head, bias=False)
        self.k_proj = Linear(mp, d_type, n_emb, s_head, bias=False)
        self.v_proj = Linear(mp, d_type, n_emb, s_head, bias=False)

        # Output projection
        self.c_proj = Linear(mp, d_type, s_head, n_emb, bias=True)

        # === Enhanced Semantic Components ===

        # Per-head learnable refinement parameters (entropy vs alignment balance)
        # Shape: (1, n_heads, 1, 1) - each head learns its own balance
        self.refine_scale = mp.ones((1, n_heads, 1, 1)) * r_refine
        self.entropy_weight = mp.ones((1, n_heads, 1, 1)) * 0.1   # Per-head entropy importance
        self.alignment_weight = mp.ones((1, n_heads, 1, 1)) * 1.0 # Per-head semantic alignment
        self.value_weight = mp.ones((1, n_heads, 1, 1)) * 0.3     # Per-head value coherence

        # Learnable relative position bias for semantic distance awareness
        # Captures that nearby tokens often share semantic context
        # Shape: (1, n_heads, n_ctx, n_ctx) - but we'll use a more efficient representation
        # We use log-linear distance buckets for efficiency
        self.n_pos_buckets = 32
        self.pos_bias = mp.zeros((1, n_heads, self.n_pos_buckets))  # Learnable position biases

        # Local semantic gate - learns to blend local vs global attention
        self.local_gate = mp.ones((1, n_heads, 1, 1)) * 0.5

        # Dropout layers
        self.attn_dropout = Dropout(mp, r_dropout)
        self.resid_dropout = Dropout(mp, r_dropout)

        # Causal mask
        causal_mask = mp.triu(mp.ones((n_ctx, n_ctx)) * -1e9, k=1)
        self.causal_mask = causal_mask

        # Pre-compute relative position bucket indices
        self._init_position_buckets()

        # KV cache for inference
        self.kv_cache = None

        # Momentum buffer for refinement (cleared each forward)
        self._momentum_buffer = None

    def _init_position_buckets(self):
        """
        Initialize relative position bucket indices using log-linear bucketing.
        This efficiently encodes relative distances for position bias.

        Bucket scheme:
        - Exact positions for small distances (0 to n_pos_buckets//2)
        - Logarithmically spaced buckets for larger distances
        """
        n_exact = self.n_pos_buckets // 2

        # Create position indices matrix
        positions = self.mp.arange(self.n_ctx)
        # relative_pos[i, j] = i - j (query position - key position)
        relative_pos = positions[:, None] - positions[None, :]

        # For causal attention, we only care about non-negative relative positions
        # (query can only attend to earlier or same position keys)
        relative_pos = self.mp.abs(relative_pos)

        # Bucket assignment
        bucket_indices = self.mp.zeros((self.n_ctx, self.n_ctx), dtype=self.mp.int32)

        # Exact buckets for small distances
        is_small = relative_pos < n_exact
        bucket_indices = self.mp.where(is_small, relative_pos, bucket_indices)

        # Log buckets for larger distances
        max_distance = self.n_ctx
        log_ratio = mt.log(max_distance / n_exact) / mt.log(2)
        log_pos = n_exact + (
            self.mp.log(relative_pos.astype(self.mp.float32) / n_exact + 1e-6) /
            mt.log(2) * (self.n_pos_buckets - n_exact) / log_ratio
        ).astype(self.mp.int32)
        log_pos = self.mp.clip(log_pos, n_exact, self.n_pos_buckets - 1)

        bucket_indices = self.mp.where(~is_small, log_pos, bucket_indices)
        self.pos_bucket_indices = bucket_indices  # (n_ctx, n_ctx)

    def _get_position_bias(self, T_q, T_kv):
        """
        Get relative position bias for given sequence lengths.

        Args:
            T_q: Query sequence length
            T_kv: Key/Value sequence length

        Returns:
            Position bias tensor of shape (1, n_heads, T_q, T_kv)
        """
        # Get relevant bucket indices
        bucket_idx = self.pos_bucket_indices[:T_q, :T_kv]  # (T_q, T_kv)

        # Gather position biases: (1, n_heads, n_buckets) -> (1, n_heads, T_q, T_kv)
        # Expand bucket indices to match head dimension
        bias = self.mp.zeros((1, self.n_heads, T_q, T_kv))
        for h in range(self.n_heads):
            bias[0, h] = self.pos_bias[0, h, bucket_idx]

        return bias

    def set(self, mode=True):
        """Sets the attention module and its sub-modules to training/eval mode."""
        super().set(mode)
        self.q_proj.set(mode)
        self.k_proj.set(mode)
        self.v_proj.set(mode)
        self.c_proj.set(mode)
        self.attn_dropout.set(mode)
        self.resid_dropout.set(mode)

        # Clear cache and momentum buffer when switching to training mode
        if mode:
            self.clear_cache()
            self._momentum_buffer = None

    def parameters(self):
        """Returns all parameters of the attention module."""
        params = (self.q_proj.parameters() +
                  self.k_proj.parameters() +
                  self.v_proj.parameters() +
                  self.c_proj.parameters())
        # Add learnable semantic refinement parameters
        params.extend([
            self.refine_scale,
            self.entropy_weight,
            self.alignment_weight,
            self.value_weight,
            self.pos_bias,
            self.local_gate
        ])
        return params

    def clear_cache(self):
        """Clear the KV cache and momentum buffer."""
        self.kv_cache = None
        self._momentum_buffer = None

    def flops(self, batch_size, training):
        """
        Estimate FLOPs for this HDA v2 layer.
        Includes additional cost for enhanced semantic refinement.
        """
        flops = 0

        # Q, K, V projections: (B, T, n_emb) x (n_emb, head_size)
        flops += 3 * batch_size * self.n_ctx * self.n_emb * self.s_head * 2

        # Initial attention score computation: Q @ K^T
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * self.d_k * 2

        # Position bias lookup and addition
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx

        # Softmax over attention scores
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * 5

        # Enhanced differential refinement steps (per step):
        refinement_flops_per_step = (
            # V @ V^T for value coherence
            batch_size * self.n_heads * self.n_ctx * self.d_k * self.n_ctx * 2 +
            # Local attention mask computation
            batch_size * self.n_heads * self.n_ctx * self.n_ctx * 2 +
            # Semantic gradient ops (entropy, alignment, value coherence)
            batch_size * self.n_heads * self.n_ctx * self.n_ctx * 8 +
            # Momentum update
            batch_size * self.n_heads * self.n_ctx * self.n_ctx * 3 +
            # Re-softmax
            batch_size * self.n_heads * self.n_ctx * self.n_ctx * 5
        )
        flops += self.n_refine_steps * refinement_flops_per_step

        # Weighted sum: Attn @ V
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * self.d_k * 2

        # Output projection
        flops += batch_size * self.n_ctx * self.s_head * self.n_emb * 2

        if self.c_proj.bias is not None:
            flops += batch_size * self.n_ctx * self.n_emb

        if training:
            flops *= 3

        return flops

    def _compute_local_mask(self, T_q, T_kv):
        """
        Create a local attention mask for capturing nearby semantic context.

        Returns a soft mask that emphasizes local tokens within the window,
        allowing the model to blend local and global attention patterns.

        Args:
            T_q: Query sequence length
            T_kv: Key/Value sequence length

        Returns:
            Local mask of shape (1, 1, T_q, T_kv) with values in [0, 1]
        """
        # Create distance matrix
        q_pos = self.mp.arange(T_q)[:, None]
        k_pos = self.mp.arange(T_kv)[None, :]
        distance = self.mp.abs(q_pos - k_pos)

        # Soft local mask using Gaussian-like decay
        # Tokens within window get high weight, outside get low weight
        half_window = self.s_local_window // 2
        local_weight = self.mp.exp(-0.5 * (distance / (half_window + 1e-6)) ** 2)

        return local_weight[None, None, :, :]  # (1, 1, T_q, T_kv)

    def _compute_value_coherence(self, V, attn_weights):
        """
        Compute value-space semantic coherence signal.

        This measures how well the current attention distribution
        respects the semantic structure in value space. Tokens with
        similar values should receive similar attention patterns.

        Args:
            V: Value tensor (B, n_heads, T_kv, d_k)
            attn_weights: Current attention weights (B, n_heads, T_q, T_kv)

        Returns:
            Value coherence gradient for attention weights
        """
        # Compute value similarity matrix: V @ V^T
        # This captures which keys have semantically similar values
        V_norm = V / (self.mp.sqrt(self.mp.sum(V * V, axis=-1, keepdims=True)) + 1e-6)
        value_sim = self.mp.matmul(V_norm, V_norm.transpose(0, 1, 3, 2))  # (B, n_heads, T_kv, T_kv)

        # For each query position, compute expected attention based on value similarity
        # If attending to key_i, we should also attend to keys similar to key_i
        # A_value_target[q, k] = sum_j(A[q,j] * V_sim[j,k]) / sum_j(A[q,j])
        weighted_sim = self.mp.matmul(attn_weights, value_sim)  # (B, n_heads, T_q, T_kv)

        # Normalize to get target distribution
        weighted_sim_sum = self.mp.sum(weighted_sim, axis=-1, keepdims=True) + 1e-9
        A_value_target = weighted_sim / weighted_sim_sum

        # Gradient: encourage attention to align with value coherence
        value_coherence_grad = 2.0 * (attn_weights - A_value_target)

        return value_coherence_grad

    def _compute_energy_gradient(self, Q, K, V, scores, mask, pos_bias, local_mask):
        """
        Compute enhanced energy gradient for semantic-aware attention refinement.

        Energy function combines multiple semantic signals:
        E = E_entropy + E_alignment + E_value + E_local

        Where:
        - E_entropy: Encourages sharp, focused attention
        - E_alignment: Preserves Q-K semantic relevance
        - E_value: Respects value-space semantic coherence
        - E_local: Balances local vs global attention patterns

        Args:
            Q: Query tensor (B, n_heads, T, d_k)
            K: Key tensor (B, n_heads, T_kv, d_k)
            V: Value tensor (B, n_heads, T_kv, d_k)
            scores: Attention scores (B, n_heads, T, T_kv)
            mask: Causal mask
            pos_bias: Relative position bias
            local_mask: Local attention soft mask

        Returns:
            grad_scores: Gradient of energy w.r.t. scores
        """
        # Compute current attention weights with position bias
        masked_scores = scores + mask + pos_bias
        attn_weights = softmax(self.mp, masked_scores, axis=-1)

        # === 1. Entropy Gradient (Sharpness) ===
        # Encourages peaked attention distributions
        log_attn = self.mp.log(attn_weights + 1e-9)
        entropy_grad = 1.0 + log_attn

        # === 2. Q-K Semantic Alignment Gradient ===
        # Preserves the fundamental Q-K similarity structure
        scale = mt.sqrt(self.d_k) * self.r_temp
        QK_sim = self.mp.matmul(Q, K.transpose(0, 1, 3, 2)) / scale
        QK_sim_with_pos = QK_sim + pos_bias + mask
        A_target_qk = softmax(self.mp, QK_sim_with_pos, axis=-1)
        alignment_grad = 2.0 * (attn_weights - A_target_qk)

        # === 3. Value Coherence Gradient ===
        # Ensures attention respects semantic relationships in value space
        value_coherence_grad = self._compute_value_coherence(V, attn_weights)

        # === 4. Local-Global Balance ===
        # Blend local attention pattern with global
        # local_gate controls how much to emphasize local context
        local_target = local_mask * attn_weights
        local_target = local_target / (self.mp.sum(local_target, axis=-1, keepdims=True) + 1e-9)

        # Interpolate between current attention and local-focused attention
        blended_target = self.local_gate * local_target + (1 - self.local_gate) * attn_weights
        local_grad = 2.0 * (attn_weights - blended_target)

        # === Combine Gradients with Per-Head Learnable Weights ===
        grad_A = (
            self.entropy_weight * entropy_grad +
            self.alignment_weight * alignment_grad +
            self.value_weight * value_coherence_grad +
            0.1 * local_grad  # Small fixed weight for local balance
        )

        # Backprop through softmax
        grad_scores = softmax_prime(self.mp, grad_A, attn_weights)

        return grad_scores, attn_weights

    def forward(self, x, use_cache):
        """
        Forward pass with enhanced semantic-aware differential attention refinement.

        Key enhancements over standard attention:
        1. Relative position bias for distance-aware attention
        2. Multi-signal refinement (entropy, alignment, value coherence, local)
        3. Momentum-based stable refinement
        4. Per-head learnable semantic balancing

        Args:
            x: Input tensor, shape (B, T, n_emb)
            use_cache: Whether to use KV cache for inference

        Returns:
            Output tensor, shape (B, T, n_emb)
        """
        _, T, _ = x.shape

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

        # 4. Compute initial scaled dot-product attention scores
        scores = self.mp.matmul(Q, K.transpose(0, 1, 3, 2)) / (mt.sqrt(self.d_k) * self.r_temp)

        # 5. Get relative position bias for semantic distance awareness
        pos_bias = self._get_position_bias(T, actual_seq_len)

        # 6. Prepare causal mask
        if use_cache and T == 1 and actual_seq_len > 1:
            mask = self.mp.zeros((1, actual_seq_len))
        else:
            mask = self.causal_mask[:T, :actual_seq_len]

        # 7. Compute local attention mask for multi-scale semantics
        local_mask = self._compute_local_mask(T, actual_seq_len)

        # 8. ENHANCED DIFFERENTIAL REFINEMENT with momentum
        scores_history = [scores]
        attn_history = []

        # Initialize momentum buffer for stable refinement
        momentum = self.mp.zeros_like(scores)

        for _ in range(self.n_refine_steps):
            # Compute enhanced energy gradient with all semantic signals
            energy_grad, attn_at_step = self._compute_energy_gradient(
                Q, K, V, scores, mask, pos_bias, local_mask
            )
            attn_history.append(attn_at_step)

            # Momentum-based update for stable convergence
            # momentum = beta * momentum + (1 - beta) * grad
            momentum = self.r_momentum * momentum + (1 - self.r_momentum) * energy_grad

            # Update scores with momentum
            scores = scores - self.refine_scale * momentum
            scores_history.append(scores)

        # 9. Apply final position bias, causal mask, and softmax
        final_scores = scores + mask + pos_bias
        attn_weights = softmax(self.mp, final_scores, axis=-1)

        # 10. Apply dropout to attention weights
        attn_weights_dropped = self.attn_dropout.forward(attn_weights)

        # 11. Compute weighted sum of values
        out = self.mp.matmul(attn_weights_dropped, V)

        # 12. Merge heads
        out_combined = merge_heads(self, out)

        # 13. Final linear projection
        c_proj_out = self.c_proj.forward(out_combined)

        # 14. Apply residual dropout
        dropped_out = self.resid_dropout.forward(c_proj_out)

        # 15. Cache intermediate values for backward pass
        if self.setting:
            self._cache = (
                x, Q_orig, K_orig, V_orig, Q, K_new, V_new, K, V,
                scores_history, attn_history, mask, pos_bias, local_mask,
                final_scores, attn_weights, attn_weights_dropped, out, out_combined
            )

        return dropped_out

    def backward(self, grad_output):
        """
        Backward pass for enhanced HDA.

        Computes gradients through the multi-signal differential refinement,
        including gradients for:
        - Q, K, V projections
        - Per-head semantic weights (entropy, alignment, value, local_gate)
        - Position bias parameters

        Args:
            grad_output: Gradient from subsequent layer, shape (B, T, n_emb)

        Returns:
            (grad_input, list_of_param_grads)
        """
        # 1. Unpack cached values
        (_x, _Q_orig, _K_orig, _V_orig, Q, _K_new, _V_new, K, V,
         _scores_history, attn_history, mask, pos_bias, local_mask,
         _final_scores, attn_weights, attn_weights_dropped,
         _out, _out_combined) = self._cache

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

        # 7. Backward through softmax (final)
        grad_final_scores = softmax_prime(self.mp, grad_attn_weights, attn_weights)

        # Gradient flows through position bias addition
        grad_pos_bias = self.mp.sum(grad_final_scores, axis=0, keepdims=True)
        grad_scores = grad_final_scores  # Position bias and mask don't affect score gradient

        # 8. Backward through differential refinement (unroll the steps)
        # Accumulate gradients for Q, K, V from refinement
        grad_Q_refine = self.mp.zeros_like(Q)
        grad_K_refine = self.mp.zeros_like(K)
        grad_V_refine = self.mp.zeros_like(V)

        # Gradients for learnable semantic parameters
        grad_entropy_weight = self.mp.zeros_like(self.entropy_weight)
        grad_alignment_weight = self.mp.zeros_like(self.alignment_weight)
        grad_value_weight = self.mp.zeros_like(self.value_weight)
        grad_local_gate = self.mp.zeros_like(self.local_gate)
        grad_refine_scale = self.mp.zeros_like(self.refine_scale)

        for step in range(self.n_refine_steps - 1, -1, -1):
            attn_at_step = attn_history[step] if step < len(attn_history) else attn_weights

            # Recompute intermediate values for gradient
            scale = mt.sqrt(self.d_k) * self.r_temp
            QK_sim = self.mp.matmul(Q, K.transpose(0, 1, 3, 2)) / scale
            QK_sim_with_pos = QK_sim + pos_bias + mask
            A_target_qk = softmax(self.mp, QK_sim_with_pos, axis=-1)

            # Value coherence target
            V_norm = V / (self.mp.sqrt(self.mp.sum(V * V, axis=-1, keepdims=True)) + 1e-6)
            value_sim = self.mp.matmul(V_norm, V_norm.transpose(0, 1, 3, 2))
            weighted_sim = self.mp.matmul(attn_at_step, value_sim)
            A_value_target = weighted_sim / (self.mp.sum(weighted_sim, axis=-1, keepdims=True) + 1e-9)

            # Local target
            local_target = local_mask * attn_at_step
            local_target = local_target / (self.mp.sum(local_target, axis=-1, keepdims=True) + 1e-9)

            # === Gradient accumulation for semantic weights ===
            log_attn = self.mp.log(attn_at_step + 1e-9)
            entropy_grad_component = 1.0 + log_attn
            alignment_grad_component = 2.0 * (attn_at_step - A_target_qk)
            value_grad_component = 2.0 * (attn_at_step - A_value_target)

            blended_target = self.local_gate * local_target + (1 - self.local_gate) * attn_at_step
            local_grad_component = 2.0 * (attn_at_step - blended_target)

            # Accumulate parameter gradients (simplified first-order approximation)
            grad_entropy_weight += self.mp.sum(
                grad_scores * entropy_grad_component * (-self.refine_scale),
                axis=(0, 2, 3), keepdims=True
            )
            grad_alignment_weight += self.mp.sum(
                grad_scores * alignment_grad_component * (-self.refine_scale),
                axis=(0, 2, 3), keepdims=True
            )
            grad_value_weight += self.mp.sum(
                grad_scores * value_grad_component * (-self.refine_scale),
                axis=(0, 2, 3), keepdims=True
            )

            # Gradient for refine_scale
            full_energy_grad = (
                self.entropy_weight * entropy_grad_component +
                self.alignment_weight * alignment_grad_component +
                self.value_weight * value_grad_component +
                0.1 * local_grad_component
            )
            energy_grad_through_softmax = softmax_prime(self.mp, full_energy_grad, attn_at_step)
            grad_refine_scale += self.mp.sum(
                grad_scores * (-energy_grad_through_softmax),
                axis=(0, 2, 3), keepdims=True
            )

            # Gradient contribution to Q and K through A_target_qk
            grad_A_target = -2.0 * self.alignment_weight * self.refine_scale
            grad_QK_sim = softmax_prime(self.mp,
                grad_A_target * (attn_at_step - A_target_qk), A_target_qk)

            grad_Q_refine = grad_Q_refine + self.mp.matmul(grad_QK_sim, K) / scale
            grad_K_refine = grad_K_refine + self.mp.matmul(
                grad_QK_sim.transpose(0, 1, 3, 2), Q) / scale

            # Gradient contribution to V through value coherence
            grad_V_refine = grad_V_refine + self.mp.matmul(
                attn_at_step.transpose(0, 1, 3, 2),
                -2.0 * self.value_weight * self.refine_scale * (attn_at_step - A_value_target)
            ) * 0.1  # Scaled down for stability

        # 9. Backward through initial score computation
        scale = mt.sqrt(self.d_k) * self.r_temp
        grad_Q = self.mp.matmul(grad_scores, K) / scale + grad_Q_refine
        grad_K = self.mp.matmul(grad_scores.transpose(0, 1, 3, 2), Q) / scale + grad_K_refine

        # Add V gradient from refinement
        grad_V = grad_V + grad_V_refine

        # 10. Merge head gradients
        grad_Q_orig = merge_heads(self, grad_Q)
        grad_K_orig = merge_heads(self, grad_K)
        grad_V_orig = merge_heads(self, grad_V)

        # 11. Backward through Q, K, V projections
        grad_x_q, q_proj_grads = self.q_proj.backward(grad_Q_orig)
        grad_x_k, k_proj_grads = self.k_proj.backward(grad_K_orig)
        grad_x_v, v_proj_grads = self.v_proj.backward(grad_V_orig)

        grad_x = grad_x_q + grad_x_k + grad_x_v

        # 12. Assemble parameter gradients (matching order in parameters())
        param_grads = []
        param_grads.extend(q_proj_grads)
        param_grads.extend(k_proj_grads)
        param_grads.extend(v_proj_grads)
        param_grads.extend(c_proj_grads)

        # Add gradients for semantic refinement parameters
        param_grads.extend([
            grad_refine_scale,
            grad_entropy_weight,
            grad_alignment_weight,
            grad_value_weight,
            grad_pos_bias,
            grad_local_gate
        ])

        return grad_x, param_grads

    def from_dict(self, weights_dict, i):
        """Load weights from dictionary."""
        self.q_proj.weight = weights_dict[f'block_{i}_hda_q_weight']
        self.k_proj.weight = weights_dict[f'block_{i}_hda_k_weight']
        self.v_proj.weight = weights_dict[f'block_{i}_hda_v_weight']
        self.c_proj.weight = weights_dict[f'block_{i}_hda_c_weight']
        self.c_proj.bias = weights_dict[f'block_{i}_hda_c_bias']

        # Load semantic refinement parameters if saved
        if f'block_{i}_hda_refine_scale' in weights_dict:
            self.refine_scale = weights_dict[f'block_{i}_hda_refine_scale']
        if f'block_{i}_hda_entropy_weight' in weights_dict:
            self.entropy_weight = weights_dict[f'block_{i}_hda_entropy_weight']
        if f'block_{i}_hda_alignment_weight' in weights_dict:
            self.alignment_weight = weights_dict[f'block_{i}_hda_alignment_weight']
        if f'block_{i}_hda_value_weight' in weights_dict:
            self.value_weight = weights_dict[f'block_{i}_hda_value_weight']
        if f'block_{i}_hda_pos_bias' in weights_dict:
            self.pos_bias = weights_dict[f'block_{i}_hda_pos_bias']
        if f'block_{i}_hda_local_gate' in weights_dict:
            self.local_gate = weights_dict[f'block_{i}_hda_local_gate']

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

        # Save semantic refinement parameters
        weights_dict[f'block_{i}_hda_refine_scale'] = self.refine_scale
        weights_dict[f'block_{i}_hda_entropy_weight'] = self.entropy_weight
        weights_dict[f'block_{i}_hda_alignment_weight'] = self.alignment_weight
        weights_dict[f'block_{i}_hda_value_weight'] = self.value_weight
        weights_dict[f'block_{i}_hda_pos_bias'] = self.pos_bias
        weights_dict[f'block_{i}_hda_local_gate'] = self.local_gate
