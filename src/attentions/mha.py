#########################
# Multi Head Attention (MHA)
# Author: Koureas Stavros
#########################

import math as mt
from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import split_heads, merge_heads, softmax, softmax_prime

class MHA(Module):
    """
    Multi-Head Self-Attention mechanism.
    Computes attention scores and combines information from different "heads".
    """
    def __init__(self, mp, n_ctx, n_emb, r_dropout, head_size, n_heads):
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

        # Linear projections for Query, Key, Value
        self.q_proj = Linear(mp, n_emb, head_size, bias=False)  #W^Q
        self.k_proj = Linear(mp, n_emb, head_size, bias=False)  #W^K
        self.v_proj = Linear(mp, n_emb, head_size, bias=False)  #W^V

        # Output projection
        self.c_proj = Linear(mp, head_size, n_emb, bias=True)

        # Dropout layers
        self.attn_dropout = Dropout(mp, r_dropout)
        self.resid_dropout = Dropout(mp, r_dropout)

        # Causal mask to prevent looking ahead in sequence (for decoder-only models)
        causal_mask = mp.triu(mp.ones((n_ctx, n_ctx)) * -1e9, k=1)
        self.causal_mask = causal_mask

        # KV cache for inference
        self.kv_cache = None

    def set(self, mode=True):
        """Sets the attention module and its sub-modules to training/eval mode."""
        super().set(mode) # Call base Module set to set self.set
        self.q_proj.set(mode)
        self.k_proj.set(mode)
        self.v_proj.set(mode)
        self.c_proj.set(mode)
        self.attn_dropout.set(mode)
        self.resid_dropout.set(mode)

        # Clear cache when switching to training mode
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
    
    def flops(self, batch_size, training):
        """
        Estimate FLOPs for this MHA layer.
        seq_len: actual sequence length (defaults to self.n_ctx for worst-case)
        batch_size: number of sequences in the batch
        training: if True, include backward/update cost (~3x forward)
        """

        flops = 0

        # Q, K, V projections: (B, T, n_emb) x (n_emb, head_size)
        # Multiply–add counted as 2 FLOPs
        flops += 3 * batch_size * self.n_ctx * self.n_emb * self.head_size * 2

        # Attention score computation: Q @ K^T
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * self.d_k * 2

        # Softmax over attention scores
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * 5  # exp + sum + div approx

        # Weighted sum: Attn @ V
        flops += batch_size * self.n_heads * self.n_ctx * self.n_ctx * self.d_k * 2

        # Output projection: (B, T, head_size) x (head_size, n_emb)
        flops += batch_size * self.n_ctx * self.head_size * self.n_emb * 2

        # Bias add for output projection
        if self.c_proj.bias is not None:
            flops += batch_size * self.n_ctx * self.n_emb

        if training:
            flops *= 3  # forward + backward + update

        return flops

    def forward(self, x, use_cache):
        """
        x: input tensor, shape (B, T, n_emb)
        Returns: output tensor, shape (B, T, n_emb)
        """
        B, T, _ = x.shape

        # 1. Project input to Q, K, V
        Q_orig = self.q_proj.forward(x)  #Q = X * W^Q (B, T, head_size)
        K_orig = self.k_proj.forward(x)  #K = X * W^K (B, T, head_size)
        V_orig = self.v_proj.forward(x)  #V = X * W^V (B, T, head_size)

        # 2. Helper function to split heads and transpose
        Q = split_heads(self, Q_orig)
        K_new = split_heads(self, K_orig)
        V_new = split_heads(self, V_orig)

        # Handle KV cache for inference
        if use_cache and not self.setting:
            if self.kv_cache is not None:
                # Concatenate with cached K, V
                K_cached, V_cached = self.kv_cache
                K = self.mp.concatenate([K_cached, K_new], axis=2)  # Concat along sequence dimension
                V = self.mp.concatenate([V_cached, V_new], axis=2)  # Concat along sequence dimension
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

        # 5. Apply softmax to get attention weights
        attn_weights = softmax(self.mp, masked_scores, axis=-1)

        # 6. Apply dropout to attention weights
        attn_weights_dropped = self.attn_dropout.forward(attn_weights)

        # 7. Compute weighted sum of values
        # (B, n_heads, T, T) @ (B, n_heads, T, d_k) -> (B, n_heads, T, d_k)
        out = self.mp.matmul(attn_weights_dropped, V)

        # 8. Recombine heads: transpose and reshape back to (B, T, head_size)
        out_combined = merge_heads(self, out)   # (B, T, head_size)

        # 9. Final linear projection
        c_proj_out = self.c_proj.forward(out_combined)

        # 10. Apply residual dropout
        dropped_out = self.resid_dropout.forward(c_proj_out)

        # 11. Cache intermediate values for backward pass
        if self.setting:
            self._cache = (x, Q_orig, K_orig, V_orig, Q, K_new, V_new, scores, masked_scores, attn_weights, attn_weights_dropped, out, out_combined)

        return dropped_out

    def backward(self, grad_output):
        """
        grad_output: gradient from the subsequent layer, shape (B, T, n_emb)
        Returns: (grad_input, list_of_param_grads)
        """

        # 1. Unpack cached values
        (x, Q_orig, K_orig, V_orig, Q, K, V, scores, masked_scores, attn_weights, attn_weights_dropped, out, out_combined) = self._cache

        # 2. Backward through residual dropout
        grad_out_dropped, _ = self.resid_dropout.backward(grad_output) # Dropout has no params

        # 3. Backward through final linear projection
        grad_out_combined, c_proj_grads = self.c_proj.backward(grad_out_dropped)

        # 4. Backward through undo reshape/transpose for o_combined to get grad_o
        # grad_out_combined: (B, T, head_size)
        # grad_out: (B, n_heads, T, d_k)
        B, T, H = grad_out_combined.shape
        grad_out = split_heads(self, grad_out_combined)

        # 5. Backward through matmul(attn_weights_dropped, V)
        # o = A @ V  => dL/dA = dL/do @ V.T, dL/dV = A.T @ dL/do
        grad_attn_weights_dropped = self.mp.matmul(grad_out, V.transpose(0, 1, 3, 2))
        grad_V = self.mp.matmul(attn_weights_dropped.transpose(0, 1, 3, 2), grad_out)

        # 6. Backward through attn_dropout
        grad_attn_weights, _ = self.attn_dropout.backward(grad_attn_weights_dropped)

        # 7. Backward through softmax (attn_weights = softmax(masked_scores))
        # dL/dx = y * (dL/dy - sum(dL/dy * y)) where y = softmax(x)
        grad_masked_scores = softmax_prime(self.mp, grad_attn_weights, attn_weights)
        
        # 8. Backward through causal_mask (causal_mask is constant, so its gradient is 0)
        grad_scores = grad_masked_scores

        # 9. Backward through scaled dot-product: scores = (Q @ K.T) / sqrt(d_k)
        # Let S = Q @ K.T / sqrt(d_k)
        # dL/dQ = (dL/dS @ K) / sqrt(d_k)
        # dL/dK = (dL/dS.T @ Q) / sqrt(d_k)
        grad_Q = self.mp.matmul(grad_scores, K) / mt.sqrt(self.d_k)
        grad_K = self.mp.matmul(grad_scores.transpose(0, 1, 3, 2), Q) / mt.sqrt(self.d_k)

        # 10. Undo split_heads for Q, K, V to get gradients for original Q_orig, K_orig, V_orig
        grad_Q_orig = merge_heads(self, grad_Q)
        grad_K_orig = merge_heads(self, grad_K)
        grad_V_orig = merge_heads(self, grad_V)

        # 11. Backward through q_proj, k_proj, v_proj
        grad_x_q, q_proj_grads = self.q_proj.backward(grad_Q_orig)
        grad_x_k, k_proj_grads = self.k_proj.backward(grad_K_orig)
        grad_x_v, v_proj_grads = self.v_proj.backward(grad_V_orig)

        # Sum gradients for the input 'x' from all three paths (Q, K, V)
        grad_x = grad_x_q + grad_x_k + grad_x_v

        # Assemble gradients in the correct order: q_proj, k_proj, v_proj, c_proj
        param_grads = []
        param_grads.extend(q_proj_grads)
        param_grads.extend(k_proj_grads)
        param_grads.extend(v_proj_grads)
        param_grads.extend(c_proj_grads)

        return grad_x, param_grads
    
    def from_dict(self, weights_dict, i):
        self.q_proj.weight = weights_dict[f'block_{i}_mha_q_weight']
        self.k_proj.weight = weights_dict[f'block_{i}_mha_k_weight']
        self.v_proj.weight = weights_dict[f'block_{i}_mha_v_weight']
        self.c_proj.weight = weights_dict[f'block_{i}_mha_c_weight']
        self.c_proj.bias = weights_dict[f'block_{i}_mha_c_bias']

        self.q_proj.synchronize()
        self.k_proj.synchronize()
        self.v_proj.synchronize()
        self.c_proj.synchronize()

    def towa_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_mha_q_weight'] = self.q_proj.weight
        weights_dict[f'block_{i}_mha_k_weight'] = self.k_proj.weight
        weights_dict[f'block_{i}_mha_v_weight'] = self.v_proj.weight
        weights_dict[f'block_{i}_mha_c_weight'] = self.c_proj.weight
        weights_dict[f'block_{i}_mha_c_bias'] = self.c_proj.bias