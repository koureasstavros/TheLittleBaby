#########################
# Multi Head Attention (MHA)
# Author: Koureas Stavros
#########################

import math as mt
from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import softmax

class MHA(Module):
    """
    Multi-Head Self-Attention mechanism.
    Computes attention scores and combines information from different "heads".
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

        # Linear projections for Query, Key, Value
        self.q_proj = Linear(mp, n_emb, head_size, bias=False)
        self.k_proj = Linear(mp, n_emb, head_size, bias=False)
        self.v_proj = Linear(mp, n_emb, head_size, bias=False)

        # Output projection
        self.c_proj = Linear(mp, head_size, n_emb)

        # Dropout layers
        self.attn_dropout = Dropout(mp, p_dropout)
        self.resid_dropout = Dropout(mp, p_dropout)

        # Causal mask to prevent looking ahead in sequence (for decoder-only models)
        causal_mask = mp.triu(mp.ones((n_ctx, n_ctx)) * -1e9, k=1)
        self.causal_mask = causal_mask

        # KV cache for inference
        self.kv_cache = None

    def clear_cache(self):
        """Clear the KV cache."""
        self.kv_cache = None

    def parameters(self):
        """Returns all parameters of the attention module."""
        return (self.q_proj.parameters() +
                self.k_proj.parameters() +
                self.v_proj.parameters() +
                self.c_proj.parameters())

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

    def forward(self, x, use_cache):
        """
        x: input tensor, shape (B, T, n_emb)
        Returns: output tensor, shape (B, T, n_emb)
        """
        B, T, _ = x.shape

        # Project input to Q, K, V
        Q_orig = self.q_proj.forward(x)  # (B, T, head_size)
        K_orig = self.k_proj.forward(x)
        V_orig = self.v_proj.forward(x)

        # Helper function to split heads and transpose
        def split_heads(z):
            B_s, T_s, H_s = z.shape
            z = z.reshape(B_s, T_s, self.n_heads, self.d_k)
            return z.transpose(0, 2, 1, 3) # (B, n_heads, T, d_k)

        Q = split_heads(Q_orig)
        K_new = split_heads(K_orig)
        V_new = split_heads(V_orig)

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
        
        # Compute scaled dot-product attention scores
        # (B, n_heads, T, d_k) @ (B, n_heads, d_k, actual_seq_len) -> (B, n_heads, T, actual_seq_len)
        scores = self.mp.matmul(Q, K.transpose(0, 1, 3, 2)) / mt.sqrt(self.d_k)

        # Apply causal mask (prevents attending to future tokens)
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

        attn_weights = softmax(self.mp, masked_scores, axis=-1)
        attn_weights_dropped = self.attn_dropout.forward(attn_weights)

        # Compute weighted sum of values
        # (B, n_heads, T, T) @ (B, n_heads, T, d_k) -> (B, n_heads, T, d_k)
        o = self.mp.matmul(attn_weights_dropped, V)

        # Recombine heads: transpose and reshape back to (B, T, head_size)
        o_combined = o.transpose(0, 2, 1, 3).reshape(B, T, self.head_size)

        # Final linear projection
        out = self.c_proj.forward(o_combined)
        out_dropped = self.resid_dropout.forward(out)

        # Store all intermediate values needed for backward pass (unchanged for training)
        if self.setting:  # Only cache for backward pass during training
            self._cache = (x, Q_orig, K_orig, V_orig, Q, K_new, V_new, scores, masked_scores, attn_weights, attn_weights_dropped, o, o_combined)
            
        return out_dropped

    def backward(self, grad_output):
        """
        grad_output: gradient from the subsequent layer, shape (B, T, n_emb)
        Returns: (grad_input, list_of_param_grads)
        """
        (x, Q_orig, K_orig, V_orig, Q, K, V, scores, masked_scores, attn_weights, attn_weights_dropped, o, o_combined) = self._cache

        # Gradients will be collected in the order of self.parameters(): q_proj, k_proj, v_proj, c_proj_dn
        current_mha_param_grads = []

        # 1. Backward through resid_dropout
        grad_out_dropped, _ = self.resid_dropout.backward(grad_output) # Dropout has no params

        # 2. Backward through c_proj (output linear layer)
        grad_o_combined, c_proj_grads = self.c_proj.backward(grad_out_dropped)

        # 3. Undo reshape/transpose for o_combined to get grad_o
        # grad_o_combined: (B, T, head_size)
        # grad_o: (B, n_heads, T, d_k)
        B, T, H = grad_o_combined.shape
        grad_o = grad_o_combined.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # 4. Backward through matmul(attn_weights_dropped, V)
        # o = A @ V  => dL/dA = dL/do @ V.T, dL/dV = A.T @ dL/do
        grad_attn_weights_dropped = self.mp.matmul(grad_o, V.transpose(0, 1, 3, 2))
        grad_V = self.mp.matmul(attn_weights_dropped.transpose(0, 1, 3, 2), grad_o)

        # 5. Backward through attn_dropout
        grad_attn_weights, _ = self.attn_dropout.backward(grad_attn_weights_dropped)

        # 6. Backward through softmax (attn_weights = softmax(masked_scores))
        # dL/dx = y * (dL/dy - sum(dL/dy * y)) where y = softmax(x)
        grad_masked_scores = grad_attn_weights * attn_weights - self.mp.sum(grad_attn_weights * attn_weights, axis=-1, keepdims=True) * attn_weights

        # 7. Backward through scores + causal_mask (causal_mask is constant, so its gradient is 0)
        grad_scores = grad_masked_scores

        # 8. Backward through scaled dot-product: scores = (Q @ K.T) / sqrt(d_k)
        # Let S = Q @ K.T / sqrt(d_k)
        # dL/dQ = (dL/dS @ K) / sqrt(d_k)
        # dL/dK = (dL/dS.T @ Q) / sqrt(d_k)
        grad_Q = self.mp.matmul(grad_scores, K) / mt.sqrt(self.d_k)
        grad_K = self.mp.matmul(grad_scores.transpose(0, 1, 3, 2), Q) / mt.sqrt(self.d_k)

        # 9. Undo split_heads for Q, K, V to get gradients for original Q_orig, K_orig, V_orig
        def un_split_heads(z_grad, original_shape):
            B_s, NH_s, T_s, DK_s = z_grad.shape
            z_grad = z_grad.transpose(0, 2, 1, 3) # (B, T, n_heads, d_k)
            return z_grad.reshape(original_shape) # (B, T, head_size)

        grad_Q_orig = un_split_heads(grad_Q, Q_orig.shape)
        grad_K_orig = un_split_heads(grad_K, K_orig.shape)
        grad_V_orig = un_split_heads(grad_V, V_orig.shape)

        # 10. Backward through q_proj, k_proj, v_proj
        grad_x_q, q_proj_grads = self.q_proj.backward(grad_Q_orig)
        grad_x_k, k_proj_grads = self.k_proj.backward(grad_K_orig)
        grad_x_v, v_proj_grads = self.v_proj.backward(grad_V_orig)

        # Sum gradients for the input 'x' from all three paths (Q, K, V)
        grad_x = grad_x_q + grad_x_k + grad_x_v

        # Assemble gradients in the correct order: q_proj, k_proj, v_proj, c_proj
        current_mha_param_grads.extend(q_proj_grads)
        current_mha_param_grads.extend(k_proj_grads)
        current_mha_param_grads.extend(v_proj_grads)
        current_mha_param_grads.extend(c_proj_grads)

        return grad_x, current_mha_param_grads
    
    def from_dict(self, weights_dict, i):
        self.q_proj.weight = weights_dict[f'block_{i}_mha_q_weight']
        self.k_proj.weight = weights_dict[f'block_{i}_mha_k_weight']
        self.v_proj.weight = weights_dict[f'block_{i}_mha_v_weight']
        self.c_proj.weight = weights_dict[f'block_{i}_mha_c_weight']
        if weights_dict[f'block_{i}_mha_c_bias'] is not None:
            self.c_proj.bias = weights_dict[f'block_{i}_mha_c_bias']

        self.q_proj._parameters = [self.q_proj.weight]
        self.k_proj._parameters = [self.k_proj.weight]
        self.v_proj._parameters = [self.v_proj.weight]
        self.c_proj._parameters = [self.c_proj.weight]
        if self.c_proj.bias is not None:
            self.c_proj._parameters.append(self.c_proj.bias)

    def to_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_mha_q_weight'] = self.q_proj.weight
        weights_dict[f'block_{i}_mha_k_weight'] = self.k_proj.weight
        weights_dict[f'block_{i}_mha_v_weight'] = self.v_proj.weight
        weights_dict[f'block_{i}_mha_c_weight'] = self.c_proj.weight
        weights_dict[f'block_{i}_mha_c_bias'] = self.c_proj.bias if self.c_proj.bias is not None else None