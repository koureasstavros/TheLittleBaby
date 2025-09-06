#########################
# Recurrent Focused Attention (RFA)
# Author: Koureas Stavros
#########################

import math as mt
from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import softmax, softmax_prime

class RFA(Module):
    """
    Recurrent Focused Attention:
    - Local sliding window attention for short-term context
    - Recurrent memory vector per head for long-term context
    """
    def __init__(self, mp, n_ctx, n_emb, r_dropout, head_size, n_heads, window_size):
        super().__init__()
        assert head_size % n_heads == 0, "head_size must be divisible by n_heads"
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.r_dropout = r_dropout
        self.head_size = head_size
        self.n_heads = n_heads
        self.d_k = head_size // n_heads
        self.window_size = window_size

        self.q_proj = Linear(mp, n_emb, head_size, bias=False)
        self.k_proj = Linear(mp, n_emb, head_size, bias=False)
        self.v_proj = Linear(mp, n_emb, head_size, bias=False)
        self.c_proj = Linear(mp, head_size, n_emb, bias=True)

        # Dropout layers
        self.attn_dropout = Dropout(mp, r_dropout)
        self.resid_dropout = Dropout(mp, r_dropout)

        # KV cache for inference
        self.kv_cache = None

        # Recurrent memory per head
        self.memory = self.mp.zeros((1, n_heads, 1, self.d_k))

    def clear_cache(self):
        """Clear the KV cache."""
        self.kv_cache = None
        self.memory = self.mp.zeros_like(self.memory)

    def parameters(self):
        """Returns all parameters of the attention module."""
        return (self.q_proj.parameters() +
                self.k_proj.parameters() +
                self.v_proj.parameters() +
                self.c_proj.parameters())

    def flops(self, batch_size, training):
        """
        Estimate FLOPs for this RFA layer.
        Uses local sliding window attention instead of full sequence attention.
        batch_size: number of sequences in the batch
        training: if True, include backward/update cost (~3x forward)
        """
        flops = 0

        # Q, K, V projections
        flops += 3 * batch_size * self.n_ctx * self.n_emb * self.head_size * 2

        # Attention score computation: Q @ K^T (local window)
        flops += batch_size * self.n_heads * self.n_ctx * self.window_size * self.d_k * 2

        # Masking
        flops += batch_size * self.n_heads * self.n_ctx * self.window_size

        # Softmax over local window
        flops += batch_size * self.n_heads * self.n_ctx * self.window_size * 5

        # Weighted sum: Attn @ V
        flops += batch_size * self.n_heads * self.n_ctx * self.window_size * self.d_k * 2

        # Output projection
        flops += batch_size * self.n_ctx * self.head_size * self.n_emb * 2

        # Bias add for output projection
        if self.c_proj.bias is not None:
            flops += batch_size * self.n_ctx * self.n_emb

        # Dropout (approximate)
        flops += batch_size * self.n_ctx * self.n_emb

        if training:
            flops *= 3  # forward + backward + update

        return flops
    
    def forward_split_heads(self, z):
        B, T, _ = z.shape
        return z.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def backward_unsplit_heads(self, z_grad, shape):
        return z_grad.transpose(0, 2, 1, 3).reshape(shape)
    
    def forward(self, x, use_cache=False):
        B, T, _ = x.shape

        # 1. Project input to Q, K, V
        Q = self.forward_split_heads(self.q_proj.forward(x))  # Split and transpose Q
        K = self.forward_split_heads(self.k_proj.forward(x))  # Split and transpose K
        V = self.forward_split_heads(self.v_proj.forward(x))  # Split and transpose V

        # Handle KV cache for inference
        if use_cache and self.kv_cache is not None:
            # Concatenate with cached K, V
            K_cached, V_cached = self.kv_cache
            K = self.mp.concatenate([K_cached, K], axis=2)
            V = self.mp.concatenate([V_cached, V], axis=2)
        self.kv_cache = (K, V)

        # 2. Compute attention scores
        scores = self.mp.matmul(Q, K.transpose(0, 1, 3, 2)) / mt.sqrt(self.d_k)

        # 3. Create and apply masks (local sliding window or causal mask)
        actual_seq_len = K.shape[2]
        idxs = self.mp.arange(actual_seq_len)

        # 4. Apply causal mask (prevents attending to future tokens)
        if use_cache:
            # Always sliding window when using KV cache
            local_mask = (idxs[None, :] >= idxs[:, None] - self.window_size) & (idxs[None, :] <= idxs[:, None])
            local_mask = self.mp.where(local_mask, 0, -1e9)
            # In cache mode with T==1, only keep last query row
            if T == 1 and actual_seq_len > 1:
                mask = local_mask[-1:, :]
            else:
                mask = local_mask[-T:, :]
        else:
            # Full causal mask when not using cache
            causal_mask = idxs[None, :] <= idxs[:, None]
            mask = self.mp.where(causal_mask, 0, -1e9)

        masked_scores = scores + mask[None, None, :, :]

        # 5. Compute attention weights using softmax
        attn_weights = softmax(self.mp, masked_scores, axis=-1)

        # 6. Apply dropout to attention weights
        attn_weights_dropped = self.attn_dropout.forward(attn_weights)

        # 7. Compute weighted sum of values
        o = self.mp.matmul(attn_weights_dropped, V)

        # 8. Recombine heads and reshape to (B, T, head_size)
        o_combined = o.transpose(0, 2, 1, 3).reshape(B, T, self.head_size)

        # 9. Final output projection
        out = self.c_proj.forward(o_combined)

        # 10. Apply residual dropout
        out_dropped = self.resid_dropout.forward(out)

        # 11. Update recurrent memory with the last token's value
        self.memory = V[:, :, -1:, :].mean(axis=0, keepdims=True)

        # 12. Cache intermediate values for backward pass (if needed)
        if self.setting:
            self._cache = (x, Q, K, V, scores, masked_scores, attn_weights, attn_weights_dropped, o, o_combined)

        return out_dropped

    def backward(self, grad_output):

        # 1. Unpack cached values
        (x, Q, K, V, scores, masked_scores, attn_weights, attn_weights_dropped, o, o_combined) = self._cache

        # 2. Backward through gradients for recurrent memory
        grad_memory = self.memory  # Gradient for recurrent memory
        self.memory -= grad_memory  # Update memory gradient

        # 3. Backward through residual dropout
        grad_out_dropped, _ = self.resid_dropout.backward(grad_output)

        # 4. Backward through final output projection
        grad_o_combined, c_proj_grads = self.c_proj.backward(grad_out_dropped)

        # 5. Backward through reshape/transpose for o_combined to get grad_o
        B, T, _ = grad_o_combined.shape
        grad_o = grad_o_combined.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # 6. Backward through weighted sum of values
        grad_attn_weights_dropped = self.mp.matmul(grad_o, V.transpose(0, 1, 3, 2))
        grad_V = self.mp.matmul(attn_weights_dropped.transpose(0, 1, 3, 2), grad_o)

        # 7. Backward through dropout to attention weights
        grad_attn_weights, _ = self.attn_dropout.backward(grad_attn_weights_dropped)

        # 8. Backward through softmax (attn_weights = softmax(masked_scores))
        grad_masked_scores = softmax_prime(self.mp, grad_attn_weights, attn_weights)
        
        # 9. Backward through causal_mask (causal_mask is constant, so its gradient is 0)
        grad_scores = grad_masked_scores

        # 10. Backward through scaled dot-product: scores = (Q @ K.T) / sqrt(d_k)
        grad_Q = self.mp.matmul(grad_scores, K) / mt.sqrt(self.d_k)
        grad_K = self.mp.matmul(grad_scores.transpose(0, 1, 3, 2), Q) / mt.sqrt(self.d_k)

        # 11. Undo split_heads for Q, K, V to get gradients for original Q_orig, K_orig, V_orig
        grad_Q_orig = self.backward_unsplit_heads(grad_Q, (B, T, self.head_size))
        grad_K_orig = self.backward_unsplit_heads(grad_K, (B, T, self.head_size))
        grad_V_orig = self.backward_unsplit_heads(grad_V, (B, T, self.head_size))

        # 12. Backward through q_proj, k_proj, v_proj
        grad_x_q, q_proj_grads = self.q_proj.backward(grad_Q_orig)
        grad_x_k, k_proj_grads = self.k_proj.backward(grad_K_orig)
        grad_x_v, v_proj_grads = self.v_proj.backward(grad_V_orig)

        # Combine gradients
        grad_x = grad_x_q + grad_x_k + grad_x_v
        param_grads = q_proj_grads + k_proj_grads + v_proj_grads + c_proj_grads

        return grad_x, param_grads

    def from_dict(self, weights_dict, i):
        self.q_proj.weight = weights_dict[f'block_{i}_rfa_q_weight']        
        self.k_proj.weight = weights_dict[f'block_{i}_rfa_k_weight']                  
        self.v_proj.weight = weights_dict[f'block_{i}_rfa_v_weight']        
        self.c_proj.weight = weights_dict[f'block_{i}_rfa_c_weight']
        self.c_proj.bias = weights_dict[f'block_{i}_rfa_c_bias']
        
        self.q_proj.synchronize()
        self.k_proj.synchronize()
        self.v_proj.synchronize()
        self.c_proj.synchronize()

    def towa_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_rfa_q_weight'] = self.q_proj.weight
        weights_dict[f'block_{i}_rfa_k_weight'] = self.k_proj.weight
        weights_dict[f'block_{i}_rfa_v_weight'] = self.v_proj.weight
        weights_dict[f'block_{i}_rfa_c_weight'] = self.c_proj.weight
        weights_dict[f'block_{i}_rfa_c_bias'] = self.c_proj.bias