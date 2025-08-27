#########################
# Recurrent Focused Attention (RFA)
# Author: Koureas Stavros
#########################

import math as mt
from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import softmax

class RFA(Module):
    """
    Recurrent Focused Attention:
    - Local sliding window attention for short-term context
    - Recurrent memory vector per head for long-term context
    """
    def __init__(self, mp, n_emb, n_ctx, p_dropout, head_size, n_heads, window_size):
        super().__init__()
        assert head_size % n_heads == 0
        self.mp = mp
        self.n_emb = n_emb
        self.n_ctx = n_ctx
        self.head_size = head_size
        self.n_heads = n_heads
        self.d_k = head_size // n_heads
        self.window_size = window_size
        
        self.p_dropout = p_dropout
        self.k_cache = None
        self.v_cache = None

        self.q_proj = Linear(mp, n_emb, head_size, bias=False)
        self.k_proj = Linear(mp, n_emb, head_size, bias=False)
        self.v_proj = Linear(mp, n_emb, head_size, bias=False)
        self.c_proj = Linear(mp, head_size, n_emb)

        self.attn_dropout = Dropout(mp, p_dropout)
        self.resid_dropout = Dropout(mp, p_dropout)

        # Recurrent memory per head
        self.memory = self.mp.zeros((1, n_heads, 1, self.d_k))

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.memory = self.mp.zeros_like(self.memory)

    def parameters(self):
        return (self.q_proj.parameters() +
                self.k_proj.parameters() +
                self.v_proj.parameters() +
                self.c_proj.parameters())

    def forward(self, x, use_cache=False):
        B, T, _ = x.shape
        Q = self.q_proj.forward(x).reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = self.k_proj.forward(x).reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = self.v_proj.forward(x).reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Append recurrent memory to K/V
        if use_cache and self.k_cache is not None:
            K = self.mp.concatenate([self.k_cache, K], axis=2)
            V = self.mp.concatenate([self.v_cache, V], axis=2)
        self.k_cache = K
        self.v_cache = V      

        # Compute attention scores
        scores = self.mp.matmul(Q, K.transpose(0, 1, 3, 2)) / mt.sqrt(self.d_k)

        # Get actual sequence length after KV cache append
        actual_seq_len = K.shape[2]
        idxs = self.mp.arange(actual_seq_len)

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

        # Apply mask
        masked_scores = scores + mask[None, None, :, :]

        # Compute attention weights
        attn_weights = softmax(self.mp, masked_scores, axis=-1)
        attn_weights_dropped = self.attn_dropout.forward(attn_weights)

        o = self.mp.matmul(attn_weights_dropped, V)
        o_combined = o.transpose(0, 2, 1, 3).reshape(B, T, self.head_size)
        out = self.c_proj.forward(o_combined)
        out_dropped = self.resid_dropout.forward(out)

        # Update recurrent memory with last token's value
        self.memory = V[:, :, -1:, :].mean(axis=0, keepdims=True)

        if self.setting:
            self._cache = (x, Q, K, V, scores, masked_scores, attn_weights, attn_weights_dropped, o, o_combined)

        return out_dropped

    def backward(self, grad_output):
        (x, Q, K, V, scores, masked_scores, attn_weights, attn_weights_dropped, o, o_combined) = self._cache

        grad_out_dropped, _ = self.resid_dropout.backward(grad_output)
        grad_o_combined, c_proj_grads = self.c_proj.backward(grad_out_dropped)
        B, T, _ = grad_o_combined.shape
        grad_o = grad_o_combined.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        grad_attn_weights_dropped = self.mp.matmul(grad_o, V.transpose(0, 1, 3, 2))
        grad_V = self.mp.matmul(attn_weights_dropped.transpose(0, 1, 3, 2), grad_o)
        grad_attn_weights, _ = self.attn_dropout.backward(grad_attn_weights_dropped)

        grad_scores = grad_attn_weights * attn_weights - self.mp.sum(grad_attn_weights * attn_weights, axis=-1, keepdims=True) * attn_weights
        grad_Q = self.mp.matmul(grad_scores, K) / mt.sqrt(self.d_k)
        grad_K = self.mp.matmul(grad_scores.transpose(0, 1, 3, 2), Q) / mt.sqrt(self.d_k)

        def un_split(z_grad, shape):
            return z_grad.transpose(0, 2, 1, 3).reshape(shape)

        grad_Q_orig = un_split(grad_Q, (B, T, self.head_size))
        grad_K_orig = un_split(grad_K, (B, T, self.head_size))
        grad_V_orig = un_split(grad_V, (B, T, self.head_size))

        grad_x_q, q_proj_grads = self.q_proj.backward(grad_Q_orig)
        grad_x_k, k_proj_grads = self.k_proj.backward(grad_K_orig)
        grad_x_v, v_proj_grads = self.v_proj.backward(grad_V_orig)
        grad_x = grad_x_q + grad_x_k + grad_x_v

        return grad_x, q_proj_grads + k_proj_grads + v_proj_grads + c_proj_grads

    def from_dict(self, weights_dict, i):
        self.q_proj.weight = weights_dict[f'block_{i}_rfa_q_weight']
        self.k_proj.weight = weights_dict[f'block_{i}_rfa_k_weight']
        self.v_proj.weight = weights_dict[f'block_{i}_rfa_v_weight']
        self.c_proj.weight = weights_dict[f'block_{i}_rfa_c_weight']
        self.q_proj._parameters = [self.q_proj.weight]
        self.k_proj._parameters = [self.k_proj.weight]
        self.v_proj._parameters = [self.v_proj.weight]
        self.c_proj._parameters = [self.c_proj.weight]

    def to_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_rfa_q_weight'] = self.q_proj.weight
        weights_dict[f'block_{i}_rfa_k_weight'] = self.k_proj.weight
        weights_dict[f'block_{i}_rfa_v_weight'] = self.v_proj.weight
        weights_dict[f'block_{i}_rfa_c_weight'] = self.c_proj.weight