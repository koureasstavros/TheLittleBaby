#########################
# Linear Diagonal Attention (LDA)
# Author: Koureas Stavros
#########################

from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.layers.normalization import Normalization
from src.layers.convolution import DepthwiseConv1D
from src.functions.process import relu, relu_prime

class LDA(Module):
    """
    Linear Diagonal Attention:
    - No QK^T, no cumsum
    - Uses local depthwise convolution over V to simulate attention
    - Complexity: O(B·T·D)
    Params order: k_proj, v_proj, c_proj
    """
    def __init__(self, mp, n_emb, n_ctx, p_dropout, kernel_size):
        super().__init__()
        self.mp = mp
        self.n_emb = n_emb
        self.n_ctx = n_ctx
        self.kernel_size = kernel_size

        # Projections
        self.k_proj = Linear(mp, n_emb, n_emb, bias=False)
        self.v_proj = Linear(mp, n_emb, n_emb, bias=False)
        self.c_proj = Linear(mp, n_emb, n_emb)

        # Depthwise conv layer
        self.depthwise_conv = DepthwiseConv1D(mp, n_emb, kernel_size)

        # Dropouts
        self.attn_dropout = Dropout(mp, p_dropout)
        self.resid_dropout = Dropout(mp, p_dropout)

        # Normalization layer
        self.norm = Normalization(mp, n_emb)

        # KV cache for inference
        self.kv_cache = None
        
    def clear_cache(self):
        """Clear the KV cache."""
        self.kv_cache = None

    def parameters(self):
        params = (self.k_proj.parameters() +
              self.v_proj.parameters() +
              self.c_proj.parameters() +
              self.depthwise_conv.parameters() +
              self.norm.parameters()
              )
        return params

    def set(self, mode=True):
        super().set(mode)
        for m in (self.k_proj, self.v_proj, self.c_proj,
                  self.attn_dropout, self.resid_dropout):
            m.set(mode)

        # Clear cache when switching to training mode
        if mode:
            self.clear_cache()

    def forward(self, x, use_cache):
        # Project K,V
        k_lin = self.k_proj.forward(x)
        v_lin = self.v_proj.forward(x)

        # Elementwise gate
        gated_v = k_lin * v_lin
        if use_cache and not self.setting:
            if self.kv_cache is None:
                self.kv_cache = gated_v
            else:
                # Append and keep only last kernel_size-1 + current tokens
                self.kv_cache = self.mp.concat([self.kv_cache, gated_v], axis=1)
                max_len = self.kernel_size - 1 + gated_v.shape[1]
                if self.kv_cache.shape[1] > max_len:
                    self.kv_cache = self.kv_cache[:, -max_len:, :]
            # Always use the cache for context
            context = self.kv_cache
        else:
            # Use current gated_v as context during training
            context = gated_v

        # Normalize context before depthwise convolution
        norm_context = self.norm.forward(context)        
        
        mixed_conv = self.depthwise_conv.forward(norm_context)

        # Add nonlinearity after depthwise convolution
        mixed = relu(self.mp, mixed_conv)

        # Dropout + output projection
        mixed_d = self.attn_dropout.forward(mixed)
        out = self.c_proj.forward(mixed_d)
        out = self.resid_dropout.forward(out)

        if self.setting:
            self._cache = (x, k_lin, v_lin, gated_v, mixed)
        return out
    
    def backward(self, grad_output):
        """
        Backward pass for LDA.
        grad_output: gradient from subsequent layer, shape (B, T, D)
        Returns: (grad_input, list_of_param_grads)
        """
        x, k_lin, v_lin, gated_v, mixed = self._cache
        param_grads = []

        # 1. Backward through resid_dropout
        grad_out, _ = self.resid_dropout.backward(grad_output)

        # 2. Backward through c_proj
        grad_mixed_d, c_proj_grads = self.c_proj.backward(grad_out)

        # 3. Backward through attn_dropout
        grad_mixed, _ = self.attn_dropout.backward(grad_mixed_d)

        # 4. Backward through depthwise conv
        grad_mixed = grad_mixed * relu_prime(self.mp, mixed)
        grad_norm_context, conv_grads = self.depthwise_conv.backward(grad_mixed)

        # 5. Backward through normalization
        grad_context, norm_grads = self.norm.backward(grad_norm_context)

        # 6. Backward through gating (elementwise multiply k_lin * v_lin)
        grad_k_lin = grad_context * v_lin
        grad_v_lin = grad_context * k_lin

        # 7. Backward through v_proj
        grad_x_v, v_proj_grads = self.v_proj.backward(grad_v_lin)

        # 8. Backward through k_proj
        grad_x_k, k_proj_grads = self.k_proj.backward(grad_k_lin)

        # 9. Sum grads from both branches
        grad_x = grad_x_v + grad_x_k

        # Assemble grads in same order as parameters()
        param_grads.extend(k_proj_grads)
        param_grads.extend(v_proj_grads)
        param_grads.extend(c_proj_grads)
        param_grads.extend(conv_grads)
        param_grads.extend(norm_grads)

        return grad_x, param_grads
    
    def from_dict(self, weights_dict, i):
        self.kernel_size = int(weights_dict[f'block_{i}_lda_kernel_size'])
        self.k_proj.weight = weights_dict[f'block_{i}_lda_k_weight']
        self.v_proj.weight = weights_dict[f'block_{i}_lda_v_weight']
        self.c_proj.weight = weights_dict[f'block_{i}_lda_c_weight']
        if weights_dict[f'block_{i}_lda_c_bias'] is not None:
            self.c_proj.bias = weights_dict[f'block_{i}_lda_c_bias']
        self.depthwise_conv.weight = weights_dict[f'block_{i}_lda_conv_weight']
        self.depthwise_conv.bias = weights_dict[f'block_{i}_lda_conv_bias']
        self.norm.gamma = weights_dict[f'block_{i}_lda_norm_gamma']
        self.norm.beta = weights_dict[f'block_{i}_lda_norm_beta']

        self.k_proj._parameters = [self.k_proj.weight]
        self.v_proj._parameters = [self.v_proj.weight]
        self.c_proj._parameters = [self.c_proj.weight]
        if self.c_proj.bias is not None:
            self.c_proj._parameters.append(self.c_proj.bias)

    def to_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_lda_kernel_size'] = self.kernel_size
        weights_dict[f'block_{i}_lda_k_weight'] = self.k_proj.weight
        weights_dict[f'block_{i}_lda_v_weight'] = self.v_proj.weight
        weights_dict[f'block_{i}_lda_c_weight'] = self.c_proj.weight
        weights_dict[f'block_{i}_lda_c_bias'] = self.c_proj.bias if self.c_proj.bias is not None else None
        weights_dict[f'block_{i}_lda_conv_weight'] = self.depthwise_conv.weight
        weights_dict[f'block_{i}_lda_conv_bias'] = self.depthwise_conv.bias
        weights_dict[f'block_{i}_lda_norm_gamma'] = self.norm.gamma
        weights_dict[f'block_{i}_lda_norm_beta'] = self.norm.beta