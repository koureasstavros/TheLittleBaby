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
    def __init__(self, mp, d_type, n_ctx, n_emb, r_dropout, r_temp, s_kernel):
        super().__init__()
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.r_dropout = r_dropout
        self.r_temp = r_temp
        self.s_kernel = s_kernel

        # Projections
        self.k_proj = Linear(mp, d_type, n_emb, n_emb, bias=False)
        self.v_proj = Linear(mp, d_type, n_emb, n_emb, bias=False)
        self.c_proj = Linear(mp, d_type, n_emb, n_emb, bias=True)

        # Depthwise conv layer
        self.depthwise_conv = DepthwiseConv1D(mp, d_type, n_emb, s_kernel)

        # Dropouts
        self.attn_dropout = Dropout(mp, r_dropout)
        self.resid_dropout = Dropout(mp, r_dropout)

        # Normalization layer
        self.norm = Normalization(mp, d_type, n_emb)

        # KV cache for inference
        self.kv_cache = None
    
    def set(self, mode=True):
        super().set(mode)
        self.k_proj.set(mode)
        self.v_proj.set(mode)
        self.c_proj.set(mode)
        self.attn_dropout.set(mode)
        self.resid_dropout.set(mode)

        # Clear cache when switching to training mode
        if mode:
            self.clear_cache()

    def parameters(self):
        params = (self.k_proj.parameters() +
                  self.v_proj.parameters() +
                  self.c_proj.parameters() +
                  self.depthwise_conv.parameters() +
                  self.norm.parameters())
        return params
        
    def clear_cache(self):
        """Clear the KV cache."""
        self.kv_cache = None

    def flops(self, batch_size, training):
        """
        Estimate FLOPs for this LDA layer.
        Includes K/V projections, elementwise gating, normalization,
        depthwise convolution, and output projection.
        batch_size: number of sequences in the batch
        training: if True, include backward/update cost (~3x forward)
        """
        flops = 0

         # K and V projections: (B, T, D) x (D, D)
        flops += 2 * batch_size * self.n_ctx * self.n_emb * self.n_emb * 2

        # Elementwise gating (K * V)
        flops += batch_size * self.n_ctx * self.n_emb

        # Normalization (gamma*x + beta) ~2 FLOPs per element
        flops += 2 * batch_size * self.n_ctx * self.n_emb

        # Depthwise convolution: each channel has kernel_size multiplications per position
        flops += batch_size * self.n_ctx * self.n_emb * self.s_kernel * 2  # multiply–add ×2

        # Output projection: (B, T, D) x (D, D)
        flops += batch_size * self.n_ctx * self.n_emb * self.n_emb * 2

        # Bias add for output projection
        if self.c_proj.bias is not None:
            flops += batch_size * self.n_ctx * self.n_emb

        # Dropout (approximate)
        flops += batch_size * self.n_ctx * self.n_emb

        if training:
            flops *= 3  # forward + backward + update

        return flops

    def forward(self, x, use_cache):
        # 1. Project K,V
        k_lin = self.k_proj.forward(x)
        v_lin = self.v_proj.forward(x)

        # Apply temperature scaling to k_lin before gating
        k_lin_scaled = k_lin / self.r_temp

        # 2. Elementwise gate
        gated_v = k_lin_scaled * v_lin

        # Handle KV cache for inference
        if use_cache and not self.setting:
            if self.kv_cache is None:
                self.kv_cache = gated_v
            else:
                # Append and keep only last kernel_size-1 + current tokens
                self.kv_cache = self.mp.concat([self.kv_cache, gated_v], axis=1)
                max_len = self.s_kernel - 1 + gated_v.shape[1]
                if self.kv_cache.shape[1] > max_len:
                    self.kv_cache = self.kv_cache[:, -max_len:, :]
            # Always use the cache for context
            context = self.kv_cache
        else:
            # Use current gated_v as context during training
            context = gated_v

        # 3. Normalize context
        norm_context = self.norm.forward(context)

        # 4. Depthwise convolution
        mixed_conv = self.depthwise_conv.forward(norm_context)

        # 5. Add nonlinearity
        mixed = relu(self.mp, mixed_conv)

        # 6. Dropout output projection
        mixed_d = self.attn_dropout.forward(mixed)

        # 7. Output projection
        c_proj_out = self.c_proj.forward(mixed_d)

        # 8. Residual dropout
        dropped_out = self.resid_dropout.forward(c_proj_out)

        # 9. Cache intermediate values for backward pass
        if self.setting:
            self._cache = (x, k_lin, v_lin, gated_v, mixed)
        
        return dropped_out
    
    def backward(self, grad_output):
        """
        Backward pass for LDA.
        grad_output: gradient from subsequent layer, shape (B, T, D)
        Returns: (grad_input, list_of_param_grads)
        """

        # 1. Unpack cached values
        x, k_lin, v_lin, gated_v, mixed = self._cache
        param_grads = []

        # 2. Backward through residual dropout
        grad_out, _ = self.resid_dropout.backward(grad_output)

        # 3. Backward through output projection
        grad_mixed_d, c_proj_grads = self.c_proj.backward(grad_out)

        # 4. Backward through dropout output projection
        grad_mixed, _ = self.attn_dropout.backward(grad_mixed_d)

        # 5. Backward through add nonlinearity
        grad_mixed = grad_mixed * relu_prime(self.mp, mixed)

        # 6. Backward through depthwise convolution
        grad_norm_context, conv_grads = self.depthwise_conv.backward(grad_mixed)

        # 7. Backward through normalization
        grad_context, norm_grads = self.norm.backward(grad_norm_context)

        # 8. Backward through gating (elementwise multiply k_lin * v_lin)
        grad_k_lin = grad_context * v_lin
        grad_v_lin = grad_context * k_lin

        # 9. Backward through k_proj, v_proj 
        grad_x_v, v_proj_grads = self.v_proj.backward(grad_v_lin)
        grad_x_k, k_proj_grads = self.k_proj.backward(grad_k_lin)

        # Sum grads from both branches
        grad_x = grad_x_v + grad_x_k

        # Assemble grads in same order as parameters()
        param_grads.extend(k_proj_grads)
        param_grads.extend(v_proj_grads)
        param_grads.extend(c_proj_grads)
        param_grads.extend(conv_grads)
        param_grads.extend(norm_grads)

        return grad_x, param_grads
    
    def from_dict(self, weights_dict, i):
        self.s_kernel = int(weights_dict[f'block_{i}_lda_kernel_size'])
        self.k_proj.weight = weights_dict[f'block_{i}_lda_k_weight']
        self.v_proj.weight = weights_dict[f'block_{i}_lda_v_weight']
        self.c_proj.weight = weights_dict[f'block_{i}_lda_c_weight']
        self.c_proj.bias = weights_dict[f'block_{i}_lda_c_bias']
        self.depthwise_conv.weight = weights_dict[f'block_{i}_lda_conv_weight']
        self.depthwise_conv.bias = weights_dict[f'block_{i}_lda_conv_bias']
        self.norm.gamma = weights_dict[f'block_{i}_lda_norm_gamma']
        self.norm.beta = weights_dict[f'block_{i}_lda_norm_beta']

        self.k_proj.synchronize()
        self.v_proj.synchronize()
        self.c_proj.synchronize()

    def towa_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_lda_kernel_size'] = self.s_kernel
        weights_dict[f'block_{i}_lda_k_weight'] = self.k_proj.weight
        weights_dict[f'block_{i}_lda_v_weight'] = self.v_proj.weight
        weights_dict[f'block_{i}_lda_c_weight'] = self.c_proj.weight
        weights_dict[f'block_{i}_lda_c_bias'] = self.c_proj.bias
        weights_dict[f'block_{i}_lda_conv_weight'] = self.depthwise_conv.weight
        weights_dict[f'block_{i}_lda_conv_bias'] = self.depthwise_conv.bias
        weights_dict[f'block_{i}_lda_norm_gamma'] = self.norm.gamma
        weights_dict[f'block_{i}_lda_norm_beta'] = self.norm.beta