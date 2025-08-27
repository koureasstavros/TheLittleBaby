#########################
# Normalization Layer
# Author: Koureas Stavros
#########################

from src.module import Module

class Normalization(Module):
    """
    Layer Normalization layer.
    Normalizes features across the last dimension.
    It does use trainable parameters (gamma and beta).
    """
    def __init__(self, mp, dims, eps=1e-5):
        super().__init__()
        self.mp = mp
        self.eps = eps
        
        self.gamma = self.mp.ones(dims)  # Learnable scaling parameter
        self.beta = self.mp.zeros(dims)  # Learnable shifting parameter
        self._parameters = [self.gamma, self.beta]

    def forward(self, x):
        """
        Forward pass for LayerNorm.
        x: input tensor, shape (..., dims)
        Returns: normalized tensor, same shape as x
        """
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std = self.mp.sqrt(var + self.eps)
        x_norm = (x - mean) / std # Normalized input
        out = self.gamma * x_norm + self.beta

        # Store intermediate values for backward pass
        self._cache = (x, mean, std, x_norm)
        return out

    def backward(self, grad_output):
        """
        Backward pass for LayerNorm.
        grad_output: gradient from subsequent layer, same shape as forward output.
        Returns: (grad_input, [grad_gamma, grad_beta])
        """
        x, mean, std, x_norm = self._cache

        # Gradients for gamma and beta (sum over all dimensions except the last)
        grad_gamma = self.mp.sum(grad_output * x_norm, axis=tuple(range(grad_output.ndim - 1)))
        grad_beta = self.mp.sum(grad_output, axis=tuple(range(grad_output.ndim - 1)))

        # Gradient for x_norm (before gamma/beta scaling)
        grad_x_norm = grad_output * self.gamma

        # Gradient for x (through normalization formula)
        # This is a common and numerically stable way to compute LayerNorm's grad_x
        N = x.shape[-1] # Number of features in the last dimension

        # Part 1: Gradient through (x - mean) / std
        grad_x = grad_x_norm / std

        # Part 2: Gradient through std (which comes from var)
        grad_var = self.mp.sum(grad_x_norm * (x - mean) * (-0.5) * (std**(-3)), axis=-1, keepdims=True)

        # Part 3: Gradient through mean
        grad_mean = self.mp.sum(grad_x_norm * (-1 / std), axis=-1, keepdims=True) + grad_var * (-2 / N) * self.mp.sum(x - mean, axis=-1, keepdims=True)

        grad_x += (2 / N) * grad_var * (x - mean)
        grad_x += (1 / N) * grad_mean

        return grad_x, [grad_gamma, grad_beta]