#########################
# Convolutional Layer
# Author: Koureas Stavros
#########################

from src.module import Module

class DepthwiseConv1D(Module):
    """
    Depthwise Convolution layer.
    Applies a depthwise convolution over the input.
    It does use trainable parameters (weights and bias).
    """
    def __init__(self, mp, n_channel, s_kernel=None, dtype=None):
        super().__init__()
        self.mp = mp
        self.n_channel = n_channel
        self.s_kernel = s_kernel
        self.dtype = dtype
        
        # Initialize weights with small random values
        self.weight = mp.random.randn(n_channel, s_kernel) / mp.sqrt(n_channel)
        # Initialize bias with small random values
        self.bias = mp.zeros(n_channel)
        # Syncronize Parameters
        self.synchronize()

    def synchronize(self):
        self._parameters = [self.weight]
        if self.bias is not None:
            self._parameters.append(self.bias)

    def forward(self, x):
        x = x.astype(self.dtype)
        B, T, D = x.shape
        pad = self.s_kernel // 2
        x_padded = self.mp.pad(x, ((0,0),(pad,pad),(0,0)), mode='constant')
        out = self.mp.zeros_like(x)
        for i in range(self.s_kernel):
            out += x_padded[:, i:i+T, :] * self.weight[:, i]
        self._cache = (x, x_padded)
        return out + self.bias

    def backward(self, grad_out):
        x, x_padded = self._cache
        B, T, D = x.shape
        pad = self.s_kernel // 2
        grad_x_padded = self.mp.zeros_like(x_padded)
        grad_weight = self.mp.zeros_like(self.weight)
        grad_bias = grad_out.sum(axis=(0, 1))
        for i in range(self.s_kernel):
            grad_weight[:, i] += (grad_out * x_padded[:, i:i+T, :]).sum(axis=(0, 1))
            grad_x_padded[:, i:i+T, :] += grad_out * self.weight[:, i]
        grad_x = grad_x_padded[:, pad:pad+T, :]
        return grad_x, [grad_weight, grad_bias]