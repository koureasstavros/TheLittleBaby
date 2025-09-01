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
    def __init__(self, mp, channels, kernel_size):
        super().__init__()
        self.mp = mp
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = mp.random.randn(channels, kernel_size) / mp.sqrt(channels)
        self.bias = mp.zeros(channels)
        self.synchronize()

    def synchronize(self):
        self._parameters = [self.weight]
        if self.bias is not None:
            self._parameters.append(self.bias)

    def forward(self, x):
        B, T, D = x.shape
        pad = self.kernel_size // 2
        x_padded = self.mp.pad(x, ((0,0),(pad,pad),(0,0)), mode='constant')
        out = self.mp.zeros_like(x)
        for i in range(self.kernel_size):
            out += x_padded[:, i:i+T, :] * self.weight[:, i]
        self._cache = (x, x_padded)
        return out + self.bias

    def backward(self, grad_out):
        x, x_padded = self._cache
        B, T, D = x.shape
        pad = self.kernel_size // 2
        grad_x_padded = self.mp.zeros_like(x_padded)
        grad_weight = self.mp.zeros_like(self.weight)
        grad_bias = grad_out.sum(axis=(0, 1))
        for i in range(self.kernel_size):
            grad_weight[:, i] += (grad_out * x_padded[:, i:i+T, :]).sum(axis=(0, 1))
            grad_x_padded[:, i:i+T, :] += grad_out * self.weight[:, i]
        grad_x = grad_x_padded[:, pad:pad+T, :]
        # Return flat list of grads
        return grad_x, [grad_weight, grad_bias]