########################
# Linear Layer
########################

from src.module import Module

class Linear(Module):
    """
    Linear (fully connected) layer.
    Performs y = x @ W + b.
    It does use trainable parameters (weights and bias).
    """
    def __init__(self, mp, in_features, out_features, bias=True):
        super().__init__()
        self.mp = mp
        
        # Initialize weights with small random values
        self.weight = self.mp.random.randn(in_features, out_features) * 0.02
        self.bias = self.mp.zeros(out_features) if bias else None
        self.synchronize()

    def synchronize(self):
        self._parameters = [self.weight]
        if self.bias is not None:
            self._parameters.append(self.bias)

    def forward(self, x):
        """
        Forward pass for Linear layer.
        x: input tensor, shape (..., in_features)
        Returns: output tensor, shape (..., out_features)
        """
        self._cache = x # Store input for backward pass
        out = x.dot(self.weight)
        
        if self.bias is not None:
            out = out + self.bias

        return out

    def backward(self, grad_output):
        """
        Backward pass for Linear layer.
        grad_output: gradient from the subsequent layer, shape (..., out_features)
        Returns: (grad_input, [grad_weight, grad_bias])
        """
        x = self._cache # Retrieve input
        original_x_shape = x.shape

        # Reshape x and grad_output to 2D for matrix multiplication for gradients
        # This handles arbitrary leading dimensions (B, T, ...)
        x_reshaped = x.reshape(-1, original_x_shape[-1]) # (N, in_features)
        grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1]) # (N, out_features)

        # Gradient for weight: dL/dW = x.T @ dL/dy
        grad_weight = x_reshaped.T @ grad_output_reshaped

        grad_bias = None
        if self.bias is not None:
            # Gradient for bias: dL/db = sum(dL/dy) along all dimensions except the last
            grad_bias = self.mp.sum(grad_output_reshaped, axis=0)

        # Gradient for input: dL/dx = dL/dy @ W.T
        grad_input = grad_output_reshaped @ self.weight.T
        grad_input = grad_input.reshape(original_x_shape) # Reshape back to original input shape

        param_grads = [grad_weight]
        if grad_bias is not None:
            param_grads.append(grad_bias)

        return grad_input, param_grads