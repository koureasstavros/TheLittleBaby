#########################
# Dropout Layer
# Author: Koureas Stavros
#########################

from src.module import Module

class Dropout(Module):
    """
    Dropout layer.
    Randomly sets a fraction of input units to zero during training.
    """
    def __init__(self, mp, p):
        super().__init__()
        self.mp = mp
        self.p = p # Dropout probability

    def forward(self, x):
        """
        Forward pass for Dropout.
        x: input tensor
        Returns: output tensor with dropout applied (if training)
        """
        if self.setting and self.p > 0:
            # Create a mask: True for elements to keep, False for elements to drop
            # Scale by 1/(1-p) during training (inverted dropout)
            mask = (self.mp.random.rand(*x.shape) >= self.p) / (1.0 - self.p)
            self._cache = mask # Store mask for backward pass
            return x * mask
        self._cache = None # No mask if not training or p=0
        return x

    def backward(self, grad_output):
        """
        Backward pass for Dropout.
        grad_output: gradient from subsequent layer.
        Returns: (grad_input, []) - no parameters to update.
        """
        if self._cache is not None: # Only apply mask if dropout was active in forward
            mask = self._cache
            grad_input = grad_output * mask
        else: # If dropout was not active, simply pass gradient through
            grad_input = grad_output
        return grad_input, [] # Dropout has no parameters
