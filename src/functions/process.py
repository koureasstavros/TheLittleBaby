#########################
# Process Functions
# Author: Koureas Stavros
#########################

import math as mt
from src.functions.runtime import is_debug

def gelu(mp, x):
    """Gaussian Error Linear Unit (GELU) activation function."""
    return 0.5 * x * (1.0 + mp.tanh(mt.sqrt(2/mt.pi) * (x + 0.044715 * mp.power(x, 3))))

def gelu_prime(mp, x):
    """Derivative of the GELU activation function."""
    k = mt.sqrt(2/mt.pi) * (x + 0.044715 * mp.power(x, 3))
    sech_sq = 1 / mp.cosh(k)**2 # sech^2(x) = 1 / cosh^2(x)
    k_prime = mt.sqrt(2/mt.pi) * (1 + 3 * 0.044715 * mp.power(x, 2))
    return 0.5 * (1 + mp.tanh(k)) + 0.5 * x * sech_sq * k_prime

def sigmoid(mp, x):
    """Compute the sigmoid activation function."""
    return 1 / (1 + mp.exp(-x))

def sigmoid_prime(mp, x):
    """Derivative of the sigmoid function."""
    s = 1 / (1 + mp.exp(-x))
    return s * (1 - s)

# Softmax (along given axis)
def softmax(mp, x, axis=-1):
    """Computes softmax probabilities along a given axis for numerical stability."""
    x_max = mp.max(x, axis=axis, keepdims=True)
    e_x = mp.exp(x - x_max)
    return e_x / mp.sum(e_x, axis=axis, keepdims=True)

def cross_entropy_loss(mp, logits, targets):
    """
    Computes cross-entropy loss and its gradient with respect to logits.
    logits: (B, T, vocab_size) - raw predictions from the model
    targets: (B, T) - true token IDs
    Returns: (loss_value, grad_logits)
    """
    B, T, C = logits.shape
    logits_flat = logits.reshape(B * T, C)
    targets_flat = targets.reshape(B * T)

    # For numerical stability: subtract max logit from all logits before exponentiation
    logits_max = mp.max(logits_flat, axis=1, keepdims=True)
    exp_logits = mp.exp(logits_flat - logits_max)

    sum_exp_logits = mp.sum(exp_logits, axis=1, keepdims=True)
    probs = exp_logits / sum_exp_logits # Softmax probabilities

    # Compute loss: - sum(target_one_hot * log(probs))
    log_probs = mp.log(probs + 1e-9) # Add epsilon for numerical stability to avoid log(0)
    loss = -mp.mean(log_probs[mp.arange(B * T), targets_flat])

    # Compute gradient of cross-entropy loss with respect to logits
    # The derivative of Cross-Entropy + Softmax is (probs - one_hot_targets)
    one_hot_targets = mp.zeros_like(probs)
    one_hot_targets[mp.arange(B * T), targets_flat] = 1

    grad_logits = probs - one_hot_targets # Shape: (B*T, C)
    grad_logits = grad_logits.reshape(B, T, C) # Reshape back to (B, T, C)

    return loss, grad_logits

def value_and_grad(self, x, y, use_cache):
    """
    Performs a forward pass to compute loss and then a backward pass
    to compute gradients for all model parameters.
    """
    # Forward pass
    is_debug("Start Forward pass")
    logits = self.forward(x, use_cache)
    is_debug("Stop Forward pass")
    # Compute loss and get initial gradient for logits from the loss function
    is_debug("Start Cross entropy")
    loss, grad_logits = cross_entropy_loss(self.mp, logits, y)
    is_debug("Stop Cross entropy")

    # Backward pass: The model's backward method takes the gradient from the loss
    # and propagates it back through all layers, returning gradients for parameters.
    is_debug("Start Backward pass")
    _, grads = self.backward(grad_logits) # grad_input for model is None
    is_debug("Stop Backward pass")

    return loss, grads

def value_and_nograd(self, x, y, use_cache):
    """
    Performs a forward pass to compute loss
    """
    # Forward pass
    is_debug("Start Forward pass")
    logits = self.forward(x, use_cache)
    is_debug("Stop Forward pass")
    # Compute loss from the loss function
    is_debug("Start Cross entropy")
    loss, _ = cross_entropy_loss(self.mp, logits, y)
    is_debug("Stop Cross entropy")

    return loss, _