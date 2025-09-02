#########################
# Adam Optimizer
# Author: Koureas Stavros
#########################

class AdamW:
    """
    AdamW optimizer implementation.
    Includes adaptive learning rates and weight decay.
    """
    def __init__(self, mp, parameters, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.mp = mp
        self.params = parameters
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0 # Timestep counter

        # Initialize first and second moment estimates for each parameter
        self.m = {id(p): mp.zeros_like(p) for p in self.params}
        self.v = {id(p): mp.zeros_like(p) for p in self.params}

    def step(self, grads):
        """
        Performs a single optimization step (parameter update).
        grads: a list of gradients, corresponding to self.params.
        """
        self.t += 1 # Increment timestep
        for p, g in zip(self.params, grads):
            pid = id(p) # Use object ID for unique parameter identification

            # Skip weight decay for 1D params (biases, LayerNorm gamma/beta)
            if self.weight_decay and p.ndim > 1:
                # Apply weight decay (L2 regularization)
                # This is applied directly to the gradient before the Adam update
                g = g + self.weight_decay * p

            # Update biased first moment estimate
            self.m[pid] = self.beta1 * self.m[pid] + (1 - self.beta1) * g
            # Update biased second raw moment estimate
            self.v[pid] = self.beta2 * self.v[pid] + (1 - self.beta2) * (g * g)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[pid] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[pid] / (1 - self.beta2 ** self.t)

            # Update parameters with bias-corrected estimates - gradient descent
            p -= self.lr * m_hat / (self.mp.sqrt(v_hat) + self.eps)