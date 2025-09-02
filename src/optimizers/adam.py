#########################
# Adam Optimizer
# Author: Koureas Stavros
#########################

class AdamW:
    """
    AdamW optimizer implementation.
    Includes adaptive learning rates and weight decay.
    """
    def __init__(self, mp, parameters, r_learn, beta1, beta2, eps, weight_decay):
        self.mp = mp
        self.params = parameters
        self.lr = r_learn
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

    def set_r_learn(self, r_learn, c_step=None, a_step=None, s_warmup=None):
        if s_warmup == "none":
            self.lr = r_learn
        elif s_warmup == "auto":
            s_warmup = a_step
            self.lr = r_learn * (c_step / s_warmup) if c_step <= s_warmup else r_learn
        else:
            s_warmup = int(s_warmup)
            self.lr = r_learn * (c_step / s_warmup) if c_step <= s_warmup else r_learn