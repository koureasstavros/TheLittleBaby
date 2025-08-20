#########################
# Embedding Layer
# Author: Koureas Stavros
#########################

from src.module import Module

class Embedding(Module):
    """
    A simple Embedding layer.
    Maps integer indices to dense vectors.
    """
    def __init__(self, mp, num_embeddings, embedding_dim):
        super().__init__()
        self.mp = mp
        
        # Initialize weights with small random values
        self.weight = self.mp.random.randn(num_embeddings, embedding_dim) * 0.02
        self._parameters = [self.weight] # Register weight as a parameter

    def forward(self, x):
        """
        Forward pass for Embedding layer.
        x: input indices (e.g., token IDs), shape (B, T) or (T,)
        Returns: embedded vectors, shape (B, T, embedding_dim) or (T, embedding_dim)
        """
        self._cache = x # Store input indices for backward pass
        return self.weight[x]

    def backward(self, grad_output):
        """
        Backward pass for Embedding layer.
        grad_output: gradient from the subsequent layer, shape (B, T, embedding_dim)
        Returns: (grad_input, [grad_weight])
        """
        x = self._cache # Retrieve input indices
        grad_weight = self.mp.zeros_like(self.weight) # Initialize gradient for weights

        # Accumulate gradients for each embedding used.
        # This is a sparse update: only update rows corresponding to input indices.
        # if x.ndim == 1: # Handle (T,) input case
        #     for i, idx in enumerate(x):
        #         grad_weight[idx] += grad_output[i]
        # else: # Handle (B, T) input case
        #     for b in range(x.shape[0]):
        #         for t in range(x.shape[1]):
        #             grad_weight[x[b, t]] += grad_output[b, t]

        # Efficiently accumulate gradients using self.mp.add.at
        # Remove loops in Embedding.backward (lets BLAS + vectorized add handle parallelism)
        grad_weight = self.mp.zeros_like(self.weight)
        if x.ndim == 1:
            self.mp.add.at(grad_weight, x, grad_output)
        else:
            self.mp.add.at(grad_weight, x.reshape(-1), grad_output.reshape(-1, grad_output.shape[-1]))

        # For embedding layer, there's no gradient to pass back to the input (it's just indices).
        return None, [grad_weight]