#########################
# Multi-Layer Perceptron (MLP)
# Author: Koureas Stavros
#########################

from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import gelu, gelu_prime

class MLP(Module):
    """
    Multi-Layer Perceptron block, typically used in Transformer after attention.
    Consists of two linear layers with GELU activation and dropout.
    """
    def __init__(self, mp, n_emb, p_dropout, n_expansion):
        super().__init__()
        self.mp = mp
        
        # First linear layer (expands dimension)
        self.c_proj_up = Linear(mp, n_emb, n_expansion * n_emb)
        # Second linear layer (projects back to original dimension)
        self.c_proj_dn = Linear(mp, n_expansion * n_emb, n_emb)
        # Dropout layer
        self.p_dropout = Dropout(mp, p_dropout)

    def parameters(self):
        """Returns all parameters of the MLP module."""
        return self.c_proj_up.parameters() + self.c_proj_dn.parameters()

    def set(self, mode=True):
        """Sets the MLP module and its sub-modules to training/eval mode."""
        super().set(mode)
        self.c_proj_up.set(mode)
        self.c_proj_dn.set(mode)
        self.p_dropout.set(mode)

    def forward(self, x):
        """
        Forward pass for MLP.
        x: input tensor, shape (B, T, n_emb)
        Returns: output tensor, shape (B, T, n_emb)
        """
        self._cache_x = x # Store input to MLP for backward pass

        fc_out = self.c_proj_up.forward(x)
        gelu_out = gelu(self.mp, fc_out)
        proj_out = self.c_proj_dn.forward(gelu_out)
        dropped_out = self.p_dropout.forward(proj_out)

        # Store intermediate results for backward pass
        self._cache = (fc_out, gelu_out, proj_out)
        return dropped_out

    def backward(self, grad_output):
        """
        Backward pass for MLP.
        grad_output: gradient from subsequent layer.
        Returns: (grad_input, list_of_param_grads)
        """
        x = self._cache_x
        fc_out, gelu_out, proj_out = self._cache

        # Gradients will be collected in the order of self.parameters(): c_proj_up, c_proj_dn
        current_mlp_param_grads = []

        # 1. Backward through dropout
        grad_proj_out, _ = self.p_dropout.backward(grad_output)

        # 2. Backward through c_proj_dn
        grad_gelu_out, c_proj_dn_grads = self.c_proj_dn.backward(grad_proj_out)

        # 3. Backward through GELU activation
        # dL/dx = dL/dy * gelu_prime(x)
        grad_fc_out = grad_gelu_out * gelu_prime(self.mp, fc_out)

        # 4. Backward through c_proj_up
        grad_x, c_proj_up_grads = self.c_proj_up.backward(grad_fc_out)

        # Assemble gradients in the correct order: c_proj_up, c_proj_dn
        current_mlp_param_grads.extend(c_proj_up_grads)
        current_mlp_param_grads.extend(c_proj_dn_grads)

        return grad_x, current_mlp_param_grads
    
    def from_dict(self, weights_dict, i):
        self.c_proj_up.weight = weights_dict[f'block_{i}_mlp_proj_up_weight']
        if weights_dict[f'block_{i}_mlp_proj_up_bias'] is not None:
            self.c_proj_up.bias = weights_dict[f'block_{i}_mlp_proj_up_bias']
        self.c_proj_dn.weight = weights_dict[f'block_{i}_mlp_proj_dn_weight']
        if weights_dict[f'block_{i}_mlp_proj_dn_bias'] is not None:
            self.c_proj_dn.bias = weights_dict[f'block_{i}_mlp_proj_dn_bias']

        self.c_proj_up._parameters = [self.c_proj_up.weight]
        if self.c_proj_up.bias is not None:
            self.c_proj_up._parameters.append(self.c_proj_up.bias)
        self.c_proj_dn._parameters = [self.c_proj_dn.weight]
        if self.c_proj_dn.bias is not None:
            self.c_proj_dn._parameters.append(self.c_proj_dn.bias)

    def to_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_mlp_proj_up_weight'] = self.c_proj_up.weight
        weights_dict[f'block_{i}_mlp_proj_up_bias'] = self.c_proj_up.bias if self.c_proj_up.bias is not None else None
        weights_dict[f'block_{i}_mlp_proj_dn_weight'] = self.c_proj_dn.weight
        weights_dict[f'block_{i}_mlp_proj_dn_bias'] = self.c_proj_dn.bias if self.c_proj_dn.bias is not None else None