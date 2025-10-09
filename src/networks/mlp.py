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
    def __init__(self, mp, d_type, n_ctx, n_emb, r_dropout, n_expansion):
        super().__init__()
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.r_dropout = r_dropout

        # First linear layer (expands dimension)
        self.c_proj_up = Linear(mp, d_type, n_emb, n_expansion * n_emb, bias=True)
        # Second linear layer (projects back to original dimension)
        self.c_proj_dn = Linear(mp, d_type, n_expansion * n_emb, n_emb, bias=True)
        
        # Dropout layer
        self.dropout = Dropout(mp, r_dropout)
    
    def set(self, mode=True):
        """Sets the MLP module and its sub-modules to training/eval mode."""
        super().set(mode)
        self.c_proj_up.set(mode)
        self.c_proj_dn.set(mode)
        self.dropout.set(mode)

    def parameters(self):
        """Returns all parameters of the MLP module."""
        return self.c_proj_up.parameters() + self.c_proj_dn.parameters()

    def flops(self, batch_size, training):
        """
        Estimate FLOPs for the MLP forward pass.
        Multiply-adds are counted as 2 FLOPs.
        training: if True, include backward/update cost (~3x forward)
        """

        flops = 0

        def linear_flops(in_f, out_f):
            return 2 * batch_size * self.n_ctx * in_f * out_f

        n_emb = self.c_proj_dn.weight.shape[0]  # output dim of MLP = embedding size
        n_hidden = self.c_proj_up.weight.shape[1]  # expanded hidden size

        # First projection (up)
        f_up = linear_flops(n_emb, n_hidden)

        # GELU activation (~4 FLOPs per element)
        f_gelu = 4 * batch_size * self.n_ctx * n_hidden

        # Second projection (down)
        f_down = linear_flops(n_hidden, n_emb)

        flops = f_up + f_gelu + f_down

        if training:
            flops *= 3  # forward + backward + update

        return flops

    def forward(self, x):
        """
        Forward pass for MLP.
        x: input tensor, shape (B, T, n_emb)
        Returns: output tensor, shape (B, T, n_emb)
        """

        # 1. Forward through c_proj_up
        fc_out = self.c_proj_up.forward(x)

        # 2. Apply GELU activation
        gelu_out = gelu(self.mp, fc_out)

        # 3. Project back to original dimension
        proj_out = self.c_proj_dn.forward(gelu_out)

        # 4. Apply dropout
        dropped_out = self.dropout.forward(proj_out)

        # 5. Store intermediate results for backward pass
        self._cache = (fc_out, gelu_out, proj_out)

        return dropped_out

    def backward(self, grad_output):
        """
        Backward pass for MLP.
        grad_output: gradient from subsequent layer.
        Returns: (grad_input, list_of_param_grads)
        """
        
        # 1. Unpack cached values
        fc_out, gelu_out, proj_out = self._cache

        # 2. Backward through dropout
        grad_proj_out, _ = self.dropout.backward(grad_output)

        # 3. Backward through c_proj_dn
        grad_gelu_out, c_proj_dn_grads = self.c_proj_dn.backward(grad_proj_out)

        # 4. Backward through GELU activation
        # dL/dx = dL/dy * gelu_prime(x)
        grad_fc_out = grad_gelu_out * gelu_prime(self.mp, fc_out)

        # 5. Backward through c_proj_up
        grad_x, c_proj_up_grads = self.c_proj_up.backward(grad_fc_out)

        # Assemble gradients in the correct order: c_proj_up, c_proj_dn
        param_grads = []
        param_grads.extend(c_proj_up_grads)
        param_grads.extend(c_proj_dn_grads)

        return grad_x, param_grads
    
    def from_dict(self, weights_dict, i):
        self.c_proj_up.weight = weights_dict[f'block_{i}_mlp_proj_up_weight']
        self.c_proj_up.bias = weights_dict[f'block_{i}_mlp_proj_up_bias']
        self.c_proj_dn.weight = weights_dict[f'block_{i}_mlp_proj_dn_weight']
        self.c_proj_dn.bias = weights_dict[f'block_{i}_mlp_proj_dn_bias']

        self.c_proj_up.synchronize()
        self.c_proj_dn.synchronize()

    def towa_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_mlp_proj_up_weight'] = self.c_proj_up.weight
        weights_dict[f'block_{i}_mlp_proj_up_bias'] = self.c_proj_up.bias
        weights_dict[f'block_{i}_mlp_proj_dn_weight'] = self.c_proj_dn.weight
        weights_dict[f'block_{i}_mlp_proj_dn_bias'] = self.c_proj_dn.bias