#########################
# Gated Linear Network (GLN)
# Author: Koureas Stavros
#########################

from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import sigmoid, sigmoid_prime

class GLN(Module):
    """
    Gated Linear Network (GLN)
    """
    def __init__(self, mp, d_type, n_ctx, n_emb, r_dropout):
        super().__init__()
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.r_dropout = r_dropout

        # Main projection
        self.c_proj = Linear(mp, d_type, n_emb, n_emb, bias=True)

        # Gating mechanism
        self.g_proj = Linear(mp, d_type, n_emb, n_emb, bias=True)

        # Dropout layer
        self.dropout = Dropout(mp, r_dropout)
    
    def set(self, mode=True):
        super().set(mode)
        self.c_proj.set(mode)
        self.g_proj.set(mode)
        self.dropout.set(mode)

    def parameters(self):
        return self.c_proj.parameters() + self.g_proj.parameters()
        
    def flops(self, batch_size, training):
        """
        Estimate FLOPs for the GLN forward pass.
        Multiply-adds are counted as 2 FLOPs.
        training: if True, include backward/update cost (~3x forward)
        """
        def linear_flops(in_f, out_f):
            return 2 * batch_size * self.n_ctx * in_f * out_f

        flops = 0

        # Main projection
        flops += linear_flops(self.n_emb, self.n_emb)

        # Gate projection
        flops += linear_flops(self.n_emb, self.n_emb)
        # Sigmoid activation (~4 FLOPs per element)
        flops += 4 * batch_size * self.n_ctx * self.n_emb
        # Elementwise multiply with main projection
        flops += batch_size * self.n_ctx * self.n_emb

        if training:
            flops *= 3  # forward + backward + update

        return flops

    def forward(self, x):
        """
        x: (B,T,D)
        returns: (B,T,D)
        """

        # 1. Main projection
        self.c_proj_out = self.c_proj.forward(x)

        # 2. Gating projection        
        self.g_proj_out = self.g_proj.forward(x)
        self.g_sig = sigmoid(self.mp, self.g_proj_out)
        out = self.c_proj_out * self.g_sig

        # 3. Apply dropout
        out = self.dropout.forward(out)

        return out

    def backward(self, grad_output):
        """
        Backward pass:
        grad_output: Gradient of the output
        Returns: (grad_x, param_grads)
        """

        # 1. Backward through dropout
        grad_out, _ = self.dropout.backward(grad_output)

        # 2. Backward through gating
        grad_c_proj = grad_out * self.g_sig
        grad_g_sig = grad_out * self.c_proj_out
        g_proj = grad_g_sig * sigmoid_prime(self.mp, self.g_sig)

        grad_x_c, c_proj_grads = self.c_proj.backward(grad_c_proj)
        grad_x_g, g_proj_grads = self.g_proj.backward(g_proj)

        grad_x = grad_x_c + grad_x_g
        param_grads = c_proj_grads + g_proj_grads

        return grad_x, param_grads

    def from_dict(self, weights_dict, i):
        self.c_proj.weight = weights_dict[f'block_{i}_gln_c_weight']
        self.c_proj.bias = weights_dict[f'block_{i}_gln_c_bias']
        self.g_proj.weight = weights_dict[f'block_{i}_gln_g_weight']
        self.g_proj.bias = weights_dict[f'block_{i}_gln_g_bias']

        self.c_proj.synchronize()        
        self.g_proj.synchronize()

    def towa_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_gln_c_weight'] = self.c_proj.weight
        weights_dict[f'block_{i}_gln_c_bias'] = self.c_proj.bias
        weights_dict[f'block_{i}_gln_g_weight'] = self.g_proj.weight
        weights_dict[f'block_{i}_gln_g_bias'] = self.g_proj.bias