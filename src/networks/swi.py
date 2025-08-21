#########################
# SwiGLU Network Module
# Author: Koureas Stavros
#########################
from src.module import Module
from src.layers.linear import Linear
from src.functions.process import sigmoid, sigmoid_prime

class SWI(Module):
    """
    SwiGLU: Gated MLP variant using Swish activation.
    y = Linear1(x) * swish(Linear2(x)), then projected by Linear3.
    """
    def __init__(self, mp, n_emb_in, n_emb_out, n_expansion):
        super().__init__()
        self.mp = mp
        self.n_emb_in = n_emb_in
        n_emb_hid = n_emb_in * n_expansion
        self.n_emb_hid = n_emb_hid
        self.n_emb_out = n_emb_out

        self.c_proj_up = Linear(mp, n_emb_in, n_emb_hid)
        self.c_proj_gt = Linear(mp, n_emb_in, n_emb_hid)
        self.linear_dn = Linear(mp, n_emb_hid, n_emb_out)

    def parameters(self):
        return self.c_proj_up.parameters() + self.c_proj_gt.parameters() + self.linear_dn.parameters()

    def set(self, mode=True):
        self.c_proj_up.set(mode)
        self.c_proj_gt.set(mode)
        self.linear_dn.set(mode)

    def swish(self, x):
        return x * sigmoid(self.mp, x)

    def swish_grad(self, x):
        sig = sigmoid(self.mp, x)
        return sig + x * sigmoid_prime(self.mp, x)

    def forward(self, x):
        self.x = x  # cache for backward
        self.h1 = self.c_proj_up.forward(x)
        self.h2 = self.c_proj_gt.forward(x)
        self.gate = self.swish(self.h2)
        self.h = self.h1 * self.gate
        self.out = self.linear_dn.forward(self.h)
        return self.out

    def backward(self, grad_output):
        # Backprop through c_proj_dn
        grad_h, c_proj_dn_grads = self.linear_dn.backward(grad_output)  # grad_h: (batch, hidden_features)

        # h = h1 * gate
        grad_h1 = grad_h * self.gate
        grad_gate = grad_h * self.h1

        # Backprop through swish
        grad_h2 = grad_gate * self.swish_grad(self.h2)

        # Backprop through c_proj_up and c_proj_gt
        grad_x1, c_proj_up_grads = self.c_proj_up.backward(grad_h1)
        grad_x2, c_proj_gt_grads = self.c_proj_gt.backward(grad_h2)

        # Total gradient w.r.t input
        grad_x = grad_x1 + grad_x2

        # Return gradient w.r.t input and all parameter gradients in correct order
        return grad_x, c_proj_up_grads + c_proj_gt_grads + c_proj_dn_grads
    
    def from_dict(self, weights_dict, i):
        self.c_proj_up.weight = weights_dict[f'block_{i}_swi_c_proj_up_weight']
        if weights_dict.get(f'block_{i}_swi_c_proj_up_bias') is not None:
            self.c_proj_up.bias = weights_dict[f'block_{i}_swi_c_proj_up_bias']
        self.c_proj_gt.weight = weights_dict[f'block_{i}_swi_c_proj_gt_weight']
        if weights_dict.get(f'block_{i}_swi_c_proj_gt_bias') is not None:
            self.c_proj_gt.bias = weights_dict[f'block_{i}_swi_c_proj_gt_bias']
        self.linear_dn.weight = weights_dict[f'block_{i}_swi_c_proj_dn_weight']
        if weights_dict.get(f'block_{i}_swi_c_proj_dn_bias') is not None:
            self.linear_dn.bias = weights_dict[f'block_{i}_swi_c_proj_dn_bias']

        self.c_proj_up._parameters = [self.c_proj_up.weight]
        if self.c_proj_up.bias is not None:
            self.c_proj_up._parameters.append(self.c_proj_up.bias)
        self.c_proj_gt._parameters = [self.c_proj_gt.weight]
        if self.c_proj_gt.bias is not None:
            self.c_proj_gt._parameters.append(self.c_proj_gt.bias)
        self.linear_dn._parameters = [self.linear_dn.weight]
        if self.linear_dn.bias is not None:
            self.linear_dn._parameters.append(self.linear_dn.bias)

    def to_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_swi_c_proj_up_weight'] = self.c_proj_up.weight
        weights_dict[f'block_{i}_swi_c_proj_up_bias'] = self.c_proj_up.bias if self.c_proj_up.bias is not None else None
        weights_dict[f'block_{i}_swi_c_proj_gt_weight'] = self.c_proj_gt.weight
        weights_dict[f'block_{i}_swi_c_proj_gt_bias'] = self.c_proj_gt.bias if self.c_proj_gt.bias is not None else None
        weights_dict[f'block_{i}_swi_c_proj_dn_weight'] = self.linear_dn.weight
        weights_dict[f'block_{i}_swi_c_proj_dn_bias'] = self.linear_dn.bias if self.linear_dn.bias is not None else None