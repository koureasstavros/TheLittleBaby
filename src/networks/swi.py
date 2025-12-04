#########################
# Swish-Gated Linear (SWI)
# Author: Koureas Stavros
#########################
from src.module import Module
from src.layers.linear import Linear
from src.functions.process import sigmoid, sigmoid_prime

class SWI(Module):
    """
    Swish-Gated Linear (SWI)
    """
    def __init__(self, mp, d_type, n_ctx, n_emb_in, n_emb_out, n_expansion):
        super().__init__()
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb_in = n_emb_in
        n_emb_hid = n_emb_in * n_expansion
        self.n_emb_hid = n_emb_hid
        self.n_emb_out = n_emb_out

        # Projection layers
        self.c_proj_up = Linear(mp, d_type, n_emb_in, n_emb_hid, bias=True)
        self.c_proj_gt = Linear(mp, d_type, n_emb_in, n_emb_hid, bias=True)
        self.c_proj_dn = Linear(mp, d_type, n_emb_hid, n_emb_out, bias=True)
    
    def set(self, mode=True):
        self.c_proj_up.set(mode)
        self.c_proj_gt.set(mode)
        self.c_proj_dn.set(mode)

    def parameters(self):
        return self.c_proj_up.parameters() + self.c_proj_gt.parameters() + self.c_proj_dn.parameters()
    
    def flops(self, batch_size, training):
        """
        Estimate FLOPs for the SWI (SwiGLU) forward pass.
        Multiply-adds are counted as 2 FLOPs.
        batch_size: number of sequences in the batch
        seq_len: sequence length
        training: if True, include backward/update cost (~3x forward)
        """
        def linear_flops(in_f, out_f):
            return 2 * batch_size * self.n_ctx * in_f * out_f

        flops = 0

        # First projection (up branch)
        flops += linear_flops(self.n_emb_in, self.n_emb_hid)

        # Second projection (gate branch)
        flops += linear_flops(self.n_emb_in, self.n_emb_hid)

        # Swish activation: sigmoid (~4 FLOPs) + multiply (~1 FLOP)
        flops += 5 * batch_size * self.n_ctx * self.n_emb_hid

        # Elementwise multiply h1 * gate
        flops += batch_size * self.n_ctx * self.n_emb_hid

        # Final projection down
        flops += linear_flops(self.n_emb_hid, self.n_emb_out)

        if training:
            flops *= 3  # forward + backward + update

        return flops

    def forward(self, x):
        """
        Forward pass for the SwiGLU layer.
        """

        # 1. Projections
        self.h1 = self.c_proj_up.forward(x)
        self.h2 = self.c_proj_gt.forward(x)

        # 2. Gating Swish Mechanism
        self.sig_h2 = sigmoid(self.mp, self.h2)
        self.gate = self.h2 * self.sig_h2

        # 3. Gated Hidden State
        self.h = self.h1 * self.gate

        # 4. Final Projection
        self.out = self.c_proj_dn.forward(self.h)

        return self.out

    def backward(self, grad_output):
        """
        Backward pass for the SwiGLU layer.
        """

        # 1. Backward through final projection
        grad_h, c_proj_dn_grads = self.c_proj_dn.backward(grad_output)  # grad_h: (batch, hidden_features)

        # 2. Backward through gated hidden state
        grad_h1 = grad_h * self.gate
        grad_gate = grad_h * self.h1

        # 3. Backward through Swish Mechanism
        grad_h2 = grad_gate * (self.sig_h2 + self.h2 * sigmoid_prime(self.mp, self.sig_h2))

        # 4. Backward through projections
        grad_x1, c_proj_up_grads = self.c_proj_up.backward(grad_h1)
        grad_x2, c_proj_gt_grads = self.c_proj_gt.backward(grad_h2)

        # Total gradient w.r.t input
        grad_x = grad_x1 + grad_x2

        # Return gradient w.r.t input and all parameter gradients in correct order
        param_grads =  c_proj_up_grads + c_proj_gt_grads + c_proj_dn_grads
        
        return grad_x, param_grads
    
    def from_dict(self, weights_dict, i):
        self.c_proj_up.weight = weights_dict[f'block_{i}_swi_c_proj_up_weight']
        self.c_proj_up.bias = weights_dict[f'block_{i}_swi_c_proj_up_bias']
        self.c_proj_gt.weight = weights_dict[f'block_{i}_swi_c_proj_gt_weight']
        self.c_proj_gt.bias = weights_dict[f'block_{i}_swi_c_proj_gt_bias']
        self.c_proj_dn.weight = weights_dict[f'block_{i}_swi_c_proj_dn_weight']
        self.c_proj_dn.bias = weights_dict[f'block_{i}_swi_c_proj_dn_bias']

        self.c_proj_up.synchronize()        
        self.c_proj_gt.synchronize()
        self.c_proj_dn.synchronize()

    def towa_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_swi_c_proj_up_weight'] = self.c_proj_up.weight
        weights_dict[f'block_{i}_swi_c_proj_up_bias'] = self.c_proj_up.bias
        weights_dict[f'block_{i}_swi_c_proj_gt_weight'] = self.c_proj_gt.weight
        weights_dict[f'block_{i}_swi_c_proj_gt_bias'] = self.c_proj_gt.bias
        weights_dict[f'block_{i}_swi_c_proj_dn_weight'] = self.c_proj_dn.weight
        weights_dict[f'block_{i}_swi_c_proj_dn_bias'] = self.c_proj_dn.bias