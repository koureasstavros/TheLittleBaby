#########################
# LoRA Network Module
# Author: Koureas Stavros
#########################
from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout

class LOR(Module):
    """
    LoRA module for adapting a pretrained linear layer with low-rank matrices.
    Formula:
        y = W(x) + alpha / r * (B(A(x)))
    where:
        W is the frozen pretrained weight,
        A and B are trainable low-rank matrices,
        r is the rank,
        alpha is a scaling factor.
    """
    def __init__(self, mp, n_ctx, n_emb, n_out, r_dropout, rank=4, alpha=1.0):
        super().__init__()
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.n_out = n_out
        self.r_dropout = r_dropout
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen base projection (pretrained)
        self.c_proj = Linear(mp, n_emb, n_out, bias=True)
        self.c_proj.set(mode=False)  # Keep frozen by default

        # LoRA trainable adapters
        self.c_proj_dn = Linear(mp, n_emb, rank, bias=False)
        self.c_proj_up = Linear(mp, rank, n_out, bias=False)

        # Dropout for regularization
        self.dropout = Dropout(mp, r_dropout)

    def set(self, mode=True):
        """Set mode for LoRA adapters and dropout (base_proj stays frozen)."""
        super().set(mode)
        self.c_proj_dn.set(mode)
        self.c_proj_up.set(mode)
        self.dropout.set(mode)

    def parameters(self):
        """Return only trainable LoRA parameters (A and B)."""
        return self.c_proj_dn.parameters() + self.c_proj_up.parameters()

    def flops(self, batch_size, training):
        """
        Estimate FLOPs for the LoRA forward pass.
        Multiply-adds are counted as 2 FLOPs.
        training: if True, include backward/update cost (~3x forward)
        """
        def linear_flops(in_f, out_f):
            return 2 * batch_size * self.n_ctx * in_f * out_f

        flops = 0

        # Base frozen projection (still computed)
        flops += linear_flops(self.n_emb, self.n_out)

        # LoRA down projection (A)
        flops += linear_flops(self.n_emb, self.rank)

        # LoRA up projection (B)
        flops += linear_flops(self.rank, self.n_out)

        if training:
            flops *= 3  # forward + backward + update

        return flops

    def forward(self, x):
        """
        Forward pass:
        x: (B, T, n_emb)
        returns: (B, T, n_out)
        """

        # 1. Base frozen projection
        base_out = self.c_proj.forward(x)

        # 2. LoRA adaptation
        lora_out = self.c_proj_up.forward(self.c_proj_dn.forward(x)) * self.scaling

        # 3. Combine and apply dropout
        out = base_out + lora_out

        # 4. Apply dropout
        out = self.dropout.forward(out)

        return out

    def backward(self, grad_output):
        """
        Backward pass for LoRA.
        Returns: (grad_x, param_grads)
        """

        # 1. Backward through dropout
        grad_combined, _ = self.dropout.backward(grad_output)

        # 2. Backward through split gradient (base_proj is frozen, so no grads)
        grad_lora = grad_combined

        # 3. Backward through LoRA adaptation
        grad_A_out, c_proj_up_grads = self.c_proj_up.backward(grad_lora * self.scaling)
        grad_x_lora, c_proj_dn_grads = self.c_proj_dn.backward(grad_A_out)

        # 4. Backward through base_proj (ignored for params, but needed for grad_x)
        grad_x_base, _ = self.c_proj.backward(grad_combined)

        # Assemble grad_x
        grad_x = grad_x_base + grad_x_lora

        # Assemble gradients
        param_grads = c_proj_dn_grads + c_proj_up_grads

        return grad_x, param_grads
    
    def from_dict(self, weights_dict, i):
        self.c_proj.weight = weights_dict[f'block_{i}_lor_c_proj_weight']
        self.c_proj.bias = weights_dict[f'block_{i}_lor_c_proj_bias']        
        self.c_proj_dn.weight = weights_dict[f'block_{i}_lor_c_proj_dn_weight']
        self.c_proj_up.weight = weights_dict[f'block_{i}_lor_c_proj_up_weight']
        self.rank = int(weights_dict[f'block_{i}_lor_rank'])
        self.alpha = float(weights_dict[f'block_{i}_lor_alpha'])

        self.c_proj.synchronize()
        self.c_proj_dn.synchronize()
        self.c_proj_up.synchronize()

    def towa_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_lor_c_proj_weight'] = self.c_proj.weight
        weights_dict[f'block_{i}_lor_c_proj_bias'] = self.c_proj.bias
        weights_dict[f'block_{i}_lor_c_proj_dn_weight'] = self.c_proj_dn.weight
        weights_dict[f'block_{i}_lor_c_proj_up_weight'] = self.c_proj_up.weight
        weights_dict[f'block_{i}_lor_rank'] = self.rank
        weights_dict[f'block_{i}_lor_alpha'] = self.alpha