#########################
# Transformer Block
# Author: Koureas Stavros
#########################

from src.module import Module
from src.attention import Attention
from src.network import Network
from src.layers.normalization import Normalization

class Block(Module):
    """
    A single Transformer Block.
    Consists of Norm, Attention, Network
    Includes residual connections.
    """
    def __init__(self, mp, c_sequence, c_attention, c_network, n_emb, n_ctx, p_dropout, head_size, n_heads):
        super().__init__()
        self.c_sequence = c_sequence  # "pre" or "post"
        self.c_attention = c_attention  # "mha" or "moh"
        self.c_network = c_network  # "mlp" or "moe"

        self.ln_1 = Normalization(mp, n_emb)
        self.att = Attention(mp, c_attention, n_ctx, n_emb, p_dropout, head_size, n_heads)
        self.ln_2 = Normalization(mp, n_emb)
        self.net = Network(mp, c_network, n_ctx, n_emb, p_dropout)

    def parameters(self):
        """Returns all parameters of the Transformer Block."""
        return (self.ln_1.parameters() +
                self.att.parameters() +
                self.ln_2.parameters() +
                self.net.parameters())
    
    def flops(self, batch_size, training):
        """
        Estimate FLOPs for this Transformer Block.
        Includes normalization, attention, and network.
        """
        flops = 0

        # Normalization FLOPs ~ 2 * n_emb per token (mean + variance + scale + shift)
        norm_flops = 4 * batch_size * self.att.n_ctx * self.att.n_emb
        flops += norm_flops  # ln_1
        flops += self.att.flops(batch_size, training)
        flops += norm_flops  # ln_2
        flops += self.net.flops(batch_size, training)

        return flops
    
    def set(self, mode=True):
        """Sets the block and its sub-modules to training/eval mode."""
        super().set(mode)
        self.ln_1.set(mode)
        self.ln_2.set(mode)
        self.att.set(mode)
        self.net.set(mode)

    def forward(self, x, use_cache):
        """
        Forward pass for a Transformer Block.
        x: input tensor, shape (B, T, n_emb)
        Returns: output tensor, shape (B, T, n_emb)
        """
        self._cache_x = x # Store input for the first residual connection

        match self.c_sequence:
            case "pre":
                # Pre-Norm path
                # First residual connection: x + ATT(LayerNorm(x))
                ln1_out = self.ln_1.forward(x)
                att_out = self.att.forward(ln1_out, use_cache)
                x_res1 = x + att_out # Residual connection 1
                # Second residual connection: x_res1 + NET(LayerNorm(x_res1))
                ln2_out = self.ln_2.forward(x_res1)
                # Call net with cache only if it's AFT
                if self.c_network == "nft":
                    net_out = self.net.forward(ln2_out, use_cache)
                else:
                    net_out = self.net.forward(ln2_out)
                out = x_res1 + net_out # Residual connection 2
            case "post":
                # Post-Norm path
                # First residual connection: x + ATT(LayerNorm(x))
                att_out = self.att.forward(x, use_cache)
                res1 = x + att_out
                ln1_out = self.ln_1.forward(res1)
                # Second residual connection: ln1_out + NET(LayerNorm(ln1_out))
                if self.c_network == "nft":
                    net_out = self.net.forward(ln1_out, use_cache)
                else:
                    net_out = self.net.forward(ln1_out)
                res2 = ln1_out + net_out
                ln2_out = self.ln_2.forward(res2)
                self._cache = ("post", att_out, res1, ln1_out, net_out, res2)
                return ln2_out

        # Store intermediate values for backward pass
        self._cache = (self.c_sequence, ln1_out, att_out, ln2_out, net_out, x_res1)
        return out

    def backward(self, grad_output):
        """
        Backward pass for a Transformer Block.
        grad_output: gradient from subsequent layer.
        Returns: (grad_input, list_of_param_grads)
        """
        c_sequence_cache = self._cache[0]
        x = self._cache_x
        
        # Gradients will be collected in the order of self.parameters(): ln_1, mha, ln_2, mlp
        current_block_param_grads = []

        match c_sequence_cache:
            case "pre":
                # Pre-Norm path
                _, ln1_out, att_out, ln2_out, net_out, x_res1 = self._cache

                # 1. Backward through second residual connection and NET
                grad_x_res1_from_res2 = grad_output
                grad_net_out = grad_output

                grad_ln2_out, net_grads = self.net.backward(grad_net_out)

                grad_x_res1_from_ln2, ln2_grads = self.ln_2.backward(grad_ln2_out)

                # Sum gradients for x_res1 from both paths
                grad_x_res1 = grad_x_res1_from_res2 + grad_x_res1_from_ln2

                # 2. Backward through first residual connection and ATT
                grad_x_from_res1 = grad_x_res1
                grad_att_out = grad_x_res1

                grad_ln1_out, att_grads = self.att.backward(grad_att_out)

                grad_x_from_ln1, ln1_grads = self.ln_1.backward(grad_ln1_out)

                # Sum gradients for the initial input 'x' from both paths
                grad_x = grad_x_from_res1 + grad_x_from_ln1

                # Assemble gradients in the correct order: ln_1, mha, ln_2, mlp
                current_block_param_grads.extend(ln1_grads)
                current_block_param_grads.extend(att_grads)
                current_block_param_grads.extend(ln2_grads)
                current_block_param_grads.extend(net_grads)
                
            case c_sequence_cache:            
                # Post-Norm path
                _, att_out, res1, ln1_out, net_out, res2 = self._cache

                # 1. Backward through second residual connection and NET
                grad_res2, ln2_grads = self.ln_2.backward(grad_output)

                grad_ln1_out_from_res2 = grad_res2
                grad_net_out = grad_res2

                grad_ln1_out_from_net, net_grads = self.net.backward(grad_net_out)
                grad_ln1_out_total = grad_ln1_out_from_res2 + grad_ln1_out_from_net

                # 2. Backward through first residual connection and ATT
                grad_res1, ln1_grads = self.ln_1.backward(grad_ln1_out_total)

                grad_x_from_res1 = grad_res1
                grad_att_out = grad_res1

                grad_x_from_att, att_grads = self.att.backward(grad_att_out)

                grad_x = grad_x_from_res1 + grad_x_from_att

                # Order: att, ln1, net, ln2
                current_block_param_grads.extend(att_grads)
                current_block_param_grads.extend(ln1_grads)
                current_block_param_grads.extend(net_grads)
                current_block_param_grads.extend(ln2_grads)

        return grad_x, current_block_param_grads