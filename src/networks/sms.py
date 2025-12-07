#########################
# Continuum Memory SwiGLU (SMS)
# Author: Koureas Stavros
#########################

from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import sigmoid, sigmoid_prime


class SMS(Module):
    """
    Continuum Memory SwiGLU (SMS)
    """

    def __init__(self, mp, d_type, n_ctx, n_emb, r_dropout, n_expansion=4,
                 tau_fast=0.9, tau_med=0.99, tau_slow=0.999):
        super().__init__()
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.r_dropout = r_dropout
        self.n_expansion = n_expansion

        # Memory decay rates (tau values for EMA)
        self.tau_fast = tau_fast
        self.tau_med = tau_med
        self.tau_slow = tau_slow

        # Hidden dimension for SwiGLU
        n_hidden = n_emb * n_expansion
        self.n_hidden = n_hidden

        # ========================================
        # Multi-Rate Memory Banks
        # ========================================
        # Each memory bank projects input to hidden dimension
        # They share architecture but maintain separate weights
        # that evolve at different rates during training

        # Fast memory bank (tau = 0.9, decays quickly)
        # Captures recent, rapidly changing patterns
        self.memory_fast = Linear(mp, d_type, n_emb, n_hidden, bias=True)

        # Medium memory bank (tau = 0.99)
        # Captures mid-term dependencies
        self.memory_med = Linear(mp, d_type, n_emb, n_hidden, bias=True)

        # Slow memory bank (tau = 0.999, persistent)
        # Captures long-term, stable patterns
        self.memory_slow = Linear(mp, d_type, n_emb, n_hidden, bias=True)

        # ========================================
        # Memory Combiner with Learnable Gating
        # ========================================
        # Projects concatenated memories to combined representation
        # Input: 3 * n_hidden (fast + med + slow)
        # Output: n_hidden (combined memory)
        self.combiner = Linear(mp, d_type, 3 * n_hidden, n_hidden, bias=True)

        # Gating projection for memory importance weighting
        # Learns which memory banks are most relevant for each token
        self.gate_proj = Linear(mp, d_type, n_emb, 3, bias=True)

        # ========================================
        # SwiGLU Output Stage
        # ========================================
        # Gate projection for SwiGLU (parallel to combiner output)
        self.swi_gate = Linear(mp, d_type, n_emb, n_hidden, bias=True)

        # Final projection back to embedding dimension
        self.c_proj_dn = Linear(mp, d_type, n_hidden, n_emb, bias=True)

        # Dropout layer
        self.dropout = Dropout(mp, r_dropout)

        # ========================================
        # Memory State (for EMA updates)
        # ========================================
        # Running averages of memory outputs (initialized to None)
        # These are updated during training to implement continuum memory
        self.memory_state_fast = None
        self.memory_state_med = None
        self.memory_state_slow = None

    def set(self, mode=True):
        """Sets the SMS module and its sub-modules to training/eval mode."""
        super().set(mode)
        self.memory_fast.set(mode)
        self.memory_med.set(mode)
        self.memory_slow.set(mode)
        self.combiner.set(mode)
        self.gate_proj.set(mode)
        self.swi_gate.set(mode)
        self.c_proj_dn.set(mode)
        self.dropout.set(mode)

        # Reset memory states when switching to training mode
        if mode:
            self.memory_state_fast = None
            self.memory_state_med = None
            self.memory_state_slow = None

    def parameters(self):
        """Returns all parameters of the SMS module."""
        return (self.memory_fast.parameters() +
                self.memory_med.parameters() +
                self.memory_slow.parameters() +
                self.combiner.parameters() +
                self.gate_proj.parameters() +
                self.swi_gate.parameters() +
                self.c_proj_dn.parameters())

    def flops(self, batch_size, training):
        """
        Estimate FLOPs for the SMS forward pass.
        Multiply-adds are counted as 2 FLOPs.
        """
        def linear_flops(in_f, out_f):
            return 2 * batch_size * self.n_ctx * in_f * out_f

        flops = 0

        # Three memory bank projections: n_emb -> n_hidden
        flops += 3 * linear_flops(self.n_emb, self.n_hidden)

        # Gate projection: n_emb -> 3
        flops += linear_flops(self.n_emb, 3)

        # Softmax over gates (~5 FLOPs per element)
        flops += 5 * batch_size * self.n_ctx * 3

        # Weighted memory combination (element-wise multiply and sum)
        flops += 3 * batch_size * self.n_ctx * self.n_hidden

        # Combiner projection: 3 * n_hidden -> n_hidden
        flops += linear_flops(3 * self.n_hidden, self.n_hidden)

        # SwiGLU gate projection: n_emb -> n_hidden
        flops += linear_flops(self.n_emb, self.n_hidden)

        # Swish activation: sigmoid (~4 FLOPs) + multiply (~1 FLOP)
        flops += 5 * batch_size * self.n_ctx * self.n_hidden

        # Gated multiplication
        flops += batch_size * self.n_ctx * self.n_hidden

        # Final projection: n_hidden -> n_emb
        flops += linear_flops(self.n_hidden, self.n_emb)

        # EMA updates (if training): ~3 operations per element per memory
        if training:
            flops += 3 * 3 * batch_size * self.n_ctx * self.n_hidden
            flops *= 3  # forward + backward + update

        return flops

    def _update_memory_state(self, memory_output, state, tau):
        """
        Update memory state using exponential moving average.

        new_state = tau * old_state + (1 - tau) * memory_output

        Args:
            memory_output: Current memory bank output (B, T, n_hidden)
            state: Previous memory state or None
            tau: Decay rate (higher = slower decay, more persistent)

        Returns:
            Updated memory state
        """
        if state is None:
            return memory_output
        else:
            # Handle batch size mismatch (e.g., last batch may be smaller)
            current_batch_size = memory_output.shape[0]
            state_batch_size = state.shape[0]
            if current_batch_size != state_batch_size:
                # Truncate state to match current batch size
                state = state[:current_batch_size]
            return tau * state + (1 - tau) * memory_output

    def forward(self, x):
        """
        Forward pass for SMS (Continuum Memory SwiGLU).

        Args:
            x: Input tensor, shape (B, T, n_emb)

        Returns:
            Output tensor, shape (B, T, n_emb)
        """
        # ========================================
        # 1. Query all memory banks
        # ========================================
        m_fast = self.memory_fast.forward(x)   # (B, T, n_hidden)
        m_med = self.memory_med.forward(x)     # (B, T, n_hidden)
        m_slow = self.memory_slow.forward(x)   # (B, T, n_hidden)

        # ========================================
        # 2. Update memory states with EMA (during training)
        # ========================================
        # This implements the "continuum" aspect - memories at different timescales
        if self.setting:  # Training mode
            self.memory_state_fast = self._update_memory_state(
                m_fast, self.memory_state_fast, self.tau_fast)
            self.memory_state_med = self._update_memory_state(
                m_med, self.memory_state_med, self.tau_med)
            self.memory_state_slow = self._update_memory_state(
                m_slow, self.memory_state_slow, self.tau_slow)

        # ========================================
        # 3. Compute memory importance gates
        # ========================================
        # Learn which memory banks are most relevant for each token
        gate_logits = self.gate_proj.forward(x)  # (B, T, 3)

        # Softmax to get probability distribution over memories
        gate_max = self.mp.max(gate_logits, axis=-1, keepdims=True)
        gate_exp = self.mp.exp(gate_logits - gate_max)
        gate_probs = gate_exp / self.mp.sum(gate_exp, axis=-1, keepdims=True)  # (B, T, 3)

        # ========================================
        # 4. Weighted combination of memories
        # ========================================
        # Scale each memory by its learned importance
        m_fast_weighted = m_fast * gate_probs[:, :, 0:1]   # (B, T, n_hidden)
        m_med_weighted = m_med * gate_probs[:, :, 1:2]     # (B, T, n_hidden)
        m_slow_weighted = m_slow * gate_probs[:, :, 2:3]   # (B, T, n_hidden)

        # Concatenate all memories
        m_concat = self.mp.concatenate([m_fast_weighted, m_med_weighted, m_slow_weighted], axis=-1)  # (B, T, 3*n_hidden)

        # Project combined memories
        m_combined = self.combiner.forward(m_concat)  # (B, T, n_hidden)

        # ========================================
        # 5. SwiGLU activation
        # ========================================
        # Gate branch: swish(gate_proj(x))
        swi_gate_out = self.swi_gate.forward(x)  # (B, T, n_hidden)
        sig_gate = sigmoid(self.mp, swi_gate_out)
        swish_gate = swi_gate_out * sig_gate  # Swish activation

        # Gated hidden state
        h_gated = m_combined * swish_gate  # (B, T, n_hidden)

        # ========================================
        # 6. Final projection and dropout
        # ========================================
        out = self.c_proj_dn.forward(h_gated)  # (B, T, n_emb)
        out = self.dropout.forward(out)

        # ========================================
        # 7. Cache for backward pass
        # ========================================
        self._cache = (x, m_fast, m_med, m_slow, gate_logits, gate_probs,
                       m_fast_weighted, m_med_weighted, m_slow_weighted,
                       m_concat, m_combined, swi_gate_out, sig_gate, swish_gate, h_gated)

        return out

    def backward(self, grad_output):
        """
        Backward pass for SMS.

        Args:
            grad_output: Gradient from subsequent layer, shape (B, T, n_emb)

        Returns:
            (grad_input, list_of_param_grads)
        """
        # 1. Unpack cached values
        (_x, m_fast, m_med, m_slow, _gate_logits, gate_probs,
         _m_fast_weighted, _m_med_weighted, _m_slow_weighted,
         _m_concat, m_combined, swi_gate_out, sig_gate, swish_gate, _h_gated) = self._cache

        # 2. Backward through dropout
        grad_out, _ = self.dropout.backward(grad_output)

        # 3. Backward through final projection
        grad_h_gated, c_proj_dn_grads = self.c_proj_dn.backward(grad_out)

        # 4. Backward through gated multiplication: h_gated = m_combined * swish_gate
        grad_m_combined = grad_h_gated * swish_gate
        grad_swish_gate = grad_h_gated * m_combined

        # 5. Backward through swish activation: swish_gate = swi_gate_out * sig_gate
        # d(swish)/d(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        #               = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        grad_swi_gate_out = grad_swish_gate * (sig_gate + swi_gate_out * sigmoid_prime(self.mp, sig_gate))

        # 6. Backward through SwiGLU gate projection
        grad_x_swi, swi_gate_grads = self.swi_gate.backward(grad_swi_gate_out)

        # 7. Backward through combiner
        grad_m_concat, combiner_grads = self.combiner.backward(grad_m_combined)

        # 8. Split gradient back to weighted memories
        grad_m_fast_weighted = grad_m_concat[:, :, :self.n_hidden]
        grad_m_med_weighted = grad_m_concat[:, :, self.n_hidden:2*self.n_hidden]
        grad_m_slow_weighted = grad_m_concat[:, :, 2*self.n_hidden:]

        # 9. Backward through weighted multiplication
        # m_fast_weighted = m_fast * gate_probs[:, :, 0:1]
        grad_m_fast = grad_m_fast_weighted * gate_probs[:, :, 0:1]
        grad_m_med = grad_m_med_weighted * gate_probs[:, :, 1:2]
        grad_m_slow = grad_m_slow_weighted * gate_probs[:, :, 2:3]

        # Gradient w.r.t. gate_probs
        grad_gate_probs_0 = self.mp.sum(grad_m_fast_weighted * m_fast, axis=-1, keepdims=True)
        grad_gate_probs_1 = self.mp.sum(grad_m_med_weighted * m_med, axis=-1, keepdims=True)
        grad_gate_probs_2 = self.mp.sum(grad_m_slow_weighted * m_slow, axis=-1, keepdims=True)
        grad_gate_probs = self.mp.concatenate([grad_gate_probs_0, grad_gate_probs_1, grad_gate_probs_2], axis=-1)

        # 10. Backward through softmax
        # d(softmax)/d(logits) = softmax * (grad - sum(grad * softmax))
        sum_term = self.mp.sum(grad_gate_probs * gate_probs, axis=-1, keepdims=True)
        grad_gate_logits = gate_probs * (grad_gate_probs - sum_term)

        # 11. Backward through gate projection
        grad_x_gate, gate_proj_grads = self.gate_proj.backward(grad_gate_logits)

        # 12. Backward through memory banks
        grad_x_fast, memory_fast_grads = self.memory_fast.backward(grad_m_fast)
        grad_x_med, memory_med_grads = self.memory_med.backward(grad_m_med)
        grad_x_slow, memory_slow_grads = self.memory_slow.backward(grad_m_slow)

        # 13. Sum all gradients w.r.t. input x
        grad_x = grad_x_fast + grad_x_med + grad_x_slow + grad_x_gate + grad_x_swi

        # 14. Assemble parameter gradients in correct order
        param_grads = []
        param_grads.extend(memory_fast_grads)
        param_grads.extend(memory_med_grads)
        param_grads.extend(memory_slow_grads)
        param_grads.extend(combiner_grads)
        param_grads.extend(gate_proj_grads)
        param_grads.extend(swi_gate_grads)
        param_grads.extend(c_proj_dn_grads)

        return grad_x, param_grads

    def from_dict(self, weights_dict, i):
        """Load weights from dictionary."""
        # Memory banks
        self.memory_fast.weight = weights_dict[f'block_{i}_sms_memory_fast_weight']
        self.memory_fast.bias = weights_dict[f'block_{i}_sms_memory_fast_bias']
        self.memory_med.weight = weights_dict[f'block_{i}_sms_memory_med_weight']
        self.memory_med.bias = weights_dict[f'block_{i}_sms_memory_med_bias']
        self.memory_slow.weight = weights_dict[f'block_{i}_sms_memory_slow_weight']
        self.memory_slow.bias = weights_dict[f'block_{i}_sms_memory_slow_bias']

        # Combiner
        self.combiner.weight = weights_dict[f'block_{i}_sms_combiner_weight']
        self.combiner.bias = weights_dict[f'block_{i}_sms_combiner_bias']

        # Gate projection
        self.gate_proj.weight = weights_dict[f'block_{i}_sms_gate_proj_weight']
        self.gate_proj.bias = weights_dict[f'block_{i}_sms_gate_proj_bias']

        # SwiGLU gate
        self.swi_gate.weight = weights_dict[f'block_{i}_sms_swi_gate_weight']
        self.swi_gate.bias = weights_dict[f'block_{i}_sms_swi_gate_bias']

        # Final projection
        self.c_proj_dn.weight = weights_dict[f'block_{i}_sms_c_proj_dn_weight']
        self.c_proj_dn.bias = weights_dict[f'block_{i}_sms_c_proj_dn_bias']

        # Tau values (if saved)
        if f'block_{i}_sms_tau_fast' in weights_dict:
            self.tau_fast = float(weights_dict[f'block_{i}_sms_tau_fast'])
            self.tau_med = float(weights_dict[f'block_{i}_sms_tau_med'])
            self.tau_slow = float(weights_dict[f'block_{i}_sms_tau_slow'])

        # Synchronize all layers
        self.memory_fast.synchronize()
        self.memory_med.synchronize()
        self.memory_slow.synchronize()
        self.combiner.synchronize()
        self.gate_proj.synchronize()
        self.swi_gate.synchronize()
        self.c_proj_dn.synchronize()

    def towa_dict(self, weights_dict, i):
        """Save weights to dictionary."""
        # Memory banks
        weights_dict[f'block_{i}_sms_memory_fast_weight'] = self.memory_fast.weight
        weights_dict[f'block_{i}_sms_memory_fast_bias'] = self.memory_fast.bias
        weights_dict[f'block_{i}_sms_memory_med_weight'] = self.memory_med.weight
        weights_dict[f'block_{i}_sms_memory_med_bias'] = self.memory_med.bias
        weights_dict[f'block_{i}_sms_memory_slow_weight'] = self.memory_slow.weight
        weights_dict[f'block_{i}_sms_memory_slow_bias'] = self.memory_slow.bias

        # Combiner
        weights_dict[f'block_{i}_sms_combiner_weight'] = self.combiner.weight
        weights_dict[f'block_{i}_sms_combiner_bias'] = self.combiner.bias

        # Gate projection
        weights_dict[f'block_{i}_sms_gate_proj_weight'] = self.gate_proj.weight
        weights_dict[f'block_{i}_sms_gate_proj_bias'] = self.gate_proj.bias

        # SwiGLU gate
        weights_dict[f'block_{i}_sms_swi_gate_weight'] = self.swi_gate.weight
        weights_dict[f'block_{i}_sms_swi_gate_bias'] = self.swi_gate.bias

        # Final projection
        weights_dict[f'block_{i}_sms_c_proj_dn_weight'] = self.c_proj_dn.weight
        weights_dict[f'block_{i}_sms_c_proj_dn_bias'] = self.c_proj_dn.bias

        # Tau values
        weights_dict[f'block_{i}_sms_tau_fast'] = self.tau_fast
        weights_dict[f'block_{i}_sms_tau_med'] = self.tau_med
        weights_dict[f'block_{i}_sms_tau_slow'] = self.tau_slow
