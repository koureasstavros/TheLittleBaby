#########################
# HOPE: Hierarchical Online Predictive Encoding
# Based on: "Nested Learning: The Illusion of Deep Learning Architectures"
# by Behrouz et al. (NeurIPS 2025)
# Author: Koureas Stavros
#########################

from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import sigmoid, sigmoid_prime


class HOPE(Module):
    """
    Hierarchical Online Predictive Encoding (HOPE)
    """

    def __init__(self, mp, d_type, n_ctx, n_emb, r_dropout, n_expansion, n_memory_levels, tau_min, tau_max, r_delta, use_predictive_coding=True):
        super().__init__()
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.r_dropout = r_dropout
        self.n_expansion = n_expansion
        self.n_memory_levels = n_memory_levels
        self.r_delta = r_delta
        self.use_predictive_coding = use_predictive_coding

        # Hidden dimension
        n_hidden = n_emb * n_expansion
        self.n_hidden = n_hidden

        # ========================================
        # Continuum Memory System
        # ========================================
        # Instead of discrete short/long-term memory, we use a continuum
        # of memory banks with logarithmically spaced decay rates.
        # tau_i = tau_min^(1-i/(n-1)) * tau_max^(i/(n-1))

        self.tau_values = []
        for i in range(n_memory_levels):
            if n_memory_levels > 1:
                ratio = i / (n_memory_levels - 1)
            else:
                ratio = 0.5
            tau = tau_min ** (1 - ratio) * tau_max ** ratio
            self.tau_values.append(tau)

        # Create memory banks for each level
        self.memory_banks = []
        for i in range(n_memory_levels):
            bank = Linear(mp, d_type, n_emb, n_hidden, bias=True)
            self.memory_banks.append(bank)

        # ========================================
        # Deep Memory (Delta Rule Associative Memory)
        # ========================================
        # Matrix-valued associative memory that compresses context
        # Uses delta rule for better capacity management

        # The deep memory projects input to a compressed representation
        self.deep_memory_key = Linear(mp, d_type, n_emb, n_hidden // 2, bias=False)
        self.deep_memory_value = Linear(mp, d_type, n_emb, n_hidden // 2, bias=False)

        # Deep memory state (M matrix) - initialized per forward
        self.deep_memory_state = None

        # ========================================
        # Predictive Coding Module
        # ========================================
        # Computes prediction error (surprise signal) for memory updates

        if use_predictive_coding:
            # Prediction head: predicts next hidden state
            self.predictor = Linear(mp, d_type, n_hidden, n_hidden, bias=True)
            # Surprise encoding: transforms prediction error
            self.surprise_encoder = Linear(mp, d_type, n_hidden, n_hidden // 2, bias=True)

        # ========================================
        # Memory Combiner with Learnable Gating
        # ========================================
        # Combines outputs from all memory levels

        # Gate projection: learns importance of each memory level
        self.gate_proj = Linear(mp, d_type, n_emb, n_memory_levels, bias=True)

        # Memory combiner: aggregates weighted memory outputs
        self.combiner = Linear(mp, d_type, n_memory_levels * n_hidden + n_hidden // 2, n_hidden, bias=True)

        # ========================================
        # SwiGLU Output Stage
        # ========================================
        # Gate projection for SwiGLU
        self.swi_gate = Linear(mp, d_type, n_emb, n_hidden, bias=True)

        # Final projection back to embedding dimension
        self.c_proj_dn = Linear(mp, d_type, n_hidden, n_emb, bias=True)

        # Dropout layer
        self.dropout = Dropout(mp, r_dropout)

        # ========================================
        # Memory State (for EMA updates)
        # ========================================
        self.memory_states = [None] * n_memory_levels

        # Learnable parameters for delta rule
        self.delta_scale = mp.ones((1, 1, n_hidden // 2)) * r_delta

    def set(self, mode=True):
        """Sets the HOPE module and its sub-modules to training/eval mode."""
        super().set(mode)

        for bank in self.memory_banks:
            bank.set(mode)

        self.deep_memory_key.set(mode)
        self.deep_memory_value.set(mode)

        if self.use_predictive_coding:
            self.predictor.set(mode)
            self.surprise_encoder.set(mode)

        self.gate_proj.set(mode)
        self.combiner.set(mode)
        self.swi_gate.set(mode)
        self.c_proj_dn.set(mode)
        self.dropout.set(mode)

        # Reset memory states when switching to training mode
        if mode:
            self.memory_states = [None] * self.n_memory_levels
            self.deep_memory_state = None

    def parameters(self):
        """Returns all parameters of the HOPE module."""
        params = []

        for bank in self.memory_banks:
            params.extend(bank.parameters())

        params.extend(self.deep_memory_key.parameters())
        params.extend(self.deep_memory_value.parameters())

        if self.use_predictive_coding:
            params.extend(self.predictor.parameters())
            params.extend(self.surprise_encoder.parameters())

        params.extend(self.gate_proj.parameters())
        params.extend(self.combiner.parameters())
        params.extend(self.swi_gate.parameters())
        params.extend(self.c_proj_dn.parameters())

        # Add learnable delta scale
        params.append(self.delta_scale)

        return params

    def flops(self, batch_size, training):
        """
        Estimate FLOPs for the HOPE forward pass.
        """
        def linear_flops(in_f, out_f):
            return 2 * batch_size * self.n_ctx * in_f * out_f

        flops = 0

        # Memory bank projections
        flops += self.n_memory_levels * linear_flops(self.n_emb, self.n_hidden)

        # Deep memory projections
        flops += 2 * linear_flops(self.n_emb, self.n_hidden // 2)

        # Deep memory read/write (delta rule operations)
        flops += 3 * batch_size * self.n_ctx * (self.n_hidden // 2) ** 2

        # Predictive coding
        if self.use_predictive_coding:
            flops += linear_flops(self.n_hidden, self.n_hidden)  # Predictor
            flops += linear_flops(self.n_hidden, self.n_hidden // 2)  # Surprise encoder

        # Gate projection
        flops += linear_flops(self.n_emb, self.n_memory_levels)

        # Softmax
        flops += 5 * batch_size * self.n_ctx * self.n_memory_levels

        # Combiner
        flops += linear_flops(self.n_memory_levels * self.n_hidden + self.n_hidden // 2, self.n_hidden)

        # SwiGLU
        flops += linear_flops(self.n_emb, self.n_hidden)
        flops += 5 * batch_size * self.n_ctx * self.n_hidden  # Swish activation
        flops += batch_size * self.n_ctx * self.n_hidden  # Gated multiplication

        # Final projection
        flops += linear_flops(self.n_hidden, self.n_emb)

        # EMA updates
        if training:
            flops += self.n_memory_levels * 3 * batch_size * self.n_ctx * self.n_hidden
            flops *= 3

        return flops

    def _update_memory_state(self, memory_output, state, tau):
        """
        Update memory state using exponential moving average.

        new_state = tau * old_state + (1 - tau) * memory_output

        Args:
            memory_output: Current memory bank output (B, T, n_hidden)
            state: Previous memory state or None
            tau: Decay rate

        Returns:
            Updated memory state
        """
        if state is None:
            return memory_output
        else:
            current_batch_size = memory_output.shape[0]
            state_batch_size = state.shape[0]
            if current_batch_size != state_batch_size:
                state = state[:current_batch_size]
            return tau * state + (1 - tau) * memory_output

    def _delta_rule_memory_update(self, M, k, v):
        """
        Update deep memory using delta rule.

        Delta rule: M_new = M + η * (v - M @ k) @ k^T

        This implements the "more expressive association" from the paper.

        Args:
            M: Memory matrix (B, d_k, d_k) or None
            k: Keys (B, T, d_k)
            v: Values (B, T, d_k)

        Returns:
            Updated memory matrix
        """
        B, T, d_k = k.shape

        if M is None:
            M = self.mp.zeros((B, d_k, d_k))

        # Compute mean key and value for stable update
        k_mean = self.mp.mean(k, axis=1)  # (B, d_k)
        v_mean = self.mp.mean(v, axis=1)  # (B, d_k)

        # Prediction: M @ k
        M_pred = self.mp.matmul(M, k_mean[:, :, None])[:, :, 0]  # (B, d_k)

        # Delta (prediction error)
        delta = v_mean - M_pred  # (B, d_k)

        # Delta rule update
        update = self.mp.matmul(
            delta[:, :, None],  # (B, d_k, 1)
            k_mean[:, None, :]  # (B, 1, d_k)
        )  # (B, d_k, d_k)

        # Apply learnable scale
        scale = self.mp.mean(self.delta_scale)
        M_new = M + scale * update

        return M_new

    def _read_deep_memory(self, M, q):
        """
        Read from deep memory using query.

        Args:
            M: Memory matrix (B, d_k, d_k)
            q: Query (B, T, d_k)

        Returns:
            Memory output (B, T, d_k)
        """
        B, T, d_k = q.shape

        if M is None:
            return self.mp.zeros((B, T, d_k))

        # Read: output = M @ q^T -> transpose back
        output = self.mp.matmul(M, q.transpose(0, 2, 1))  # (B, d_k, T)
        return output.transpose(0, 2, 1)  # (B, T, d_k)

    def forward(self, x):
        """
        Forward pass for HOPE.

        Args:
            x: Input tensor, shape (B, T, n_emb)

        Returns:
            Output tensor, shape (B, T, n_emb)
        """
        B, T, _ = x.shape

        # 1. Query all continuum memory banks

        memory_outputs = []
        for i, bank in enumerate(self.memory_banks):
            m_out = bank.forward(x)  # (B, T, n_hidden)

            # Update memory state with EMA
            if self.setting:
                self.memory_states[i] = self._update_memory_state(
                    m_out, self.memory_states[i], self.tau_values[i]
                )

            memory_outputs.append(m_out)

        # 2. Deep Memory with Delta Rule
        # Project to key and value space
        deep_k = self.deep_memory_key.forward(x)  # (B, T, n_hidden//2)
        deep_v = self.deep_memory_value.forward(x)  # (B, T, n_hidden//2)

        # Update deep memory
        if self.setting:
            self.deep_memory_state = self._delta_rule_memory_update(
                self.deep_memory_state, deep_k, deep_v
            )

        # Read from deep memory
        deep_memory_out = self._read_deep_memory(
            self.deep_memory_state, deep_k
        )  # (B, T, n_hidden//2)

        # 3. Predictive Coding (Surprise Signal)
        if self.use_predictive_coding and len(memory_outputs) > 0:
            # Predict next state from combined memories
            combined_for_pred = memory_outputs[0]  # Use fastest memory for prediction
            prediction = self.predictor.forward(combined_for_pred)  # (B, T, n_hidden)

            # Compute prediction error (surprise)
            # Use slowest memory as "target" (what should have been predicted)
            target = memory_outputs[-1]
            surprise = target - prediction

            # Encode surprise signal
            surprise_encoded = self.surprise_encoder.forward(surprise)  # (B, T, n_hidden//2)

            # Modulate deep memory with surprise
            deep_memory_out = deep_memory_out + 0.1 * surprise_encoded
        else:
            surprise = None
            surprise_encoded = None

        # 4. Compute memory importance gates
        gate_logits = self.gate_proj.forward(x)  # (B, T, n_memory_levels)

        # Softmax to get probability distribution
        gate_max = self.mp.max(gate_logits, axis=-1, keepdims=True)
        gate_exp = self.mp.exp(gate_logits - gate_max)
        gate_probs = gate_exp / self.mp.sum(gate_exp, axis=-1, keepdims=True)

        # 5. Weighted combination of memories
        weighted_memories = []
        for i, m_out in enumerate(memory_outputs):
            weighted = m_out * gate_probs[:, :, i:i+1]
            weighted_memories.append(weighted)

        # Concatenate all weighted memories plus deep memory
        m_concat = self.mp.concatenate(weighted_memories + [deep_memory_out], axis=-1)

        # Project combined memories
        m_combined = self.combiner.forward(m_concat)  # (B, T, n_hidden)

        # 6. SwiGLU activation
        swi_gate_out = self.swi_gate.forward(x)  # (B, T, n_hidden)
        sig_gate = sigmoid(self.mp, swi_gate_out)
        swish_gate = swi_gate_out * sig_gate  # Swish activation

        # Gated hidden state
        h_gated = m_combined * swish_gate  # (B, T, n_hidden)

        # 7. Final projection and dropout
        out = self.c_proj_dn.forward(h_gated)  # (B, T, n_emb)
        out = self.dropout.forward(out)

        # 8. Cache for backward pass
        self._cache = (
            x, memory_outputs, deep_k, deep_v, deep_memory_out,
            gate_logits, gate_probs, weighted_memories, m_concat, m_combined,
            swi_gate_out, sig_gate, swish_gate, h_gated,
            surprise, surprise_encoded if self.use_predictive_coding else None
        )

        return out

    def backward(self, grad_output):
        """
        Backward pass for HOPE.

        Args:
            grad_output: Gradient from subsequent layer, shape (B, T, n_emb)

        Returns:
            (grad_input, list_of_param_grads)
        """
        # 1. Unpack cached values
        (x, memory_outputs, deep_k, deep_v, deep_memory_out,
         gate_logits, gate_probs, weighted_memories, m_concat, m_combined,
         swi_gate_out, sig_gate, swish_gate, h_gated,
         surprise, surprise_encoded) = self._cache

        # 2. Backward through dropout
        grad_out, _ = self.dropout.backward(grad_output)

        # 3. Backward through final projection
        grad_h_gated, c_proj_dn_grads = self.c_proj_dn.backward(grad_out)

        # 4. Backward through gated multiplication
        grad_m_combined = grad_h_gated * swish_gate
        grad_swish_gate = grad_h_gated * m_combined

        # 5. Backward through swish
        grad_swi_gate_out = grad_swish_gate * (sig_gate + swi_gate_out * sigmoid_prime(self.mp, sig_gate))

        # 6. Backward through SwiGLU gate projection
        grad_x_swi, swi_gate_grads = self.swi_gate.backward(grad_swi_gate_out)

        # 7. Backward through combiner
        grad_m_concat, combiner_grads = self.combiner.backward(grad_m_combined)

        # 8. Split gradient back to weighted memories and deep memory
        split_idx = 0
        grad_weighted_memories = []
        for i in range(self.n_memory_levels):
            grad_weighted_memories.append(
                grad_m_concat[:, :, split_idx:split_idx + self.n_hidden]
            )
            split_idx += self.n_hidden
        grad_deep_memory_out = grad_m_concat[:, :, split_idx:]

        # 9. Backward through predictive coding (if used)
        surprise_encoder_grads = []
        predictor_grads = []
        if self.use_predictive_coding and surprise_encoded is not None:
            grad_surprise_encoded = 0.1 * grad_deep_memory_out
            grad_surprise, surprise_encoder_grads = self.surprise_encoder.backward(grad_surprise_encoded)

            # Gradient for predictor (through surprise = target - prediction)
            grad_prediction = -grad_surprise
            grad_combined_for_pred, predictor_grads = self.predictor.backward(grad_prediction)

            # Add gradient to first memory output
            grad_weighted_memories[0] = grad_weighted_memories[0] + grad_combined_for_pred

            # Add gradient to last memory output (target)
            grad_weighted_memories[-1] = grad_weighted_memories[-1] + grad_surprise

        # 10. Backward through weighted multiplication
        grad_memory_outputs = []
        grad_gate_probs_list = []

        for i in range(self.n_memory_levels):
            m_out = memory_outputs[i]
            grad_weighted = grad_weighted_memories[i]

            grad_m_out = grad_weighted * gate_probs[:, :, i:i+1]
            grad_memory_outputs.append(grad_m_out)

            grad_gate_prob_i = self.mp.sum(grad_weighted * m_out, axis=-1, keepdims=True)
            grad_gate_probs_list.append(grad_gate_prob_i)

        grad_gate_probs = self.mp.concatenate(grad_gate_probs_list, axis=-1)

        # 11. Backward through softmax
        sum_term = self.mp.sum(grad_gate_probs * gate_probs, axis=-1, keepdims=True)
        grad_gate_logits = gate_probs * (grad_gate_probs - sum_term)

        # 12. Backward through gate projection
        grad_x_gate, gate_proj_grads = self.gate_proj.backward(grad_gate_logits)

        # 13. Backward through deep memory (simplified)
        # Deep memory read backward
        grad_deep_k_from_read = grad_deep_memory_out  # Simplified

        # Deep memory key/value projections
        grad_x_deep_k, deep_memory_key_grads = self.deep_memory_key.backward(grad_deep_k_from_read)
        grad_x_deep_v, deep_memory_value_grads = self.deep_memory_value.backward(
            self.mp.zeros_like(deep_v)  # Simplified gradient for values
        )

        # Gradient for delta scale (simplified)
        grad_delta_scale = self.mp.zeros_like(self.delta_scale)

        # 14. Backward through memory banks
        grad_x_banks = self.mp.zeros_like(x)
        memory_bank_grads = []

        for i, bank in enumerate(self.memory_banks):
            grad_x_bank, bank_grads = bank.backward(grad_memory_outputs[i])
            grad_x_banks = grad_x_banks + grad_x_bank
            memory_bank_grads.extend(bank_grads)

        # 15. Sum all gradients w.r.t. input x
        grad_x = grad_x_banks + grad_x_gate + grad_x_swi + grad_x_deep_k + grad_x_deep_v

        # 16. Assemble parameter gradients in correct order
        param_grads = []
        param_grads.extend(memory_bank_grads)
        param_grads.extend(deep_memory_key_grads)
        param_grads.extend(deep_memory_value_grads)

        if self.use_predictive_coding:
            param_grads.extend(predictor_grads)
            param_grads.extend(surprise_encoder_grads)

        param_grads.extend(gate_proj_grads)
        param_grads.extend(combiner_grads)
        param_grads.extend(swi_gate_grads)
        param_grads.extend(c_proj_dn_grads)
        param_grads.append(grad_delta_scale)

        return grad_x, param_grads

    def from_dict(self, weights_dict, i):
        """Load weights from dictionary."""
        # Memory banks
        for j, bank in enumerate(self.memory_banks):
            bank.weight = weights_dict[f'block_{i}_hope_memory_{j}_weight']
            bank.bias = weights_dict[f'block_{i}_hope_memory_{j}_bias']
            bank.synchronize()

        # Deep memory projections
        self.deep_memory_key.weight = weights_dict[f'block_{i}_hope_deep_key_weight']
        self.deep_memory_value.weight = weights_dict[f'block_{i}_hope_deep_value_weight']

        # Predictive coding
        if self.use_predictive_coding:
            if f'block_{i}_hope_predictor_weight' in weights_dict:
                self.predictor.weight = weights_dict[f'block_{i}_hope_predictor_weight']
                self.predictor.bias = weights_dict[f'block_{i}_hope_predictor_bias']
                self.surprise_encoder.weight = weights_dict[f'block_{i}_hope_surprise_weight']
                self.surprise_encoder.bias = weights_dict[f'block_{i}_hope_surprise_bias']
                self.predictor.synchronize()
                self.surprise_encoder.synchronize()

        # Gate and combiner
        self.gate_proj.weight = weights_dict[f'block_{i}_hope_gate_weight']
        self.gate_proj.bias = weights_dict[f'block_{i}_hope_gate_bias']
        self.combiner.weight = weights_dict[f'block_{i}_hope_combiner_weight']
        self.combiner.bias = weights_dict[f'block_{i}_hope_combiner_bias']

        # SwiGLU gate
        self.swi_gate.weight = weights_dict[f'block_{i}_hope_swi_gate_weight']
        self.swi_gate.bias = weights_dict[f'block_{i}_hope_swi_gate_bias']

        # Final projection
        self.c_proj_dn.weight = weights_dict[f'block_{i}_hope_c_proj_dn_weight']
        self.c_proj_dn.bias = weights_dict[f'block_{i}_hope_c_proj_dn_bias']

        # Delta scale
        if f'block_{i}_hope_delta_scale' in weights_dict:
            self.delta_scale = weights_dict[f'block_{i}_hope_delta_scale']

        # Tau values
        if f'block_{i}_hope_tau_values' in weights_dict:
            self.tau_values = list(weights_dict[f'block_{i}_hope_tau_values'])

        # Synchronize all layers
        self.deep_memory_key.synchronize()
        self.deep_memory_value.synchronize()
        self.gate_proj.synchronize()
        self.combiner.synchronize()
        self.swi_gate.synchronize()
        self.c_proj_dn.synchronize()

    def towa_dict(self, weights_dict, i):
        """Save weights to dictionary."""
        # Memory banks
        for j, bank in enumerate(self.memory_banks):
            weights_dict[f'block_{i}_hope_memory_{j}_weight'] = bank.weight
            weights_dict[f'block_{i}_hope_memory_{j}_bias'] = bank.bias

        # Deep memory projections
        weights_dict[f'block_{i}_hope_deep_key_weight'] = self.deep_memory_key.weight
        weights_dict[f'block_{i}_hope_deep_value_weight'] = self.deep_memory_value.weight

        # Predictive coding
        if self.use_predictive_coding:
            weights_dict[f'block_{i}_hope_predictor_weight'] = self.predictor.weight
            weights_dict[f'block_{i}_hope_predictor_bias'] = self.predictor.bias
            weights_dict[f'block_{i}_hope_surprise_weight'] = self.surprise_encoder.weight
            weights_dict[f'block_{i}_hope_surprise_bias'] = self.surprise_encoder.bias

        # Gate and combiner
        weights_dict[f'block_{i}_hope_gate_weight'] = self.gate_proj.weight
        weights_dict[f'block_{i}_hope_gate_bias'] = self.gate_proj.bias
        weights_dict[f'block_{i}_hope_combiner_weight'] = self.combiner.weight
        weights_dict[f'block_{i}_hope_combiner_bias'] = self.combiner.bias

        # SwiGLU gate
        weights_dict[f'block_{i}_hope_swi_gate_weight'] = self.swi_gate.weight
        weights_dict[f'block_{i}_hope_swi_gate_bias'] = self.swi_gate.bias

        # Final projection
        weights_dict[f'block_{i}_hope_c_proj_dn_weight'] = self.c_proj_dn.weight
        weights_dict[f'block_{i}_hope_c_proj_dn_bias'] = self.c_proj_dn.bias

        # Delta scale
        weights_dict[f'block_{i}_hope_delta_scale'] = self.delta_scale

        # Tau values
        weights_dict[f'block_{i}_hope_tau_values'] = self.mp.array(self.tau_values)
