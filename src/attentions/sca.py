#########################
# Semantic Centroid Attention (SCA)
# Author: Koureas Stavros + Copilot
#########################

import math as mt
from src.module import Module
from src.layers.linear import Linear
from src.layers.dropout import Dropout
from src.functions.process import softmax, softmax_prime, cosine_similarity, cosine_similarity_prime

class SCA(Module):
    """
    Semantic Centroid Attention:
    - Computes a global semantic centroid vector
    - Blends it with local attention over a small window
    - O(n·w) complexity instead of O(n²)
    """
    def __init__(self, mp, n_ctx, n_emb, r_dropout, head_size, window_size):
        super().__init__()
        self.mp = mp
        self.n_ctx = n_ctx
        self.n_emb = n_emb
        self.r_dropout = r_dropout
        self.head_size = head_size
        self.window_size = window_size

        # Linear projections
        self.q_proj = Linear(mp, n_emb, head_size, bias=False)
        self.k_proj = Linear(mp, n_emb, head_size, bias=False)
        self.v_proj = Linear(mp, n_emb, head_size, bias=False)
        self.c_proj = Linear(mp, head_size, n_emb, bias=True)

        # Dropouts
        self.resid_dropout = Dropout(mp, r_dropout)

        # Learnable blend factor
        self.gate = mp.array([0.5], dtype=mp.float32)

    def parameters(self):
        return (self.q_proj.parameters() +
                self.k_proj.parameters() +
                self.v_proj.parameters() +
                self.c_proj.parameters() +
                [self.gate])

    def forward(self, x, use_cache=False):
        """
        x: (B, T, n_emb)
        """
        B, T, _ = x.shape

        # 1. Projections
        Q = self.q_proj.forward(x)  # (B, T, head_size)
        K = self.k_proj.forward(x)
        V = self.v_proj.forward(x)

        # 2. Global semantic centroid
        mean_query = self.mp.mean(Q, axis=1, keepdims=True)  # (B, 1, head_size)
        sim = cosine_similarity(self.mp, K, mean_query)  # (B, T, 1)
        centroid = self.mp.sum(V * sim, axis=1, keepdims=True) / (self.mp.sum(sim, axis=1, keepdims=True) + 1e-6)

        # 3. Local attention
        local_out = self.mp.zeros_like(V)
        for i in range(T):
            start = max(0, i - self.window_size)
            end = min(T, i + self.window_size + 1)
            q_i = Q[:, i:i+1, :]  # (B, 1, head_size)
            k_local = K[:, start:end, :]
            v_local = V[:, start:end, :]
            attn_scores = self.mp.matmul(q_i, k_local.transpose(0, 2, 1)) / mt.sqrt(self.head_size)
            attn_weights = softmax(self.mp, attn_scores, axis=-1)
            local_out[:, i:i+1, :] = self.mp.matmul(attn_weights, v_local)

        # 4. Blend global + local
        out = self.gate * local_out + (1 - self.gate) * centroid

        # 5. Output projection + dropout
        out_proj = self.c_proj.forward(out)

        # 6. Residual dropout
        out_drop = self.resid_dropout.forward(out_proj)

        # 7. Cache intermediate values for backward pass
        if self.setting:
            self._cache = (x, Q, K, V, sim, centroid, local_out, out, out_proj)

        return out_drop

    def backward(self, grad_output):
        """
        Backward through SCA.
        """

        # 1. Unpack cached values
        (x, Q, K, V, sim, centroid, local_out, out, out_proj) = self._cache

        # 2. Backward through dropout
        out = self.resid_dropout.backward(grad_output)[0]

        # 3. Backward through output projection
        grad_out_proj, c_proj_grads = self.c_proj.backward(out)

        # 4. Backward through gradient wrt gate
        grad_gate = self.mp.sum((local_out - centroid) * grad_out_proj)
        grad_local_out = grad_out_proj * self.gate
        grad_centroid = grad_out_proj * (1 - self.gate)
        denom = (self.mp.sum(sim, axis=1, keepdims=True) + 1e-6)
        grad_V_centroid = grad_centroid * sim / denom
        grad_sim = grad_centroid * (V - centroid) / denom  # derivative wrt sim
        grad_K_centroid, grad_mean_query = cosine_similarity_prime(self.mp, grad_sim, K, self.mp.mean(Q, axis=1, keepdims=True))
        grad_Q_centroid = grad_mean_query / Q.shape[1]

        # 5. Backward through local attention
        grad_Q_local = self.mp.zeros_like(Q)
        grad_K_local = self.mp.zeros_like(K)
        grad_V_local = self.mp.zeros_like(V)

        B, T, _ = Q.shape
        for i in range(T):
            start = max(0, i - self.window_size)
            end = min(T, i + self.window_size + 1)
            q_i = Q[:, i:i+1, :]
            k_local = K[:, start:end, :]
            v_local = V[:, start:end, :]

            attn_scores = self.mp.matmul(q_i, k_local.transpose(0, 2, 1)) / mt.sqrt(self.head_size)
            attn_weights = softmax(self.mp, attn_scores, axis=-1)

            grad_v_local = self.mp.matmul(attn_weights.transpose(0, 2, 1), grad_local_out[:, i:i+1, :])
            grad_attn_weights = self.mp.matmul(grad_local_out[:, i:i+1, :], v_local.transpose(0, 2, 1))

            grad_scores = softmax_prime(self.mp, grad_attn_weights, attn_weights) / mt.sqrt(self.head_size)
            grad_q_i = self.mp.matmul(grad_scores, k_local)
            grad_k_local = self.mp.matmul(grad_scores.transpose(0, 2, 1), q_i)

            grad_Q_local[:, i:i+1, :] += grad_q_i
            grad_K_local[:, start:end, :] += grad_k_local
            grad_V_local[:, start:end, :] += grad_v_local

        # 6. Combine grads from centroid + local
        grad_V_total = grad_V_local + grad_V_centroid
        grad_Q_total = grad_Q_local + grad_Q_centroid
        grad_K_total = grad_K_local + grad_K_centroid

        # 7. Backprop through projections
        grad_x_q, q_proj_grads = self.q_proj.backward(grad_Q_total)
        grad_x_k, k_proj_grads = self.k_proj.backward(grad_K_total)
        grad_x_v, v_proj_grads = self.v_proj.backward(grad_V_total)

        # Combine gradient
        grad_x = grad_x_q + grad_x_k + grad_x_v

        # Combine parameter gradients
        param_grads = []
        param_grads.extend(q_proj_grads)
        param_grads.extend(k_proj_grads)
        param_grads.extend(v_proj_grads)
        param_grads.extend(c_proj_grads)
        param_grads.append(grad_gate)

        return grad_x, param_grads

    def from_dict(self, weights_dict, i):
        self.q_proj.weight = weights_dict[f'block_{i}_sca_q_weight']
        self.k_proj.weight = weights_dict[f'block_{i}_sca_k_weight']
        self.v_proj.weight = weights_dict[f'block_{i}_sca_v_weight']
        self.c_proj.weight = weights_dict[f'block_{i}_sca_c_weight']
        self.c_proj.bias = weights_dict[f'block_{i}_sca_c_bias']
        self.gate[...] = weights_dict[f'block_{i}_sca_gate']

        self.q_proj.synchronize()
        self.k_proj.synchronize()
        self.v_proj.synchronize()
        self.c_proj.synchronize()

    def towa_dict(self, weights_dict, i):
        weights_dict[f'block_{i}_sca_q_weight'] = self.q_proj.weight
        weights_dict[f'block_{i}_sca_k_weight'] = self.k_proj.weight
        weights_dict[f'block_{i}_sca_v_weight'] = self.v_proj.weight
        weights_dict[f'block_{i}_sca_c_weight'] = self.c_proj.weight
        weights_dict[f'block_{i}_sca_c_bias'] = self.c_proj.bias
        weights_dict[f'block_{i}_sca_gate'] = self.gate

