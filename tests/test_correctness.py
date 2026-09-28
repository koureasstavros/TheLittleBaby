import unittest

import numpy as np

from src.attentions.aft import AFT
from src.attentions.gla import GLA
from src.attentions.gqa import GQA
from src.attentions.mha import MHA
from src.attentions.moh import MOH
from src.attentions.rfa import RFA
from src.block import Block
from src.functions.process import cross_entropy_loss
from src.gpt import GPT
from src.layers.convolution import DepthwiseConv1D


class CorrectnessTests(unittest.TestCase):
    def test_cross_entropy_gradient_matches_mean_loss(self):
        rng = np.random.default_rng(3)
        logits = rng.normal(size=(2, 3, 4)).astype(np.float64)
        targets = np.array([[0, 2, 1], [3, 1, 0]], dtype=np.int32)

        _, grad = cross_entropy_loss(np, logits, targets)
        numerical = np.zeros_like(logits)
        epsilon = 1e-6
        for index in np.ndindex(logits.shape):
            plus = logits.copy()
            minus = logits.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            loss_plus, _ = cross_entropy_loss(np, plus, targets)
            loss_minus, _ = cross_entropy_loss(np, minus, targets)
            numerical[index] = (loss_plus - loss_minus) / (2 * epsilon)

        np.testing.assert_allclose(grad, numerical, rtol=1e-5, atol=1e-7)

    def test_post_norm_gradients_match_parameter_order(self):
        np.random.seed(5)
        block = Block(np, 1, 0, "post", "mha", "mlp", np.float64,
                      4, 3, 0.0, 0.8, 4, 2)
        block.set(True)
        x = np.random.normal(size=(1, 3, 4))
        output = block.forward(x, False)
        _, grads = block.backward(np.ones_like(output))

        params = block.parameters()
        self.assertEqual(len(grads), len(params))
        for param, grad in zip(params, grads):
            self.assertEqual(param.shape, grad.shape)

    def test_attention_input_gradients_include_temperature(self):
        constructors = (
            lambda: MHA(np, np.float64, 3, 4, 0.0, 0.7, 4, 2),
            lambda: GQA(np, np.float64, 3, 4, 0.0, 0.7, 4, 2, 1),
            lambda: MOH(np, np.float64, 3, 4, 0.0, 0.7, 4, 2),
            lambda: RFA(np, np.float64, 3, 4, 0.0, 0.7, 4, 2, 2),
            lambda: AFT(np, np.float64, 3, 4, 0.0, 0.7, 10.0),
            lambda: GLA(np, np.float64, 3, 4, 0.0, 0.7, 3),
        )
        for make_attention in constructors:
            with self.subTest(attention=make_attention().__class__.__name__):
                np.random.seed(11)
                attention = make_attention()
                attention.set(True)
                x = np.random.normal(scale=0.2, size=(1, 3, 4))
                grad_output = np.random.normal(size=(1, 3, 4))
                output = attention.forward(x, False)
                grad_x, _ = attention.backward(grad_output)

                numerical = np.zeros_like(x)
                epsilon = 1e-6
                for index in np.ndindex(x.shape):
                    plus = x.copy()
                    minus = x.copy()
                    plus[index] += epsilon
                    minus[index] -= epsilon
                    loss_plus = np.sum(attention.forward(plus, False) * grad_output)
                    loss_minus = np.sum(attention.forward(minus, False) * grad_output)
                    numerical[index] = (loss_plus - loss_minus) / (2 * epsilon)

                self.assertEqual(output.shape, grad_x.shape)
                np.testing.assert_allclose(grad_x, numerical, rtol=2e-3, atol=2e-5)

    def test_depthwise_convolution_is_causal(self):
        np.random.seed(17)
        conv = DepthwiseConv1D(np, np.float64, 2, 3)
        x = np.random.normal(size=(1, 5, 2))
        changed_future = x.copy()
        changed_future[:, 4, :] += 100

        original = conv.forward(x)
        changed = conv.forward(changed_future)

        np.testing.assert_allclose(original[:, :4], changed[:, :4])

    def test_gla_cached_chunks_match_full_causal_forward(self):
        np.random.seed(23)
        attention = GLA(np, np.float64, 5, 4, 0.0, 0.9, 3)
        attention.set(False)
        x = np.random.normal(scale=0.2, size=(1, 5, 4))

        full = attention.forward(x, False)
        attention.clear_cache()
        chunks = [attention.forward(x[:, :2], True),
                  attention.forward(x[:, 2:4], True),
                  attention.forward(x[:, 4:], True)]
        cached = np.concatenate(chunks, axis=1)

        self.assertEqual(cached.shape, full.shape)
        np.testing.assert_allclose(cached, full, rtol=1e-10, atol=1e-10)

    def test_aft_cached_chunks_match_full_forward_with_temperature(self):
        np.random.seed(27)
        attention = AFT(np, np.float64, 5, 4, 0.0, 0.7, 10.0)
        attention.set(False)
        x = np.random.normal(scale=0.1, size=(1, 5, 4))

        full = attention.forward(x, False)
        attention.clear_cache()
        chunks = [attention.forward(x[:, :2], True),
                  attention.forward(x[:, 2:4], True),
                  attention.forward(x[:, 4:], True)]
        cached = np.concatenate(chunks, axis=1)

        np.testing.assert_allclose(cached, full, rtol=1e-10, atol=1e-10)

    def test_cached_generation_matches_uncached_generation(self):
        class GenerationHarness:
            generate = GPT.generate

            def __init__(self):
                self.mp = np
                self.n_ctx = 4
                self.cache = []

            def eval(self):
                pass

            def clear_cache(self):
                self.cache = []

            def forward(self, tokens, use_cache):
                current = tokens.reshape(-1).tolist()
                if use_cache:
                    self.cache.extend(current)
                    context = self.cache
                else:
                    context = current

                next_token = sum(context[-2:]) % 2
                logits = np.zeros((1, tokens.shape[1], 2))
                logits[0, -1, next_token] = 20.0
                return logits

        model = GenerationHarness()
        np.random.seed(29)
        cached = model.generate({"a": 0, "b": 1}, "ab", True, 2)
        np.random.seed(29)
        uncached = model.generate({"a": 0, "b": 1}, "ab", False, 2)

        np.testing.assert_array_equal(cached, uncached)


if __name__ == "__main__":
    unittest.main()