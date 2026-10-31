import math
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    raise unittest.SkipTest("PyTorch is not installed; run the tests with .venv/bin/python")

from lm.config import PRESETS, ModelConfig
from lm.generate import generate, next_token_distribution
from lm.model import GPT, RMSNorm, apply_rope, explicit_attention, rope_angles

SMALL = ModelConfig(vocab_size=64, context_length=16, n_layers=2, d_model=32, n_heads=4, mlp_hidden=64)


def small_model(seed=0):
    torch.manual_seed(seed)
    return GPT(SMALL)


class ParameterCountTests(unittest.TestCase):
    def test_main_model_is_124m_and_matches_formula(self):
        with torch.device("meta"):  # Shapes only: no memory for 124M real weights.
            model = GPT(PRESETS["main_124m"])
        self.assertEqual(model.num_parameters(), 124_275_456)
        self.assertEqual(model.num_parameters(), PRESETS["main_124m"].expected_parameters())

    def test_formula_matches_every_preset(self):
        for name, config in [*PRESETS.items(), ("small", SMALL)]:
            with self.subTest(name=name), torch.device("meta"):
                self.assertEqual(GPT(config).num_parameters(), config.expected_parameters())

    def test_output_layer_is_tied_to_embedding(self):
        names = [name for name, _ in small_model().named_parameters()]
        self.assertEqual([name for name in names if "embedding" in name], ["embedding.weight"])
        self.assertEqual(len(names), len(set(names)))


class ComponentTests(unittest.TestCase):
    def test_rmsnorm_gives_unit_root_mean_square(self):
        x = torch.randn(3, 5, 32) * 7 + 2
        rms = RMSNorm(32, 1e-5)(x).pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones(3, 5), atol=1e-4, rtol=0)

    def test_rope_keeps_length_and_depends_only_on_relative_position(self):
        config = ModelConfig(vocab_size=8, context_length=32, n_layers=1, d_model=16, n_heads=2, mlp_hidden=8)
        cos, sin = rope_angles(config)
        q, k = torch.randn(config.head_dim), torch.randn(config.head_dim)

        def score(q_position, k_position):
            # Put q and k at chosen positions in a length-32 sequence, rotate, take the dot product.
            sequence = torch.zeros(1, 1, 32, config.head_dim)
            sequence[0, 0, q_position], sequence[0, 0, k_position] = q, k
            rotated = apply_rope(sequence, cos, sin)[0, 0]
            torch.testing.assert_close(rotated[q_position].norm(), q.norm())  # Rotation keeps length.
            return rotated[q_position] @ rotated[k_position]

        torch.testing.assert_close(score(5, 2), score(25, 22), atol=1e-4, rtol=1e-4)  # Both 3 apart.
        self.assertFalse(torch.isclose(score(5, 2), score(5, 4), atol=1e-3))

    def test_explicit_attention_on_hand_computed_example(self):
        # One head, Dh=1, two positions. Position 0 can only see itself;
        # position 1 weighs v0 and v1 by softmax of q1·k / √1 = (1·0, 1·2).
        q = torch.tensor([[[[0.0], [1.0]]]])
        k = torch.tensor([[[[0.0], [2.0]]]])
        v = torch.tensor([[[[10.0], [20.0]]]])
        w1 = math.exp(2) / (1 + math.exp(2))
        expected = torch.tensor([[[[10.0], [10 * (1 - w1) + 20 * w1]]]])
        torch.testing.assert_close(explicit_attention(q, k, v), expected)

    def test_fused_attention_matches_explicit(self):
        model = small_model()
        ids = torch.randint(0, SMALL.vocab_size, (2, 16))
        fused, _ = model(ids)
        explicit, _ = model(ids, explicit_attention=True)
        torch.testing.assert_close(fused, explicit, atol=1e-5, rtol=1e-4)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)  # Random test inputs are the same on every run.

    def test_shapes(self):
        logits, loss = small_model()(torch.zeros(3, 10, dtype=torch.long))
        self.assertEqual(logits.shape, (3, 10, SMALL.vocab_size))
        self.assertIsNone(loss)

    def test_rejects_sequences_longer_than_context(self):
        with self.assertRaises(ValueError):
            small_model()(torch.zeros(1, SMALL.context_length + 1, dtype=torch.long))

    def test_future_tokens_cannot_change_earlier_logits(self):
        model = small_model()
        ids = torch.randint(0, SMALL.vocab_size, (1, 16))
        for t in (0, 7, 14):
            with self.subTest(changed_position=t + 1):
                changed = ids.clone()
                changed[0, t + 1] = (changed[0, t + 1] + 1) % SMALL.vocab_size
                before, after = model(ids)[0], model(changed)[0]
                torch.testing.assert_close(before[0, :t + 1], after[0, :t + 1])
                self.assertFalse(torch.allclose(before[0, t + 1], after[0, t + 1]))

    def test_initial_loss_is_near_uniform_guessing(self):
        # Unrelated random targets: an untrained model should be no better than
        # guessing uniformly, a loss of ln(vocab). (Using the inputs as targets
        # scores lower: with tied embeddings each token already favors itself.)
        ids, targets = torch.randint(0, SMALL.vocab_size, (2, 4, 16))
        _, loss = small_model()(ids, targets)
        self.assertLess(abs(loss.item() - math.log(SMALL.vocab_size)), 0.2)

    def test_every_parameter_gets_a_finite_nonzero_gradient(self):
        model = small_model()
        ids = torch.randint(0, SMALL.vocab_size, (2, 16))
        _, loss = model(ids[:, :-1], ids[:, 1:])
        loss.backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_overfits_a_single_batch(self):
        model = small_model()
        ids = torch.randint(0, SMALL.vocab_size, (2, 17))
        inputs, targets = ids[:, :-1], ids[:, 1:]  # Predict each next token.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0)
        for _ in range(150):
            _, loss = model(inputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), 0.05)

    def test_save_and_load_give_identical_logits(self):
        model = small_model()
        ids = torch.randint(0, SMALL.vocab_size, (2, 16))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save(path)
            restored = GPT.load(path)
        self.assertEqual(restored.config, SMALL)
        torch.testing.assert_close(model(ids)[0], restored(ids)[0], atol=0, rtol=0)


class FavoriteTokenModel(torch.nn.Module):
    """Stand-in model that always prefers one token, so sampling is predictable."""

    def __init__(self, favorite, vocab_size=8):
        super().__init__()
        self.config = ModelConfig(vocab_size=vocab_size, context_length=4, n_layers=1, d_model=2,
                                  n_heads=1, mlp_hidden=2)
        self.favorite = favorite
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids):
        logits = torch.zeros(*ids.shape, self.config.vocab_size)
        logits[..., self.favorite] = 5.0
        return logits, None


class GenerationTests(unittest.TestCase):
    def test_greedy_respects_max_tokens_and_stops_at_eos(self):
        self.assertEqual(generate(FavoriteTokenModel(3), [1, 2], 5, temperature=0), [3] * 5)
        self.assertEqual(generate(FavoriteTokenModel(3), [1, 2], 5, temperature=0, eos_id=3), [])

    def test_top_k_and_top_p_keep_only_the_top_token(self):
        logits = torch.tensor([1.0, 3.0, 2.0, 0.0])
        for probabilities in (next_token_distribution(logits, torch.tensor([]), top_k=1),
                              next_token_distribution(logits, torch.tensor([]), top_p=0.5)):
            torch.testing.assert_close(probabilities, torch.tensor([0.0, 1.0, 0.0, 0.0]))

    def test_top_p_keeps_tokens_until_probability_is_reached(self):
        probabilities = next_token_distribution(torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05])),
                                                torch.tensor([]), top_p=0.7)
        torch.testing.assert_close(probabilities, torch.tensor([0.625, 0.375, 0.0, 0.0]))

    def test_repetition_penalty_lowers_seen_tokens(self):
        logits = torch.tensor([2.0, 2.0, -1.0, -1.0])
        probabilities = next_token_distribution(logits, torch.tensor([0, 2]), repetition_penalty=2.0)
        self.assertLess(probabilities[0], probabilities[1])
        self.assertLess(probabilities[2], probabilities[3])

    def test_seeded_sampling_is_reproducible_and_crops_long_prompts(self):
        model = small_model()
        prompt = list(range(40))  # Longer than the 16-token context.
        runs = [generate(model, prompt, 10, top_k=20, generator=torch.Generator().manual_seed(7))
                for _ in range(2)]
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(len(runs[0]), 10)


if __name__ == "__main__":
    unittest.main()
