import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

try:
    import torch
except ImportError:
    raise unittest.SkipTest("PyTorch is not installed; run the tests with .venv/bin/python")

from lm import chat_format
from lm.data import IGNORE_INDEX, ShardWriter, mark_trained, split_loss_mask, write_manifest
from lm.model import GPT
from lm.config import ModelConfig
from lm.train import TrainConfig, train
from tokenizer import ByteBPETokenizer

ROOT = Path(__file__).resolve().parent.parent
CONVERSATION = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "4."}, {"role": "user", "content": "And 3 + 3?"},
                {"role": "assistant", "content": "It is 6."}]
SMALL_MODEL = {"vocab_size": 64, "context_length": 16, "n_layers": 2, "d_model": 32, "n_heads": 4, "mlp_hidden": 64}


class ChatFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = ByteBPETokenizer.load(ROOT / "tokenizers" / "fineweb_edu_32k.json")

    def show(self, ids):
        """Decode, writing the special tokens out (decode itself leaves them out)."""
        names = {self.tokenizer.BOS_ID: "<bos>", self.tokenizer.EOS_ID: "<eos>"}
        return "".join(names.get(i) or self.tokenizer.decode([i]) for i in ids)

    def test_conversation_text_and_trained_tokens(self):
        ids, trained = chat_format.encode_conversation(self.tokenizer, CONVERSATION)
        self.assertEqual(self.show(ids), "<bos>System: Be brief.\n\nUser: What is 2 + 2?\n\n"
                         "Assistant: 4.<eos>User: And 3 + 3?\n\nAssistant: It is 6.<eos>")
        scored = [i for i, t in zip(ids, trained) if t]
        # Only the replies and the <eos> that ends each one are scored.
        self.assertEqual(self.show(scored), " 4.<eos> It is 6.<eos>")

    def test_chat_prompts_match_the_training_tokens(self):
        # What the chat sends must be exactly the start of what training saw,
        # or the model is asked to continue text it never learned from.
        full, _ = chat_format.encode_conversation(self.tokenizer, CONVERSATION)
        for turns in (2, 4):
            prompt = chat_format.prompt_ids(self.tokenizer, CONVERSATION[:turns])
            self.assertEqual(prompt, full[:len(prompt)])
            self.assertEqual(self.tokenizer.decode(full[len(prompt):len(prompt) + 1])[0], " ")
        with self.assertRaises(ValueError):
            chat_format.prompt_ids(self.tokenizer, CONVERSATION[:3])


class DataMixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("prepare_sft", ROOT / "scripts" / "prepare_sft.py")
        cls.prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.prepare)

    def test_filter_drops_code_and_rewriting(self):
        keep = self.prepare.keep

        def chat(user, reply="Sure."):
            return [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]

        self.assertTrue(keep(chat("Why is the sky blue?"), "smol-magpie-ultra-short"))
        self.assertFalse(keep(chat("Why is the sky blue?"), "self-oss-instruct"))  # A coding source.
        self.assertFalse(keep(chat("Sort a list", "```python\nsorted(x)\n```"), "openhermes-50k"))
        self.assertFalse(keep(chat("Please edit this paragraph for clarity: ..."), "smol-magpie-ultra-short"))
        self.assertTrue(keep(chat("Who was the editor of the Times?"), "openhermes-50k"))  # "editor" is not "edit".

    def test_small_talk_is_mixed_in_evenly_and_cycles(self):
        mixed = list(self.prepare.mix_in(iter(range(7)), ["a", "b"], every=2))
        self.assertEqual(mixed, [0, 1, "a", 2, 3, "b", 4, 5, "a", 6])
        self.assertEqual(list(self.prepare.mix_in(iter(range(3)), [], every=2)), [0, 1, 2])


class LossMaskTests(unittest.TestCase):
    def test_mask_bit_round_trip(self):
        rows = torch.tensor([mark_trained([5, 6, 7, 8, 32_767], [False, True, False, True, True])])
        inputs, targets = split_loss_mask(rows[:, :-1], rows[:, 1:])
        self.assertEqual(inputs.tolist(), [[5, 6, 7, 8]])
        self.assertEqual(targets.tolist(), [[6, IGNORE_INDEX, 8, 32_767]])
        with self.assertRaises(ValueError):
            mark_trained([32_768], [True])

    def test_loss_counts_only_scored_targets(self):
        torch.manual_seed(0)
        model = GPT(ModelConfig(**SMALL_MODEL))
        ids = torch.randint(0, 64, (2, 8))
        targets = torch.randint(0, 64, (2, 8))
        logits, _ = model(ids)
        keep = torch.rand(2, 8) < 0.5
        _, loss = model(ids, torch.where(keep, targets, IGNORE_INDEX))
        expected = torch.nn.functional.cross_entropy(logits[keep], targets[keep])
        torch.testing.assert_close(loss, expected)
        _, nothing = model(ids, torch.full_like(targets, IGNORE_INDEX))
        self.assertEqual(nothing.item(), 0.0)  # Not NaN: such a batch must not poison training.


def make_masked_dataset(directory, tokens=20_000):
    """Pairs (random prompt token, answer = 3 x prompt mod 64): only the answers are scored.

    The prompts are unpredictable, so the loss can only fall to near zero if
    training ignores them and learns the answers.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(0)
    splits = {}
    for split, count in (("train", tokens), ("val", 2_000)):
        prompts = torch.randint(0, 64, (count // 2,), generator=generator).tolist()
        ids, trained = [], []
        for p in prompts:
            ids += [p, 3 * p % 64]
            trained += [False, True]
        writer = ShardWriter(directory, split, 6_000)
        writer.add(mark_trained(ids, trained))
        splits[split] = {"shards": writer.close(), "documents": 1, "tokens": len(ids), "text_bytes": len(ids)}
    write_manifest(directory, {"file": "tokenizer.json", "sha256": "test", "vocab_size": 64, "eos_id": 0},
                   splits, {}, {"loss_mask": "top_bit", "chat_format": chat_format.FORMAT})


class FineTuneTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        make_masked_dataset(self.root / "data")
        torch.manual_seed(1)
        GPT(ModelConfig(**SMALL_MODEL)).save(self.root / "start.pt")
        self.config = TrainConfig(data_dir=str(self.root / "data"), out_dir=str(self.root / "run"),
                                  model=SMALL_MODEL, micro_batch_size=8, grad_accum_steps=1,
                                  max_tokens=150 * 8 * 16, learning_rate=1e-2, warmup_fraction=0.05,
                                  eval_every=50, eval_batches=4, log_every=50, device="cpu", precision="fp32",
                                  init_from=str(self.root / "start.pt"))

    def tearDown(self):
        self.directory.cleanup()

    def run_training(self, **changes):
        from dataclasses import replace
        with redirect_stdout(io.StringIO()):
            return train(replace(self.config, **changes))

    def test_learns_the_scored_tokens_and_records_the_chat_format(self):
        result = self.run_training()
        # Uniform guessing is ln(64) = 4.16; the answers are fully predictable.
        self.assertLess(result["best_val_loss"], 0.5)
        tuned = GPT.load(self.root / "run" / "final_model.pt")
        self.assertEqual(tuned.metadata, {"chat_format": chat_format.FORMAT})

    def test_starts_from_the_given_weights(self):
        # With a learning rate of zero nothing changes, so the result is the start.
        self.run_training(learning_rate=0.0, max_tokens=8 * 16)
        start = GPT.load(self.root / "start.pt").state_dict()
        tuned = GPT.load(self.root / "run" / "final_model.pt").state_dict()
        for name in start:
            torch.testing.assert_close(tuned[name], start[name], atol=0, rtol=0, msg=name)


if __name__ == "__main__":
    unittest.main()
