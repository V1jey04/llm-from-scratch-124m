import json
import os
import socket
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path

try:
    import torch
except ImportError:
    raise unittest.SkipTest("PyTorch is not installed; run the tests with .venv/bin/python")

from lm.data import ShardWriter, TokenShards, WindowSampler, write_manifest
from lm.train import TrainConfig, learning_rate, train

SMALL_MODEL = {"vocab_size": 64, "context_length": 16, "n_layers": 2, "d_model": 32, "n_heads": 4, "mlp_hidden": 64}


def make_dataset(directory, train_tokens=20_000, val_tokens=2_000, shard_tokens=6_000, kind="random"):
    """Token IDs written the way prepare_data.py writes them.

    kind: "random" (nothing to learn), "counting" (0, 1, 2, ...) or "pattern",
    where each token is mostly (5 x previous + 3) mod 64: learnable, not trivial.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(0)
    splits = {}
    for split, count in (("train", train_tokens), ("val", val_tokens)):
        if kind == "counting":
            ids = (torch.arange(count) % 60_000).tolist()
        elif kind == "pattern":
            noise = torch.rand(count, generator=generator) < 0.1
            ids = [0]
            for i in range(1, count):
                ids.append(int(torch.randint(0, 64, (), generator=generator)) if noise[i] else (5 * ids[-1] + 3) % 64)
        else:
            ids = torch.randint(0, 64, (count,), generator=generator).tolist()
        writer = ShardWriter(directory, split, shard_tokens)
        writer.add(ids)
        splits[split] = {"shards": writer.close(), "documents": 1, "tokens": count, "text_bytes": count * 4}
    write_manifest(directory, {"file": "tokenizer.json", "sha256": "test", "vocab_size": 64, "eos_id": 0}, splits, {})


def quiet_train(config):
    with redirect_stdout(io.StringIO()):
        return train(config)


def final_weights(out_dir):
    return torch.load(Path(out_dir) / "checkpoint.pt", weights_only=True)["model"]


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        make_dataset(self.directory.name, train_tokens=1_000, shard_tokens=300, kind="counting")
        self.shards = TokenShards(self.directory.name, "train")

    def tearDown(self):
        self.directory.cleanup()

    def test_shards_round_trip(self):
        self.assertEqual(len(self.shards.arrays), 4)  # 300 + 300 + 300 + 100 tokens.
        self.assertEqual(self.shards.tokens, 1_000)

    def test_windows_are_contiguous_and_each_used_once_per_epoch(self):
        sampler = WindowSampler(self.shards, context_length=16, seed=1)
        # (300 - 1) // 16 = 18 per full shard, (100 - 1) // 16 = 6 in the last:
        # a window never crosses into the next shard.
        self.assertEqual(sampler.num_windows, 18 + 18 + 18 + 6)
        seen = []
        for _ in range(sampler.num_windows // 4):
            inputs, targets = sampler.next_batch(4)
            torch.testing.assert_close(targets, inputs + 1)  # Targets are the inputs shifted by one.
            torch.testing.assert_close(inputs[:, 1:], inputs[:, :-1] + 1)  # Consecutive tokens.
            seen.extend(inputs[:, 0].tolist())
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(sampler.epoch, 0)
        sampler.next_batch(4)
        self.assertEqual(sampler.epoch, 1)

    def test_processes_split_the_single_process_batch(self):
        single, rank0, rank1 = (WindowSampler(self.shards, 16, seed=3) for _ in range(3))
        for _ in range(5):
            expected, _ = single.next_batch(4)
            first, _ = rank0.next_batch(2, rank=0, world_size=2)
            second, _ = rank1.next_batch(2, rank=1, world_size=2)
            torch.testing.assert_close(torch.cat([first, second]), expected)

    def test_state_restores_the_data_order(self):
        original, restored = WindowSampler(self.shards, 16, seed=5), WindowSampler(self.shards, 16, seed=5)
        for _ in range(17):  # Crosses into the second epoch.
            original.next_batch(4)
        restored.load_state_dict(original.state_dict())
        torch.testing.assert_close(original.next_batch(4)[0], restored.next_batch(4)[0])


class ScheduleTests(unittest.TestCase):
    def test_warmup_then_cosine_to_ten_percent(self):
        config = TrainConfig(data_dir="", out_dir="", learning_rate=1.0, warmup_fraction=0.1)
        rates = [learning_rate(step, 100, config) for step in range(100)]
        self.assertAlmostEqual(rates[0], 0.1)
        self.assertAlmostEqual(rates[9], 1.0)
        self.assertTrue(all(a >= b for a, b in zip(rates[9:], rates[10:])))
        self.assertAlmostEqual(learning_rate(100, 100, config), 0.1)

    def test_overrides_from_the_command_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"data_dir": "d", "out_dir": "o"}')
            config = TrainConfig.from_json(path, ["max_tokens=1_000", "compile=true", "out_dir=/x/y", "model=tiny"])
            with self.assertRaises(ValueError):
                TrainConfig.from_json(path, ["no_such_setting=1"])
        self.assertEqual((config.max_tokens, config.compile, config.out_dir, config.model),
                         (1_000, True, "/x/y", "tiny"))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        make_dataset(root / "data", kind="pattern")
        # 6 steps of 4 windows x 16 tokens.
        self.config = TrainConfig(data_dir=str(root / "data"), out_dir=str(root / "run"), model=SMALL_MODEL,
                                  micro_batch_size=4, grad_accum_steps=1, max_tokens=6 * 4 * 16,
                                  learning_rate=3e-3, warmup_fraction=0.2, eval_every=3, eval_batches=2,
                                  log_every=2, device="cpu", precision="fp32")

    def tearDown(self):
        self.directory.cleanup()

    def run_in(self, name, **changes):
        return quiet_train(replace(self.config, out_dir=str(Path(self.directory.name) / name), **changes))

    def test_training_lowers_the_loss_and_saves_models(self):
        result = self.run_in("long", max_tokens=60 * 4 * 16, learning_rate=1e-2)
        self.assertEqual((result["status"], result["step"]), ("finished", 60))
        # ln(64) = 4.16 is uniform guessing; the pattern is right 90% of the time.
        self.assertLess(result["best_val_loss"], 2.5)
        for name in ("checkpoint.pt", "best_model.pt", "final_model.pt", "log.jsonl"):
            self.assertTrue((Path(self.directory.name) / "long" / name).exists())

    def test_mixed_precision_trains(self):
        # fp16 runs the GradScaler path the Kaggle T4s use; bf16 is for newer GPUs.
        for precision in ("fp16", "bf16"):
            with self.subTest(precision=precision):
                result = self.run_in(precision, precision=precision, max_tokens=40 * 4 * 16, learning_rate=1e-2)
                self.assertLess(result["best_val_loss"], 2.5)
                checkpoint = torch.load(Path(self.directory.name) / precision / "checkpoint.pt", weights_only=True)
                self.assertEqual(bool(checkpoint["scaler"]), precision == "fp16")  # Scaler state is saved.

    def test_stopping_and_resuming_gives_exactly_the_same_model(self):
        self.run_in("uninterrupted")
        self.assertEqual(self.run_in("interrupted", stop_after_steps=3)["status"], "paused")
        self.assertEqual(self.run_in("interrupted")["status"], "finished")
        expected = final_weights(Path(self.directory.name) / "uninterrupted")
        resumed = final_weights(Path(self.directory.name) / "interrupted")
        for name in expected:
            torch.testing.assert_close(resumed[name], expected[name], atol=0, rtol=0, msg=name)

    def test_first_log_after_resume_averages_only_the_steps_it_ran(self):
        # Every step logged separately, as the reference.
        self.run_in("every_step", log_every=1)
        # Stop at step 3 with log_every=4: after resuming, step 4 is the only step
        # in its logging window, so its logged loss is that one step's loss.
        self.run_in("resumed", log_every=4, stop_after_steps=3)
        self.run_in("resumed", log_every=4)

        def train_losses(name):
            lines = (Path(self.directory.name) / name / "log.jsonl").read_text().splitlines()
            return {r["step"]: r["loss"] for r in map(json.loads, lines) if "loss" in r}

        self.assertAlmostEqual(train_losses("resumed")[4], train_losses("every_step")[4], places=5)

    def test_resume_refuses_a_different_model(self):
        self.run_in("run", stop_after_steps=2)
        with self.assertRaises(ValueError):
            self.run_in("run", model={**SMALL_MODEL, "n_layers": 3})

    def test_gradient_accumulation_matches_one_big_batch(self):
        self.run_in("big_batch")
        self.run_in("accumulated", micro_batch_size=2, grad_accum_steps=2)
        expected = final_weights(Path(self.directory.name) / "big_batch")
        accumulated = final_weights(Path(self.directory.name) / "accumulated")
        for name in expected:
            torch.testing.assert_close(accumulated[name], expected[name], atol=2e-5, rtol=1e-4, msg=name)

    def test_two_process_ddp_matches_one_process(self):
        self.run_in("single")
        ddp_config = replace(self.config, out_dir=str(Path(self.directory.name) / "ddp"), micro_batch_size=2)
        with socket.socket() as probe:  # Find a free port for the processes to meet on.
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        torch.multiprocessing.spawn(ddp_worker, args=(2, port, ddp_config), nprocs=2)
        expected = final_weights(Path(self.directory.name) / "single")
        ddp = final_weights(Path(self.directory.name) / "ddp")
        for name in expected:
            torch.testing.assert_close(ddp[name], expected[name], atol=2e-5, rtol=1e-4, msg=name)


def ddp_worker(rank, world_size, port, config):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    quiet_train(config)


if __name__ == "__main__":
    unittest.main()
