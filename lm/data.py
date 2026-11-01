"""Token IDs on disk, and the order in which training reads them.

A dataset directory holds `manifest.json`, the tokenizer, and shard files of
raw little-endian uint16 token IDs (`train_0000.bin`, `val_0000.bin`, ...).
Documents are joined with `<eos>` between them.

Instruction-tuning data uses the same files. Token IDs stay below 32,768, so
the top bit of each uint16 is free: there it marks the tokens the loss counts
(the assistant's replies). The manifest then says "loss_mask": "top_bit".
"""

from array import array
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


LOSS_MASK_BIT = 1 << 15
IGNORE_INDEX = -100  # Targets with this value add nothing to the loss.


def mark_trained(ids: list[int], trained: list[bool]) -> list[int]:
    """Set the top bit on IDs the loss should count."""
    if any(i >= LOSS_MASK_BIT for i in ids):
        raise ValueError("Token IDs must stay below 32,768 to leave room for the loss-mask bit.")
    return [i | LOSS_MASK_BIT if t else i for i, t in zip(ids, trained)]


def split_loss_mask(inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove the mask bit; targets without it become IGNORE_INDEX."""
    clear = LOSS_MASK_BIT - 1
    counted = (targets & LOSS_MASK_BIT) != 0
    return inputs & clear, torch.where(counted, targets & clear, IGNORE_INDEX)


class ShardWriter:
    """Collect token IDs and write them out in shards of `shard_tokens` tokens."""

    def __init__(self, directory: str | Path, split: str, shard_tokens: int):
        self.directory = Path(directory)
        self.split = split
        self.shard_tokens = shard_tokens
        self.buffer = array("H")  # uint16: raises OverflowError for IDs >= 65,536.
        self.shards = []
        self.tokens = 0

    def add(self, ids: list[int]) -> None:
        self.buffer.extend(ids)
        self.tokens += len(ids)
        while len(self.buffer) >= self.shard_tokens:
            self._write(self.buffer[:self.shard_tokens])
            del self.buffer[:self.shard_tokens]

    def close(self) -> list[dict]:
        if self.buffer:
            self._write(self.buffer)
            self.buffer = array("H")
        return self.shards

    def _write(self, tokens: array) -> None:
        if sys.byteorder != "little":
            raise RuntimeError("Shards are little-endian; this machine is not.")
        name = f"{self.split}_{len(self.shards):04d}.bin"
        data = tokens.tobytes()
        (self.directory / name).write_bytes(data)
        self.shards.append({"file": name, "tokens": len(tokens), "sha256": hashlib.sha256(data).hexdigest()})


def write_manifest(directory: str | Path, tokenizer: dict, splits: dict, source: dict,
                   extra: dict | None = None) -> None:
    """splits maps a name to {"shards", "documents", "tokens", "text_bytes"}.

    extra adds top-level fields, such as "loss_mask" and "chat_format".
    """
    manifest = {"format": "uint16-token-shards", "version": 1, "tokenizer": tokenizer,
                "splits": splits, "source": source, **(extra or {})}
    Path(directory, "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


class TokenShards:
    """Read one split's shards without loading them: numpy memory-maps each file."""

    def __init__(self, directory: str | Path, split: str):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        info = self.manifest["splits"][split]
        self.arrays = [np.memmap(directory / shard["file"], dtype="<u2", mode="r") for shard in info["shards"]]
        self.tokens = sum(len(tokens) for tokens in self.arrays)
        self.text_bytes = info["text_bytes"]
        self.split_tokens = info["tokens"]


class WindowSampler:
    """Cut the shards into windows of T + 1 tokens and visit them in shuffled order.

    Window i holds inputs i*T ... i*T+T-1 and, shifted by one, their targets, so
    windows never repeat inside an epoch: every token is trained on once before
    any is seen again. The shuffle comes only from (seed, epoch), so the state
    to save in a checkpoint is just two numbers.

    For data-parallel training every process calls `next_batch` in step and
    takes its own slice of the same global batch: the processes see different
    data, and together exactly what a single process would see.
    """

    def __init__(self, shards: TokenShards, context_length: int, seed: int):
        self.shards = shards
        self.T = context_length
        self.seed = seed
        windows_per_shard = [max(0, (len(tokens) - 1) // context_length) for tokens in shards.arrays]
        self.shard_starts = np.cumsum([0] + windows_per_shard)  # First window index of each shard.
        self.num_windows = int(self.shard_starts[-1])
        if self.num_windows == 0:
            raise ValueError("Not enough tokens for a single window.")
        self.epoch = 0
        self.position = 0  # Windows consumed so far in this epoch, across all processes.
        self._order = None

    def window(self, index: int) -> torch.Tensor:
        shard = int(np.searchsorted(self.shard_starts, index, side="right")) - 1
        start = (index - int(self.shard_starts[shard])) * self.T
        return torch.from_numpy(self.shards.arrays[shard][start:start + self.T + 1].astype(np.int64))

    def _shuffled(self) -> torch.Tensor:
        if self._order is None:
            generator = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
            self._order = torch.randperm(self.num_windows, generator=generator)
        return self._order

    def next_batch(self, batch_size: int, rank: int = 0, world_size: int = 1):
        """Return (inputs, targets), each (batch_size, T), for this process."""
        global_size = batch_size * world_size
        if global_size > self.num_windows:
            raise ValueError("The batch is larger than the dataset.")
        if self.position + global_size > self.num_windows:  # Drop the incomplete tail.
            self.epoch, self.position, self._order = self.epoch + 1, 0, None
        first = self.position + rank * batch_size
        rows = torch.stack([self.window(int(i)) for i in self._shuffled()[first:first + batch_size]])
        self.position += global_size
        return rows[:, :-1], rows[:, 1:]

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state: dict) -> None:
        self.epoch, self.position, self._order = state["epoch"], state["position"], None
