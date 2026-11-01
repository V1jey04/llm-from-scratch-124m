"""Train the model: mixed precision, gradient accumulation, DDP, exact resume.

One process (Mac, CPU, one GPU):
    .venv/bin/python -m lm.train --config configs/smoke.json
Two GPUs on Kaggle, one process each:
    torchrun --nproc_per_node=2 -m lm.train --config configs/main_124m.json
Change any setting without editing the file:
    ... --set data_dir=/kaggle/input/tokens out_dir=/kaggle/working/run max_tokens=1_000_000

Running the same command again resumes from `out_dir/checkpoint.pt`.
Each optimizer step uses world_size x micro_batch_size x grad_accum_steps
windows of context_length tokens.
"""

import argparse
import ast
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import PRESETS, ModelConfig
from .data import TokenShards, WindowSampler, split_loss_mask
from .model import GPT


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    model: str | dict = "main_124m"  # A preset name, or ModelConfig fields.
    micro_batch_size: int = 8
    grad_accum_steps: int = 16
    max_tokens: int = 2_000_000_000
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1  # Cosine decay ends at 10% of the peak.
    warmup_fraction: float = 0.02
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "auto"  # auto, fp32, bf16 or fp16
    device: str = "auto"
    compile: bool = False
    eval_every: int = 250
    eval_batches: int = 20
    log_every: int = 10
    checkpoint_minutes: float = 30.0
    time_limit_hours: float | None = None  # Save and stop before a Kaggle session ends.
    stop_after_steps: int | None = None  # Save and stop early (tests and pilots).
    seed: int = 1337
    wandb_project: str | None = None
    init_from: str | None = None  # A saved model to start from (fine-tuning) instead of random weights.

    def model_config(self) -> ModelConfig:
        return PRESETS[self.model] if isinstance(self.model, str) else ModelConfig(**self.model)

    @classmethod
    def from_json(cls, path, overrides=()) -> "TrainConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        names = {field.name for field in fields(cls)}
        for override in overrides:
            key, _, raw = override.partition("=")
            if key not in names:
                raise ValueError(f"Unknown setting: {key}")
            try:
                values[key] = json.loads(raw)  # Numbers, true/false, null.
            except json.JSONDecodeError:
                try:
                    values[key] = ast.literal_eval(raw)  # Python spellings such as 1_000_000.
                except (ValueError, SyntaxError):
                    values[key] = raw  # A bare string such as a path.
        return cls(**values)


# Changing any of these mid-run would silently alter the schedule or data order.
SCHEDULE_SETTINGS = ("max_tokens", "learning_rate", "min_lr_ratio", "warmup_fraction", "weight_decay",
                     "beta1", "beta2", "seed")


def learning_rate(step: int, total_steps: int, config: TrainConfig) -> float:
    """Linear warmup to the peak, then cosine decay down to min_lr_ratio x peak."""
    warmup = max(1, round(config.warmup_fraction * total_steps))
    peak, low = config.learning_rate, config.learning_rate * config.min_lr_ratio
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return low + 0.5 * (peak - low) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: GPT, config: TrainConfig, device: torch.device) -> torch.optim.AdamW:
    """Weight decay pulls weights toward zero; apply it to the matrices only.

    Norm weights are scales around 1, and the embedding is a lookup table where
    rare tokens would just be pushed toward zero, so both are left alone.
    """
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.dim() >= 2 and "embedding" not in name else no_decay).append(parameter)
    groups = [{"params": decay, "weight_decay": config.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=config.learning_rate, betas=(config.beta1, config.beta2),
                             fused=device.type == "cuda")


def pick_device(config: TrainConfig, local_rank: int) -> torch.device:
    if config.device != "auto":
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def pick_precision(config: TrainConfig, device: torch.device) -> str:
    if config.precision != "auto":
        return config.precision
    if device.type == "cuda":
        # T4s (compute capability 7.5) have no fast bfloat16, so they use fp16.
        return "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"
    return "fp32"


def batch_for_model(inputs: torch.Tensor, targets: torch.Tensor, masked: bool, device: torch.device):
    if masked:  # Instruction tuning: score only the assistant's tokens.
        inputs, targets = split_loss_mask(inputs, targets)
    return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model: GPT, sampler: WindowSampler, config: TrainConfig, autocast, rank, world_size, device,
             masked: bool = False) -> float:
    """Mean loss over the first windows of the validation split, always the same ones."""
    model.eval()
    count = min(sampler.num_windows, config.eval_batches * config.micro_batch_size * world_size)
    mine = list(range(rank, count, world_size))  # Each process takes every world_size-th window.
    totals = torch.zeros(2, device=device)  # Sum of losses, number of windows.
    for first in range(0, len(mine), config.micro_batch_size):
        rows = torch.stack([sampler.window(i) for i in mine[first:first + config.micro_batch_size]])
        with autocast:
            _, loss = model(*batch_for_model(rows[:, :-1], rows[:, 1:], masked, device))
        totals += torch.stack([loss.float() * len(rows), torch.tensor(float(len(rows)), device=device)])
    if world_size > 1:
        dist.all_reduce(totals)
    model.train()
    return (totals[0] / totals[1]).item()


def save_atomically(state: dict, path: Path) -> None:
    """Write to a temporary file and rename: a crash mid-save never leaves a broken checkpoint."""
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


class Logger:
    """Print, append JSON lines to out_dir/log.jsonl, and optionally send to Weights & Biases."""

    def __init__(self, config: TrainConfig, enabled: bool, run_id: str | None):
        self.enabled = enabled
        self.path = Path(config.out_dir) / "log.jsonl"
        self.wandb = None
        self.run_id = run_id
        if enabled and config.wandb_project:
            import wandb  # Needs WANDB_API_KEY; on Kaggle, read it from Kaggle Secrets.

            self.wandb = wandb.init(project=config.wandb_project, id=run_id, resume="allow", config=asdict(config))
            self.run_id = self.wandb.id

    def log(self, record: dict) -> None:
        if not self.enabled:
            return
        print("  ".join(f"{key} {value:.4g}" if isinstance(value, float) else f"{key} {value}"
                        for key, value in record.items()), flush=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        if self.wandb:
            self.wandb.log(record, step=record["step"])


def train(config: TrainConfig) -> dict:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank, local_rank = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    device = pick_device(config, local_rank)
    if world_size > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    main_process = rank == 0
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_config = config.model_config()
    train_shards = TokenShards(config.data_dir, "train")
    tokenizer_info = train_shards.manifest["tokenizer"]
    if tokenizer_info["vocab_size"] > model_config.vocab_size:
        raise ValueError("The tokenizer has more IDs than the model's vocabulary.")
    T = model_config.context_length
    sampler = WindowSampler(train_shards, T, config.seed)
    val_sampler = WindowSampler(TokenShards(config.data_dir, "val"), T, seed=0)
    val_tokens_per_byte = val_sampler.shards.split_tokens / val_sampler.shards.text_bytes
    masked = train_shards.manifest.get("loss_mask") == "top_bit"

    tokens_per_step = world_size * config.micro_batch_size * config.grad_accum_steps * T
    total_steps = max(1, config.max_tokens // tokens_per_step)

    torch.manual_seed(config.seed)  # Identical starting weights in every process.
    model = GPT(model_config).to(device)
    if config.init_from and not (out_dir / "checkpoint.pt").exists():  # A resume loads its own weights.
        start = GPT.load(config.init_from)
        if start.config != model_config:
            raise ValueError("init_from was saved with a different model configuration.")
        model.load_state_dict(start.state_dict())
    if "chat_format" in train_shards.manifest:  # Lets chat.py pick the right prompt format.
        model.metadata = {"chat_format": train_shards.manifest["chat_format"]}
    optimizer = make_optimizer(model, config, device)
    precision = pick_precision(config, device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    autocast = torch.autocast(device.type, dtype=dtype) if dtype else nullcontext()
    # fp16 gradients underflow to zero; the scaler multiplies the loss first and
    # divides the gradients back, skipping any step whose gradients overflowed.
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")

    step, tokens_seen, best_val_loss, run_id = 0, 0, float("inf"), None
    checkpoint_path = out_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint["model_config"] != model_config.to_dict():
            raise ValueError("The checkpoint was made with a different model configuration.")
        if checkpoint["tokenizer_sha256"] != tokenizer_info["sha256"]:
            raise ValueError("The checkpoint was made with a different tokenizer.")
        if checkpoint["tokens_per_step"] != tokens_per_step:
            raise ValueError("Tokens per step changed; the schedule and data order would not match.")
        # The learning-rate schedule and data order must also continue unchanged.
        # Settings such as time_limit_hours may differ between sessions.
        saved = checkpoint["train_config"]
        changed = [key for key in SCHEDULE_SETTINGS if saved[key] != getattr(config, key)]
        if changed:
            raise ValueError(f"Resuming with different {', '.join(changed)} than the checkpoint "
                             f"({', '.join(f'{key}={saved[key]}' for key in changed)}).")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        sampler.load_state_dict(checkpoint["sampler"])
        step, tokens_seen = checkpoint["step"], checkpoint["tokens_seen"]
        best_val_loss, run_id = checkpoint["best_val_loss"], checkpoint["wandb_run_id"]

    raw_model = model
    if config.compile:
        model = torch.compile(model)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    logger = Logger(config, main_process, run_id)
    if main_process:
        print(f"{raw_model.num_parameters():,} parameters | {device} {precision} | world {world_size} | "
              f"{tokens_per_step:,} tokens/step | step {step}/{total_steps}"
              f"{' (resumed)' if step else ''} | {sampler.num_windows * T:,} training tokens available")

    def checkpoint_state():
        return {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "sampler": sampler.state_dict(), "step": step, "tokens_seen": tokens_seen,
                "best_val_loss": best_val_loss, "wandb_run_id": logger.run_id,
                "model_config": model_config.to_dict(), "train_config": asdict(config),
                "tokenizer_sha256": tokenizer_info["sha256"], "tokens_per_step": tokens_per_step}

    def run_eval():
        nonlocal best_val_loss
        val_loss = evaluate(raw_model, val_sampler, config, autocast, rank, world_size, device, masked)
        record = {"step": step, "tokens": tokens_seen, "val_loss": val_loss}
        if not masked:  # With a loss mask only some tokens are scored, so per-byte figures do not apply.
            # Bits per byte: nats -> bits, then per token -> per byte of original text.
            # It uses the split's average bytes per token, so it is approximate here;
            # the final evaluation scores whole documents exactly.
            record["val_bpb"] = val_loss / math.log(2) * val_tokens_per_byte
        logger.log(record)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if main_process:
                raw_model.save(out_dir / "best_model.pt")

    start_time = last_save = last_log = time.time()
    loss_sum = torch.zeros((), device=device)
    steps_since_log = 0  # After a resume this can be fewer than log_every.
    status = "finished"
    while step < total_steps:
        lr = learning_rate(step, total_steps, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        for micro_step in range(config.grad_accum_steps):
            inputs, targets = batch_for_model(*sampler.next_batch(config.micro_batch_size, rank, world_size),
                                              masked, device)
            # With DDP, only the last micro-step averages gradients across GPUs.
            last_micro = micro_step == config.grad_accum_steps - 1
            sync = nullcontext() if world_size == 1 or last_micro else model.no_sync()
            with sync:
                with autocast:
                    _, loss = model(inputs, targets)
                loss = loss / config.grad_accum_steps  # Sum of micro-steps = mean over the batch.
                scaler.scale(loss).backward()
            loss_sum += loss.detach()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        steps_since_log += 1
        tokens_seen += tokens_per_step

        if step % config.log_every == 0 or step == total_steps:
            if world_size > 1:
                dist.all_reduce(loss_sum)
                loss_sum /= world_size
            now = time.time()
            record = {"step": step, "tokens": tokens_seen, "loss": loss_sum.item() / steps_since_log, "lr": lr,
                      "grad_norm": grad_norm.item(),
                      "tokens_per_s": round(steps_since_log * tokens_per_step / (now - last_log)),
                      "scale": scaler.get_scale() if scaler.is_enabled() else 1.0}
            if device.type == "cuda":  # Peak GPU memory of this process so far.
                record["max_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
            logger.log(record)
            loss_sum.zero_()
            steps_since_log = 0
            last_log = time.time()
        if step % config.eval_every == 0 or step == total_steps:
            run_eval()
            last_log = time.time()  # Keep evaluation time out of tokens/s.

        # Rank 0 decides when to save or stop, so every process does it on the same step.
        elapsed_hours = (time.time() - start_time) / 3600
        stop = (config.stop_after_steps is not None and step >= config.stop_after_steps) or \
               (config.time_limit_hours is not None and elapsed_hours >= config.time_limit_hours)
        save = stop or step == total_steps or time.time() - last_save >= 60 * config.checkpoint_minutes
        flags = torch.tensor([float(save), float(stop)], device=device)
        if world_size > 1:
            dist.broadcast(flags, src=0)
        if flags[0] and main_process:
            save_atomically(checkpoint_state(), checkpoint_path)
            last_save = time.time()
        if flags[1] and step < total_steps:
            status = "paused"
            break

    if main_process and status == "finished":
        raw_model.save(out_dir / "final_model.pt")
        print(f"Finished {step} steps, {tokens_seen:,} tokens. Best validation loss {best_val_loss:.4f}")
    elif main_process:
        print(f"Paused at step {step}/{total_steps}; run the same command again to resume.")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return {"status": status, "step": step, "tokens_seen": tokens_seen, "best_val_loss": best_val_loss}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    train(TrainConfig.from_json(args.config, args.set))


if __name__ == "__main__":
    main()
