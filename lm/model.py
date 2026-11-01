"""A decoder-only Transformer in the LLaMA style, written out piece by piece.

Shapes use B = batch size, T = sequence length (time), D = d_model,
H = number of heads, and Dh = head_dim = D / H.
"""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Divide each vector by its root-mean-square, then scale by a learned weight.

    Unlike LayerNorm there is no mean subtraction and no bias. The arithmetic is
    done in float32 because squaring fp16 numbers overflows easily.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normalized = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized.type_as(x) * self.weight


def rope_angles(config: ModelConfig, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of the rotation angle for every (position, pair), shape (T, Dh/2).

    Pair i of the head vector turns by angle position * base^(-2i/Dh): the
    first pairs spin fast (fine position detail), the last ones slowly.
    """
    pair_frequencies = config.rope_base ** (-torch.arange(0, config.head_dim, 2, device=device) / config.head_dim)
    positions = torch.arange(config.context_length, device=device)
    angles = torch.outer(positions, pair_frequencies)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each (even, odd) pair of x, shape (B, H, T, Dh), by its angle.

    Rotating q at position m and k at position n makes q·k depend only on m - n:
    attention sees relative positions without any position embedding table.
    """
    T = x.shape[-2]
    cos, sin = cos[:T], sin[:T]
    even, odd = x[..., 0::2].float(), x[..., 1::2].float()
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).type_as(x)


def explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """softmax(q kᵀ / √Dh + causal mask) v, written out. Inputs are (B, H, T, Dh).

    The model uses the fused `scaled_dot_product_attention`; tests check that
    it computes the same thing as this version.
    """
    T = q.shape[-2]
    scores = (q @ k.transpose(-2, -1)).float() / math.sqrt(q.shape[-1])  # (B, H, T, T)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))  # Position t may not look at t+1, t+2, ...
    return scores.softmax(dim=-1).type_as(v) @ v


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                explicit: bool = False) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        # (B, T, D) -> (B, T, H, Dh) -> (B, H, T, Dh): each head attends separately.
        q, k, v = (t.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if explicit:
            y = explicit_attention(q, k, v)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))  # Concatenate heads, then mix them.


class SwiGLU(nn.Module):
    """w2(silu(w1 x) ⊙ w3 x): one branch decides how much of the other passes."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.mlp_hidden, bias=False)
        self.w3 = nn.Linear(config.d_model, config.mlp_hidden, bias=False)
        self.w2 = nn.Linear(config.mlp_hidden, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """Pre-norm residual block: normalize, transform, add back to the input."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin, explicit=False):
        x = x + self.attention(self.attention_norm(x), cos, sin, explicit)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.metadata = {}  # Saved alongside the weights; see save().
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        # Fixed rotation tables: not learned and not saved (persistent=False).
        cos, sin = rope_angles(config)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # GPT-2's trick: shrink the layers that write into the residual stream,
        # so its size does not grow with depth at the start of training.
        for block in self.blocks:
            for layer in (block.attention.out, block.mlp.w2):
                nn.init.normal_(layer.weight, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None,
                explicit_attention: bool = False):
        """ids: (B, T) token IDs. Returns logits (B, T, vocab) and, with targets, the loss."""
        if ids.shape[1] > self.config.context_length:
            raise ValueError(f"Sequence length {ids.shape[1]} exceeds context {self.config.context_length}.")
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, explicit_attention)
        # Tied weights: the output layer reuses the embedding matrix, so a token's
        # input vector is also what the model compares against to predict it.
        logits = F.linear(self.final_norm(x), self.embedding.weight)
        if targets is None:
            return logits, None
        # Targets of -100 (prompt text during instruction tuning) are skipped. Sum
        # and divide by the count ourselves: a batch with nothing to score gives
        # 0 rather than the NaN that the built-in mean would return.
        targets = targets.flatten()
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets, ignore_index=-100, reduction="sum")
        return logits, loss / (targets != -100).sum().clamp(min=1)

    def save(self, path: str | Path) -> None:
        """self.metadata holds extra facts for whoever loads the model, such as its chat format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config.to_dict(), "model": self.state_dict(), "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path: str | Path, device="cpu") -> "GPT":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(ModelConfig.from_dict(checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["model"])
        model.metadata = checkpoint.get("metadata", {})  # Older files have none.
        return model
