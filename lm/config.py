"""Model sizes, and the parameter count worked out by hand."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32_768
    context_length: int = 1_024
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    mlp_hidden: int = 2_560
    rope_base: float = 10_000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.head_dim % 2:
            raise ValueError("RoPE rotates pairs of numbers, so head_dim must be even.")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def expected_parameters(self) -> int:
        """Count by formula; the tests check it against the real model."""
        embedding = self.vocab_size * self.d_model  # Shared with the output layer.
        attention = 4 * self.d_model * self.d_model  # Q, K, V and output projections.
        mlp = 3 * self.d_model * self.mlp_hidden  # SwiGLU has three matrices.
        norms = 2 * self.d_model  # Two RMSNorm weight vectors per block.
        return embedding + self.n_layers * (attention + mlp + norms) + self.d_model  # + final norm

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        return cls(**values)


# The final model (about 124.3M parameters) and small ones for tests and ablations.
PRESETS = {
    "main_124m": ModelConfig(),
    "ablation_20m": ModelConfig(n_layers=6, d_model=384, n_heads=6, mlp_hidden=1_024),
    "tiny": ModelConfig(vocab_size=512, context_length=128, n_layers=4, d_model=128, n_heads=4,
                        mlp_hidden=384),
}
