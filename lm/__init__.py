"""A small GPT-style language model, written for learning."""

from .config import PRESETS, ModelConfig
from .generate import generate
from .model import GPT

__all__ = ["GPT", "ModelConfig", "PRESETS", "generate"]
