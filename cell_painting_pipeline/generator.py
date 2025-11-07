"""Simplified conditional generator placeholder."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import GeneratorConfig


@dataclass(slots=True)
class GeneratorState:
    config: GeneratorConfig
    vocabulary: list[str]


class SeqDiffMorph:
    """A minimal stand-in for the SeqDiff-Morph model."""

    def __init__(self, state: GeneratorState):
        self.state = state

    def train_step(self, batch: Iterable[tuple[str, np.ndarray]]) -> float:
        """Pretend to perform a training step and return a loss value."""

        return random.random()

    def sample(self, condition: np.ndarray) -> str:
        """Generate a pseudo SMILES sequence conditioned on the embedding."""

        random.seed(int(np.sum(condition) * 1000) % 2**32)
        length = random.randint(8, min(self.state.config.max_seq_length, 32))
        return "".join(random.choice(self.state.vocabulary) for _ in range(length))

