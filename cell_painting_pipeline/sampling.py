"""Sampling logic for conditional generation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .config import SamplerConfig
from .generator import SeqDiffMorph


def sample_molecules(model: SeqDiffMorph, conditions: Iterable[np.ndarray], config: SamplerConfig) -> list[str]:
    """Generate SMILES strings for a batch of conditions."""

    outputs: list[str] = []
    for condition in conditions:
        attempt = 0
        while attempt < config.max_attempts:
            candidate = model.sample(condition)
            if _is_valid(candidate):
                outputs.append(candidate)
                break
            attempt += 1
        else:
            raise RuntimeError("Failed to sample a valid molecule within the attempt limit.")
    return outputs


def _is_valid(smiles: str) -> bool:
    """Very light validity checks to mimic bracket balancing."""

    return smiles.count("(") == smiles.count(")")

