"""Lightweight wrapper around a multi-channel encoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class EmbeddingResult:
    embedding: np.ndarray
    metadata: dict[str, str]


class MockEncoder:
    """A stand-in encoder that produces deterministic embeddings."""

    def __call__(self, images: Iterable[np.ndarray]) -> np.ndarray:
        arrays = list(images)
        if not arrays:
            raise ValueError("No images provided to the encoder.")
        flattened = [array.flatten() for array in arrays]
        stacked = np.stack([np.mean(chunk) for chunk in flattened])
        return np.array([np.mean(stacked), np.std(stacked)])


def encode_sample(images: Iterable[np.ndarray], encoder: MockEncoder) -> np.ndarray:
    """Encode a list of channel images into a single embedding."""

    return encoder(images)

