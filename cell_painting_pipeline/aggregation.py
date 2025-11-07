"""Aggregate site-level embeddings into well-level profiles."""

from __future__ import annotations

import numpy as np

from .config import AggregationConfig


def aggregate_embeddings(embeddings: np.ndarray, config: AggregationConfig) -> np.ndarray:
    """Aggregate embeddings using robust statistics."""

    if embeddings.size == 0:
        raise ValueError("No embeddings provided for aggregation.")
    if config.aggregation == "median":
        return np.median(embeddings, axis=0)
    if config.aggregation == "mean":
        return np.mean(embeddings, axis=0)
    if config.aggregation == "trimmed":
        fraction = config.outlier_fraction
        lower = int(len(embeddings) * fraction)
        upper = len(embeddings) - lower
        sorted_embeddings = np.sort(embeddings, axis=0)
        return np.mean(sorted_embeddings[lower:upper], axis=0)
    raise ValueError(f"Unsupported aggregation method: {config.aggregation}")

