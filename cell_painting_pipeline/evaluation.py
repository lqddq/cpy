"""Reporting helpers for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import EvaluationConfig
from .selection import SelectionResult


def summarize_results(results: Iterable[SelectionResult], config: EvaluationConfig) -> dict[str, float]:
    """Produce summary statistics for selected molecules."""

    scores = [candidate.attributes.get("score", 0.0) for candidate in results]
    novelty = [candidate.attributes.get("novelty", 0.0) for candidate in results]
    summary = {
        "count": float(len(scores)),
        "score_mean": float(np.mean(scores)) if scores else 0.0,
        "score_std": float(np.std(scores)) if scores else 0.0,
        "novelty_mean": float(np.mean(novelty)) if novelty else 0.0,
    }
    path = config.report_path
    path.write_text("\n".join(f"{key}: {value}" for key, value in summary.items()), encoding="utf-8")
    return summary

