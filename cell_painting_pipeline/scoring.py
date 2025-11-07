"""Score molecules against phenotype embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class ScoreResult:
    smiles: str
    score: float
    uncertainty: float


class ConsistencyScorer:
    """Compute similarity between molecule fingerprints and embeddings."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def fingerprint(self, smiles: str) -> np.ndarray:
        values = [ord(char) for char in smiles]
        return np.array([np.mean(values), np.std(values) or 1.0])

    def score(self, smiles: str, embedding: np.ndarray) -> ScoreResult:
        fp = self.fingerprint(smiles)
        similarity = float(np.dot(fp, embedding) / (np.linalg.norm(fp) * np.linalg.norm(embedding)))
        uncertainty = math.exp(-abs(similarity)) / self.temperature
        return ScoreResult(smiles=smiles, score=similarity, uncertainty=uncertainty)

    def batch_score(self, smiles_list: Iterable[str], embedding: np.ndarray) -> list[ScoreResult]:
        return [self.score(smiles, embedding) for smiles in smiles_list]

