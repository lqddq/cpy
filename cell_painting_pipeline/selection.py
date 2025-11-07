"""Multi-objective ranking and diversity filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import SelectionConfig
from .scoring import ScoreResult


@dataclass(slots=True)
class SelectionResult:
    smiles: str
    attributes: dict[str, float]


def pareto_front(candidates: Iterable[SelectionResult], objectives: Sequence[str]) -> list[SelectionResult]:
    """Compute a Pareto front given objective directions."""

    front: list[SelectionResult] = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if candidate is other:
                continue
            if _dominates(other, candidate, objectives):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def _dominates(a: SelectionResult, b: SelectionResult, objectives: Sequence[str]) -> bool:
    better_in_all = True
    better_in_any = False
    for objective in objectives:
        sign = -1.0 if objective.startswith("-") else 1.0
        key = objective.lstrip("-")
        score_a = a.attributes.get(key, 0.0) * sign
        score_b = b.attributes.get(key, 0.0) * sign
        if score_a < score_b:
            better_in_all = False
        if score_a > score_b:
            better_in_any = True
    return better_in_all and better_in_any


def select_diverse(front: Iterable[SelectionResult], config: SelectionConfig) -> list[SelectionResult]:
    """Greedy max-min diversity selection."""

    selected: list[SelectionResult] = []
    for candidate in front:
        if not selected:
            selected.append(candidate)
            continue
        distance = min(
            abs(candidate.attributes.get("novelty", 0.0) - existing.attributes.get("novelty", 0.0))
            for existing in selected
        )
        if distance >= config.min_scaffold_distance:
            selected.append(candidate)
        if len(selected) >= config.subset_size:
            break
    return selected

