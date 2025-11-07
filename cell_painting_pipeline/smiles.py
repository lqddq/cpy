"""Utilities for working with SMILES strings."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class MoleculeRecord:
    smiles: str
    metadata: dict[str, str]

    @property
    def scaffold(self) -> str:
        """Return a simple Murcko-like scaffold identifier."""

        alphabet = "".join(sorted(set(filter(str.isalpha, self.smiles))))
        return alphabet or "unknown"

    def hash(self) -> str:
        return hashlib.sha256(self.smiles.encode("utf-8")).hexdigest()


def deduplicate(records: Iterable[MoleculeRecord]) -> list[MoleculeRecord]:
    """Remove duplicate SMILES entries preserving order."""

    seen: set[str] = set()
    unique: list[MoleculeRecord] = []
    for record in records:
        key = record.hash()
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def stratified_split(records: list[MoleculeRecord], config) -> tuple[list[MoleculeRecord], list[MoleculeRecord], list[MoleculeRecord]]:
    """Split molecules into train/validation/test buckets by scaffold."""

    scaffolds: dict[str, list[MoleculeRecord]] = {}
    for record in records:
        scaffolds.setdefault(record.scaffold, []).append(record)
    scaffold_items = sorted(scaffolds.items())
    train: list[MoleculeRecord] = []
    valid: list[MoleculeRecord] = []
    test: list[MoleculeRecord] = []
    for index, (_, group) in enumerate(scaffold_items):
        bucket = index % 10
        if bucket < int(config.validation_fraction * 10):
            valid.extend(group)
        elif bucket < int((config.validation_fraction + config.test_fraction) * 10):
            test.extend(group)
        else:
            train.extend(group)
    return train, valid, test

