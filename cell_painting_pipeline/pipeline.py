"""High-level orchestration of the Cell Painting workflow."""

from __future__ import annotations

import numpy as np

from .aggregation import aggregate_embeddings
from .config import PipelineConfig
from .evaluation import summarize_results
from .generator import GeneratorState, SeqDiffMorph
from .manifest import build_manifest
from .preprocessing import ImagePreprocessor
from .sampling import sample_molecules
from .scoring import ConsistencyScorer
from .selection import SelectionResult, pareto_front, select_diverse
from .smiles import MoleculeRecord, deduplicate


class Pipeline:
    """Glue together all modules to form an executable workflow."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self) -> dict[str, float]:
        manifest_entries = build_manifest(self.config.manifest)
        preprocessor = ImagePreprocessor(self.config.preprocess)
        encoder_outputs: list[np.ndarray] = []
        for entry in manifest_entries:
            processed = preprocessor.process(entry.channels.values())
            if not processed:
                continue
            encoder_outputs.append(np.mean(processed, axis=0))
        if not encoder_outputs:
            raise RuntimeError("No embeddings produced from manifest entries.")
        aggregated = aggregate_embeddings(np.stack(encoder_outputs), self.config.aggregation)
        state = GeneratorState(config=self.config.generator, vocabulary=list("CNOPSH[]=()"))
        model = SeqDiffMorph(state)
        molecules = sample_molecules(model, [aggregated], self.config.sampler)
        deduped = deduplicate([MoleculeRecord(smiles=molecule, metadata={}) for molecule in molecules])
        scorer = ConsistencyScorer()
        scored = [
            SelectionResult(
                smiles=record.smiles,
                attributes={
                    "score": scorer.score(record.smiles, aggregated).score,
                    "qed": 0.5,
                    "sa": 3.0,
                    "novelty": float(index) / max(1, len(deduped) - 1),
                },
            )
            for index, record in enumerate(deduped)
        ]
        front = pareto_front(scored, self.config.selection.objectives)
        selected = select_diverse(front, self.config.selection)
        if self.config.evaluation is None:
            return {"selected": float(len(selected))}
        return summarize_results(selected, self.config.evaluation)

