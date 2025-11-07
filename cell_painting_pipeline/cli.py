"""Command-line interface for running the pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    AggregationConfig,
    ChannelConfig,
    EvaluationConfig,
    ManifestConfig,
    PipelineConfig,
    PreprocessConfig,
)
from .pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Cell Painting pipeline")
    parser.add_argument("input_root", type=Path, help="Directory containing raw plate data")
    parser.add_argument("output_report", type=Path, help="Path to write the evaluation report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    channels = [
        ChannelConfig(name="dna", pattern="DNA"),
        ChannelConfig(name="er", pattern="ER"),
        ChannelConfig(name="rna", pattern="RNA"),
        ChannelConfig(name="agp", pattern="AGP"),
        ChannelConfig(name="mito", pattern="Mito"),
    ]
    manifest_config = ManifestConfig(
        input_root=args.input_root,
        output_path=args.input_root / "manifest.csv",
        channels=channels,
        metadata_extractors=(),
    )
    preprocess_config = PreprocessConfig()
    aggregation_config = AggregationConfig()
    evaluation_config = EvaluationConfig(report_path=args.output_report)
    pipeline = Pipeline(
        PipelineConfig(
            manifest=manifest_config,
            preprocess=preprocess_config,
            aggregation=aggregation_config,
            evaluation=evaluation_config,
        )
    )
    pipeline.run()


if __name__ == "__main__":
    main()

