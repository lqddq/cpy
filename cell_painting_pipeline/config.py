"""Configuration objects for the Cell Painting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence


@dataclass(slots=True)
class ChannelConfig:
    """Describe an imaging channel used in Cell Painting experiments."""

    name: str
    pattern: str
    description: str = ""


@dataclass(slots=True)
class ManifestConfig:
    """Parameters required to build the manifest file."""

    input_root: Path
    output_path: Path
    channels: Sequence[ChannelConfig]
    metadata_extractors: Sequence[Callable[[Path], dict]] = field(default_factory=tuple)
    strict: bool = True
    extensions: Sequence[str] = (".png", ".tiff", ".tif")


@dataclass(slots=True)
class PreprocessConfig:
    """Settings for image pre-processing and batch correction."""

    illumination_reference: Optional[Path] = None
    intensity_clip: Optional[tuple[float, float]] = None
    zscore_reference: Optional[str] = None
    batch_key: str = "plate"
    qc_thresholds: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(slots=True)
class AggregationConfig:
    """Settings that control how embeddings are aggregated."""

    group_keys: Sequence[str] = ("plate", "well", "treatment", "dose", "time")
    aggregation: str = "median"
    outlier_fraction: float = 0.05


@dataclass(slots=True)
class ScaffoldSplitConfig:
    """Settings for scaffold aware data splitting."""

    test_fraction: float = 0.1
    validation_fraction: float = 0.1
    random_state: int = 42


@dataclass(slots=True)
class TrainingConfig:
    """Generic hyper-parameters shared by multiple training components."""

    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 20
    gradient_clip: Optional[float] = None
    device: str = "cpu"


@dataclass(slots=True)
class GeneratorConfig(TrainingConfig):
    """Configuration values specific to the conditional generator."""

    max_seq_length: int = 256
    diffusion_steps: int = 1000
    classifier_free_prob: float = 0.1
    guidance_scale: float = 1.5


@dataclass(slots=True)
class SamplerConfig:
    """Parameters controlling the sampling process."""

    temperature: float = 1.0
    top_k: Optional[int] = None
    guidance_scale: float = 1.5
    max_attempts: int = 10


@dataclass(slots=True)
class SelectionConfig:
    """Settings for Pareto and diversity based selection."""

    objectives: Sequence[str] = ("score", "qed", "-sa", "novelty")
    diversity_metric: str = "tanimoto"
    subset_size: int = 32
    min_scaffold_distance: float = 0.3


@dataclass(slots=True)
class EvaluationConfig:
    """Parameters for downstream evaluation and reporting."""

    report_path: Path
    include_umap: bool = True
    include_pareto: bool = True


@dataclass(slots=True)
class PipelineConfig:
    """Collect all configuration options for the complete workflow."""

    manifest: ManifestConfig
    preprocess: PreprocessConfig
    aggregation: AggregationConfig
    scaffold_split: ScaffoldSplitConfig = field(default_factory=ScaffoldSplitConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    evaluation: Optional[EvaluationConfig] = None

