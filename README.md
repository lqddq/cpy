# Cell Painting Morphology-to-Molecule Pipeline

This repository provides a modular Python implementation of the workflow described in the
specification for mapping Cell Painting perturbation phenotypes to candidate small molecules.

The code base mirrors the conceptual modules outlined in the design document:

1. **Manifest construction** (`manifest.py`) – discovers channel-aligned images and exports
   a manifest of `(plate, well, site)` samples.
2. **Image pre-processing** (`preprocessing.py`) – performs illumination correction,
   normalization, and quality control filters.
3. **Embedding** (`embedding.py`) – demonstrates how multi-channel images can be mapped to a
   compact representation. A deterministic mock encoder is supplied for demonstration.
4. **Aggregation** (`aggregation.py`) – reduces site-level embeddings to robust well-level
   profiles.
5. **SMILES utilities** (`smiles.py`) – normalizes, deduplicates, and performs scaffold-aware
   splitting of molecular annotations.
6. **Conditional generator** (`generator.py`) – a placeholder `SeqDiffMorph` class that
   mirrors the API of the diffusion-based SMILES generator.
7. **Sampling** (`sampling.py`) – repeatedly queries the generator while enforcing lightweight
   validity checks.
8. **Scoring** (`scoring.py`) – calculates morphology–molecule consistency scores and
   uncertainty estimates.
9. **Selection** (`selection.py`) – builds Pareto fronts and performs diversity-aware subset
   selection.
10. **Evaluation** (`evaluation.py`) – writes summary statistics to a report file.
11. **Pipeline orchestration** (`pipeline.py`) – connects all components into a coherent
    workflow.
12. **CLI** (`cli.py`) – entry point for running the pipeline from the command line.

Although the numerical methods in this reference implementation are intentionally simple,
all modules are designed to be replaceable with production-grade counterparts (e.g. a
CLOOME encoder, diffusion generator, graph-based scoring). The primary goal is to provide
an executable scaffold that mirrors the modular design and demonstrates how the pieces fit
together in code.

## Usage

The repository ships as a pure-Python package, so you only need a recent version of
Python (3.9+) together with NumPy and ImageIO. The sections below walk through the
pipeline end-to-end.

### 1. Prepare the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Installing the package in editable mode exposes the `cell_painting_pipeline` package and
its `cli` entry point so you can run the workflow as a script or import the modules in a
notebook.

### 2. Arrange the raw Cell Painting data

Organise the imaging data in a directory where folders are nested as `plate/well/site`
and each site directory contains five image files—one for each Cell Painting channel:

```
/path/to/raw_data/
├── PlateA/
│   ├── A01/
│   │   ├── Site1/
│   │   │   ├── sample_DNA.tiff
│   │   │   ├── sample_ER.tiff
│   │   │   ├── sample_RNA.tiff
│   │   │   ├── sample_AGP.tiff
│   │   │   └── sample_Mito.tiff
│   │   └── Site2/
│   │       └── ...
│   └── ...
└── PlateB/
    └── ...
```

The default CLI configuration matches channel files whose names contain the substrings
`DNA`, `ER`, `RNA`, `AGP`, and `Mito`. Adjust the patterns in `cli.py` if your filenames
use different conventions.【F:cell_painting_pipeline/cli.py†L24-L37】

### 3. Run the end-to-end pipeline

```bash
python -m cell_painting_pipeline.cli /path/to/raw_data ./report.txt
```

When executed, the CLI performs the following steps:

1. Scans the directory tree and writes a manifest (`manifest.csv`) next to the raw data,
   listing the five channel paths for every `(plate, well, site)` tuple.【F:cell_painting_pipeline/manifest.py†L15-L122】
2. Applies illumination correction, intensity normalisation, and quality-control checks
   using the defaults in `PreprocessConfig`. These stubs are designed to be swapped for
   production routines.【F:cell_painting_pipeline/preprocessing.py†L1-L57】
3. Aggregates the per-site embeddings into a single morphology vector and initialises the
   mock `SeqDiffMorph` generator.【F:cell_painting_pipeline/pipeline.py†L22-L41】
4. Samples candidate SMILES, deduplicates them, scores morphology consistency, performs
   Pareto filtering, and writes a summary report to the requested path.【F:cell_painting_pipeline/pipeline.py†L41-L55】

The generated report is a human-readable text file containing the selected molecules and
their toy attributes. Re-run the command to overwrite the report after adjusting any
settings or data.

### 4. Inspect intermediate artefacts

During execution the pipeline produces several artefacts that can guide debugging:

* **Manifest (`manifest.csv`)** – the curated list of channel-aligned samples.
* **Console logs** – progress messages for manifest creation, preprocessing, and sampling.
* **Evaluation report** – the final metrics and selected candidates written to the path
  you pass as the second CLI argument.【F:cell_painting_pipeline/evaluation.py†L1-L29】

### 5. Customise the configuration

Every stage is controlled by dataclasses in `config.py`. You can override any default by
writing a short Python launcher instead of relying on the stock CLI:

```python
from pathlib import Path

from cell_painting_pipeline.cli import ChannelConfig
from cell_painting_pipeline.config import (
    AggregationConfig,
    EvaluationConfig,
    ManifestConfig,
    PipelineConfig,
    PreprocessConfig,
)
from cell_painting_pipeline.pipeline import Pipeline

channels = [
    ChannelConfig(name="dna", pattern="DAPI"),
    ChannelConfig(name="er", pattern="ER"),
    ChannelConfig(name="rna", pattern="RNA"),
    ChannelConfig(name="agp", pattern="AGP"),
    ChannelConfig(name="mito", pattern="Mito"),
]

config = PipelineConfig(
    manifest=ManifestConfig(
        input_root=Path("/path/to/raw_data"),
        output_path=Path("/path/to/raw_data/manifest.csv"),
        channels=channels,
    ),
    preprocess=PreprocessConfig(batch_key="plate"),
    aggregation=AggregationConfig(aggregation="median"),
    evaluation=EvaluationConfig(report_path=Path("./report.txt")),
)

Pipeline(config).run()
```

This pattern lets you set illumination references, customise aggregation statistics, or
alter sampling parameters before initiating the workflow.【F:cell_painting_pipeline/config.py†L10-L79】【F:cell_painting_pipeline/pipeline.py†L17-L56】

### 6. Use the modules interactively

Because each module is importable, you can stitch together bespoke notebooks: for example,
call `build_manifest` to preview channel coverage, feed the entries into
`ImagePreprocessor`, or run `sample_molecules` with alternative generators. Explore the
docstrings within each module for API-level details.

## Development

No third-party dependencies beyond NumPy and ImageIO are required. All modules include
comprehensive docstrings and type hints to aid further development.

