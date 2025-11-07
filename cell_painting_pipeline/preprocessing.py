"""Image pre-processing and batch effect mitigation utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import PreprocessConfig


class ImagePreprocessor:
    """Apply illumination correction, normalization, and QC filters."""

    def __init__(self, config: PreprocessConfig):
        self.config = config

    def load_image(self, path: Path) -> np.ndarray:
        """Load an image from disk using NumPy."""

        from imageio.v3 import imread

        return imread(path)

    def illumination_correct(self, image: np.ndarray) -> np.ndarray:
        if self.config.illumination_reference is None:
            return image
        reference = np.load(self.config.illumination_reference)
        corrected = image.astype(np.float32) / np.clip(reference, 1e-3, None)
        return np.clip(corrected, 0, None)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        if self.config.intensity_clip:
            low, high = self.config.intensity_clip
            image = np.clip(image, low, high)
        mean = float(np.mean(image))
        std = float(np.std(image)) or 1.0
        return (image - mean) / std

    def quality_check(self, image: np.ndarray) -> bool:
        sharpness = float(np.var(image))
        min_allowed, max_allowed = self.config.qc_thresholds.get("sharpness", (0.0, math.inf))
        return min_allowed <= sharpness <= max_allowed

    def process(self, paths: Iterable[Path]) -> list[np.ndarray]:
        processed: list[np.ndarray] = []
        for path in paths:
            image = self.load_image(path)
            image = self.illumination_correct(image)
            image = self.normalize(image)
            if not self.quality_check(image):
                continue
            processed.append(image)
        return processed

