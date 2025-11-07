"""Tools for constructing a manifest from raw Cell Painting images."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from .config import ChannelConfig, ManifestConfig


@dataclass(slots=True)
class ManifestEntry:
    """Describe the files that belong to a single (plate, well, site) sample."""

    plate: str
    well: str
    site: str
    channels: Mapping[str, Path]
    metadata: Mapping[str, str]

    def to_dict(self) -> dict[str, str]:
        """Convert the manifest entry into a flat dictionary."""

        base = {
            "plate": self.plate,
            "well": self.well,
            "site": self.site,
        }
        for channel, path in self.channels.items():
            base[f"path_{channel}"] = str(path)
        for key, value in self.metadata.items():
            base[key] = value
        return base


def _match_channels(files: Iterable[Path], channels: Iterable[ChannelConfig]) -> dict[str, Path]:
    """Match each channel to a file based on filename patterns."""

    channel_map: dict[str, Path] = {}
    for channel in channels:
        matches = [path for path in files if channel.pattern in path.name]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one file for channel {channel.name!r} using pattern {channel.pattern!r}; "
                f"found {len(matches)}"
            )
        channel_map[channel.name] = matches[0]
    return channel_map


def _extract_metadata(path: Path, extractors: Iterable) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for extractor in extractors:
        extracted = extractor(path)
        metadata.update({key: str(value) for key, value in extracted.items()})
    return metadata


def build_manifest(config: ManifestConfig) -> list[ManifestEntry]:
    """Create manifest entries based on the configuration."""

    entries: list[ManifestEntry] = []
    for plate_dir in sorted(path for path in config.input_root.iterdir() if path.is_dir()):
        for well_dir in sorted(path for path in plate_dir.iterdir() if path.is_dir()):
            grouped: dict[str, list[Path]] = {}
            for site_dir in sorted(path for path in well_dir.iterdir() if path.is_dir()):
                files = [
                    file
                    for file in site_dir.iterdir()
                    if file.suffix.lower() in config.extensions
                ]
                grouped[site_dir.name] = files
            for site, files in grouped.items():
                if len(files) < len(config.channels):
                    if config.strict:
                        raise ValueError(
                            f"Missing channels for sample plate={plate_dir.name} well={well_dir.name} site={site}"
                        )
                    continue
                channels = _match_channels(files, config.channels)
                metadata = _extract_metadata(files[0], config.metadata_extractors)
                entries.append(
                    ManifestEntry(
                        plate=plate_dir.name,
                        well=well_dir.name,
                        site=site,
                        channels=channels,
                        metadata=metadata,
                    )
                )
    return entries


def save_manifest(entries: Iterable[ManifestEntry], path: Path) -> None:
    """Persist the manifest entries to disk."""

    header_written = False
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            data = entry.to_dict()
            if not header_written:
                handle.write(",".join(data.keys()) + "\n")
                header_written = True
            handle.write(",".join(data.values()) + "\n")


def dump_manifest(entries: Iterable[ManifestEntry], path: Path) -> None:
    """Persist manifest entries as JSON for debugging."""

    serialized = [entry.to_dict() for entry in entries]
    path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

