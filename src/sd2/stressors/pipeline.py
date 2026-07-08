"""Offline stress-run materialization for image directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from sd2.stressors import (
    FrameRef,
    ImageStressor,
    SequenceStressor,
    build_stressor,
    validate_severity,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MANIFEST_NAME = "stress_manifest.json"


def run_stress(
    input_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Apply one stress config to a directory of images and write outputs."""

    source_dir = Path(input_path)
    if not source_dir.exists():
        raise FileNotFoundError(f"input path does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"input path must be a directory of images: {source_dir}")

    image_paths = _sorted_image_paths(source_dir)
    if not image_paths:
        raise ValueError(f"input directory contains no PNG/JPG images: {source_dir}")

    config = _load_stress_config(config_path)
    severity = validate_severity(config.get("severity"))
    stressor = build_stressor(config)
    rng = np.random.default_rng(seed)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(stressor, ImageStressor):
        manifest = _apply_visual_stress(
            stressor=stressor,
            image_paths=image_paths,
            output_dir=target_dir,
            severity=severity,
            seed=seed,
            rng=rng,
        )
    elif isinstance(stressor, SequenceStressor):
        manifest = _apply_temporal_stress(
            stressor=stressor,
            image_paths=image_paths,
            output_dir=target_dir,
            severity=severity,
            seed=seed,
            rng=rng,
        )
    else:
        raise ValueError(f"unsupported stressor family for {stressor.stress_type!r}")

    manifest_path = target_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _apply_visual_stress(
    stressor: ImageStressor,
    image_paths: list[Path],
    output_dir: Path,
    severity: int,
    seed: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    description = stressor.describe(severity)
    files: list[dict[str, Any]] = []
    for image_path in image_paths:
        image = _load_rgb_array(image_path)
        stressed = stressor.apply_image(image, severity, rng)
        output_path = output_dir / image_path.name
        _save_rgb_array(stressed, output_path)
        file_params = stressor.last_report.get("params", description["params"])
        files.append(
            {
                "filename": image_path.name,
                "output_filename": output_path.name,
                "params": file_params,
            }
        )

    return {
        "stress_type": stressor.stress_type,
        "family": stressor.family,
        "severity": severity,
        "seed": seed,
        "params": description["params"],
        "files": files,
    }


def _apply_temporal_stress(
    stressor: SequenceStressor,
    image_paths: list[Path],
    output_dir: Path,
    severity: int,
    seed: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    description = stressor.describe(severity)
    frames = [
        FrameRef(frame_idx=idx, timestamp=float(idx), image_path=str(image_path))
        for idx, image_path in enumerate(image_paths)
    ]
    stressed_frames = stressor.apply_sequence(frames, severity, rng)
    black_image = _black_image_like(image_paths[0])

    files: list[dict[str, Any]] = []
    length_preserved = len(stressed_frames) == len(image_paths)
    for output_position, frame in enumerate(stressed_frames):
        if length_preserved:
            output_name = image_paths[output_position].name
        elif frame.image_path is not None:
            output_name = Path(frame.image_path).name
        else:
            output_name = f"{output_position:06d}.png"

        output_path = output_dir / output_name
        if frame.blanked or frame.image_path is None:
            _save_rgb_array(black_image, output_path)
            source_filename = None
        else:
            _save_rgb_array(_load_rgb_array(Path(frame.image_path)), output_path)
            source_filename = Path(frame.image_path).name

        files.append(
            {
                "output_position": output_position,
                "output_filename": output_name,
                "source_frame_idx": frame.frame_idx,
                "source_filename": source_filename,
                "timestamp": frame.timestamp,
                "blanked": frame.blanked,
                "params": description["params"],
            }
        )

    return {
        "stress_type": stressor.stress_type,
        "family": stressor.family,
        "severity": severity,
        "seed": seed,
        "params": description["params"],
        "input_frame_count": len(image_paths),
        "output_frame_count": len(stressed_frames),
        "materialization": (
            "Temporal stressors use sorted input images as frame references. "
            "Dropped frames are omitted; blackout frames are written as black "
            "images; delayed frames hold earlier source images at the output "
            "position."
        ),
        "sequence_report": stressor.last_report,
        "files": files,
    }


def _load_stress_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"stress config must be a YAML mapping: {path}")
    return data


def _sorted_image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _load_rgb_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _save_rgb_array(image: np.ndarray, path: Path) -> None:
    Image.fromarray(image).save(path)


def _black_image_like(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        width, height = image.convert("RGB").size
    return np.zeros((height, width, 3), dtype=np.uint8)
