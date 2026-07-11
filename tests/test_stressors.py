import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from sd2.stressors import (
    CompositeImageStressor,
    FrameRef,
    ImageStressor,
    SequenceStressor,
    build_stressor,
    validate_severity,
)


VISUAL_TYPES = [
    "gaussian_noise",
    "motion_blur",
    "fog",
    "brightness_shift",
    "contrast_shift",
    "jpeg_compression",
    "low_light",
]


def _synthetic_image() -> np.ndarray:
    y_grid, x_grid = np.mgrid[0:64, 0:64]
    checker = ((x_grid // 4 + y_grid // 4) % 2) * 80
    return np.stack(
        [
            (x_grid * 4 + checker) % 256,
            (y_grid * 4 + checker // 2) % 256,
            ((x_grid + y_grid) * 2 + checker) % 256,
        ],
        axis=2,
    ).astype(np.uint8)


def _mse(left: np.ndarray, right: np.ndarray) -> float:
    diff = left.astype(np.float32) - right.astype(np.float32)
    return float(np.mean(diff * diff))


def _frames(count: int) -> list[FrameRef]:
    return [
        FrameRef(frame_idx=idx, timestamp=idx * 0.1, image_path=f"{idx:06d}.png")
        for idx in range(count)
    ]


@pytest.mark.parametrize("stress_type", VISUAL_TYPES)
def test_visual_stressors_are_deterministic_for_same_seed(stress_type: str) -> None:
    stressor = build_stressor(stress_type)
    assert isinstance(stressor, ImageStressor)
    image = _synthetic_image()

    first = stressor.apply_image(image, 3, np.random.default_rng(123))
    second = stressor.apply_image(image, 3, np.random.default_rng(123))

    assert np.array_equal(first, second)


@pytest.mark.parametrize("stress_type", VISUAL_TYPES)
def test_visual_stressor_severity_mse_is_monotonic(stress_type: str) -> None:
    stressor = build_stressor(stress_type)
    assert isinstance(stressor, ImageStressor)
    image = _synthetic_image()
    mses = [
        _mse(image, stressor.apply_image(image, severity, np.random.default_rng(321)))
        for severity in range(1, 6)
    ]

    assert all(next_mse >= current_mse for current_mse, next_mse in zip(mses, mses[1:]))


@pytest.mark.parametrize("stress_type", VISUAL_TYPES)
def test_visual_stressors_preserve_bounds_shape_and_input(stress_type: str) -> None:
    stressor = build_stressor(stress_type)
    assert isinstance(stressor, ImageStressor)
    image = _synthetic_image()
    original = image.copy()

    output = stressor.apply_image(image, 3, np.random.default_rng(99))

    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert int(output.min()) >= 0
    assert int(output.max()) <= 255
    assert np.array_equal(image, original)


@pytest.mark.parametrize(
    "stress_type", ["contrast_shift", "jpeg_compression", "low_light"]
)
def test_new_cli_visual_stressors_produce_valid_images(stress_type: str) -> None:
    stressor = build_stressor(stress_type)
    image = _synthetic_image()

    output = stressor.apply_image(image, 5, np.random.default_rng(123))

    assert output.shape == image.shape
    assert output.dtype == np.uint8


def test_composite_image_stressor_stacks_visual_corruptions() -> None:
    image = _synthetic_image()
    contrast = build_stressor("contrast_shift")
    low_light = build_stressor("low_light")
    assert isinstance(contrast, ImageStressor)
    assert isinstance(low_light, ImageStressor)
    composite = CompositeImageStressor([(contrast, 5), (low_light, 5)])

    contrast_only = contrast.apply_image(image, 5, np.random.default_rng(123))
    output = composite.apply_image(image, 5, np.random.default_rng(123))

    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert not np.array_equal(output, contrast_only)


def test_sequence_stressors_are_deterministic_for_same_seed() -> None:
    frames = _frames(12)
    for stress_type in ["frame_drop", "frame_delay", "camera_blackout", "low_fps"]:
        stressor = build_stressor(stress_type)
        assert isinstance(stressor, SequenceStressor)
        first = stressor.apply_sequence(frames, 3, np.random.default_rng(7))
        second = stressor.apply_sequence(frames, 3, np.random.default_rng(7))
        assert first == second


def test_frame_drop_drops_expected_count() -> None:
    stressor = build_stressor("frame_drop")
    assert isinstance(stressor, SequenceStressor)

    output = stressor.apply_sequence(_frames(10), 3, np.random.default_rng(5))

    assert len(output) == 7
    assert stressor.last_report["dropped_count"] == 3
    assert len(stressor.last_report["dropped_indices"]) == 3


def test_camera_blackout_blanks_expected_window() -> None:
    stressor = build_stressor("camera_blackout")
    assert isinstance(stressor, SequenceStressor)

    output = stressor.apply_sequence(_frames(10), 3, np.random.default_rng(5))
    report = stressor.last_report
    blanked_positions = [idx for idx, frame in enumerate(output) if frame.blanked]

    assert len(output) == 10
    assert len(blanked_positions) == 3
    assert blanked_positions == report["blanked_positions"]
    assert blanked_positions == list(
        range(report["window_start_position"], report["window_start_position"] + 3)
    )
    assert all(output[position].image_path is None for position in blanked_positions)


def test_frame_delay_preserves_length_with_held_frames() -> None:
    stressor = build_stressor("frame_delay")
    assert isinstance(stressor, SequenceStressor)

    output = stressor.apply_sequence(_frames(6), 2, np.random.default_rng(5))

    assert len(output) == 6
    assert [frame.frame_idx for frame in output] == [0, 0, 0, 1, 2, 3]
    assert stressor.last_report["effective_delay_frames"] == 2


def test_low_fps_decimates_as_expected() -> None:
    stressor = build_stressor("low_fps")
    assert isinstance(stressor, SequenceStressor)

    output = stressor.apply_sequence(_frames(10), 2, np.random.default_rng(5))

    assert [frame.frame_idx for frame in output] == [0, 3, 6, 9]
    assert stressor.last_report["kept_positions"] == [0, 3, 6, 9]


def test_unknown_stressor_type_lists_available_types() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_stressor("not_a_stressor")

    message = str(exc_info.value)
    assert "unknown stressor type" in message
    assert "gaussian_noise" in message
    assert "frame_drop" in message


def test_invalid_severity_is_rejected() -> None:
    with pytest.raises(ValueError, match="1..5"):
        validate_severity(0)
    with pytest.raises(ValueError, match="1..5"):
        validate_severity(6)


def test_build_stressor_from_existing_configs() -> None:
    for config_path in Path("configs/stress").glob("*.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        stressor = build_stressor(config)
        severity = validate_severity(config["severity"])
        description = stressor.describe(severity)
        assert description["stress_type"] == config["stress_type"]
        assert description["severity"] == severity


def test_cli_stress_end_to_end_and_deterministic_bytes(tmp_path: Path) -> None:
    sample_dir = Path("data/sample/images")
    assert sample_dir.is_dir()

    input_dir = tmp_path / "images"
    shutil.copytree(sample_dir, input_dir)
    output_one = tmp_path / "out_one"
    output_two = tmp_path / "out_two"

    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    command = [
        sys.executable,
        "-m",
        "sd2.cli",
        "stress",
        "--input",
        str(input_dir),
        "--config",
        "configs/stress/gaussian_noise.yaml",
        "--seed",
        "42",
    ]
    first = subprocess.run(
        command + ["--output", str(output_one)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    second = subprocess.run(
        command + ["--output", str(output_two)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    manifest_path = output_one / "stress_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stress_type"] == "gaussian_noise"
    assert manifest["severity"] == 3
    assert manifest["seed"] == 42
    assert len(manifest["files"]) == 6
    assert sorted(path.name for path in output_one.glob("*.png")) == sorted(
        path.name for path in input_dir.glob("*.png")
    )

    first_hashes = _directory_hashes(output_one)
    second_hashes = _directory_hashes(output_two)
    assert first_hashes == second_hashes


def _directory_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for file_path in sorted(item for item in path.iterdir() if item.is_file()):
        hashes[file_path.name] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return hashes
