from __future__ import annotations

import sys

import pytest

from sd2.adapters.aim_adapter import (
    aim_record_to_sd2,
    build_aim_run_metadata,
    write_sd2_jsonl,
)
from sd2.adapters.jsonl_adapter import load_run_jsonl
from sd2.core.run import pair_runs
from sd2.core.schema import FrameLog, RunMetadata
from sd2.core.stage import Stage


def test_aim_record_to_sd2_omits_unobserved_semantic_stage() -> None:
    frame_record = aim_record_to_sd2(
        {
            "frame_idx": 2,
            "timestamp": 0.1,
            "vision": {
                "image_mean": 0.42,
                "image_std": 0.11,
                "feature": [0.1, 0.2, 0.3],
                "feature_source": "mean_pooled_image_encoder",
            },
            "planning": {
                "waypoints": [[1.0, 0.0], [3.0, 0.0], [5.0, 0.2]],
                "desired_speed": 4.5,
                "target_point": [9.0, 1.0],
                "command": 4,
            },
            "control": {"steer": 0.1, "throttle": 0.4, "brake": 0.0},
            "outcome": {
                "collision": False,
                "lane_invasion": True,
                "route_progress": 0.25,
                "min_ttc": 3.2,
            },
            "ego": {"x": 1.0, "y": 2.0, "yaw": 15.0, "speed": 5.0},
        },
        run_id="aim_clean",
    )

    frame = FrameLog.model_validate(
        {key: value for key, value in frame_record.items() if key != "type"}
    )

    assert set(frame.states) == {
        Stage.VISION,
        Stage.PLANNING,
        Stage.CONTROL,
        Stage.OUTCOME,
    }
    assert Stage.SEMANTIC not in frame.states
    assert frame.states[Stage.VISION].feature == [0.1, 0.2, 0.3]
    assert frame.states[Stage.PLANNING].target_speed == 4.5
    assert frame.states[Stage.PLANNING].waypoints == [[1.0, 0.0], [3.0, 0.0], [5.0, 0.2]]
    assert frame.states[Stage.PLANNING].model_extra["target_point"] == [9.0, 1.0]
    assert frame.states[Stage.CONTROL].steer == 0.1
    assert frame.states[Stage.OUTCOME].lane_invasion is True


def test_aim_record_to_sd2_handles_missing_optional_fields() -> None:
    frame_record = aim_record_to_sd2(
        {
            "frame_idx": 0,
            "timestamp": 0.0,
            "vision": {"image_mean": 0.5, "image_std": 0.2, "feature": None},
            "planning": {},
            "control": {"steer": 0.0},
            "outcome": {"route_progress": 2.0, "min_ttc": None},
        },
        run_id="aim_clean",
    )

    frame = FrameLog.model_validate(
        {key: value for key, value in frame_record.items() if key != "type"}
    )

    assert frame.states[Stage.VISION].feature == [0.5, 0.2]
    assert frame.states[Stage.PLANNING].waypoints is None
    assert frame.states[Stage.CONTROL].throttle == 0.0
    assert frame.states[Stage.OUTCOME].route_progress == 1.0
    assert Stage.SEMANTIC not in frame.states
    assert "min_ttc" not in frame_record["states"]["outcome"]


def test_aim_control_anti_crawl_marker_is_conditional() -> None:
    marked = _synthetic_record(0, feature_delta=0.0)
    marked["control"]["anti_crawl_applied"] = True
    marked["control"]["applied_throttle"] = 0.6

    marked_frame = aim_record_to_sd2(marked, run_id="aim_clean")
    marked_control = marked_frame["states"]["control"]

    assert marked_control["anti_crawl_applied"] is True
    assert marked_control["applied_throttle"] == pytest.approx(0.6)

    clean_frame = aim_record_to_sd2(_synthetic_record(1, feature_delta=0.0), run_id="aim_clean")
    assert set(clean_frame["states"]["control"]) == {"steer", "throttle", "brake"}


def test_build_aim_run_metadata_validates_run_metadata() -> None:
    metadata_record = build_aim_run_metadata(
        run_id="aim_Town10HD_Opt_spawn0_dest10_clean_seed42",
        scenario_id="Town10HD_Opt_spawn0_dest10",
        condition="clean",
        stress_type=None,
        severity=0,
        seed=42,
        town="Town10HD_Opt",
    )

    metadata = RunMetadata.model_validate(
        {key: value for key, value in metadata_record.items() if key != "type"}
    )

    assert metadata_record["type"] == "run_metadata"
    assert metadata.model_id == "aim"
    assert metadata.scenario_id == "Town10HD_Opt_spawn0_dest10"
    assert metadata.stress_type is None


def test_aim_jsonl_round_trips_and_pairs(tmp_path) -> None:
    scenario_id = "Town10HD_Opt_spawn0_dest10"
    clean_run_id = "aim_Town10HD_Opt_spawn0_dest10_clean_seed42"
    stress_run_id = "aim_Town10HD_Opt_spawn0_dest10_gaussian_noise_s3_seed42"
    clean_metadata = build_aim_run_metadata(clean_run_id, scenario_id, "clean", None, 0, 42)
    stress_metadata = build_aim_run_metadata(
        stress_run_id,
        scenario_id,
        "stress",
        "gaussian_noise",
        3,
        42,
    )
    clean_frames = [
        aim_record_to_sd2(_synthetic_record(idx, feature_delta=0.0), clean_run_id)
        for idx in range(4)
    ]
    stress_frames = [
        aim_record_to_sd2(_synthetic_record(idx, feature_delta=0.1), stress_run_id)
        for idx in range(4)
    ]

    clean_path = tmp_path / "clean.jsonl"
    stress_path = tmp_path / "stress.jsonl"
    write_sd2_jsonl(clean_path, clean_metadata, clean_frames)
    write_sd2_jsonl(stress_path, stress_metadata, stress_frames)

    paired = pair_runs(load_run_jsonl(clean_path), load_run_jsonl(stress_path))

    assert paired.summary.paired_count == 4
    assert paired.summary.skipped_count == 0
    assert paired.pairs[0].pair_key == f"aim:{scenario_id}:42:0"


def test_aim_adapter_does_not_import_heavy_runtime_modules() -> None:
    assert "carla" not in sys.modules
    assert "torch" not in sys.modules
    assert "aim.model" not in sys.modules
    assert "team_code.aim_agent" not in sys.modules


def _synthetic_record(frame_idx: int, feature_delta: float) -> dict:
    return {
        "frame_idx": frame_idx,
        "timestamp": frame_idx * 0.05,
        "vision": {
            "image_mean": 0.4,
            "image_std": 0.1,
            "feature": [1.0 + feature_delta, 0.0, 0.0],
        },
        "planning": {
            "waypoints": [[float(frame_idx), 0.0], [float(frame_idx + 1), 0.1]],
            "target_speed": 4.0,
            "target_point": [5.0, 0.0],
        },
        "control": {"steer": 0.0, "throttle": 0.3, "brake": 0.0},
        "outcome": {
            "collision": False,
            "lane_invasion": False,
            "route_progress": frame_idx / 3,
            "min_ttc": None,
        },
        "ego": {
            "x": float(frame_idx),
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.0,
            "speed": 4.0,
        },
    }
