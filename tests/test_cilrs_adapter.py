from __future__ import annotations

import sys

from sd2.adapters.cilrs_adapter import (
    build_cilrs_run_metadata,
    cilrs_record_to_sd2,
    write_sd2_jsonl,
)
from sd2.adapters.jsonl_adapter import load_run_jsonl
from sd2.core.run import pair_runs
from sd2.core.schema import FrameLog, RunMetadata
from sd2.core.stage import Stage


def test_cilrs_record_to_sd2_omits_semantic_and_records_predicted_velocity() -> None:
    frame_record = cilrs_record_to_sd2(
        {
            "frame_idx": 2,
            "timestamp": 0.1,
            "vision": {
                "image_mean": 0.42,
                "image_std": 0.11,
                "feature": [0.1, 0.2, 0.3],
                "feature_source": "mean_pooled_encoder",
            },
            "planning": {
                "velocity_pred": 4.75,
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
        run_id="cilrs_clean",
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
    assert frame.states[Stage.PLANNING].waypoints is None
    assert frame.states[Stage.PLANNING].target_speed == 4.75
    assert frame.states[Stage.PLANNING].model_extra["velocity_pred"] == 4.75
    assert frame.states[Stage.PLANNING].model_extra["planning_source"] == "predicted_velocity"
    assert frame.states[Stage.PLANNING].model_extra["target_point"] == [9.0, 1.0]


def test_cilrs_record_to_sd2_handles_missing_optional_fields() -> None:
    frame_record = cilrs_record_to_sd2(
        {
            "frame_idx": 0,
            "timestamp": 0.0,
            "vision": {"image_mean": 0.5, "image_std": 0.2, "feature": None},
            "planning": {},
            "control": {"steer": 0.0},
            "outcome": {"route_progress": 2.0, "min_ttc": None},
        },
        run_id="cilrs_clean",
    )

    frame = FrameLog.model_validate(
        {key: value for key, value in frame_record.items() if key != "type"}
    )

    assert frame.states[Stage.VISION].feature == [0.5, 0.2]
    assert frame.states[Stage.PLANNING].target_speed is None
    assert frame.states[Stage.PLANNING].model_extra["planning_source"] == "predicted_velocity"
    assert frame.states[Stage.CONTROL].throttle == 0.0
    assert frame.states[Stage.OUTCOME].route_progress == 1.0
    assert Stage.SEMANTIC not in frame.states


def test_build_cilrs_run_metadata_validates_run_metadata() -> None:
    metadata_record = build_cilrs_run_metadata(
        run_id="cilrs_Town10HD_Opt_spawn0_dest10_clean_seed42",
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
    assert metadata.model_id == "cilrs"
    assert metadata.scenario_id == "Town10HD_Opt_spawn0_dest10"
    assert metadata.stress_type is None


def test_cilrs_jsonl_round_trips_and_pairs(tmp_path) -> None:
    scenario_id = "Town10HD_Opt_spawn0_dest10"
    clean_run_id = "cilrs_Town10HD_Opt_spawn0_dest10_clean_seed42"
    stress_run_id = "cilrs_Town10HD_Opt_spawn0_dest10_gaussian_noise_s3_seed42"
    clean_metadata = build_cilrs_run_metadata(clean_run_id, scenario_id, "clean", None, 0, 42)
    stress_metadata = build_cilrs_run_metadata(
        stress_run_id,
        scenario_id,
        "stress",
        "gaussian_noise",
        3,
        42,
    )
    clean_frames = [
        cilrs_record_to_sd2(_synthetic_record(idx, feature_delta=0.0), clean_run_id)
        for idx in range(4)
    ]
    stress_frames = [
        cilrs_record_to_sd2(_synthetic_record(idx, feature_delta=0.1), stress_run_id)
        for idx in range(4)
    ]

    clean_path = tmp_path / "clean.jsonl"
    stress_path = tmp_path / "stress.jsonl"
    write_sd2_jsonl(clean_path, clean_metadata, clean_frames)
    write_sd2_jsonl(stress_path, stress_metadata, stress_frames)

    paired = pair_runs(load_run_jsonl(clean_path), load_run_jsonl(stress_path))

    assert paired.summary.paired_count == 4
    assert paired.summary.skipped_count == 0
    assert paired.pairs[0].pair_key == f"cilrs:{scenario_id}:42:0"


def test_cilrs_adapter_does_not_import_heavy_runtime_modules() -> None:
    assert "carla" not in sys.modules
    assert "torch" not in sys.modules
    assert "cilrs.model" not in sys.modules
    assert "team_code.cilrs_agent" not in sys.modules


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
            "velocity_pred": 4.0 + frame_idx * 0.1,
            "target_point": [5.0, 0.0],
            "command": 4,
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
