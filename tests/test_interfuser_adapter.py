from __future__ import annotations

import sys

from sd2.adapters.interfuser_adapter import (
    build_interfuser_run_metadata,
    interfuser_record_to_sd2,
    write_sd2_jsonl,
)
from sd2.adapters.jsonl_adapter import load_run_jsonl
from sd2.core.run import pair_runs
from sd2.core.schema import FrameLog, RunMetadata
from sd2.core.stage import Stage


def test_interfuser_record_to_sd2_validates_all_observed_stages() -> None:
    frame_record = interfuser_record_to_sd2(
        {
            "frame_idx": 2,
            "timestamp": 0.1,
            "vision": {
                "image_mean": 0.42,
                "image_std": 0.11,
                "feature": [0.1, 0.2, 0.3],
                "feature_source": "mean_pooled_bev_feature",
            },
            "semantic": {
                "object_density_summary": {
                    "vehicle": 2,
                    "bike": 0,
                    "pedestrian": 1,
                    "occupied_cells": 5,
                },
                "num_objects": 3,
                "junction": 0.8,
                "traffic_light_state": 0.74,
                "stop_sign": 0.2,
            },
            "planning": {
                "waypoints": [[1.0, 0.0], [2.0, 0.2]],
                "target_speed": 4.5,
                "target_point": [9.0, 1.0],
                "command": 4,
            },
            "control": {"steer": 0.1, "throttle": 0.4, "brake": 0.0},
            "outcome": {
                "collision": False,
                "lane_invasion": True,
                "off_route": True,
                "route_progress": 0.25,
                "min_ttc": 3.2,
            },
            "ego": {"x": 1.0, "y": 2.0, "yaw": 15.0, "speed": 5.0},
        },
        run_id="interfuser_clean",
    )

    assert frame_record["type"] == "frame"
    frame = FrameLog.model_validate(
        {key: value for key, value in frame_record.items() if key != "type"}
    )

    assert set(frame.states) == {
        Stage.VISION,
        Stage.SEMANTIC,
        Stage.PLANNING,
        Stage.CONTROL,
        Stage.OUTCOME,
    }
    vision = frame.states[Stage.VISION]
    assert vision.feature == [0.1, 0.2, 0.3]
    assert vision.model_extra["image_mean"] == 0.42
    assert vision.model_extra["feature_source"] == "mean_pooled_bev_feature"

    semantic = frame.states[Stage.SEMANTIC]
    assert semantic.objects == ["pedestrian", "vehicle"]
    assert semantic.critical_object == "vehicle"
    assert semantic.traffic_light_state == "red_or_yellow"
    assert semantic.model_extra["traffic_light_state_score"] == 0.74
    assert semantic.model_extra["junction"] == 0.8
    assert semantic.model_extra["stop_sign"] == 0.2

    planning = frame.states[Stage.PLANNING]
    assert planning.waypoints == [[1.0, 0.0], [2.0, 0.2]]
    assert planning.target_speed == 4.5
    assert planning.model_extra["ego"]["yaw"] == 15.0
    assert planning.model_extra["target_point"] == [9.0, 1.0]

    control = frame.states[Stage.CONTROL]
    assert control.steer == 0.1
    assert control.throttle == 0.4
    assert control.brake == 0.0

    outcome = frame.states[Stage.OUTCOME]
    assert outcome.collision is False
    assert outcome.lane_invasion is True
    assert outcome.off_route is True
    assert outcome.route_progress == 0.25
    assert outcome.min_ttc == 3.2


def test_interfuser_record_to_sd2_handles_missing_optional_fields() -> None:
    frame_record = interfuser_record_to_sd2(
        {
            "frame_idx": 0,
            "timestamp": 0.0,
            "vision": {"image_mean": 0.5, "image_std": 0.2, "feature": None},
            "semantic": {
                "object_density_summary": {},
                "traffic_light_state": None,
            },
            "planning": {},
            "control": {"steer": 0.0},
            "outcome": {"route_progress": 2.0, "min_ttc": None},
        },
        run_id="interfuser_clean",
    )

    frame = FrameLog.model_validate(
        {key: value for key, value in frame_record.items() if key != "type"}
    )

    assert frame.states[Stage.VISION].feature == [0.5, 0.2]
    assert frame.states[Stage.VISION].model_extra["feature_source"] == "image_stats"
    assert frame.states[Stage.SEMANTIC].objects is None
    assert frame.states[Stage.PLANNING].waypoints is None
    assert frame.states[Stage.CONTROL].throttle == 0.0
    assert frame.states[Stage.OUTCOME].route_progress == 1.0
    assert "min_ttc" not in frame_record["states"]["outcome"]


def test_build_interfuser_run_metadata_validates_run_metadata() -> None:
    metadata_record = build_interfuser_run_metadata(
        run_id="interfuser_Town10HD_Opt_spawn0_dest10_clean_seed42",
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
    assert metadata.model_id == "interfuser"
    assert metadata.scenario_id == "Town10HD_Opt_spawn0_dest10"
    assert metadata.condition == "clean"
    assert metadata.stress_type is None


def test_interfuser_jsonl_round_trips_and_pairs(tmp_path) -> None:
    scenario_id = "Town10HD_Opt_spawn0_dest10"
    clean_run_id = "interfuser_Town10HD_Opt_spawn0_dest10_clean_seed42"
    stress_run_id = "interfuser_Town10HD_Opt_spawn0_dest10_gaussian_noise_s3_seed42"
    clean_metadata = build_interfuser_run_metadata(
        clean_run_id,
        scenario_id,
        "clean",
        None,
        0,
        42,
    )
    stress_metadata = build_interfuser_run_metadata(
        stress_run_id,
        scenario_id,
        "stress",
        "gaussian_noise",
        3,
        42,
    )

    clean_frames = [
        interfuser_record_to_sd2(_synthetic_record(idx, feature_delta=0.0), clean_run_id)
        for idx in range(4)
    ]
    stress_frames = [
        interfuser_record_to_sd2(_synthetic_record(idx, feature_delta=0.1), stress_run_id)
        for idx in range(4)
    ]

    clean_path = tmp_path / "clean.jsonl"
    stress_path = tmp_path / "stress.jsonl"
    write_sd2_jsonl(clean_path, clean_metadata, clean_frames)
    write_sd2_jsonl(stress_path, stress_metadata, stress_frames)

    clean_run = load_run_jsonl(clean_path)
    stress_run = load_run_jsonl(stress_path)
    paired = pair_runs(clean_run, stress_run)

    assert paired.summary.paired_count == 4
    assert paired.summary.skipped_count == 0
    assert paired.pairs[0].pair_key == f"interfuser:{scenario_id}:42:0"


def test_interfuser_adapter_does_not_import_heavy_runtime_modules() -> None:
    assert "carla" not in sys.modules
    assert "torch" not in sys.modules
    assert "timm" not in sys.modules
    assert "team_code.interfuser_agent" not in sys.modules


def _synthetic_record(frame_idx: int, feature_delta: float) -> dict:
    return {
        "frame_idx": frame_idx,
        "timestamp": frame_idx * 0.05,
        "vision": {
            "image_mean": 0.4,
            "image_std": 0.1,
            "feature": [1.0 + feature_delta, 0.0, 0.0],
        },
        "semantic": {
            "object_density_summary": {"vehicle": 1, "pedestrian": 0},
            "num_objects": 1,
            "junction": 0.9,
            "traffic_light_state": 0.1,
            "stop_sign": 0.8,
        },
        "planning": {
            "waypoints": [[float(frame_idx), 0.0], [float(frame_idx + 1), 0.1]],
            "target_speed": 4.0,
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
