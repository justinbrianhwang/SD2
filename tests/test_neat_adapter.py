from __future__ import annotations

import sys

import pytest

from sd2.adapters.jsonl_adapter import load_run_jsonl
from sd2.adapters.neat_adapter import (
    build_neat_run_metadata,
    neat_record_to_sd2,
    write_sd2_jsonl,
)
from sd2.core.run import pair_runs
from sd2.core.schema import FrameLog, RunMetadata
from sd2.core.stage import Stage
from sd2.metrics.semantic import SemanticCompositeMetric


def test_neat_record_to_sd2_records_bev_semantic_stage() -> None:
    frame_record = neat_record_to_sd2(
        {
            "frame_idx": 2,
            "timestamp": 0.1,
            "vision": {
                "image_mean": 0.42,
                "image_std": 0.11,
                "feature": [0.1, 0.2, 0.3],
                "feature_source": "mean_pooled_encoder_tokens",
            },
            "semantic": {
                "bev_seg_summary": _bev_summary(90, 5, 5, 0, 0),
                "red_light_occ": 2,
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
        run_id="neat_clean",
    )

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
    semantic = frame.states[Stage.SEMANTIC]
    assert semantic.objects is None
    assert semantic.model_extra["bev_seg_summary"]["class_0"] == 90
    assert semantic.model_extra["bev_seg_summary"]["class_4"] == 0
    assert semantic.model_extra["red_light_occ"] == 2.0
    assert frame.states[Stage.PLANNING].target_speed == 4.5
    assert frame.states[Stage.PLANNING].model_extra["target_point"] == [9.0, 1.0]


def test_neat_record_to_sd2_handles_missing_optional_fields() -> None:
    frame_record = neat_record_to_sd2(
        {
            "frame_idx": 0,
            "timestamp": 0.0,
            "vision": {"image_mean": 0.5, "image_std": 0.2, "feature": None},
            "semantic": {},
            "planning": {},
            "control": {"steer": 0.0},
            "outcome": {"route_progress": 2.0, "min_ttc": None},
        },
        run_id="neat_clean",
    )

    frame = FrameLog.model_validate(
        {key: value for key, value in frame_record.items() if key != "type"}
    )

    assert frame.states[Stage.VISION].feature == [0.5, 0.2]
    assert frame.states[Stage.SEMANTIC].objects is None
    assert frame.states[Stage.PLANNING].waypoints is None
    assert frame.states[Stage.CONTROL].throttle == 0.0
    assert frame.states[Stage.OUTCOME].route_progress == 1.0


def test_neat_control_anti_crawl_marker_is_conditional() -> None:
    marked = _synthetic_record(0, feature_delta=0.0, bev=_bev_summary(90, 5, 5, 0, 0))
    marked["control"]["anti_crawl_applied"] = True
    marked["control"]["applied_throttle"] = 0.6

    marked_frame = neat_record_to_sd2(marked, run_id="neat_clean")
    marked_control = marked_frame["states"]["control"]

    assert marked_control["anti_crawl_applied"] is True
    assert marked_control["applied_throttle"] == pytest.approx(0.6)

    clean_frame = neat_record_to_sd2(
        _synthetic_record(1, feature_delta=0.0, bev=_bev_summary(90, 5, 5, 0, 0)),
        run_id="neat_clean",
    )
    assert set(clean_frame["states"]["control"]) == {"steer", "throttle", "brake"}


def test_build_neat_run_metadata_validates_run_metadata() -> None:
    metadata_record = build_neat_run_metadata(
        run_id="neat_Town10HD_Opt_spawn0_dest10_clean_seed42",
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
    assert metadata.model_id == "neat"
    assert metadata.scenario_id == "Town10HD_Opt_spawn0_dest10"
    assert metadata.stress_type is None


def test_neat_bev_seg_summary_drives_semantic_composite() -> None:
    clean_frame = FrameLog.model_validate(
        {
            key: value
            for key, value in neat_record_to_sd2(
                _synthetic_record(0, feature_delta=0.0, bev=_bev_summary(100, 0, 0, 0, 0)),
                "neat_clean",
            ).items()
            if key != "type"
        }
    )
    stress_frame = FrameLog.model_validate(
        {
            key: value
            for key, value in neat_record_to_sd2(
                _synthetic_record(0, feature_delta=0.0, bev=_bev_summary(50, 50, 0, 0, 0)),
                "neat_stress",
            ).items()
            if key != "type"
        }
    )

    metric = SemanticCompositeMetric(stage=Stage.SEMANTIC)
    result = metric.compute(
        clean_frame.states[Stage.SEMANTIC],
        stress_frame.states[Stage.SEMANTIC],
    )

    assert result.missing is False
    assert result.raw_score == pytest.approx(0.5)
    assert result.details["components"] == ["seg_tv"]


def test_neat_jsonl_round_trips_and_pairs(tmp_path) -> None:
    scenario_id = "Town10HD_Opt_spawn0_dest10"
    clean_run_id = "neat_Town10HD_Opt_spawn0_dest10_clean_seed42"
    stress_run_id = "neat_Town10HD_Opt_spawn0_dest10_gaussian_noise_s3_seed42"
    clean_metadata = build_neat_run_metadata(clean_run_id, scenario_id, "clean", None, 0, 42)
    stress_metadata = build_neat_run_metadata(
        stress_run_id,
        scenario_id,
        "stress",
        "gaussian_noise",
        3,
        42,
    )
    clean_frames = [
        neat_record_to_sd2(
            _synthetic_record(idx, feature_delta=0.0, bev=_bev_summary(90, 5, 5, 0, 0)),
            clean_run_id,
        )
        for idx in range(4)
    ]
    stress_frames = [
        neat_record_to_sd2(
            _synthetic_record(idx, feature_delta=0.1, bev=_bev_summary(60, 35, 5, 0, 0)),
            stress_run_id,
        )
        for idx in range(4)
    ]

    clean_path = tmp_path / "clean.jsonl"
    stress_path = tmp_path / "stress.jsonl"
    write_sd2_jsonl(clean_path, clean_metadata, clean_frames)
    write_sd2_jsonl(stress_path, stress_metadata, stress_frames)

    paired = pair_runs(load_run_jsonl(clean_path), load_run_jsonl(stress_path))

    assert paired.summary.paired_count == 4
    assert paired.summary.skipped_count == 0
    assert paired.pairs[0].pair_key == f"neat:{scenario_id}:42:0"


def test_neat_adapter_does_not_import_heavy_runtime_modules() -> None:
    assert "carla" not in sys.modules
    assert "torch" not in sys.modules
    assert "neat.architectures" not in sys.modules
    assert "team_code.neat_agent" not in sys.modules


def _synthetic_record(frame_idx: int, feature_delta: float, bev: dict) -> dict:
    return {
        "frame_idx": frame_idx,
        "timestamp": frame_idx * 0.05,
        "vision": {
            "image_mean": 0.4,
            "image_std": 0.1,
            "feature": [1.0 + feature_delta, 0.0, 0.0],
        },
        "semantic": {
            "bev_seg_summary": bev,
            "red_light_occ": 0,
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


def _bev_summary(c0: int, c1: int, c2: int, c3: int, c4: int) -> dict:
    counts = [c0, c1, c2, c3, c4]
    total = sum(counts)
    summary = {f"class_{idx}": count for idx, count in enumerate(counts)}
    summary["nonzero_fraction"] = (total - c0) / total if total else 0.0
    summary["dominant_class"] = max(range(len(counts)), key=lambda idx: counts[idx])
    return summary
