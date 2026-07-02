"""Generate deterministic sample SD2 JSONL run logs."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
FRAME_COUNT = 30
SEED = 42


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    _write_run(SAMPLE_DIR / "clean_run.jsonl", _metadata("clean"), _frames("clean", rng))
    rng = random.Random(SEED)
    _write_run(SAMPLE_DIR / "stress_run.jsonl", _metadata("stress"), _frames("stress", rng))


def _metadata(condition: str) -> dict[str, Any]:
    if condition == "clean":
        return {
            "type": "run_metadata",
            "run_id": "openemma_town05_route01_clean_seed42",
            "model_id": "openemma",
            "scenario_id": "town05_route01",
            "condition": "clean",
            "stress_type": None,
            "severity": 0,
            "seed": SEED,
            "timestamp_start": "2026-01-01T00:00:00",
        }
    return {
        "type": "run_metadata",
        "run_id": "openemma_town05_route01_gaussian_noise_s3_seed42",
        "model_id": "openemma",
        "scenario_id": "town05_route01",
        "condition": "stress",
        "stress_type": "gaussian_noise",
        "severity": 3,
        "seed": SEED,
        "timestamp_start": "2026-01-01T00:00:00",
    }


def _frames(condition: str, rng: random.Random) -> list[dict[str, Any]]:
    return [_frame(condition, idx, rng) for idx in range(FRAME_COUNT)]


def _frame(condition: str, idx: int, rng: random.Random) -> dict[str, Any]:
    timestamp = round(idx * 0.1, 3)
    stress = condition == "stress"
    drift = max(0, idx - 14) if stress else 0
    base_progress = idx / (FRAME_COUNT - 1)

    clean_embedding = [
        round(0.10 + 0.004 * idx + rng.uniform(-0.002, 0.002), 4),
        round(0.03 + 0.002 * math.sin(idx / 4), 4),
        round(0.94 - 0.003 * idx + rng.uniform(-0.002, 0.002), 4),
    ]
    embedding = [
        round(value + (0.006 * drift if stress and pos == 0 else 0.0), 4)
        for pos, value in enumerate(clean_embedding)
    ]

    objects = ["lane", "vehicle", "traffic_light"]
    if 10 <= idx <= 21:
        objects.append("pedestrian")
    semantic_objects = objects.copy()
    if stress and 15 <= idx <= 18:
        semantic_objects = [obj for obj in semantic_objects if obj != "pedestrian"]

    clean_intent = "slow_down" if 10 <= idx <= 21 else "follow_lane"
    stress_intent = clean_intent
    if stress and idx >= 15:
        stress_intent = "maintain_speed" if idx < 20 else "change_lane_left"

    waypoint_shift = 0.04 * drift
    waypoints = []
    for step in range(4):
        x = idx * 0.45 + step * 1.2
        y = 0.08 * math.sin(idx / 5) + step * 0.05
        if stress:
            y += waypoint_shift * (step + 1)
        waypoints.append([round(x, 3), round(y, 3)])

    clean_target_speed = 5.8 if idx < 10 else 4.2 if idx <= 21 else 5.0
    target_speed = clean_target_speed
    if stress and idx >= 15:
        target_speed = max(2.0, clean_target_speed + 0.15 * drift)

    clean_brake = 0.05
    if 10 <= idx <= 21:
        clean_brake = 0.28
    brake = clean_brake
    if stress and idx >= 15:
        brake = 0.08 if idx < 19 else min(1.0, 0.35 + 0.08 * (idx - 19))

    throttle = max(0.0, round(0.45 - brake * 0.7, 3))
    steer = round(0.02 * math.sin(idx / 3) + (0.018 * drift if stress else 0.0), 3)

    collision = bool(stress and idx >= 25)
    lane_invasion = bool(stress and idx >= 22)
    route_progress = round(base_progress - (0.08 if stress and idx >= 24 else 0.0), 3)

    critical_object = "pedestrian" if 10 <= idx <= 21 else "vehicle"
    mentioned = not (stress and idx >= 15)
    reasoning_text = (
        "A pedestrian is near the lane, so the ego vehicle should slow down."
        if clean_intent == "slow_down"
        else "The lane is clear, so the ego vehicle should follow the lane."
    )
    if stress and idx >= 15:
        reasoning_text = (
            "The forward path appears open, so maintain speed through the segment."
            if idx < 20
            else "The object can be avoided by changing lane left."
        )

    return {
        "type": "frame",
        "run_id": (
            "openemma_town05_route01_gaussian_noise_s3_seed42"
            if stress
            else "openemma_town05_route01_clean_seed42"
        ),
        "frame_idx": idx,
        "timestamp": timestamp,
        "states": {
            "vision": {
                "image_path": f"frames/{idx:06d}.png",
                "embedding": embedding,
                "noise_level": round(0.02 + (0.12 if stress else 0.0), 3),
            },
            "semantic": {
                "objects": semantic_objects,
                "critical_object": critical_object,
                "traffic_light_state": "green",
                "lane_state": "centered" if not lane_invasion else "left_drift",
            },
            "reasoning": {
                "text": reasoning_text,
                "intent": stress_intent,
                "critical_object_mentioned": mentioned,
            },
            "planning": {
                "waypoints": waypoints,
                "target_speed": round(target_speed, 3),
                "selected_maneuver": stress_intent,
            },
            "control": {
                "steer": steer,
                "throttle": throttle,
                "brake": round(brake, 3),
            },
            "outcome": {
                "collision": collision,
                "lane_invasion": lane_invasion,
                "route_progress": route_progress,
                "driving_score": round(1.0 - (0.35 if collision else 0.0) - (0.1 if lane_invasion else 0.0), 3),
                "min_ttc": round(max(0.2, 4.0 - 0.12 * idx - (1.0 if stress and idx >= 15 else 0.0)), 3),
            },
        },
    }


def _write_run(path: Path, metadata: dict[str, Any], frames: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, separators=(",", ":")) + "\n")
        for frame in frames:
            handle.write(json.dumps(frame, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
