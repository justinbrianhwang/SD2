"""Every recorder's IMU must fire on every simulation tick.

A ``sensor_tick`` equal to (or larger than) the world delta makes CARLA skip a tick
occasionally, because it fires the sensor only once the accumulated elapsed time
reaches ``sensor_tick`` and floating-point accumulation eventually falls short. In
synchronous mode the recorder then blocks in ``SensorBuffer.read`` waiting for a
reading that can never arrive: no further tick is issued while it waits. Long runs
died with ``TimeoutError: timed out waiting for sensor 'imu'`` roughly 400-500 frames
in; short runs happened to finish first.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RECORDERS = (
    "aim_record.py",
    "cilrs_record.py",
    "interfuser_record.py",
    "neat_record.py",
    "tcp_record.py",
    "transfuser_record.py",
)

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


TICKED_SENSORS = ("sensor.other.imu", "sensor.other.gnss")


def _sensor_ticks(source: str, sensor_type: str) -> list[ast.expr]:
    """Return the sensor_tick value node of every dict literal for ``sensor_type``."""

    found: list[ast.expr] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        entries = {
            key.value: value
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        type_node = entries.get("type")
        if not isinstance(type_node, ast.Constant):
            continue
        if type_node.value != sensor_type:
            continue
        if "sensor_tick" in entries:
            found.append(entries["sensor_tick"])
    return found


@pytest.mark.parametrize("recorder", RECORDERS)
@pytest.mark.parametrize("sensor_type", TICKED_SENSORS)
def test_sensor_tick_fires_every_frame(recorder: str, sensor_type: str) -> None:
    source = (EXPERIMENTS / recorder).read_text(encoding="utf-8")
    ticks = _sensor_ticks(source, sensor_type)
    assert ticks, f"{recorder}: no {sensor_type} spec found"

    for node in ticks:
        assert isinstance(node, ast.Constant), (
            f"{recorder}: {sensor_type} sensor_tick must be a literal 0.0, not a computed "
            f"value such as the model's frame rate, which equals the world delta"
        )
        assert node.value == 0.0, (
            f"{recorder}: {sensor_type} sensor_tick is {node.value!r}; it must be 0.0 so "
            f"the sensor fires on every tick"
        )


def _load_validate_sensor_ticks():
    """Load just the runtime guard, without importing carla or torch."""

    source = (EXPERIMENTS / "_carla_e2e_common.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "validate_sensor_ticks":
            module = ast.Module(body=[node], type_ignores=[])
            namespace: dict[str, object] = {}
            exec(compile(module, "_carla_e2e_common.py", "exec"), {"Any": object}, namespace)  # noqa: S102
            return namespace["validate_sensor_ticks"]
    raise AssertionError("validate_sensor_ticks not found")


def test_runtime_guard_rejects_sensor_tick_at_world_delta() -> None:
    validate = _load_validate_sensor_ticks()
    specs = ({"type": "sensor.other.imu", "id": "imu", "sensor_tick": 0.05},)
    with pytest.raises(ValueError, match="sensor_tick"):
        validate(specs, 0.05)


def test_runtime_guard_accepts_zero_sensor_tick() -> None:
    validate = _load_validate_sensor_ticks()
    specs = (
        {"type": "sensor.other.imu", "id": "imu", "sensor_tick": 0.0},
        {"type": "sensor.camera.rgb", "id": "rgb"},
        {"type": "sensor.speedometer", "id": "speed", "reading_frequency": 20},
    )
    validate(specs, 0.05)


def test_runtime_guard_rejects_tick_above_delta() -> None:
    validate = _load_validate_sensor_ticks()
    specs = ({"type": "sensor.other.gnss", "id": "gps", "sensor_tick": 0.01},)
    with pytest.raises(ValueError):
        validate(specs, 0.01)
