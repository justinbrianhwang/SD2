"""The global plan must live in the same frame as CARLA 0.9.16's GNSS sensor.

Measured against a live server (Town10HD_Opt, ego at world x=-64.64, y=+24.47):

    GNSS reading, scaled by RoutePlanner: [ +24.48, -64.67 ]

so latitude increases with **+y** and longitude with +x. The CARLA 0.9.10 leaderboard
code these recorders were ported from computed ``my -= location.y``, which mirrors the
whole route about y=0. With that negation the plan node sat at [-24.51, -64.83], a
48.99 m error, so ``next_wp`` and therefore ``target_point`` pointed at a reflected goal
and every model drove off route. After the fix the same probe reported a 0.16 m delta.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"

# Recorders that build a GPS global plan, and the name of their converter.
SOURCES = {
    "_carla_e2e_common.py": "location_to_gps",
    "interfuser_record.py": "_location_to_gps",
    "transfuser_record.py": "_location_to_gps",
}


@dataclass
class _Location:
    x: float
    y: float
    z: float = 0.0


def _load_function(filename: str, func_name: str):
    """Extract and exec just the converter, so we never import torch or carla."""

    source = (EXPERIMENTS / filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            module = ast.Module(body=[node], type_ignores=[])
            namespace: dict[str, object] = {}
            exec(compile(module, filename, "exec"), {"math": __import__("math"), "Any": object}, namespace)  # noqa: S102
            return namespace[func_name]
    raise AssertionError(f"{filename}: {func_name} not found")


@pytest.mark.parametrize("filename,func_name", sorted(SOURCES.items()))
def test_latitude_increases_with_positive_y(filename: str, func_name: str) -> None:
    to_gps = _load_function(filename, func_name)

    north = to_gps(0.0, 0.0, _Location(x=0.0, y=100.0))
    south = to_gps(0.0, 0.0, _Location(x=0.0, y=-100.0))

    assert north["lat"] > 0.0, (
        f"{filename}: latitude must increase with +y to match CARLA 0.9.16's GNSS; "
        f"got lat={north['lat']} at y=+100"
    )
    assert south["lat"] < 0.0
    assert north["lat"] == pytest.approx(-south["lat"], rel=1e-6)


@pytest.mark.parametrize("filename,func_name", sorted(SOURCES.items()))
def test_longitude_increases_with_positive_x(filename: str, func_name: str) -> None:
    to_gps = _load_function(filename, func_name)

    east = to_gps(0.0, 0.0, _Location(x=100.0, y=0.0))
    west = to_gps(0.0, 0.0, _Location(x=-100.0, y=0.0))

    assert east["lon"] > 0.0
    assert west["lon"] < 0.0


@pytest.mark.parametrize("filename,func_name", sorted(SOURCES.items()))
def test_scaled_plan_matches_world_metres(filename: str, func_name: str) -> None:
    """RoutePlanner scales lat/lon into metres; check the round trip lands on (y, x)."""

    to_gps = _load_function(filename, func_name)
    lat_scale, lon_scale = 111324.60662786, 111319.490945

    gps = to_gps(0.0, 0.0, _Location(x=-64.644844, y=24.471010))
    scaled_north = gps["lat"] * lat_scale
    scaled_east = gps["lon"] * lon_scale

    assert scaled_north == pytest.approx(24.471010, abs=0.5)
    assert scaled_east == pytest.approx(-64.644844, abs=0.5)
