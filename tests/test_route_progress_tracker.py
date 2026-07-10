from __future__ import annotations

from dataclasses import dataclass

import pytest

from experiments._carla_e2e_common import RouteProgressTracker


@dataclass(frozen=True)
class FakeLocation:
    x: float
    y: float
    z: float = 0.0


def test_self_approaching_route_does_not_snap_to_later_nearby_segment() -> None:
    route = _self_approaching_route()
    tracker = RouteProgressTracker(route)

    assert _distance(route[12], route[250]) == pytest.approx(2.0)

    for idx in range(21):
        ego = FakeLocation(route[idx].x, route[idx].y + 1.8, route[idx].z)
        progress = tracker.progress(ego)

        assert tracker.last_index <= idx + 60
        assert progress < 0.15


def test_progress_is_monotone_when_ego_overshoots_and_comes_back() -> None:
    route = [FakeLocation(float(idx), 0.0) for idx in range(101)]
    tracker = RouteProgressTracker(route)
    tracker.reset_initial(route[0])

    progress_values = [
        tracker.progress(route[idx])
        for idx in [*range(61), *range(59, 29, -1)]
    ]

    assert all(
        current >= previous
        for previous, current in zip(progress_values, progress_values[1:])
    )


def test_off_route_flag_trips_after_repeated_corridor_gate_failures() -> None:
    route = [FakeLocation(float(idx), 0.0) for idx in range(50)]
    tracker = RouteProgressTracker(
        route,
        max_lateral_m=2.0,
        off_route_frames=3,
    )

    progress_values = [
        tracker.progress(FakeLocation(0.0, y))
        for y in (3.0, 4.0, 5.0)
    ]

    assert tracker.last_index == 0
    assert tracker.corridor_gate_failed is True
    assert tracker.consecutive_gate_failures == 3
    assert tracker.off_route is True
    assert progress_values[-1] == 0.0


def test_happy_path_reaches_full_progress_at_end_of_route() -> None:
    route = [FakeLocation(float(idx), 0.0) for idx in range(121)]
    tracker = RouteProgressTracker(route)

    progress = 0.0
    for location in route:
        progress = tracker.progress(location)

    assert tracker.last_index == len(route) - 1
    assert tracker.off_route is False
    assert progress == pytest.approx(1.0)


def _self_approaching_route() -> list[FakeLocation]:
    route = [FakeLocation(float(idx), 0.0) for idx in range(131)]
    route.extend(FakeLocation(float(262 - idx), 2.0) for idx in range(131, 301))
    return route


def _distance(left: FakeLocation, right: FakeLocation) -> float:
    return (
        (left.x - right.x) ** 2
        + (left.y - right.y) ** 2
        + (left.z - right.z) ** 2
    ) ** 0.5
