"""Record a minimal CARLA closed-loop drive as SD2 JSONL.

Clean and stress recordings use the same seed, spawn point, destination, and
frame count so SD2 can pair frames by ``frame_idx``. Heavy stress can still
change vehicle dynamics and route progress, so frame pairing is a controlled
alignment convention rather than proof that the ego visits identical states.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from threading import Lock
from typing import Any

from sd2.adapters.carla_adapter import (
    build_carla_run_metadata,
    carla_frame_to_sd2,
    write_sd2_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CARLA_PYTHON_API = (
    REPO_ROOT / "external" / "Carla" / "CARLA_0.9.16" / "PythonAPI" / "carla"
)
DEFAULT_TARGET_SPEED_KMH = 25.0
PLANNED_WAYPOINT_COUNT = 5


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.stress != "none" and not 1 <= args.stress_severity <= 5:
        raise SystemExit("--stress-severity must be in 1..5 for stress runs")

    carla, basic_agent_cls = _import_carla_modules()
    rng = random.Random(args.seed)
    control_noise_rng = random.Random(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = None
    traffic_manager = None
    ego_vehicle = None
    sensors: list[Any] = []
    frames: list[dict[str, Any]] = []
    collision_frames: set[int] = set()
    lane_invasion_frames: set[int] = set()
    event_lock = Lock()

    try:
        world = client.load_world(args.town)
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.delta
        world.apply_settings(settings)

        _apply_weather(world, carla, args.stress, args.stress_severity)

        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError(f"town {args.town!r} has no vehicle spawn points")

        spawn_index = _select_spawn_index(args.spawn_index, args.seed, len(spawn_points))
        destination_index = _select_destination_index(spawn_index, len(spawn_points))
        ego_vehicle = _spawn_ego_vehicle(
            world,
            blueprint_library,
            spawn_points,
            spawn_index,
            rng,
        )
        world.tick()

        sensors = _attach_event_sensors(
            carla,
            world,
            blueprint_library,
            ego_vehicle,
            collision_frames,
            lane_invasion_frames,
            event_lock,
        )

        agent = basic_agent_cls(ego_vehicle, target_speed=DEFAULT_TARGET_SPEED_KMH)
        agent.set_destination(spawn_points[destination_index].location)

        for _ in range(args.warmup):
            control = _agent_control(carla, agent, args, control_noise_rng)
            ego_vehicle.apply_control(control)
            world.tick()

        initial_route_distance = _remaining_route_distance(ego_vehicle, agent)
        if initial_route_distance <= 0.0:
            initial_route_distance = 1.0

        scenario_id = f"{args.town}_spawn{spawn_index}_dest{destination_index}"
        condition = "clean" if args.stress == "none" else "stress"
        stress_type = None if args.stress == "none" else args.stress
        severity = 0 if args.stress == "none" else args.stress_severity
        run_id = _build_run_id(
            scenario_id=scenario_id,
            condition=condition,
            stress_type=stress_type,
            severity=severity,
            seed=args.seed,
        )
        metadata = build_carla_run_metadata(
            run_id=run_id,
            scenario_id=scenario_id,
            condition=condition,
            stress_type=stress_type,
            severity=severity,
            seed=args.seed,
            town=args.town,
        )

        for frame_idx in range(args.frames):
            control = _agent_control(carla, agent, args, control_noise_rng)
            ego_vehicle.apply_control(control)
            carla_frame_id = world.tick()

            remaining_distance = _remaining_route_distance(ego_vehicle, agent)
            route_progress = 1.0 - (remaining_distance / initial_route_distance)
            measurement = {
                "frame_idx": frame_idx,
                "timestamp": round(frame_idx * args.delta, 6),
                "ego": _ego_measurement(ego_vehicle),
                "control": _control_measurement(control),
                "planned_waypoints": _planned_waypoints(agent, PLANNED_WAYPOINT_COUNT),
                "target_speed": _target_speed_mps(agent),
                "collision": _event_seen(collision_frames, carla_frame_id, event_lock),
                "lane_invasion": _event_seen(
                    lane_invasion_frames, carla_frame_id, event_lock
                ),
                "route_progress": _clamp(route_progress, 0.0, 1.0),
                "min_ttc": None,
            }
            frames.append(carla_frame_to_sd2(measurement, run_id=run_id))

        write_sd2_jsonl(args.output, metadata, frames)
        _print_summary(args.output, frames)
        return 0
    finally:
        _cleanup(carla, world, traffic_manager, sensors, ego_vehicle)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a minimal synchronous CARLA run as SD2 JSONL.",
        epilog=(
            "Determinism note: clean and stress runs use identical seed, route, "
            "spawn, destination, and frame count so frame_idx pairing holds. "
            "Strong stress can still make the vehicle drift away from the clean "
            "trajectory."
        ),
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument(
        "--stress",
        choices=["none", "weather_rain", "weather_fog", "control_noise"],
        default="none",
    )
    parser.add_argument("--stress-severity", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spawn-index", type=int, default=None)
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be non-negative")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.delta <= 0:
        parser.error("--delta must be positive")
    return args


def _import_carla_modules() -> tuple[Any, Any]:
    python_api_path = str(CARLA_PYTHON_API)
    if python_api_path not in sys.path:
        sys.path.insert(0, python_api_path)

    import carla
    from agents.navigation.basic_agent import BasicAgent

    return carla, BasicAgent


def _apply_weather(world: Any, carla: Any, stress: str, severity: int) -> None:
    if stress in ("none", "control_noise"):
        world.set_weather(carla.WeatherParameters.ClearNoon)
        return

    amount = _clamp(severity / 5.0, 0.0, 1.0) * 100.0
    weather = carla.WeatherParameters()
    weather.sun_altitude_angle = 45.0
    weather.cloudiness = 35.0 + amount * 0.45
    if stress == "weather_rain":
        weather.precipitation = amount
        weather.precipitation_deposits = amount
        weather.wetness = amount
        weather.wind_intensity = 15.0 + amount * 0.35
        weather.fog_density = 5.0
    elif stress == "weather_fog":
        weather.precipitation = 0.0
        weather.fog_density = amount
        weather.fog_distance = max(5.0, 80.0 - amount * 0.7)
        weather.wetness = 10.0
    world.set_weather(weather)


def _select_spawn_index(requested: int | None, seed: int, count: int) -> int:
    if requested is not None:
        return requested % count
    return random.Random(seed).randrange(count)


def _select_destination_index(spawn_index: int, count: int) -> int:
    return (spawn_index + max(1, count // 2)) % count


def _spawn_ego_vehicle(
    world: Any,
    blueprint_library: Any,
    spawn_points: list[Any],
    spawn_index: int,
    rng: random.Random,
) -> Any:
    blueprints = list(blueprint_library.filter("vehicle.tesla.model3"))
    if not blueprints:
        blueprints = list(blueprint_library.filter("vehicle.*"))
    if not blueprints:
        raise RuntimeError("no vehicle blueprints available")

    blueprint = sorted(blueprints, key=lambda item: item.id)[0]
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")
    if blueprint.has_attribute("color"):
        colors = blueprint.get_attribute("color").recommended_values
        if colors:
            blueprint.set_attribute("color", colors[rng.randrange(len(colors))])

    for offset in range(len(spawn_points)):
        transform = spawn_points[(spawn_index + offset) % len(spawn_points)]
        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot(False)
            return vehicle
    raise RuntimeError("failed to spawn ego vehicle at any map spawn point")


def _attach_event_sensors(
    carla: Any,
    world: Any,
    blueprint_library: Any,
    ego_vehicle: Any,
    collision_frames: set[int],
    lane_invasion_frames: set[int],
    event_lock: Lock,
) -> list[Any]:
    sensors: list[Any] = []
    collision_sensor = world.spawn_actor(
        blueprint_library.find("sensor.other.collision"),
        carla.Transform(),
        attach_to=ego_vehicle,
    )
    collision_sensor.listen(
        lambda event: _mark_event(collision_frames, event.frame, event_lock)
    )
    sensors.append(collision_sensor)

    lane_sensor = world.spawn_actor(
        blueprint_library.find("sensor.other.lane_invasion"),
        carla.Transform(),
        attach_to=ego_vehicle,
    )
    lane_sensor.listen(
        lambda event: _mark_event(lane_invasion_frames, event.frame, event_lock)
    )
    sensors.append(lane_sensor)
    return sensors


def _mark_event(event_frames: set[int], frame_id: int, event_lock: Lock) -> None:
    with event_lock:
        event_frames.add(frame_id)


def _event_seen(event_frames: set[int], frame_id: int, event_lock: Lock) -> bool:
    with event_lock:
        return frame_id in event_frames


def _agent_control(
    carla: Any,
    agent: Any,
    args: argparse.Namespace,
    noise_rng: random.Random,
) -> Any:
    done = getattr(agent, "done", lambda: False)
    control = carla.VehicleControl() if done() else agent.run_step()
    if args.stress == "control_noise":
        _apply_control_noise(control, args.stress_severity, noise_rng)
    return control


def _apply_control_noise(control: Any, severity: int, rng: random.Random) -> None:
    scale = _clamp(severity / 5.0, 0.0, 1.0)
    control.steer = _clamp(control.steer + rng.gauss(0.0, 0.10 * scale), -1.0, 1.0)
    control.throttle = _clamp(
        control.throttle + rng.gauss(0.0, 0.15 * scale),
        0.0,
        1.0,
    )


def _ego_measurement(vehicle: Any) -> dict[str, float]:
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return {
        "x": transform.location.x,
        "y": transform.location.y,
        "z": transform.location.z,
        "yaw": transform.rotation.yaw,
        "speed": speed,
    }


def _control_measurement(control: Any) -> dict[str, float]:
    return {
        "steer": float(control.steer),
        "throttle": float(control.throttle),
        "brake": float(control.brake),
    }


def _planned_waypoints(agent: Any, count: int) -> list[list[float]]:
    return [[loc.x, loc.y] for loc in _route_locations(agent)[:count]]


def _route_locations(agent: Any) -> list[Any]:
    local_planner = getattr(agent, "_local_planner", None)
    if local_planner is None:
        return []
    queue = getattr(local_planner, "_waypoints_queue", None)
    if queue is None:
        return []

    locations: list[Any] = []
    for item in list(queue):
        waypoint = item[0] if isinstance(item, (list, tuple)) else item
        transform = getattr(waypoint, "transform", None)
        if transform is not None:
            locations.append(transform.location)
    return locations


def _target_speed_mps(agent: Any) -> float | None:
    local_planner = getattr(agent, "_local_planner", None)
    value = None
    if local_planner is not None:
        value = getattr(local_planner, "_target_speed", None)
    if value is None:
        value = getattr(agent, "_target_speed", None)
    return None if value is None else float(value) / 3.6


def _remaining_route_distance(vehicle: Any, agent: Any) -> float:
    locations = [vehicle.get_location(), *_route_locations(agent)]
    if len(locations) < 2:
        return 0.0
    return sum(_distance(left, right) for left, right in zip(locations, locations[1:]))


def _distance(left: Any, right: Any) -> float:
    if hasattr(left, "distance"):
        return float(left.distance(right))
    return math.sqrt(
        (left.x - right.x) ** 2 + (left.y - right.y) ** 2 + (left.z - right.z) ** 2
    )


def _build_run_id(
    scenario_id: str,
    condition: str,
    stress_type: str | None,
    severity: int,
    seed: int,
) -> str:
    condition_part = "clean" if condition == "clean" else f"{stress_type}_s{severity}"
    return f"carla_basic_agent_{scenario_id}_{condition_part}_seed{seed}"


def _print_summary(path: Path, frames: list[dict[str, Any]]) -> None:
    collisions = sum(
        1 for frame in frames if frame["states"]["outcome"].get("collision") is True
    )
    lane_invasions = sum(
        1 for frame in frames if frame["states"]["outcome"].get("lane_invasion") is True
    )
    final_progress = 0.0
    if frames:
        final_progress = float(
            frames[-1]["states"]["outcome"].get("route_progress", 0.0)
        )
    print(
        "Wrote "
        f"{len(frames)} frames to {path} | collisions={collisions} "
        f"lane_invasions={lane_invasions} final_route_progress={final_progress:.3f}"
    )


def _cleanup(
    carla: Any,
    world: Any | None,
    traffic_manager: Any | None,
    sensors: list[Any],
    ego_vehicle: Any | None,
) -> None:
    for sensor in sensors:
        try:
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        except RuntimeError:
            pass

    try:
        if ego_vehicle is not None and ego_vehicle.is_alive:
            ego_vehicle.destroy()
    except RuntimeError:
        pass

    if traffic_manager is not None:
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass

    if world is not None:
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except RuntimeError:
            pass


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


if __name__ == "__main__":
    raise SystemExit(main())
