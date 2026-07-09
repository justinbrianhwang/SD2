"""Shared CARLA recorder helpers for classic E2E baselines.

This module intentionally avoids importing ``carla`` or ``torch`` at import
time. Model recorder scripts import heavy runtime packages inside guarded
``_import_runtime_modules`` functions and pass those modules into this helper.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from typing import Any, Callable, Mapping

from sd2.stressors import ImageStressor, build_stressor, validate_severity


REPO_ROOT = Path(__file__).resolve().parents[1]
CARLA_PYTHON_API = (
    REPO_ROOT / "external" / "Carla" / "CARLA_0.9.16" / "PythonAPI" / "carla"
)
DEFAULT_TARGET_SPEED_KMH = 25.0
STRESS_CHOICES = ["none", "gaussian_noise", "motion_blur", "brightness", "fog"]


class SensorBuffer:
    def __init__(self, sensor_ids: list[str]) -> None:
        self._queues: dict[str, Queue[tuple[int, Any]]] = {
            sensor_id: Queue() for sensor_id in sensor_ids
        }

    def callback(self, sensor_id: str, converter: Callable[[Any], Any]) -> Callable[[Any], None]:
        def _callback(data: Any) -> None:
            self._queues[sensor_id].put((int(data.frame), converter(data)))

        return _callback

    def read(self, frame_id: int, timeout: float = 10.0) -> dict[str, tuple[int, Any]]:
        packet: dict[str, tuple[int, Any]] = {}
        for sensor_id, queue in self._queues.items():
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out waiting for sensor {sensor_id!r} at CARLA frame {frame_id}"
                    )
                try:
                    sensor_frame, payload = queue.get(timeout=remaining)
                except Empty as exc:
                    raise TimeoutError(
                        f"timed out waiting for sensor {sensor_id!r} at CARLA frame {frame_id}"
                    ) from exc
                if sensor_frame < frame_id:
                    continue
                if sensor_frame > frame_id:
                    raise RuntimeError(
                        f"sensor {sensor_id!r} skipped target frame {frame_id}; "
                        f"received future frame {sensor_frame}"
                    )
                packet[sensor_id] = (sensor_frame, payload)
                break
        return packet


class RouteProgressTracker:
    def __init__(self, locations: list[Any]) -> None:
        self.locations = locations
        self.last_index = 0
        self.initial_remaining = max(self._polyline_distance(locations), 1.0)

    def reset_initial(self, current_location: Any) -> None:
        self.initial_remaining = max(self.remaining_distance(current_location), 1.0)

    def progress(self, current_location: Any) -> float:
        remaining = self.remaining_distance(current_location)
        return clamp(1.0 - remaining / self.initial_remaining, 0.0, 1.0)

    def remaining_distance(self, current_location: Any) -> float:
        if not self.locations:
            return 0.0
        nearest_index = self._nearest_route_index(current_location)
        self.last_index = max(self.last_index, nearest_index)
        remaining = distance(current_location, self.locations[self.last_index])
        remaining += self._polyline_distance(self.locations[self.last_index :])
        return remaining

    def _nearest_route_index(self, current_location: Any) -> int:
        start = max(0, self.last_index - 5)
        candidates = range(start, len(self.locations))
        return min(candidates, key=lambda idx: distance(current_location, self.locations[idx]))

    @staticmethod
    def _polyline_distance(locations: list[Any]) -> float:
        if len(locations) < 2:
            return 0.0
        return sum(distance(left, right) for left, right in zip(locations, locations[1:]))


def parse_record_args(
    argv: list[str] | None,
    *,
    description: str,
    default_checkpoint: Path,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--stress", choices=STRESS_CHOICES, default="none")
    parser.add_argument("--stress-severity", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spawn-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be non-negative")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.delta <= 0:
        parser.error("--delta must be positive")
    if args.stress != "none":
        try:
            validate_severity(args.stress_severity)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def run_recording(
    args: argparse.Namespace,
    modules: Any,
    runtime_factory: Callable[[argparse.Namespace, Any, ImageStressor | None, Any | None], Any],
    *,
    model_id: str,
    model_label: str,
    record_to_sd2: Callable[[dict[str, Any], str], dict[str, Any]],
    build_run_metadata: Callable[..., dict[str, Any]],
    write_sd2_jsonl: Callable[[Path, dict[str, Any], list[dict[str, Any]]], None],
    logger: logging.Logger,
) -> int:
    stressor, stress_rng = build_image_stressor(args, modules, logger)
    rng = random.Random(args.seed)

    client = modules.carla.Client(args.host, args.port)
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
        world.set_weather(modules.carla.WeatherParameters.ClearNoon)

        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError(f"town {args.town!r} has no vehicle spawn points")

        spawn_index = select_spawn_index(args.spawn_index, len(spawn_points))
        destination_index = select_destination_index(spawn_index, len(spawn_points))
        ego_vehicle = spawn_ego_vehicle(
            world,
            blueprint_library,
            spawn_points,
            spawn_index,
            rng,
        )
        world.tick()

        runtime = runtime_factory(args, modules, stressor, stress_rng)
        event_sensors = attach_event_sensors(
            modules.carla,
            world,
            blueprint_library,
            ego_vehicle,
            collision_frames,
            lane_invasion_frames,
            event_lock,
        )
        sensor_buffer, model_sensors = attach_model_sensors(
            modules,
            world,
            blueprint_library,
            ego_vehicle,
            runtime.sensor_specs(),
            model_label,
            logger,
        )
        sensors = [*event_sensors, *model_sensors]

        destination = spawn_points[destination_index].location
        basic_agent = modules.BasicAgent(ego_vehicle, target_speed=DEFAULT_TARGET_SPEED_KMH)
        basic_agent.set_destination(destination)
        route = trace_route(modules, world, ego_vehicle.get_location(), destination)
        if not route:
            route = route_from_basic_agent(basic_agent)
        route_transforms = [(waypoint.transform, road_option) for waypoint, road_option in route]
        if len(route_transforms) < 2:
            raise RuntimeError(f"failed to build a {model_label} global route")

        gps_plan = build_sparse_gps_plan(world, route_transforms)
        route_locations = [transform.location for transform, _road_option in route_transforms]
        progress_tracker = RouteProgressTracker(route_locations)
        runtime.set_global_plan(gps_plan)

        logger.info(
            "Route ready: town=%s spawn=%d dest=%d dense_points=%d sparse_points=%d",
            args.town,
            spawn_index,
            destination_index,
            len(route_transforms),
            len(gps_plan),
        )

        for warmup_idx in range(args.warmup):
            frame_id = world.tick()
            packet = sensor_buffer.read(frame_id)
            packet["speed"] = (frame_id, {"speed": ego_speed(ego_vehicle)})
            control, _extracted = runtime.run_step(
                packet,
                timestamp=warmup_idx * args.delta,
                frame_id=frame_id,
            )
            ego_vehicle.apply_control(control)

        progress_tracker.reset_initial(ego_vehicle.get_location())

        scenario_id = f"{args.town}_spawn{spawn_index}_dest{destination_index}"
        condition = "clean" if args.stress == "none" else "stress"
        stress_type = None if args.stress == "none" else args.stress
        severity = 0 if args.stress == "none" else args.stress_severity
        run_id = build_run_id(model_id, scenario_id, condition, stress_type, severity, args.seed)
        metadata = build_run_metadata(
            run_id=run_id,
            scenario_id=scenario_id,
            condition=condition,
            stress_type=stress_type,
            severity=severity,
            seed=args.seed,
            town=args.town,
        )

        for frame_idx in range(args.frames):
            frame_id = world.tick()
            packet = sensor_buffer.read(frame_id)
            packet["speed"] = (frame_id, {"speed": ego_speed(ego_vehicle)})
            control, extracted = runtime.run_step(
                packet,
                timestamp=frame_idx * args.delta,
                frame_id=frame_id,
            )
            ego_vehicle.apply_control(control)

            extracted["frame_idx"] = frame_idx
            extracted["timestamp"] = round(frame_idx * args.delta, 6)
            extracted["ego"] = ego_measurement(ego_vehicle)
            extracted["outcome"] = {
                "collision": event_seen(collision_frames, frame_id, event_lock),
                "lane_invasion": event_seen(lane_invasion_frames, frame_id, event_lock),
                "route_progress": progress_tracker.progress(ego_vehicle.get_location()),
                "min_ttc": None,
            }
            frames.append(record_to_sd2(extracted, run_id=run_id))

        write_sd2_jsonl(args.output, metadata, frames)
        print_summary(args.output, frames)
        return 0
    finally:
        cleanup(
            modules.carla if "modules" in locals() else None,
            world,
            traffic_manager,
            sensors,
            ego_vehicle,
        )


def build_image_stressor(
    args: argparse.Namespace,
    modules: Any,
    logger: logging.Logger,
) -> tuple[ImageStressor | None, Any | None]:
    if args.stress == "none":
        return None, None
    stressor_type = "brightness_shift" if args.stress == "brightness" else args.stress
    stressor = build_stressor(stressor_type)
    if not isinstance(stressor, ImageStressor):
        raise RuntimeError(f"stress {args.stress!r} is not an image stressor")
    rng = modules.np.random.default_rng(args.seed)
    logger.info("Using visual stressor: %s", stressor.describe(args.stress_severity))
    return stressor, rng


def apply_visual_stress(
    image: Any,
    stressor: ImageStressor | None,
    severity: int,
    rng: Any | None,
) -> Any:
    if stressor is None:
        return image
    return stressor.apply_image(image, severity, rng)


def attach_event_sensors(
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
        lambda event: mark_event(collision_frames, event.frame, event_lock)
    )
    sensors.append(collision_sensor)

    lane_sensor = world.spawn_actor(
        blueprint_library.find("sensor.other.lane_invasion"),
        carla.Transform(),
        attach_to=ego_vehicle,
    )
    lane_sensor.listen(
        lambda event: mark_event(lane_invasion_frames, event.frame, event_lock)
    )
    sensors.append(lane_sensor)
    return sensors


def attach_model_sensors(
    modules: Any,
    world: Any,
    blueprint_library: Any,
    ego_vehicle: Any,
    sensor_specs: tuple[dict[str, Any], ...],
    model_label: str,
    logger: logging.Logger,
) -> tuple[SensorBuffer, list[Any]]:
    active_specs = [spec for spec in sensor_specs if spec["type"] != "sensor.speedometer"]
    buffer = SensorBuffer([str(spec["id"]) for spec in active_specs])
    sensors: list[Any] = []
    for spec in active_specs:
        blueprint = blueprint_library.find(spec["type"])
        for attr in ("width", "height", "fov", "sensor_tick", "reading_frequency"):
            if attr in spec and blueprint.has_attribute(attr):
                blueprint.set_attribute(attr, str(spec[attr]))
        transform = modules.carla.Transform(
            modules.carla.Location(
                x=float(spec.get("x", 0.0)),
                y=float(spec.get("y", 0.0)),
                z=float(spec.get("z", 0.0)),
            ),
            modules.carla.Rotation(
                roll=float(spec.get("roll", 0.0)),
                pitch=float(spec.get("pitch", 0.0)),
                yaw=float(spec.get("yaw", 0.0)),
            ),
        )
        sensor = world.spawn_actor(blueprint, transform, attach_to=ego_vehicle)
        sensor_id = str(spec["id"])
        sensor.listen(buffer.callback(sensor_id, sensor_converter(modules, spec, model_label)))
        sensors.append(sensor)
    logger.info(
        "Attached %s sensor rig: %s",
        model_label,
        ", ".join(str(spec["id"]) for spec in sensor_specs),
    )
    return buffer, sensors


def sensor_converter(
    modules: Any,
    spec: Mapping[str, Any],
    model_label: str,
) -> Callable[[Any], Any]:
    sensor_type = str(spec["type"])
    if sensor_type == "sensor.camera.rgb":
        return lambda image: modules.np.frombuffer(image.raw_data, dtype=modules.np.uint8).reshape(
            (image.height, image.width, 4)
        ).copy()
    if sensor_type == "sensor.lidar.ray_cast":
        return lambda lidar: modules.np.frombuffer(lidar.raw_data, dtype=modules.np.float32).reshape(
            (-1, 4)
        ).copy()
    if sensor_type == "sensor.other.imu":
        return lambda imu: modules.np.array(
            [
                imu.accelerometer.x,
                imu.accelerometer.y,
                imu.accelerometer.z,
                imu.gyroscope.x,
                imu.gyroscope.y,
                imu.gyroscope.z,
                imu.compass,
            ],
            dtype=modules.np.float32,
        )
    if sensor_type == "sensor.other.gnss":
        return lambda gps: modules.np.array(
            [gps.latitude, gps.longitude, gps.altitude],
            dtype=modules.np.float64,
        )
    raise ValueError(f"unsupported {model_label} sensor type {sensor_type!r}")


def trace_route(modules: Any, world: Any, start: Any, destination: Any) -> list[Any]:
    try:
        planner = modules.GlobalRoutePlanner(world.get_map(), 1.0)
        return planner.trace_route(start, destination)
    except TypeError:
        if modules.GlobalRoutePlannerDAO is None:
            raise
        dao = modules.GlobalRoutePlannerDAO(world.get_map(), 1.0)
        planner = modules.GlobalRoutePlanner(dao)
        planner.setup()
        return planner.trace_route(start, destination)


def route_from_basic_agent(agent: Any) -> list[Any]:
    local_planner = getattr(agent, "_local_planner", None)
    queue = getattr(local_planner, "_waypoints_queue", None)
    if queue is None:
        return []
    return list(queue)


def build_sparse_gps_plan(
    world: Any,
    route_transforms: list[tuple[Any, Any]],
    sample_factor: float = 50.0,
) -> list[tuple[dict[str, float], Any]]:
    sampled_indices = downsample_route(route_transforms, sample_factor)
    sparse_route = [route_transforms[index] for index in sampled_indices]
    lat_ref, lon_ref = get_latlon_ref(world)
    return [
        (location_to_gps(lat_ref, lon_ref, transform.location), road_option)
        for transform, road_option in sparse_route
    ]


def downsample_route(route: list[tuple[Any, Any]], sample_factor: float) -> list[int]:
    ids_to_sample: list[int] = []
    prev_option = None
    distance_since_sample = 0.0
    for idx, point in enumerate(route):
        curr_option = point[1]
        curr_value = getattr(curr_option, "value", curr_option)
        prev_value = getattr(prev_option, "value", prev_option)
        lane_change = curr_value in (5, 6)
        prev_lane_change = prev_value in (5, 6)
        if lane_change:
            ids_to_sample.append(idx)
            distance_since_sample = 0.0
        elif prev_option is None or (prev_value != curr_value and not prev_lane_change):
            ids_to_sample.append(idx)
            distance_since_sample = 0.0
        elif distance_since_sample > sample_factor:
            ids_to_sample.append(idx)
            distance_since_sample = 0.0
        elif idx == len(route) - 1:
            ids_to_sample.append(idx)
            distance_since_sample = 0.0
        else:
            current_location = point[0].location
            previous_location = route[idx - 1][0].location
            distance_since_sample += distance(current_location, previous_location)
        prev_option = curr_option
    return sorted(set(ids_to_sample))


def location_to_gps(lat_ref: float, lon_ref: float, location: Any) -> dict[str, float]:
    earth_radius_equator = 6378137.0
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * earth_radius_equator / 180.0
    my = scale * earth_radius_equator * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    mx += location.x
    my -= location.y
    lon = mx * 180.0 / (math.pi * earth_radius_equator * scale)
    lat = 360.0 * math.atan(math.exp(my / (earth_radius_equator * scale))) / math.pi - 90.0
    return {"lat": lat, "lon": lon, "z": location.z}


def get_latlon_ref(world: Any) -> tuple[float, float]:
    tree = ET.ElementTree(ET.fromstring(world.get_map().to_opendrive()))
    lat_ref = 42.0
    lon_ref = 2.0
    for opendrive in tree.iter("OpenDRIVE"):
        for header in opendrive.iter("header"):
            for georef in header.iter("geoReference"):
                if not georef.text:
                    continue
                for item in georef.text.split(" "):
                    if "+lat_0" in item:
                        lat_ref = float(item.split("=")[1])
                    if "+lon_0" in item:
                        lon_ref = float(item.split("=")[1])
    return lat_ref, lon_ref


def select_spawn_index(requested: int, count: int) -> int:
    return int(requested) % count


def select_destination_index(spawn_index: int, count: int) -> int:
    return (spawn_index + max(1, count // 2)) % count


def spawn_ego_vehicle(
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


def route_position(np: Any, route_planner: Any, gps: Any) -> Any:
    return (gps - route_planner.mean) * route_planner.scale


def route_target_point(np: Any, pos: Any, compass: float, next_wp: Any) -> Any:
    theta = compass + np.pi / 2
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
    return rotation.T.dot(local_command_point)


def mark_event(event_frames: set[int], frame_id: int, event_lock: Lock) -> None:
    with event_lock:
        event_frames.add(frame_id)


def event_seen(event_frames: set[int], frame_id: int, event_lock: Lock) -> bool:
    with event_lock:
        return frame_id in event_frames


def ego_measurement(vehicle: Any) -> dict[str, float]:
    transform = vehicle.get_transform()
    return {
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "z": float(transform.location.z),
        "yaw": float(transform.rotation.yaw),
        "speed": ego_speed(vehicle),
    }


def ego_speed(vehicle: Any) -> float:
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def build_run_id(
    model_id: str,
    scenario_id: str,
    condition: str,
    stress_type: str | None,
    severity: int,
    seed: int,
) -> str:
    condition_part = "clean" if condition == "clean" else f"{stress_type}_s{severity}"
    return f"{model_id}_{scenario_id}_{condition_part}_seed{seed}"


def print_summary(path: Path, frames: list[dict[str, Any]]) -> None:
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


def cleanup(
    carla: Any | None,
    world: Any | None,
    traffic_manager: Any | None,
    sensors: list[Any],
    ego_vehicle: Any | None,
) -> None:
    del carla
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


def distance(left: Any, right: Any) -> float:
    if hasattr(left, "distance"):
        return float(left.distance(right))
    return math.sqrt(
        (left.x - right.x) ** 2 + (left.y - right.y) ** 2 + (left.z - right.z) ** 2
    )


def pool_feature(np: Any, feature: Any) -> list[float]:
    array = np.asarray(feature, dtype=np.float32)
    if array.ndim == 0:
        return [float(array)]
    if array.ndim == 1:
        pooled = array
    elif array.ndim == 2 and array.shape[0] == 1:
        pooled = array[0]
    else:
        pooled = array.mean(axis=tuple(range(array.ndim - 1)))
    return [float(value) for value in pooled.reshape(-1)]


def target_speed_from_waypoints(np: Any, waypoints: Any) -> float | None:
    array = np.asarray(waypoints, dtype=np.float32)
    if array.ndim < 2 or array.shape[0] < 2:
        return None
    return float(np.linalg.norm(array[0] - array[1]) * 2.0)


def shape_summary(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: shape_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [shape_summary(item) for item in value]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    return type(value).__name__


def sensor_shape_summary(packet: Mapping[str, tuple[int, Any]]) -> dict[str, Any]:
    return {
        sensor_id: {"frame": frame_id, "shape": shape_summary(payload)}
        for sensor_id, (frame_id, payload) in packet.items()
    }


def jsonable(value: Any) -> Any:
    value = tolist(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def tolist(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def scalar_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if hasattr(value, "detach"):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))
