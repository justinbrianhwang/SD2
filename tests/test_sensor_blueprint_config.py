from __future__ import annotations

import ast
import logging
import struct
from pathlib import Path
from typing import Any

import pytest

from experiments import _carla_e2e_common as e2e


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


class FakeActorAttribute:
    def __init__(self, attribute_type: str, value: Any) -> None:
        self.type = attribute_type
        self.value = str(value)

    def as_int(self) -> int:
        return int(self.value)

    def as_float(self) -> float:
        return float(self.value)

    def as_str(self) -> str:
        if self.type in {"Int", "Float"}:
            raise RuntimeError("bad attribute cast")
        return self.value

    def as_bool(self) -> bool:
        return self.value.lower() in {"1", "true", "yes"}


class FakeBlueprint:
    def __init__(
        self,
        sensor_type: str,
        attributes: dict[str, tuple[str, Any]],
        *,
        ignored_writes: set[str] | None = None,
    ) -> None:
        self.sensor_type = sensor_type
        self.attributes = {
            name: FakeActorAttribute(attribute_type, value)
            for name, (attribute_type, value) in attributes.items()
        }
        self.ignored_writes = ignored_writes or set()
        self.set_calls: list[tuple[str, str]] = []

    def has_attribute(self, name: str) -> bool:
        return name in self.attributes

    def set_attribute(self, name: str, value: str) -> None:
        self.set_calls.append((name, value))
        if name in self.ignored_writes:
            return
        self.attributes[name].value = value

    def get_attribute(self, name: str) -> FakeActorAttribute:
        return self.attributes[name]

    def int_value(self, name: str) -> int:
        return self.attributes[name].as_int()

    def float_value(self, name: str) -> float:
        return self.attributes[name].as_float()


class Float32QuantizingFakeBlueprint(FakeBlueprint):
    def set_attribute(self, name: str, value: str) -> None:
        self.set_calls.append((name, value))
        if name in self.ignored_writes:
            return
        attribute = self.attributes[name]
        if attribute.type == "Float":
            value = str(struct.unpack("f", struct.pack("f", float(value)))[0])
        attribute.value = value


class ChangedIntFakeBlueprint(FakeBlueprint):
    def set_attribute(self, name: str, value: str) -> None:
        self.set_calls.append((name, value))
        if name in self.ignored_writes:
            return
        if self.attributes[name].type == "Int":
            value = str(int(value) + 1)
        self.attributes[name].value = value


class FakeBlueprintLibrary:
    def __init__(
        self,
        *,
        blueprint_cls: type[FakeBlueprint] = FakeBlueprint,
        ignored_writes: set[str] | None = None,
    ) -> None:
        self.blueprint_cls = blueprint_cls
        self.ignored_writes = ignored_writes or set()
        self.blueprints: list[FakeBlueprint] = []

    def find(self, sensor_type: str) -> FakeBlueprint:
        blueprint = self.blueprint_cls(
            sensor_type,
            _blueprint_attributes(sensor_type),
            ignored_writes=self.ignored_writes,
        )
        self.blueprints.append(blueprint)
        return blueprint


class FakeSensor:
    def __init__(self, blueprint: FakeBlueprint) -> None:
        self.blueprint = blueprint
        self.callback: Any = None

    def listen(self, callback: Any) -> None:
        self.callback = callback


class FakeWorld:
    def __init__(self) -> None:
        self.sensors: list[FakeSensor] = []

    def spawn_actor(self, blueprint: FakeBlueprint, transform: Any, attach_to: Any) -> FakeSensor:
        sensor = FakeSensor(blueprint)
        self.sensors.append(sensor)
        return sensor


class FakeCarla:
    class Location:
        def __init__(self, **kwargs: float) -> None:
            self.kwargs = kwargs

    class Rotation:
        def __init__(self, **kwargs: float) -> None:
            self.kwargs = kwargs

    class Transform:
        def __init__(self, location: Any, rotation: Any) -> None:
            self.location = location
            self.rotation = rotation


class FakeModules:
    carla = FakeCarla


def _float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def _blueprint_attributes(sensor_type: str) -> dict[str, tuple[str, Any]]:
    if sensor_type.startswith("sensor.camera"):
        return {
            "image_size_x": ("Int", 800),
            "image_size_y": ("Int", 600),
            "fov": ("Float", 90.0),
            "lens_circle_multiplier": ("Float", 0.0),
            "lens_circle_falloff": ("Float", 0.0),
            "chromatic_aberration_intensity": ("Float", 0.0),
            "chromatic_aberration_offset": ("Float", 0.0),
            "sensor_tick": ("Float", 0.0),
        }
    if sensor_type.startswith("sensor.lidar"):
        return {
            "range": ("Float", 0.0),
            "rotation_frequency": ("Float", 0.0),
            "channels": ("Int", 0),
            "upper_fov": ("Float", 0.0),
            "lower_fov": ("Float", 0.0),
            "points_per_second": ("Int", 0),
            "atmosphere_attenuation_rate": ("Float", 0.0),
            "dropoff_general_rate": ("Float", 0.0),
            "dropoff_intensity_limit": ("Float", 0.0),
            "dropoff_zero_intensity": ("Float", 0.0),
            "sensor_tick": ("Float", 0.0),
        }
    if sensor_type.startswith("sensor.other.gnss"):
        return {
            "noise_alt_bias": ("Float", 1.0),
            "noise_lat_bias": ("Float", 1.0),
            "noise_lon_bias": ("Float", 1.0),
            "sensor_tick": ("Float", 0.0),
        }
    raise AssertionError(f"unexpected sensor type {sensor_type!r}")


def _configured_blueprint(
    spec: dict[str, Any],
    *,
    blueprint_cls: type[FakeBlueprint] = FakeBlueprint,
    ignored_writes: set[str] | None = None,
) -> FakeBlueprint:
    blueprint_library = FakeBlueprintLibrary(
        blueprint_cls=blueprint_cls,
        ignored_writes=ignored_writes,
    )
    e2e.attach_model_sensors(
        FakeModules(),
        FakeWorld(),
        blueprint_library,
        ego_vehicle=object(),
        sensor_specs=(spec,),
        model_label="test",
        logger=logging.getLogger("test_sensor_blueprint_config"),
    )
    return blueprint_library.blueprints[0]


def test_camera_width_height_translate_to_image_size_attributes() -> None:
    blueprint = _configured_blueprint(
        {"type": "sensor.camera.rgb", "id": "front", "width": 400, "height": 300, "fov": 100}
    )

    assert blueprint.int_value("image_size_x") == 400
    assert blueprint.int_value("image_size_y") == 300


def test_rgb_camera_receives_leaderboard_lens_attributes() -> None:
    blueprint = _configured_blueprint(
        {"type": "sensor.camera.rgb", "id": "front", "width": 400, "height": 300, "fov": 100}
    )

    assert blueprint.float_value("lens_circle_multiplier") == 3.0
    assert blueprint.float_value("lens_circle_falloff") == 3.0
    assert blueprint.float_value("chromatic_aberration_intensity") == 0.5
    assert blueprint.float_value("chromatic_aberration_offset") == 0.0


def test_lidar_receives_leaderboard_attributes() -> None:
    blueprint = _configured_blueprint({"type": "sensor.lidar.ray_cast", "id": "lidar"})

    assert blueprint.float_value("range") == 85.0
    assert blueprint.int_value("channels") == 64
    assert blueprint.int_value("points_per_second") == 600000
    assert blueprint.float_value("rotation_frequency") == 10.0
    assert blueprint.float_value("upper_fov") == 10.0
    assert blueprint.float_value("lower_fov") == -30.0
    assert blueprint.float_value("atmosphere_attenuation_rate") == 0.004
    assert blueprint.float_value("dropoff_general_rate") == 0.45
    assert blueprint.float_value("dropoff_intensity_limit") == 0.8
    assert blueprint.float_value("dropoff_zero_intensity") == 0.4


def test_lidar_float32_quantized_dropoff_general_rate_round_trips() -> None:
    blueprint = _configured_blueprint(
        {"type": "sensor.lidar.ray_cast", "id": "lidar"},
        blueprint_cls=Float32QuantizingFakeBlueprint,
    )

    assert blueprint.float_value("dropoff_general_rate") == _float32(0.45)


def test_zero_valued_float32_quantized_defaults_round_trip() -> None:
    camera_blueprint = _configured_blueprint(
        {"type": "sensor.camera.rgb", "id": "front", "width": 400, "height": 300, "fov": 100},
        blueprint_cls=Float32QuantizingFakeBlueprint,
    )
    gnss_blueprint = _configured_blueprint(
        {"type": "sensor.other.gnss", "id": "gps"},
        blueprint_cls=Float32QuantizingFakeBlueprint,
    )

    assert camera_blueprint.float_value("chromatic_aberration_offset") == 0.0
    assert gnss_blueprint.float_value("noise_alt_bias") == 0.0
    assert gnss_blueprint.float_value("noise_lat_bias") == 0.0
    assert gnss_blueprint.float_value("noise_lon_bias") == 0.0


def test_gnss_receives_leaderboard_noise_bias_attributes() -> None:
    blueprint = _configured_blueprint({"type": "sensor.other.gnss", "id": "gps"})

    assert blueprint.float_value("noise_alt_bias") == 0.0
    assert blueprint.float_value("noise_lat_bias") == 0.0
    assert blueprint.float_value("noise_lon_bias") == 0.0


def test_spec_value_overrides_leaderboard_default() -> None:
    blueprint = _configured_blueprint(
        {"type": "sensor.lidar.ray_cast", "id": "lidar", "rotation_frequency": 20}
    )

    assert blueprint.float_value("rotation_frequency") == 20.0


def test_spec_key_without_blueprint_attribute_raises() -> None:
    with pytest.raises(ValueError, match="front.*bad_setting"):
        _configured_blueprint(
            {
                "type": "sensor.camera.rgb",
                "id": "front",
                "width": 400,
                "height": 300,
                "fov": 100,
                "bad_setting": 1,
            }
        )


def test_silently_ignored_blueprint_write_raises() -> None:
    with pytest.raises(ValueError, match="front.*image_size_x.*round-trip"):
        _configured_blueprint(
            {
                "type": "sensor.camera.rgb",
                "id": "front",
                "width": 400,
                "height": 300,
                "fov": 100,
            },
            ignored_writes={"image_size_x"},
        )


def test_float32_quantized_ignored_lidar_write_still_raises() -> None:
    with pytest.raises(ValueError, match="lidar.*dropoff_general_rate.*round-trip"):
        _configured_blueprint(
            {"type": "sensor.lidar.ray_cast", "id": "lidar"},
            blueprint_cls=Float32QuantizingFakeBlueprint,
            ignored_writes={"dropoff_general_rate"},
        )


def test_changed_int_blueprint_write_still_raises() -> None:
    with pytest.raises(ValueError, match="front.*image_size_x.*requested 400, got 401"):
        _configured_blueprint(
            {
                "type": "sensor.camera.rgb",
                "id": "front",
                "width": 400,
                "height": 300,
                "fov": 100,
            },
            blueprint_cls=ChangedIntFakeBlueprint,
        )


def _sensor_types_in_dict_literals(source: str) -> set[str]:
    sensor_types: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        entries = {
            key.value: value
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        type_node = entries.get("type")
        if isinstance(type_node, ast.Constant) and isinstance(type_node.value, str):
            sensor_types.add(type_node.value)
    return sensor_types


def test_lidar_recorders_still_declare_ray_cast_lidar() -> None:
    for recorder in ("interfuser_record.py", "transfuser_record.py"):
        source = (EXPERIMENTS / recorder).read_text(encoding="utf-8")
        assert "sensor.lidar.ray_cast" in _sensor_types_in_dict_literals(source), recorder
