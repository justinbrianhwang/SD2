from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from experiments import _carla_e2e_common as e2e
from experiments import interfuser_record


@dataclass
class StubPidController:
    label: str
    steps: int = 0


class MutatingController:
    def __init__(self) -> None:
        self.turn_controller = StubPidController("turn")
        self.speed_controller = StubPidController("speed")
        self.stop_steps = 0
        self.forced_forward_steps = 0
        self.red_light_steps = 0
        self.block_red_light = 0
        self.in_stop_sign_effect = False
        self.block_stop_sign_distance = 0.0
        self.stop_sign_trigger_times = 0
        self.run_step_calls = 0
        self.args_seen: list[tuple[float, float, float, float]] = []

    def run_step(
        self,
        speed: float,
        waypoints: object,
        junction: float,
        traffic_light_state: float,
        stop_sign: float,
        meta_data: object,
    ) -> tuple[float, float, float, tuple[str, str, str, float]]:
        self.run_step_calls += 1
        self.stop_steps += 1
        self.red_light_steps += 2
        self.in_stop_sign_effect = True
        self.block_stop_sign_distance = 2.0
        self.stop_sign_trigger_times += 3
        self.turn_controller.steps += 1
        self.speed_controller.steps += 1
        self.args_seen.append((float(speed), junction, traffic_light_state, stop_sign))
        return 0.25, 0.5, 0.0, (
            "speed: 1.00, target_speed: 2.00",
            "semantic",
            "state",
            9.5,
        )


class FakeCarla:
    class VehicleControl:
        def __init__(self) -> None:
            self.steer = 0.0
            self.throttle = 0.0
            self.brake = 0.0


def test_interfuser_candidate_control_uses_clone_and_applied_control_mutates_real_controller() -> None:
    runtime = interfuser_record.InterFuserRuntime.__new__(
        interfuser_record.InterFuserRuntime
    )
    runtime.controller = MutatingController()

    before = deepcopy(runtime.controller.__dict__)
    control, _meta_infos = runtime._control_from_outputs(
        1.0,
        _waypoints(),
        _semantic(),
        preserve_state=True,
    )

    assert control == {"steer": 0.25, "throttle": 0.5, "brake": 0.0}
    assert runtime.controller.__dict__ == before

    runtime._control_from_outputs(
        1.0,
        _waypoints(),
        _semantic(),
        preserve_state=False,
    )

    assert runtime.controller.__dict__ != before
    assert runtime.controller.run_step_calls == 1
    assert runtime.controller.stop_steps == 1
    assert runtime.controller.in_stop_sign_effect is True
    assert runtime.controller.block_stop_sign_distance == 2.0
    assert runtime.controller.stop_sign_trigger_times == 3


def test_snapshot_pid_controllers_preserves_only_pid_controller_attributes() -> None:
    owner = SimpleNamespace(
        turn_controller=StubPidController("turn"),
        speed_controller=StubPidController("speed"),
        arbitrary_counter=0,
    )

    state = e2e.snapshot_pid_controllers(owner)
    owner.turn_controller.steps = 10
    owner.speed_controller.steps = 20
    owner.arbitrary_counter = 99
    e2e.restore_pid_controllers(owner, state)

    assert set(state) == {"turn_controller", "speed_controller"}
    assert owner.turn_controller == StubPidController("turn")
    assert owner.speed_controller == StubPidController("speed")
    assert owner.arbitrary_counter == 99


def test_interfuser_stage_none_tick_calls_real_controller_once() -> None:
    runtime = _interfuser_runtime_with_stubs()

    control, extracted = runtime.run_step({}, timestamp=0.0, frame_id=7)

    assert runtime.controller.run_step_calls == 1
    assert control.steer == 0.25
    assert control.throttle == 0.5
    assert control.brake == 0.0
    assert extracted["intervention"]["stage"] == "none"
    assert extracted["intervention"]["applied_source"] == "stress_forward"


def _interfuser_runtime_with_stubs() -> object:
    runtime = interfuser_record.InterFuserRuntime.__new__(
        interfuser_record.InterFuserRuntime
    )
    runtime.controller = MutatingController()
    runtime.modules = SimpleNamespace(
        carla=FakeCarla,
        reweight_array=np.ones((20, 20, 7), dtype=np.float32),
        find_peak_box=lambda _weighted: (
            [],
            {"car": [], "bike": [], "pedestrian": []},
        ),
    )
    runtime.np = np
    runtime.intervention = e2e.InterventionPolicy(
        model_id="interfuser",
        stage="none",
        direction=None,
        stress_type=None,
        stress_severity=0,
    )
    runtime.step = -1
    runtime.logged_shapes = True
    runtime.prev_control = None
    runtime.prev_extracted = None

    runtime._tick = lambda _sensor_packet: {
        "speed": 1.0,
        "image_mean": 0.25,
        "image_std": 0.05,
        "target_point": np.array([1.0, 0.0], dtype=np.float32),
        "next_command": 4,
    }
    runtime._with_stressed_images = lambda tick: dict(tick)
    runtime._build_model_input = lambda _tick: {}
    runtime._forward_once = lambda tick_data, _source, _model_input, _input_shapes: (
        _forward_result(runtime, tick_data)
    )
    return runtime


def _forward_result(runtime: object, tick_data: dict[str, object]) -> dict[str, object]:
    semantic = _semantic()
    control, meta_infos = runtime._control_from_outputs(
        float(tick_data["speed"]),
        _waypoints(),
        semantic,
        preserve_state=True,
    )
    return {
        "outputs": (),
        "traffic_meta": semantic["traffic_meta"],
        "bev_feature": np.array([0.1, 0.2], dtype=np.float32),
        "pred_waypoints": _waypoints(),
        "is_junction": semantic["is_junction"],
        "traffic_light_state": semantic["traffic_light_state"],
        "stop_sign": semantic["stop_sign"],
        "control": control,
        "meta_infos": meta_infos,
    }


def _semantic() -> dict[str, object]:
    return {
        "is_junction": 1.0,
        "traffic_light_state": 0.0,
        "stop_sign": 1.0,
        "traffic_meta": np.zeros((400, 7), dtype=np.float32),
    }


def _waypoints() -> np.ndarray:
    return np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
