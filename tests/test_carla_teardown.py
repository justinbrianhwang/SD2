from __future__ import annotations

import pytest

from experiments import _carla_e2e_common as e2e


class StubSettings:
    def __init__(self, synchronous_mode: bool = True) -> None:
        self.synchronous_mode = synchronous_mode
        self.fixed_delta_seconds = 0.05


class StubWorld:
    def __init__(self, log: list[str], *, synchronous_mode: bool = True) -> None:
        self.log = log
        self.settings = StubSettings(synchronous_mode)
        self.tick_count = 0
        self.applied_settings: tuple[bool, object] | None = None

    def get_settings(self) -> StubSettings:
        self.log.append("world.get_settings")
        return self.settings

    def tick(self) -> int:
        self.tick_count += 1
        self.log.append("world.tick")
        return self.tick_count

    def apply_settings(self, settings: StubSettings) -> None:
        self.applied_settings = (
            bool(settings.synchronous_mode),
            settings.fixed_delta_seconds,
        )
        self.log.append(
            f"world.apply_settings(sync={settings.synchronous_mode},delta={settings.fixed_delta_seconds})"
        )


class StubTrafficManager:
    def __init__(self, log: list[str], *, port: int = 8123) -> None:
        self.log = log
        self.port = port
        self.sync_modes: list[bool] = []

    def get_port(self) -> int:
        self.log.append("tm.get_port")
        return self.port

    def set_synchronous_mode(self, enabled: bool) -> None:
        self.sync_modes.append(bool(enabled))
        self.log.append(f"tm.set_synchronous_mode({enabled})")


class StubActor:
    def __init__(
        self,
        name: str,
        log: list[str],
        *,
        autopilot_error: bool = False,
        destroy_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.log = log
        self.is_alive = True
        self.autopilot_error = autopilot_error
        self.destroy_error = destroy_error
        self.stop_count = 0
        self.destroy_count = 0
        self.autopilot_calls: list[tuple[bool, int | None]] = []

    def stop(self) -> None:
        self.stop_count += 1
        self.log.append(f"{self.name}.stop")

    def destroy(self) -> None:
        self.destroy_count += 1
        self.log.append(f"{self.name}.destroy")
        if self.destroy_error is not None:
            raise self.destroy_error
        self.is_alive = False

    def set_autopilot(self, enabled: bool, port: int | None = None) -> None:
        self.autopilot_calls.append((bool(enabled), port))
        self.log.append(f"{self.name}.set_autopilot({enabled},{port})")
        if self.autopilot_error:
            raise RuntimeError(f"{self.name} autopilot failed")


def test_cleanup_unregisters_npc_vehicles_before_destroying_actors() -> None:
    log: list[str] = []
    world = StubWorld(log, synchronous_mode=True)
    traffic_manager = StubTrafficManager(log, port=4455)
    sensor = StubActor("sensor", log)
    controller = StubActor("controller", log)
    walker = StubActor("walker", log)
    npc_vehicles = [StubActor("npc0", log), StubActor("npc1", log)]
    ego_vehicle = StubActor("ego", log)

    e2e.cleanup(
        None,
        world,
        traffic_manager,
        [sensor],
        ego_vehicle,
        npc_vehicles,
        [walker],
        [controller],
    )

    for vehicle in npc_vehicles:
        assert vehicle.autopilot_calls == [(False, 4455)]
        assert log.index(f"{vehicle.name}.set_autopilot(False,4455)") < log.index(
            f"{vehicle.name}.destroy"
        )

    assert log.index("npc0.set_autopilot(False,4455)") < log.index("world.tick")
    assert log.index("npc1.set_autopilot(False,4455)") < log.index("world.tick")
    assert log.index("world.tick") < log.index("npc0.destroy")
    assert log.index("world.tick") < log.index("npc1.destroy")
    assert max(
        log.index("npc0.set_autopilot(False,4455)"),
        log.index("npc1.set_autopilot(False,4455)"),
    ) < min(log.index("npc0.destroy"), log.index("npc1.destroy"))
    assert world.tick_count == 1

    for actor in [sensor, controller, walker, *npc_vehicles, ego_vehicle]:
        assert actor.destroy_count == 1

    assert traffic_manager.sync_modes == [False]
    assert world.applied_settings == (False, None)
    assert log[-1] == "world.apply_settings(sync=False,delta=None)"


def test_cleanup_continues_when_one_npc_unregister_raises_runtime_error() -> None:
    log: list[str] = []
    world = StubWorld(log, synchronous_mode=True)
    traffic_manager = StubTrafficManager(log, port=4455)
    npc_vehicles = [
        StubActor("npc0", log),
        StubActor("npc1", log, autopilot_error=True),
        StubActor("npc2", log),
    ]

    e2e.cleanup(None, world, traffic_manager, [], None, npc_vehicles, [], [])

    for vehicle in npc_vehicles:
        assert vehicle.autopilot_calls == [(False, 4455)]
        assert vehicle.destroy_count == 1
        assert log.index(f"{vehicle.name}.set_autopilot(False,4455)") < log.index(
            f"{vehicle.name}.destroy"
        )

    assert world.tick_count == 1
    assert traffic_manager.sync_modes == [False]
    assert world.applied_settings == (False, None)


def test_cleanup_restores_world_settings_when_destroy_raises() -> None:
    log: list[str] = []
    world = StubWorld(log, synchronous_mode=True)
    traffic_manager = StubTrafficManager(log, port=4455)
    bad_vehicle = StubActor("npc0", log, destroy_error=ValueError("boom"))

    with pytest.raises(ValueError, match="boom"):
        e2e.cleanup(None, world, traffic_manager, [], None, [bad_vehicle], [], [])

    assert bad_vehicle.destroy_count == 1
    assert traffic_manager.sync_modes == [False]
    assert world.applied_settings == (False, None)
    assert log.index("npc0.destroy") < log.index("tm.set_synchronous_mode(False)")
    assert log[-1] == "world.apply_settings(sync=False,delta=None)"
