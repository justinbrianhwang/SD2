from __future__ import annotations

from types import SimpleNamespace

from experiments._carla_e2e_common import AntiCrawlNudger


def _args(
    *,
    anti_crawl: bool = True,
    creep_speed: float = 1.0,
    creep_frames: int = 3,
    creep_throttle: float = 0.7,
    creep_duration: int = 2,
) -> SimpleNamespace:
    return SimpleNamespace(
        anti_crawl=anti_crawl,
        creep_speed=creep_speed,
        creep_frames=creep_frames,
        creep_throttle=creep_throttle,
        creep_duration=creep_duration,
    )


def _control(throttle: float = 0.1, brake: float = 0.2) -> SimpleNamespace:
    return SimpleNamespace(throttle=throttle, brake=brake)


def test_disabled_nudger_tracks_crawl_but_never_mutates_control_or_markers() -> None:
    nudger = AntiCrawlNudger(_args(anti_crawl=False, creep_frames=2))

    for frame_idx in range(8):
        control = _control()
        extracted = {"control": {}}

        assert nudger.apply(control, extracted, current_speed=0.0) is False

        assert control.throttle == 0.1
        assert control.brake == 0.2
        assert "anti_crawl_applied" not in extracted["control"]
        assert "applied_throttle" not in extracted["control"]
        assert nudger.crawl_counter == frame_idx + 1


def test_enabled_nudger_does_not_engage_before_creep_frames() -> None:
    nudger = AntiCrawlNudger(_args(creep_frames=3, creep_duration=2))

    for frame_idx in range(2):
        control = _control()
        extracted = {"control": {}}

        assert nudger.apply(control, extracted, current_speed=0.0) is False

        assert control.throttle == 0.1
        assert control.brake == 0.2
        assert "anti_crawl_applied" not in extracted["control"]
        assert "applied_throttle" not in extracted["control"]
        assert nudger.crawl_counter == frame_idx + 1


def test_enabled_nudger_marks_and_overrides_control_on_engagement() -> None:
    nudger = AntiCrawlNudger(_args(creep_frames=2, creep_throttle=0.65, creep_duration=3))
    assert nudger.apply(_control(), {"control": {}}, current_speed=0.0) is False
    control = _control()
    extracted = {"control": {}}

    assert nudger.apply(control, extracted, current_speed=0.0) is True

    assert control.throttle == 0.65
    assert control.brake == 0.0
    assert extracted["control"]["anti_crawl_applied"] is True
    assert extracted["control"]["applied_throttle"] == 0.65


def test_burst_is_sustained_for_creep_duration_even_after_speed_recovers() -> None:
    nudger = AntiCrawlNudger(_args(creep_frames=2, creep_throttle=0.5, creep_duration=3))
    assert nudger.apply(_control(), {"control": {}}, current_speed=0.0) is False

    nudged_results = []
    for speed in (0.0, 5.0, 5.0):
        control = _control()
        extracted = {"control": {}}
        nudged_results.append(nudger.apply(control, extracted, current_speed=speed))
        assert control.throttle == 0.5
        assert control.brake == 0.0
        assert extracted["control"]["anti_crawl_applied"] is True
        assert extracted["control"]["applied_throttle"] == 0.5

    assert nudged_results == [True, True, True]


def test_after_burst_ends_control_and_markers_are_left_untouched() -> None:
    nudger = AntiCrawlNudger(_args(creep_frames=2, creep_throttle=0.5, creep_duration=1))
    assert nudger.apply(_control(), {"control": {}}, current_speed=0.0) is False
    assert nudger.apply(_control(), {"control": {}}, current_speed=0.0) is True
    control = _control(throttle=0.33, brake=0.44)
    extracted = {"control": {}}

    assert nudger.apply(control, extracted, current_speed=5.0) is False

    assert control.throttle == 0.33
    assert control.brake == 0.44
    assert "anti_crawl_applied" not in extracted["control"]
    assert "applied_throttle" not in extracted["control"]


def test_fast_frame_resets_crawl_counter() -> None:
    nudger = AntiCrawlNudger(_args(creep_frames=2, creep_duration=1))

    assert nudger.apply(_control(), {"control": {}}, current_speed=0.0) is False
    assert nudger.crawl_counter == 1
    assert nudger.apply(_control(), {"control": {}}, current_speed=1.0) is False
    assert nudger.crawl_counter == 0
    assert nudger.apply(_control(), {"control": {}}, current_speed=0.0) is False
    assert nudger.crawl_counter == 1
    assert nudger.apply(_control(), {"control": {}}, current_speed=0.0) is True
