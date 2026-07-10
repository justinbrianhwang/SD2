from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"

RECORDERS = (
    "aim_record.py",
    "cilrs_record.py",
    "interfuser_record.py",
    "neat_record.py",
    "tcp_record.py",
    "transfuser_record.py",
)

SHARED_FLAGS = (
    "--num-vehicles",
    "--num-walkers",
    "--dest-index",
    "--anti-crawl",
    "--creep-speed",
    "--creep-frames",
    "--creep-throttle",
    "--creep-duration",
    "--intervene-stage",
    "--intervene-direction",
)

COMMON_RUN_RECORDING_RECORDERS = (
    "aim_record.py",
    "cilrs_record.py",
    "neat_record.py",
    "tcp_record.py",
)

DIRECT_LOOP_RECORDERS = (
    "interfuser_record.py",
    "transfuser_record.py",
)


def _shared_flag_recorders() -> tuple[str, ...]:
    return tuple(
        sorted(
            path.name
            for path in EXPERIMENTS.glob("*_record.py")
            if "e2e.parse_record_args" in path.read_text(encoding="utf-8")
        )
    )


NPC_FLAG_RECORDERS = tuple(sorted({*RECORDERS, *_shared_flag_recorders()}))


def _env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_recorder(recorder: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXPERIMENTS / recorder), *args],
        cwd=REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _qualified_name(node.value)
        if owner:
            return f"{owner}.{node.attr}"
        return node.attr
    return ""


def _tree(recorder: str) -> ast.Module:
    return ast.parse((EXPERIMENTS / recorder).read_text(encoding="utf-8"))


def _call_names(recorder: str) -> set[str]:
    return {
        _qualified_name(node.func)
        for node in ast.walk(_tree(recorder))
        if isinstance(node, ast.Call)
    }


def _function_names(recorder: str) -> set[str]:
    return {
        node.name
        for node in ast.walk(_tree(recorder))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


@pytest.mark.parametrize("recorder", RECORDERS)
def test_recorder_help_exposes_shared_recording_flags(recorder: str) -> None:
    result = _run_recorder(recorder, "--help")

    assert result.returncode == 0, result.stderr
    help_text = result.stdout
    for flag in SHARED_FLAGS:
        assert flag in help_text, f"{recorder} help is missing {flag}"


def test_transfuser_renamed_native_creep_flags() -> None:
    result = _run_recorder("transfuser_record.py", "--help")

    assert result.returncode == 0, result.stderr
    assert "--tf-stuck-threshold" in result.stdout
    assert "--tf-creep-duration" in result.stdout
    assert "--tf-creep-speed" in result.stdout
    assert "--creep-threshold" not in result.stdout


def test_transfuser_rejects_stale_creep_threshold_flag() -> None:
    result = _run_recorder(
        "transfuser_record.py",
        "--output",
        "unused.jsonl",
        "--creep-threshold",
        "5",
    )

    assert result.returncode == 2
    assert "unrecognized arguments: --creep-threshold" in result.stderr


def test_all_recorders_route_accepted_anti_crawl_flag_to_live_drive_loop() -> None:
    for recorder in DIRECT_LOOP_RECORDERS:
        calls = _call_names(recorder)
        assert (
            "e2e.AntiCrawlNudger" in calls or "AntiCrawlNudger" in calls
        ), f"{recorder} accepts --anti-crawl but does not instantiate AntiCrawlNudger"

    for recorder in COMMON_RUN_RECORDING_RECORDERS:
        calls = _call_names(recorder)
        assert "e2e.run_recording" in calls, f"{recorder} should reach AntiCrawlNudger via run_recording"


@pytest.mark.parametrize("recorder", NPC_FLAG_RECORDERS)
def test_no_recorder_defines_local_cleanup(recorder: str) -> None:
    functions = _function_names(recorder)

    assert "_cleanup" not in functions, f"{recorder} must use e2e.cleanup instead of a local copy"


@pytest.mark.parametrize("recorder", NPC_FLAG_RECORDERS)
def test_all_recorders_route_teardown_through_shared_cleanup(recorder: str) -> None:
    calls = _call_names(recorder)

    assert (
        "e2e.cleanup" in calls or "e2e.run_recording" in calls
    ), f"{recorder} accepts shared flags but does not reach shared cleanup"


@pytest.mark.parametrize("recorder", NPC_FLAG_RECORDERS)
def test_all_recorders_route_npc_flags_to_spawn_logic(recorder: str) -> None:
    calls = _call_names(recorder)

    assert (
        "e2e.spawn_npc_traffic" in calls or "e2e.run_recording" in calls
    ), f"{recorder} accepts --num-vehicles/--num-walkers but does not spawn NPC traffic"


@pytest.mark.parametrize("recorder", NPC_FLAG_RECORDERS)
def test_all_recorders_write_scene_sidecar(recorder: str) -> None:
    calls = _call_names(recorder)

    assert (
        "e2e.write_scene_sidecar" in calls or "e2e.run_recording" in calls
    ), f"{recorder} accepts --num-vehicles/--num-walkers but does not write a scene sidecar"
