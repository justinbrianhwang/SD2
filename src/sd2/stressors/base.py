"""Base interfaces and registry for input stressors."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class FrameRef:
    """Reference to one frame in an image sequence.

    Temporal stressors operate on frame references only. They may drop entries,
    duplicate entries, or return a reference with ``blanked=True`` and
    ``image_path=None`` to represent a black camera frame.
    """

    frame_idx: int
    timestamp: float
    image_path: str | None
    blanked: bool = False


def validate_severity(severity: Any) -> int:
    """Validate and return a severity in the inclusive 1..5 range."""

    if isinstance(severity, bool) or not isinstance(severity, Integral):
        raise ValueError(f"severity must be an integer in 1..5; got {severity!r}")
    value = int(severity)
    if value < 1 or value > 5:
        raise ValueError(f"severity must be an integer in 1..5; got {severity!r}")
    return value


class Stressor(ABC):
    """Common base for visual and temporal stressors."""

    stress_type: str
    family: str

    def __init__(self, **options: Any) -> None:
        self.options = dict(options)
        self._last_report: dict[str, Any] = {}

    @abstractmethod
    def params_for_severity(self, severity: int) -> dict[str, Any]:
        """Return concrete parameters for a validated severity level."""

    def describe(self, severity: int) -> dict[str, Any]:
        """Return a manifest-friendly description for this stressor."""

        severity_value = validate_severity(severity)
        return {
            "stress_type": self.stress_type,
            "family": self.family,
            "severity": severity_value,
            "params": self.params_for_severity(severity_value),
        }

    @property
    def last_report(self) -> dict[str, Any]:
        """Return details from the most recent apply call."""

        return copy.deepcopy(self._last_report)

    def _set_report(self, report: Mapping[str, Any]) -> None:
        self._last_report = dict(report)


class ImageStressor(Stressor):
    """Abstract base for RGB image stressors.

    Implementations must accept and return ``HxWx3`` ``uint8`` RGB arrays and
    must not mutate the input array. All stochastic behavior is driven only by
    the supplied ``numpy.random.Generator``.
    """

    family = "image"

    @abstractmethod
    def apply_image(
        self,
        image: np.ndarray,
        severity: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply stress to one RGB image."""


class SequenceStressor(Stressor):
    """Abstract base for temporal frame-sequence stressors.

    Implementations operate on ``FrameRef`` lists without reading pixel data.
    They may drop, duplicate, reorder, or blank references while preserving
    deterministic behavior from the supplied ``numpy.random.Generator``.
    """

    family = "sequence"

    @abstractmethod
    def apply_sequence(
        self,
        frames: list[FrameRef],
        severity: int,
        rng: np.random.Generator,
    ) -> list[FrameRef]:
        """Apply stress to a frame-reference sequence."""


StressorFactory = type[Stressor]
_STRESSOR_REGISTRY: dict[str, StressorFactory] = {}
_BUILTINS_LOADED = False


def register_stressor(stress_type: str):
    """Register a stressor class under a config ``stress_type`` string."""

    def decorator(stressor_cls: StressorFactory) -> StressorFactory:
        if not issubclass(stressor_cls, Stressor):
            raise TypeError(f"{stressor_cls.__name__} must inherit Stressor")
        stressor_cls.stress_type = stress_type
        _STRESSOR_REGISTRY[stress_type] = stressor_cls
        return stressor_cls

    return decorator


def available_stressor_types() -> list[str]:
    """Return registered stressor type names."""

    _load_builtin_stressors()
    return sorted(_STRESSOR_REGISTRY)


def build_stressor(
    stress_type: str | Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> Stressor:
    """Build a stressor from a config block or stress type string.

    Config blocks use ``stress_type`` to mirror ``configs/stress/*.yaml``. A
    legacy ``type`` key is also accepted for consistency with metric configs.
    Unknown config keys are passed into stressor constructors and ignored by
    built-in stressors unless they are explicitly supported options.
    """

    _load_builtin_stressors()

    if isinstance(stress_type, Mapping):
        if config is not None:
            raise ValueError("stressor config must be passed as either a mapping or a type string")
        config_block: Mapping[str, Any] = stress_type
        stress_type_value = config_block.get("stress_type", config_block.get("type"))
    else:
        config_block = config or {}
        stress_type_value = stress_type

    if not isinstance(config_block, Mapping):
        raise ValueError("stressor config must be a mapping")
    if not isinstance(stress_type_value, str) or not stress_type_value:
        raise ValueError("stressor config must include a stress_type")

    stressor_cls = _STRESSOR_REGISTRY.get(stress_type_value)
    if stressor_cls is None:
        available = ", ".join(available_stressor_types()) or "<none>"
        raise ValueError(
            f"unknown stressor type {stress_type_value!r}; "
            f"available types: {available}"
        )

    kwargs = {
        key: value
        for key, value in config_block.items()
        if key not in {"stress_type", "type", "severity", "name"}
    }
    return stressor_cls(**kwargs)


def _load_builtin_stressors() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    # Imports register built-in stressors through decorators.
    import sd2.stressors.temporal  # noqa: F401
    import sd2.stressors.visual  # noqa: F401
