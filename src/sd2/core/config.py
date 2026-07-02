"""YAML configuration loading for SD2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sd2.core.stage import Stage


class SD2Config(BaseModel):
    """Small MVP config object.

    Unknown keys are allowed so later phases can extend the YAML without
    breaking Phase 1/2 loading.
    """

    model_config = ConfigDict(extra="allow")

    project_name: str = "SD2 MVP"
    stages: list[Stage] = Field(default_factory=Stage.ordered)
    pairing: dict[str, Any] = Field(default_factory=dict)
    normalization: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    diagnosis: dict[str, Any] = Field(default_factory=dict)


def load_config(path: str | Path) -> SD2Config:
    """Load an SD2 YAML config file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a YAML mapping: {config_path}")
    return SD2Config.model_validate(data)
