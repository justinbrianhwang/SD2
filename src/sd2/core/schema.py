"""Pydantic v2 schema for SD2 run metadata and frame logs."""

from __future__ import annotations

from typing import Any, Mapping, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from sd2.core.stage import Stage


class StateBase(BaseModel):
    """Base class for stage state models.

    Stage payloads allow extra fields because adapters at different
    observability tiers may expose model-specific state.
    """

    model_config = ConfigDict(extra="allow")


class VisionState(StateBase):
    """Visual perception or visual input state for a frame."""

    image_path: str | None = None
    embedding: list[float] | None = None
    feature: list[float] | None = None
    brightness: float | None = None
    noise_level: float | None = None


class SemanticState(StateBase):
    """Scene-understanding state for a frame."""

    objects: list[str] | None = None
    critical_object: str | None = None
    lane_state: str | None = None
    traffic_light_state: str | None = None
    semantic_description: str | None = None


class ReasoningState(StateBase):
    """Reasoning and decision state for a frame."""

    text: str | None = None
    decision_text: str | None = None
    intent: str | None = None
    explanation: str | None = None
    critical_object_mentioned: bool | None = None


class PlanningState(StateBase):
    """Planning state for a frame."""

    waypoints: list[list[float]] | None = None
    trajectory: list[list[float]] | None = None
    target_speed: float | None = None
    selected_maneuver: str | None = None


class ControlState(StateBase):
    """Vehicle control command state for a frame."""

    steer: float | None = None
    throttle: float | None = None
    brake: float | None = None


class OutcomeState(StateBase):
    """Driving outcome state for a frame."""

    collision: bool | None = None
    lane_invasion: bool | None = None
    route_progress: float | None = None
    driving_score: float | None = None
    distance_to_goal: float | None = None
    min_ttc: float | None = None


StageState: TypeAlias = (
    VisionState
    | SemanticState
    | ReasoningState
    | PlanningState
    | ControlState
    | OutcomeState
)

STATE_MODEL_BY_STAGE: dict[Stage, type[StateBase]] = {
    Stage.VISION: VisionState,
    Stage.SEMANTIC: SemanticState,
    Stage.REASONING: ReasoningState,
    Stage.PLANNING: PlanningState,
    Stage.CONTROL: ControlState,
    Stage.OUTCOME: OutcomeState,
}


class RunMetadata(BaseModel):
    """Run-level metadata shared by every frame in a run log."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    model_id: str
    scenario_id: str
    condition: str
    stress_type: str | None = None
    severity: int = Field(ge=0)
    seed: int
    timestamp_start: str | None = None


class FrameLog(BaseModel):
    """Frame-level log with any observable subset of stage states."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    frame_idx: int = Field(ge=0)
    timestamp: float
    states: dict[Stage, StageState] = Field(default_factory=dict)

    @field_validator("states", mode="before")
    @classmethod
    def _coerce_states(cls, value: Any) -> dict[Stage, StageState]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("states must be a mapping of stage name to state object")

        coerced: dict[Stage, StageState] = {}
        for raw_stage, raw_state in value.items():
            try:
                stage = raw_stage if isinstance(raw_stage, Stage) else Stage(str(raw_stage))
            except ValueError as exc:
                allowed = ", ".join(Stage.values())
                raise ValueError(
                    f"unknown stage {raw_stage!r}; expected one of: {allowed}"
                ) from exc

            model_cls = STATE_MODEL_BY_STAGE[stage]
            if isinstance(raw_state, model_cls):
                state = raw_state
            elif isinstance(raw_state, StateBase):
                state = model_cls.model_validate(raw_state.model_dump())
            else:
                state = model_cls.model_validate(raw_state)
            coerced[stage] = state
        return coerced

    @field_serializer("states")
    def _serialize_states(
        self, states: dict[Stage, StageState]
    ) -> dict[str, dict[str, Any]]:
        return {
            stage.value: state.model_dump(mode="json", exclude_none=True)
            for stage, state in states.items()
        }


class PairedFrameLog(BaseModel):
    """Clean/stress frame pair aligned by model, scenario, seed, and frame index."""

    model_config = ConfigDict(extra="forbid")

    pair_key: str
    frame_idx: int = Field(ge=0)
    timestamp: float
    clean: FrameLog
    stress: FrameLog
