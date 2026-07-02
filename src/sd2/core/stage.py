"""Pipeline stage identifiers and ordering helpers."""

from __future__ import annotations

from enum import StrEnum


class Stage(StrEnum):
    """Canonical SD2 functional pipeline stages."""

    VISION = "vision"
    SEMANTIC = "semantic"
    REASONING = "reasoning"
    PLANNING = "planning"
    CONTROL = "control"
    OUTCOME = "outcome"

    @classmethod
    def ordered(cls) -> list["Stage"]:
        """Return stages in upstream-to-downstream pipeline order."""

        return [
            cls.VISION,
            cls.SEMANTIC,
            cls.REASONING,
            cls.PLANNING,
            cls.CONTROL,
            cls.OUTCOME,
        ]

    @classmethod
    def values(cls) -> list[str]:
        """Return stage values in pipeline order."""

        return [stage.value for stage in cls.ordered()]

    def index(self) -> int:
        """Return this stage's pipeline index."""

        return self.ordered().index(self)

    def upstream(self) -> list["Stage"]:
        """Return stages before this stage."""

        return self.ordered()[: self.index()]

    def downstream(self) -> list["Stage"]:
        """Return stages after this stage."""

        return self.ordered()[self.index() + 1 :]
