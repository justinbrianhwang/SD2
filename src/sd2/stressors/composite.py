"""Ordered composition of visual image stressors."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from sd2.stressors.base import ImageStressor, validate_severity


class CompositeImageStressor(ImageStressor):
    """Apply multiple image stressors in order."""

    stress_type = "composite"

    def __init__(
        self,
        stressors: Iterable[tuple[ImageStressor, int]],
    ) -> None:
        super().__init__()
        self.stressors = list(stressors)
        if not self.stressors:
            raise ValueError("composite image stressor requires at least one stressor")
        for stressor, severity in self.stressors:
            if not isinstance(stressor, ImageStressor):
                raise TypeError("composite entries must contain ImageStressor instances")
            validate_severity(severity)

    def params_for_severity(self, severity: int) -> dict[str, Any]:
        severity_value = validate_severity(severity)
        return {
            "stressors": [
                stressor.describe(severity_value if index == 0 else stored_severity)
                for index, (stressor, stored_severity) in enumerate(self.stressors)
            ]
        }

    def describe(self, severity: int) -> dict[str, Any]:
        severity_value = validate_severity(severity)
        return {
            "stress_type": self.stress_type,
            "family": self.family,
            **self.params_for_severity(severity_value),
        }

    def apply_image(
        self,
        image: np.ndarray,
        severity: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        severity_value = validate_severity(severity)
        output = self._validate_image(image)
        for index, (stressor, stored_severity) in enumerate(self.stressors):
            output = stressor.apply_image(
                output,
                severity_value if index == 0 else stored_severity,
                rng,
            )
            output = self._validate_image(output)
        self._set_report(self.params_for_severity(severity_value))
        return output

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape HxWx3")
        if image.dtype != np.uint8:
            raise ValueError("image must have dtype uint8")
        return image
