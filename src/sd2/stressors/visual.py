"""Visual input stressors for RGB image arrays.

Severity mappings:

- ``gaussian_noise``: standard deviation ``[5, 10, 20, 35, 55]``.
- ``motion_blur``: directional line-kernel length ``[3, 5, 7, 9, 13]``.
- ``brightness_shift``: additive/multiplicative magnitude
  ``[0.08, 0.16, 0.24, 0.32, 0.42]``.
- ``contrast_shift``: factor around RGB midpoint 128
  ``[1.10, 1.25, 1.45, 1.70, 2.00]``.
- ``jpeg_compression``: JPEG quality ``[70, 50, 35, 20, 10]``.
- ``low_light``: multiplicative darkness factor
  ``[0.85, 0.70, 0.55, 0.40, 0.25]`` plus slight Gaussian noise.
- ``fog``: white veil blend density ``[0.08, 0.16, 0.28, 0.42, 0.60]``.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from sd2.stressors.base import ImageStressor, register_stressor, validate_severity


def _validate_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8")
    return image


def _clip_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _float_image(image: np.ndarray) -> np.ndarray:
    return _validate_rgb_uint8(image).astype(np.float32, copy=True)


@register_stressor("gaussian_noise")
class GaussianNoiseStressor(ImageStressor):
    """Additive zero-mean Gaussian noise with severity std 5, 10, 20, 35, 55."""

    STD_BY_SEVERITY = {
        1: 5.0,
        2: 10.0,
        3: 20.0,
        4: 35.0,
        5: 55.0,
    }

    def params_for_severity(self, severity: int) -> dict[str, Any]:
        severity_value = validate_severity(severity)
        return {"std": self.STD_BY_SEVERITY[severity_value]}

    def apply_image(
        self,
        image: np.ndarray,
        severity: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        params = self.params_for_severity(severity)
        base = _float_image(image)
        noise = rng.normal(0.0, params["std"], size=base.shape)
        output = _clip_uint8(base + noise)
        self._set_report({"params": params})
        return output


@register_stressor("motion_blur")
class MotionBlurStressor(ImageStressor):
    """Directional line blur with severity kernel length 3, 5, 7, 9, 13."""

    LENGTH_BY_SEVERITY = {
        1: 3,
        2: 5,
        3: 7,
        4: 9,
        5: 13,
    }

    def __init__(self, direction: str = "horizontal", **options: Any) -> None:
        super().__init__(direction=direction, **options)
        self.direction = direction

    def params_for_severity(self, severity: int) -> dict[str, Any]:
        severity_value = validate_severity(severity)
        return {
            "kernel_length": self.LENGTH_BY_SEVERITY[severity_value],
            "direction": self.direction,
        }

    def apply_image(
        self,
        image: np.ndarray,
        severity: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del rng
        params = self.params_for_severity(severity)
        base = _float_image(image)
        kernel = _line_kernel(params["kernel_length"], params["direction"])
        output = _convolve_rgb(base, kernel)
        self._set_report({"params": params})
        return _clip_uint8(output)


@register_stressor("brightness_shift")
class BrightnessShiftStressor(ImageStressor):
    """Brightness shift with severity magnitude 0.08, 0.16, 0.24, 0.32, 0.42."""

    MAGNITUDE_BY_SEVERITY = {
        1: 0.08,
        2: 0.16,
        3: 0.24,
        4: 0.32,
        5: 0.42,
    }

    def __init__(self, direction: str = "brighten", **options: Any) -> None:
        super().__init__(direction=direction, **options)
        self.direction = direction

    def params_for_severity(self, severity: int) -> dict[str, Any]:
        severity_value = validate_severity(severity)
        magnitude = self.MAGNITUDE_BY_SEVERITY[severity_value]
        return {
            "magnitude": magnitude,
            "direction": self.direction,
            "factor_delta": round(magnitude * 0.5, 4),
            "offset": round(magnitude * 80.0, 4),
        }

    def apply_image(
        self,
        image: np.ndarray,
        severity: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        params = self.params_for_severity(severity)
        sign = _brightness_sign(params["direction"], rng)
        factor = 1.0 + sign * params["factor_delta"]
        offset = sign * params["offset"]
        output = _clip_uint8(_float_image(image) * factor + offset)
        report_params = dict(params)
        report_params.update({"factor": factor, "signed_offset": offset})
        self._set_report({"params": report_params})
        return output


@register_stressor("contrast_shift")
class ContrastShiftStressor(ImageStressor):
    """Contrast increase with severity factor 1.10, 1.25, 1.45, 1.70, 2.00."""

    FACTOR_BY_SEVERITY = {
        1: 1.10,
        2: 1.25,
        3: 1.45,
        4: 1.70,
        5: 2.00,
    }

    def params_for_severity(self, severity: int) -> dict[str, Any]:
        severity_value = validate_severity(severity)
        return {"factor": self.FACTOR_BY_SEVERITY[severity_value], "midpoint": 128.0}

    def apply_image(
        self,
        image: np.ndarray,
        severity: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del rng
        params = self.params_for_severity(severity)
        base = _float_image(image)
        output = 128.0 + (base - 128.0) * params["factor"]
        self._set_report({"params": params})
        return _clip_uint8(output)


@register_stressor("jpeg_compression")
class JpegCompressionStressor(ImageStressor):
    """JPEG round-trip with severity quality 70, 50, 35, 20, 10."""

    QUALITY_BY_SEVERITY = {
        1: 70,
        2: 50,
        3: 35,
        4: 20,
        5: 10,
    }

    def params_for_severity(self, severity: int) -> dict[str, Any]:
        severity_value = validate_severity(severity)
        return {"quality": self.QUALITY_BY_SEVERITY[severity_value]}

    def apply_image(
        self,
        image: np.ndarray,
        severity: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del rng
        params = self.params_for_severity(severity)
        _validate_rgb_uint8(image)
        buffer = BytesIO()
        Image.fromarray(image).save(
            buffer,
            format="JPEG",
            quality=params["quality"],
            optimize=False,
            progressive=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            output = np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()
        self._set_report({"params": params})
        return output


@register_stressor("low_light")
class LowLightStressor(ImageStressor):
    """Darkening plus slight noise with severity factors 0.85, 0.70, 0.55, 0.40, 0.25."""

    FACTOR_BY_SEVERITY = {
        1: 0.85,
        2: 0.70,
        3: 0.55,
        4: 0.40,
        5: 0.25,
    }
    NOISE_STD_BY_SEVERITY = {
        1: 1.0,
        2: 1.5,
        3: 2.0,
        4: 3.0,
        5: 4.0,
    }

    def params_for_severity(self, severity: int) -> dict[str, Any]:
        severity_value = validate_severity(severity)
        return {
            "factor": self.FACTOR_BY_SEVERITY[severity_value],
            "noise_std": self.NOISE_STD_BY_SEVERITY[severity_value],
        }

    def apply_image(
        self,
        image: np.ndarray,
        severity: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        params = self.params_for_severity(severity)
        base = _float_image(image) * params["factor"]
        noise = rng.normal(0.0, params["noise_std"], size=base.shape)
        output = _clip_uint8(base + noise)
        self._set_report({"params": params})
        return output


@register_stressor("fog")
class FogStressor(ImageStressor):
    """Blend the image toward a white haze with severity density 0.08..0.60."""

    DENSITY_BY_SEVERITY = {
        1: 0.08,
        2: 0.16,
        3: 0.28,
        4: 0.42,
        5: 0.60,
    }

    def params_for_severity(self, severity: int) -> dict[str, Any]:
        severity_value = validate_severity(severity)
        return {"density": self.DENSITY_BY_SEVERITY[severity_value]}

    def apply_image(
        self,
        image: np.ndarray,
        severity: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del rng
        params = self.params_for_severity(severity)
        base = _float_image(image)
        haze = np.full_like(base, 255.0)
        output = base * (1.0 - params["density"]) + haze * params["density"]
        self._set_report({"params": params})
        return _clip_uint8(output)


def _brightness_sign(direction: str, rng: np.random.Generator) -> int:
    if direction == "brighten":
        return 1
    if direction == "darken":
        return -1
    if direction == "random":
        return 1 if rng.random() >= 0.5 else -1
    raise ValueError("brightness direction must be 'brighten', 'darken', or 'random'")


def _line_kernel(length: int, direction: str) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    if direction == "horizontal":
        kernel[center, :] = 1.0
    elif direction == "vertical":
        kernel[:, center] = 1.0
    elif direction == "diagonal":
        np.fill_diagonal(kernel, 1.0)
    elif direction == "anti_diagonal":
        np.fill_diagonal(np.fliplr(kernel), 1.0)
    else:
        raise ValueError(
            "motion blur direction must be 'horizontal', 'vertical', "
            "'diagonal', or 'anti_diagonal'"
        )
    return kernel / float(length)


def _convolve_rgb(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    height, width, _ = image.shape
    pad = kernel.shape[0] // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)

    for y_offset in range(kernel.shape[0]):
        for x_offset in range(kernel.shape[1]):
            weight = float(kernel[y_offset, x_offset])
            if weight == 0.0:
                continue
            output += weight * padded[
                y_offset : y_offset + height,
                x_offset : x_offset + width,
                :,
            ]
    return output
