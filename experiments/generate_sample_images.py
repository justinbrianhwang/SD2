"""Generate deterministic synthetic RGB sample images for stressor demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE_DIR = REPO_ROOT / "data" / "sample" / "images"
FRAME_COUNT = 6
IMAGE_SIZE = 64
SEED = 42


def main() -> None:
    SAMPLE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for frame_idx in range(FRAME_COUNT):
        image = _make_frame(frame_idx, rng)
        image.save(SAMPLE_IMAGE_DIR / f"frame_{frame_idx:03d}.png")


def _make_frame(frame_idx: int, rng: np.random.Generator) -> Image.Image:
    y_grid, x_grid = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    red = (x_grid * 3 + frame_idx * 18) % 256
    green = (y_grid * 3 + frame_idx * 11) % 256
    blue = ((x_grid + y_grid) * 2 + 36 + frame_idx * 9) % 256
    array = np.stack([red, green, blue], axis=2).astype(np.uint8)

    image = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(image, mode="RGBA")

    horizon = 28 + (frame_idx % 2)
    road = [(0, IMAGE_SIZE), (IMAGE_SIZE, IMAGE_SIZE), (42, horizon), (22, horizon)]
    draw.polygon(road, fill=(58, 58, 64, 190))

    lane_offset = frame_idx * 2
    draw.line(
        [(31 + lane_offset // 3, IMAGE_SIZE), (30 + lane_offset // 5, horizon + 2)],
        fill=(245, 235, 120, 220),
        width=2,
    )
    draw.line(
        [(48 - lane_offset // 4, IMAGE_SIZE), (38 - lane_offset // 6, horizon + 2)],
        fill=(235, 235, 235, 220),
        width=2,
    )

    car_x = 9 + frame_idx * 6
    car_y = 38 + (frame_idx % 3)
    car_color = tuple(int(value) for value in rng.integers(80, 225, size=3))
    draw.rounded_rectangle(
        [car_x, car_y, car_x + 14, car_y + 8],
        radius=2,
        fill=(*car_color, 235),
        outline=(24, 24, 30, 255),
        width=1,
    )
    draw.rectangle([car_x + 3, car_y - 3, car_x + 10, car_y + 1], fill=(40, 80, 120, 210))
    draw.ellipse([car_x + 2, car_y + 7, car_x + 5, car_y + 10], fill=(15, 15, 18, 255))
    draw.ellipse([car_x + 10, car_y + 7, car_x + 13, car_y + 10], fill=(15, 15, 18, 255))

    signal_x = 50
    signal_y = 12
    draw.rectangle([signal_x, signal_y, signal_x + 7, signal_y + 18], fill=(30, 34, 38, 230))
    draw.ellipse([signal_x + 2, signal_y + 3, signal_x + 5, signal_y + 6], fill=(40, 170, 80, 255))
    draw.line([(signal_x + 3, signal_y + 18), (signal_x + 3, signal_y + 28)], fill=(28, 28, 28, 255))

    pedestrian_x = 18 + frame_idx * 3
    pedestrian_y = 25
    draw.ellipse(
        [pedestrian_x, pedestrian_y, pedestrian_x + 4, pedestrian_y + 4],
        fill=(230, 190, 150, 255),
    )
    draw.line(
        [(pedestrian_x + 2, pedestrian_y + 4), (pedestrian_x + 2, pedestrian_y + 12)],
        fill=(20, 80, 180, 255),
        width=2,
    )
    draw.line(
        [(pedestrian_x + 2, pedestrian_y + 7), (pedestrian_x - 2, pedestrian_y + 10)],
        fill=(20, 80, 180, 255),
        width=1,
    )
    draw.line(
        [(pedestrian_x + 2, pedestrian_y + 7), (pedestrian_x + 6, pedestrian_y + 10)],
        fill=(20, 80, 180, 255),
        width=1,
    )

    return image


if __name__ == "__main__":
    main()
