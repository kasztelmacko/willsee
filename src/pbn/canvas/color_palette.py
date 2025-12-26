from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class ColorPalette:
    image: np.ndarray
    n_colors: int

    color_palette_dict: dict[int, tuple[int, int, int]]

    @classmethod
    def create_color_palette(
        cls,
        image: np.ndarray,
        n_colors: int
    ) -> ColorPalette:
        color_palette_dict = cls.extract_color_palette(image=image, n_colors=n_colors)

        return cls(
            image=image,
            n_colors=n_colors,
            color_palette_dict=color_palette_dict
        )

    @staticmethod
    def extract_color_palette(image: np.ndarray, n_colors: int) -> dict[int, tuple[int, int, int]]:
        pixels = image.reshape(-1, 3)
        unique_pixels = np.unique(pixels, axis=0)

        color_palette_dict = {
            color_id: (int(r), int(g), int(b))
            for color_id, (r, g, b) in enumerate(unique_pixels)
        }

        return color_palette_dict

