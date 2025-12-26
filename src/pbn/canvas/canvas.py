from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from sklearn.cluster import KMeans

from pbn.canvas.facet import Facet
from pbn.canvas.color_palette import ColorPalette
from pbn.config.pbn_config import CANVAS_SIZE_CONFIG

@dataclass(frozen=True)
class Canvas:
    input_image: Image
    canvas_orientation: str
    canvas_page_size: str
    n_colors: int

    prepared_image: np.ndarray
    clustered_image: np.ndarray
    processed_image: np.ndarray
    outlined_image: np.ndarray

    color_palette: Palette

    @classmethod
    def create_canvas(
        cls,
        input_image: Image,
        canvas_orientation: str,
        canvas_page_size: str,
        n_colors: int
    ) -> Canvas:
        prepared_image = cls._prepare_image(
            image=input_image,
            canvas_orientation=canvas_orientation,
            canvas_page_size=canvas_page_size
        )
        clustered_image, color_palette = cls._cluster_image(image=prepared_image, n_colors=n_colors)
        processed_image = cls._process_image(image=clustered_image)
        outlined_image = cls._outline_image(image=processed_image)

        return cls(
            input_image=input_image,
            canvas_orientation=canvas_orientation,
            canvas_page_size=canvas_page_size,
            n_colors=n_colors,
            prepared_image=prepared_image,
            clustered_image=clustered_image,
            processed_image=processed_image,
            outlined_image=outlined_image,
            color_palette=color_palette
        )


    @staticmethod
    def _prepare_image(
        image: Image, 
        canvas_orientation: str, 
        canvas_page_size: str
    ) -> np.ndarray:
        width = CANVAS_SIZE_CONFIG[canvas_page_size][canvas_orientation]["WIDTH"]
        height = CANVAS_SIZE_CONFIG[canvas_page_size][canvas_orientation]["HEIGHT"]

        return np.array(
                    image.resize((width, height), resample=Image.LANCZOS), 
                    dtype=np.uint8
                )

    @staticmethod
    def _cluster_image(image: np.ndarray, n_colors: int) -> tuple[np.ndarray, Palette]:
        height, width = image.shape[:2]
        
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(image.reshape(-1, 3))

        clustered_image = kmeans.labels_.reshape(height, width).astype(np.int8)
        clustered_rgb_image = kmeans.cluster_centers_[clustered_image].astype(np.uint8)

        color_palette = ColorPalette.create_color_palette(
            image=clustered_rgb_image,
            n_colors=n_colors
        )

        return clustered_image, color_palette

    @staticmethod
    def _process_image(image: np.ndarray) -> np.ndarray:
        return image

    @staticmethod
    def _outline_image(image: np.ndarray) -> np.ndarray:
        return image