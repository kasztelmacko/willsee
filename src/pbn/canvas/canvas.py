from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from PIL import Image

from pbn.canvas.canvas_utils.cluster_image import cluster_image
from pbn.canvas.canvas_utils.prepare_image import prepare_image
from pbn.canvas.canvas_utils.process_image import process_image
from pbn.canvas.canvas_utils.outline_image import (
    create_outline_mask,
    create_image_outline,
    create_image_with_color_labels,
)
import pbn.config.pbn_config as PBN_CONF

@dataclass(frozen=True)
class Canvas:
    """
    Pipeline container for preparing, clustering, processing, outlining, and labeling
    an input image for paint-by-number generation.
    """
    input_image: Image
    canvas_orientation: str
    canvas_page_size: str
    n_colors: int

    prepared_image: np.ndarray
    clustered_image: np.ndarray
    processed_image: np.ndarray
    outlined_image: np.ndarray

    @classmethod
    def create_canvas(
        cls,
        input_image: Image,
        canvas_orientation: str,
        canvas_page_size: str,
        n_colors: int
    ) -> Canvas:
        """
        End-to-end constructor that prepares, clusters, processes, outlines, and labels
        the input image according to the configured canvas parameters.
        """
        prepared_image = cls._prepare_image(
            image=input_image,
            canvas_orientation=canvas_orientation,
            canvas_page_size=canvas_page_size,
            canvas_size_config=PBN_CONF.CANVAS_SIZE_CONFIG
        )
        clustered_image = cls._cluster_image(image=prepared_image, n_colors=n_colors)
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
        )


    @staticmethod
    def _prepare_image(
        image: Image, 
        canvas_orientation: str, 
        canvas_page_size: str,
        canvas_size_config: dict
    ) -> np.ndarray:
        """
        Resize and orient the input PIL image to the target canvas settings.
        """
        return prepare_image(
            image=image,
            canvas_orientation=canvas_orientation,
            canvas_page_size=canvas_page_size,
            canvas_size_config=canvas_size_config
        )

    @staticmethod
    def _cluster_image(image: np.ndarray, n_colors: int) -> np.ndarray:
        """
        Cluster image by color and return RGB image with clustered colors.
        """
        return cluster_image(image=image, n_colors=n_colors)

    @staticmethod
    def _process_image(image: np.ndarray) -> np.ndarray:
        """
        Process clustered image by:
        1. Finding small facets and merging them
        2. Finding narrow facets and merging them
        3. Reindex facets to start at 0
        """
        return process_image(
            image=image,
            min_facet_size=PBN_CONF.MIN_FACET_PIXELS_SIZE,
            narrow_thresh_px=PBN_CONF.NARROW_FACET_THRESHOLD_PX,
        )

    @staticmethod
    def _outline_image(image: np.ndarray) -> np.ndarray:
        """
        Produce an outlined version of the processed image and render palette labels
        at facet centers onto the outline.
        """
        outline_mask = create_outline_mask(image=image)
        outline_image = create_image_outline(
            image=image,
            outline_mask=outline_mask,
            outline_color=PBN_CONF.FACET_OUTLINE_COLOR,
        )
        return create_image_with_color_labels(
            image=image,
            outline_image=outline_image,
            min_font_px=PBN_CONF.MIN_FONT_PX,
            max_font_px=PBN_CONF.MAX_FONT_PX,
            font_scale=PBN_CONF.FONT_SCALE,
            text_color=PBN_CONF.FACET_LABEL_COLOR,
        )