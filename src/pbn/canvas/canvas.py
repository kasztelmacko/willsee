from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from PIL import Image

from pbn.canvas.canvas_utils.cluster_image import cluster_image
from pbn.canvas.canvas_utils.prepare_image import prepare_image
from pbn.canvas.canvas_utils.process_image import (
    process_image,
    recolor_image_with_palette,
)
from pbn.canvas.canvas_utils.outline_image import (
    create_outline_mask,
    create_image_outline,
    create_image_with_color_labels,
)
import pbn.config.pbn_config as PBN_CONF
from pbn.canvas.color_palette import ColorPalette

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
    canvas_size_config: dict
    min_facet_size: int
    narrow_thresh_px: int
    min_font_px: int
    max_font_px: int
    font_scale: float
    facet_label_color: tuple[int, int, int]
    facet_outline_color: tuple[int, int, int]

    prepared_image: np.ndarray
    clustered_image: np.ndarray
    processed_image: np.ndarray
    outlined_image: np.ndarray

    color_palette: ColorPalette

    @classmethod
    def create_canvas(
        cls,
        input_image: Image,
        canvas_orientation: str,
        canvas_page_size: str,
        n_colors: int,
        canvas_size_config: dict = PBN_CONF.CANVAS_SIZE_CONFIG,
        min_facet_size: int = PBN_CONF.MIN_FACET_PIXELS_SIZE,
        narrow_thresh_px: int = PBN_CONF.NARROW_FACET_THRESHOLD_PX,
        min_font_px: int = PBN_CONF.MIN_FONT_PX,
        max_font_px: int = PBN_CONF.MAX_FONT_PX,
        font_scale: float = PBN_CONF.FONT_SCALE,
        facet_label_color: tuple[int, int, int] = PBN_CONF.FACET_LABEL_COLOR,
        facet_outline_color: tuple[int, int, int] = PBN_CONF.FACET_OUTLINE_COLOR,
    ) -> Canvas:
        """
        End-to-end constructor that prepares, clusters, processes, outlines, and labels
        the input image according to the configured canvas parameters.
        """
        canvas_size_config = canvas_size_config
        min_facet_size = int(min_facet_size)
        narrow_thresh_px = int(narrow_thresh_px)
        min_font_px = int(min_font_px)
        max_font_px = int(max_font_px)
        font_scale = float(font_scale)
        facet_label_color = tuple[int, ...](int(c) for c in facet_label_color)
        facet_outline_color = tuple[int, ...](int(c) for c in facet_outline_color)

        prepared_image = cls._prepare_image(
            image=input_image,
            canvas_orientation=canvas_orientation,
            canvas_page_size=canvas_page_size,
            canvas_size_config=canvas_size_config,
        )
        clustered_image = cls._cluster_image(image=prepared_image, n_colors=n_colors)
        processed_image = cls._process_image(
            image=clustered_image,
            min_facet_size=min_facet_size,
            narrow_thresh_px=narrow_thresh_px,
        )
        outlined_image, color_palette = cls._outline_image(
            image=processed_image,
            palette=None,
            min_font_px=min_font_px,
            max_font_px=max_font_px,
            font_scale=font_scale,
            text_color=facet_label_color,
            outline_color=facet_outline_color,
        )

        return cls(
            input_image=input_image,
            canvas_orientation=canvas_orientation,
            canvas_page_size=canvas_page_size,
            n_colors=n_colors,
            canvas_size_config=canvas_size_config,
            min_facet_size=min_facet_size,
            narrow_thresh_px=narrow_thresh_px,
            min_font_px=min_font_px,
            max_font_px=max_font_px,
            font_scale=font_scale,
            facet_label_color=facet_label_color,
            facet_outline_color=facet_outline_color,
            prepared_image=prepared_image,
            clustered_image=clustered_image,
            processed_image=processed_image,
            outlined_image=outlined_image,
            color_palette=color_palette,
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
    def _process_image(
        image: np.ndarray,
        min_facet_size: int,
        narrow_thresh_px: int,
    ) -> np.ndarray:
        """
        Process clustered image by:
        1. Finding small facets and merging them
        2. Finding narrow facets and merging them
        3. Reindex facets to start at 0
        """
        return process_image(
            image=image,
            min_facet_size=min_facet_size,
            narrow_thresh_px=narrow_thresh_px,
        )

    @staticmethod
    def _outline_image(
        image: np.ndarray,
        palette: ColorPalette | None = None,
        min_font_px: int = PBN_CONF.MIN_FONT_PX,
        max_font_px: int = PBN_CONF.MAX_FONT_PX,
        font_scale: float = PBN_CONF.FONT_SCALE,
        text_color: tuple[int, int, int] = PBN_CONF.FACET_LABEL_COLOR,
        outline_color: tuple[int, int, int] = PBN_CONF.FACET_OUTLINE_COLOR,
    ) -> tuple[np.ndarray, ColorPalette]:
        """
        Produce an outlined version of the processed image and render palette labels
        at facet centers onto the outline.
        """
        outline_mask = create_outline_mask(image=image)
        outline_image = create_image_outline(
            image=image,
            outline_mask=outline_mask,
            outline_color=outline_color,
        )
        outline_image, color_palette = create_image_with_color_labels(
            image=image,
            outline_image=outline_image,
            min_font_px=min_font_px,
            max_font_px=max_font_px,
            font_scale=font_scale,
            text_color=text_color,
            palette=palette,
        )
        return outline_image, color_palette

    def render_image_with_replaced_palette(self, palette: ColorPalette) -> "Canvas":
        """
        Return a new Canvas with processed and outlined images re-rendered
        using the provided palette.
        """
        recolored_image = recolor_image_with_palette(
            image=self.processed_image,
            palette=palette,
        )
        outlined_image, updated_palette = self._outline_image(
            image=recolored_image,
            palette=palette,
            min_font_px=self.min_font_px,
            max_font_px=self.max_font_px,
            font_scale=self.font_scale,
            text_color=self.facet_label_color,
            outline_color=self.facet_outline_color,
        )
        return replace(
            self,
            processed_image=recolored_image,
            outlined_image=outlined_image,
            color_palette=updated_palette,
        )

    @property
    def color_pallete(self) -> ColorPalette:
        """Backward-compatible alias for `color_palette`."""
        return self.color_palette