from __future__ import annotations
import numpy as np

from dataclasses import dataclass

from pbn.config.pbn_config import FACET_SIZE_FONT_SIZE_CONFIG
from scipy.ndimage import distance_transform_edt

@dataclass(frozen=True)
class Facet:
    facet_id: int
    facet_color_label: int
    facet_size_px: int
    facet_avg_width_px: float
    facet_avg_height_px: float
    facet_bounding_box: tuple[int, int, int, int] | None = None
    facet_label_point: tuple[int, int] | None = None
    facet_label_point_font_size: int | None = None

    @classmethod
    def create_facet(
        cls,
        facet_id: int,
        facet_color_label: int,
        facet_size_px: int,
        facet_avg_width_px: float,
        facet_avg_height_px: float
    ) -> Facet:
        return cls(
            facet_id=facet_id,
            facet_color_label=facet_color_label,
            facet_size_px=facet_size_px,
            facet_avg_width_px=facet_avg_width_px,
            facet_avg_height_px=facet_avg_height_px
        )

    @staticmethod
    def add_facet_features(
        facet: Facet,
        facet_mask: np.ndarray,
    ) -> Facet:
        """Add additional features (bounding box, label point, etc.) to a facet."""
        facet_size_px = Facet._get_facet_size(facet_mask=facet_mask)
        facet_bounding_box = Facet._get_facet_bounding_box(facet_mask=facet_mask)
        facet_label_point = Facet._get_facet_label_point(facet_mask=facet_mask)
        facet_label_point_font_size = Facet._get_facet_label_point_font_size(
            size_px=facet.facet_size_px
        )
        facet_avg_width_px, facet_avg_height_px = Facet._get_facet_avg_height_and_width(
            facet_mask=facet_mask, size_px=facet.facet_size_px
        )

        return Facet(
            facet_id=facet.facet_id,
            facet_color_label=facet.facet_color_label,
            facet_size_px=facet.facet_size_px,
            facet_bounding_box=facet_bounding_box,
            facet_label_point=facet_label_point,
            facet_label_point_font_size=facet_label_point_font_size,
            facet_avg_width_px=facet_avg_width_px,
            facet_avg_height_px=facet_avg_height_px,
        )

    @staticmethod
    def _get_facet_size(facet_mask: np.ndarray) -> int:
        """Return the number of pixels in a facet."""
        return int(facet_mask.sum())

    @staticmethod
    def _get_facet_bounding_box(facet_mask: np.ndarray) -> tuple[int, int, int, int]:
        """
        Compute bounding box as (x_min, y_min, x_max, y_max). 
        Returns the smallest rectangle that contains all True pixels in the mask.
        """
        facet_pixel_ys, facet_pixel_xs = np.nonzero(facet_mask)
        if facet_pixel_ys.size == 0:
            return (0, 0, 0, 0)
        
        facet_x_min, facet_x_max = int(facet_pixel_xs.min()), int(facet_pixel_xs.max())
        facet_y_min, facet_y_max = int(facet_pixel_ys.min()), int(facet_pixel_ys.max())

        return (facet_x_min, facet_y_min, facet_x_max, facet_y_max)

    @staticmethod
    def _get_facet_label_point(facet_mask: np.ndarray) -> tuple[int, int]:
        """
        Compute the center of the maximal inscribed circle (safe label placement).
        Falls back to centroid if the facet is too thin.
        """
        if not np.any(facet_mask):
            return (0, 0)
        
        facet_height, facet_width = facet_mask.shape
        distance = distance_transform_edt(facet_mask)
        max_indexes = int(np.argmax(distance))
        max_distance = distance.flat[max_indexes]
        label_point_y, label_point_x = divmod(max_indexes, facet_width)

        if max_distance <= 0:
            facet_pixel_ys, facet_pixel_xs = np.nonzero(facet_mask)
            label_point_y = int(np.round(facet_pixel_ys.mean()))
            label_point_x = int(np.round(facet_pixel_xs.mean()))

        return (label_point_y, label_point_x)

    @staticmethod
    def _get_facet_label_point_font_size(size_px: int) -> int:
        """Determine font size based on facet size using FACET_SIZE_FONT_SIZE_CONFIG."""
        for font_size, (min_facet_size, max_facet_size) in FACET_SIZE_FONT_SIZE_CONFIG.items():
            if size_px < min_facet_size:
                continue

            if max_facet_size is None:
                return font_size
            
            elif size_px <= max_facet_size:
                return font_size

    @staticmethod
    def _get_facet_avg_height_and_width(facet_mask: np.ndarray, size_px: int) -> tuple[float, float]:
        """
        Compute average width and height of the facet.
        
        Average width = total_pixels / number_of_rows_with_pixels
        Average height = total_pixels / number_of_columns_with_pixels
        """
        if size_px == 0:
            return (0.0, 0.0)
        
        facet_rows_count = np.count_nonzero(np.any(facet_mask, axis=1))
        facet_cols_count = np.count_nonzero(np.any(facet_mask, axis=0))

        avg_height_px = float(size_px / max(facet_cols_count, 1))
        avg_width_px = float(size_px / max(facet_rows_count, 1))

        return (avg_width_px, avg_height_px)