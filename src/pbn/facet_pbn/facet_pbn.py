from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
import pbn.pbn_config.facet_pbn_config as FACET_CONF


@dataclass(frozen=True)
class FacetPBN:
    facet_id: int
    color_rgb: tuple[int, int, int]
    color_palette_id: int

    size_px: int
    mask: np.ndarray
    mask_bounding_box: tuple[int, int, int, int]
    label_point: tuple[int, int]
    label_point_font_size: int
    avg_width_px: float
    avg_height_px: float
    neighbor_facet_ids: list[int]

    @classmethod
    def from_mask(
        cls,
        facet_id: int,
        mask: np.ndarray,
        color_rgb: tuple[int, int, int],
        color_palette_id: int,
        labels_img: np.ndarray | None = None,
    ) -> FacetPBN:
        size_px = cls._get_facet_size(mask=mask)
        mask_bounding_box = cls._get_facet_mask_bounding_box(mask=mask)
        label_point = cls._get_facet_label_point(mask=mask)
        label_point_font_size = cls._get_facet_label_point_font_size(size_px=size_px)
        avg_height_px, avg_width_px = cls._get_facet_avg_height_and_width(mask=mask, size_px=size_px)
        neighbor_facet_ids = cls._get_facet_neighbor_ids(facet_id=facet_id, mask=mask, labels_img=labels_img)
        
        return cls(
            facet_id=facet_id,
            color_rgb=color_rgb,
            color_palette_id=color_palette_id,
            size_px=size_px,
            mask=mask,
            mask_bounding_box=mask_bounding_box,
            label_point=label_point,
            label_point_font_size=label_point_font_size,
            avg_width_px=avg_width_px,
            avg_height_px=avg_height_px,
            neighbor_facet_ids=neighbor_facet_ids,
        )

    @staticmethod
    def _get_facet_size(mask: np.ndarray) -> int:
        """Return the number of pixels in a facet."""
        return int(mask.sum())

    @staticmethod
    def _get_facet_mask_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
        """
        Compute bounding box as (x_min, y_min, x_max, y_max). 
        Returns the smallest rectangle that contains all True pixels in the mask.
        """
        facet_pixel_ys, facet_pixel_xs = np.nonzero(mask)
        if facet_pixel_ys.size == 0:
            return (0, 0, 0, 0)
        
        facet_x_min, facet_x_max = int(facet_pixel_xs.min()), int(facet_pixel_xs.max())
        facet_y_min, facet_y_max = int(facet_pixel_ys.min()), int(facet_pixel_ys.max())

        return (facet_x_min, facet_y_min, facet_x_max, facet_y_max)

    @staticmethod
    def _get_facet_label_point(mask: np.ndarray) -> tuple[int, int]:
        """
        Compute the center of the maximal inscribed circle (safe label placement).
        Falls back to centroid if the facet is too thin.
        """
        if not np.any(mask):
            return (0, 0)
        
        facet_height, facet_width = mask.shape
        distance = distance_transform_edt(mask)
        max_indexes = int(np.argmax(distance))
        max_distance = distance.flat[max_indexes]
        label_point_y, label_point_x = divmod(max_indexes, facet_width)

        if max_distance <= 0:
            facet_pixel_ys, facet_pixel_xs = np.nonzero(mask)
            label_point_y = int(np.round(facet_pixel_ys.mean()))
            label_point_x = int(np.round(facet_pixel_xs.mean()))

        return (label_point_y, label_point_x)

    @staticmethod
    def _get_facet_label_point_font_size(size_px: int) -> int:
        """Determine font size based on facet size using FACET_SIZE_FONT_SIZE_CONFIG."""
        for font_size, (min_facet_size, max_facet_size) in FACET_CONF.FACET_SIZE_FONT_SIZE_CONFIG.items():
            if size_px < min_facet_size:
                continue

            if max_facet_size is None:
                return font_size
            
            elif size_px <= max_facet_size:
                return font_size

    @staticmethod
    def _get_facet_avg_height_and_width(mask: np.ndarray, size_px: int) -> tuple[float, float]:
        """
        Compute average width and height of the facet.
        
        Average width = total_pixels / number_of_rows_with_pixels
        Average height = total_pixels / number_of_columns_with_pixels
        """
        if size_px == 0:
            return (0.0, 0.0)
        
        facet_rows_count = np.count_nonzero(np.any(mask, axis=1))
        facet_cols_count = np.count_nonzero(np.any(mask, axis=0))

        avg_height_px = float(size_px / max(facet_cols_count, 1))
        avg_width_px = float(size_px / max(facet_rows_count, 1))

        return (avg_height_px, avg_width_px)

    @staticmethod
    def _get_facet_neighbor_ids(
        facet_id: int,
        mask: np.ndarray,
        labels_img: np.ndarray | None,
    ) -> list[int]:
        """Find all facet IDs that are directly adjacent (4-connected) to this facet."""
        if labels_img is None or not np.any(mask):
            return []

        canvas_height, canvas_width = labels_img.shape
        all_neighbors: list[np.ndarray] = []

        if canvas_height > 1:
            all_neighbors.append(labels_img[1:, :][mask[:-1, :]])       # Down
            all_neighbors.append(labels_img[:-1, :][mask[1:, :]])       # Up
        
        if canvas_width > 1:
            all_neighbors.append(labels_img[:, 1:][mask[:, :-1]])       # Right
            all_neighbors.append(labels_img[:, :-1][mask[:, 1:]])       # Left

        if not all_neighbors:
            return []
        
        facet_neighbor_ids: set[int] = set()
        FacetPBN._add_valid_neighbor_ids(
            neighbor_array=np.concatenate(all_neighbors),
            facet_id=facet_id,
            neighbor_set=facet_neighbor_ids
        )
        return sorted(facet_neighbor_ids)

    @staticmethod
    def _add_valid_neighbor_ids(
        neighbor_array: np.ndarray,
        facet_id: int,
        neighbor_set: set[int],
    ) -> None:
        """Filter and add valid neighbor IDs to the set."""
        valid_neighbors = neighbor_array[(neighbor_array != facet_id) & (neighbor_array >= 0)]
        neighbor_set.update(valid_neighbors)