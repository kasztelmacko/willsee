from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FacetPBN:
    facet_id: int
    color_rgb: tuple[int, int, int]
    color_palette_id: int

    size_px: int
    mask_bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    label_point: tuple[int, int]
    inradius_px: float
    avg_width_px: float
    avg_height_px: float
    neighbour_ids: list[int]

    @classmethod
    def from_mask(
        cls,
        facet_id: int,
        mask: np.ndarray,
        color_rgb: tuple[int, int, int],
        color_palette_id: int,
        labels_img: np.ndarray | None = None,
    ) -> FacetPBN:
        """
        Build a facet descriptor from a boolean mask (H, W).
        Stub: compute size, bbox, centroid, label point (max-inscribed),
        inradius, average widths, and neighbors if labels are provided.
        """
        raise NotImplementedError("FacetPBN.from_mask is not implemented yet.")