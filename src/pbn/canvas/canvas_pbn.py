from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pbn.canvas.facet_pbn import FacetPBN
from pbn.canvas.palette_pbn import PalettePBN


@dataclass(frozen=True)
class CanvasPBN:
    height: int
    width: int
    rgb_img: np.ndarray
    labels_img: np.ndarray
    facets: dict[int, FacetPBN] = field(default_factory=dict)
    palette: PalettePBN | None = None

    def __post_init__(self):
        """Automatically extract palette if not provided."""
        if self.palette is None:
            palette = self._extract_color_palette_from_canvas()
            object.__setattr__(self, 'palette', palette)

    def _extract_color_palette_from_canvas(self) -> PalettePBN:
        """Internal method to extract color palette from the canvas RGB image."""
        colors = self.rgb_img.reshape(-1, 3)
        unique_colors = sorted({tuple(int(c) for c in row) for row in colors})
        color_map = {idx + 1: color for idx, color in enumerate(unique_colors)}
        return PalettePBN(color_map=color_map)

    def extract_color_palette_from_canvas(self) -> PalettePBN:
        """Extract color palette from the canvas RGB image."""
        return self._extract_color_palette_from_canvas()

    def extract_facets_from_canvas(self) -> dict[int, FacetPBN]:
        """
        Extract facet descriptors from the label map and RGB image.
        Populates self.facets in place and returns the facets dictionary.
        """
        if self.facets:
            return self.facets
        
        labels = self.labels_img.astype(np.int32, copy=False)
        num_facets = int(labels.max()) + 1

        facets: dict[int, FacetPBN] = {}
        for facet_id in range(num_facets):
            mask = labels == facet_id
            facet_pixel_ys, facet_pixel_xs = np.nonzero(mask)

            if facet_pixel_ys.size == 0:
                continue
            
            representative_y, representative_x = int(facet_pixel_ys[0]), int(facet_pixel_xs[0])
            color_rgb = tuple(int(c) for c in self.rgb_img[representative_y, representative_x])
            palette_id = self.palette[color_rgb]
            facet = FacetPBN.from_mask(
                facet_id=facet_id,
                mask=mask,
                color_rgb=color_rgb,
                color_palette_id=palette_id,
                labels_img=labels,
            )
            facets[facet_id] = facet

        object.__setattr__(self, 'facets', facets)
        
        return facets

    def render_canvas_from_facets(self) -> np.ndarray:
        """
        Reconstruct the RGB image from labels_img and facets.
        Uses labels_img as the source of truth for pixel-to-facet mapping,
        and facets for color information.
        """
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for facet_id, facet in self.facets.items():
            mask = self.labels_img == facet_id
            canvas[mask] = facet.color_rgb

        return canvas

    def remove_small_facets(self, min_facet_size_px: int) -> CanvasPBN:
        """
        Return a new canvas with facets below the size threshold merged
        into neighboring facets. Stub only.
        """
        raise NotImplementedError("remove_small is not implemented yet.")

    def remove_narrow_facets(self, narrow_facet_threshold_px: int) -> CanvasPBN:
        """
        Return a new canvas with narrow facets merged. Stub only.
        """
        raise NotImplementedError("remove_narrow is not implemented yet.")