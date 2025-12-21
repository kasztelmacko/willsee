from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pbn.facet_pbn.facet_pbn import FacetPBN


Palette = dict[int, tuple[int, int, int]]


@dataclass(frozen=True)
class CanvasPBN:
    height: int
    width: int
    rgb_img: np.ndarray
    labels_img: np.ndarray
    facets: list[FacetPBN]
    palette: Palette

    def extract_color_palette_from_canvas(self, rgb_img: np.ndarray) -> tuple[Palette, dict[tuple[int,int,int], int]]:
        colors = rgb_img.reshape(-1, 3)
        unique_colors = sorted({tuple[int, ...](int(c) for c in row) for row in colors})
        palette = {idx + 1: color for idx, color in enumerate[tuple[int, ...]](unique_colors)}
        color_to_idx = {color: idx for idx, color in palette.items()}
        return palette, color_to_idx

    def split_canvas_into_facets(self, labels_img: np.ndarray, rgb_img: np.ndarray) -> list[FacetPBN]:
        """Rebuild and return facet descriptors from a label map and RGB image."""
        labels = labels_img.astype(np.int32, copy=False)
        num_facets = int(labels.max()) + 1

        _ , color_to_palette_idx = self.extract_color_palette_from_canvas(rgb_img=rgb_img)

        facets: list[FacetPBN] = []
        for facet_id in range(num_facets):
            mask = labels == facet_id
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue
            y0, x0 = int(ys[0]), int(xs[0])
            color_rgb = tuple[int, ...](int(c) for c in rgb_img[y0, x0])
            palette_id = color_to_palette_idx[color_rgb]
            facet = FacetPBN.from_mask(
                facet_id=facet_id,
                mask=mask,
                color_rgb=color_rgb,
                color_palette_id=palette_id,
                labels_img=labels,
            )
            facets.append(facet)

        return facets

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