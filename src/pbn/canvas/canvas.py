from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from PIL import Image

from sklearn.cluster import KMeans

from pbn.canvas.facet import Facet
from pbn.canvas.color_palette import ColorPalette
from pbn.canvas.utils.merge_facets import (
    label_facets,
    merge_facets,
    compute_small_facet_ids,
    compute_narrow_facet_ids,
)
from pbn.config.pbn_config import (
    CANVAS_SIZE_CONFIG, 
    MIN_FACET_PIXELS_SIZE, 
    NARROW_FACET_THRESHOLD_PX
)

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

    @classmethod
    def create_canvas(
        cls,
        input_image: Image,
        canvas_orientation: str,
        canvas_page_size: str,
        n_colors: int
    ) -> Canvas:
        start_total = time.perf_counter()
        
        t0 = time.perf_counter()
        prepared_image = cls._prepare_image(
            image=input_image,
            canvas_orientation=canvas_orientation,
            canvas_page_size=canvas_page_size
        )
        t1 = time.perf_counter()
        print(f"Image preparation: {t1 - t0:.3f}s")
        
        t0 = time.perf_counter()
        clustered_image = cls._cluster_image(image=prepared_image, n_colors=n_colors)
        t1 = time.perf_counter()
        print(f"KMeans clustering: {t1 - t0:.3f}s")
        
        t0 = time.perf_counter()
        processed_image = cls._process_image(image=clustered_image)
        t1 = time.perf_counter()
        print(f"Image processing: {t1 - t0:.3f}s")
        
        t0 = time.perf_counter()
        outlined_image = cls._outline_image(image=processed_image)
        t1 = time.perf_counter()
        print(f"Image outlining: {t1 - t0:.3f}s")
        
        end_total = time.perf_counter()
        print(f"\nTotal execution time: {end_total - start_total:.3f}s")
        print("=" * 50)

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
        canvas_page_size: str
    ) -> np.ndarray:
        width = CANVAS_SIZE_CONFIG[canvas_page_size][canvas_orientation]["WIDTH"]
        height = CANVAS_SIZE_CONFIG[canvas_page_size][canvas_orientation]["HEIGHT"]

        return np.array(
                    image.resize((width, height), resample=Image.LANCZOS), 
                    dtype=np.uint8
                )

    @staticmethod
    def _cluster_image(image: np.ndarray, n_colors: int) -> np.ndarray:
        """
        Cluster image by color and return RGB image with clustered colors.
        """
        height, width = image.shape[:2]
        
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(image.reshape(-1, 3))

        clustered_rgb_image = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)
        clustered_rgb_image = clustered_rgb_image.reshape(height, width, 3)

        return clustered_rgb_image

    @staticmethod
    def _process_image(image: np.ndarray) -> np.ndarray:
        """
        Process clustered image by:
        1. Finding small facets and merging them
        2. Finding narrow facets and merging them

        """
        t_small_start = time.perf_counter()
        small_merged_array = Canvas._process_small_facets(image=image, min_facet_size=MIN_FACET_PIXELS_SIZE)
        t_small_end = time.perf_counter()
        print(f"  Small facet processing: {t_small_end - t_small_start:.3f}s")
        
        t_narrow_start = time.perf_counter()
        narrow_merged_array = Canvas._process_narrow_facets(image=small_merged_array, narrow_thresh_px=NARROW_FACET_THRESHOLD_PX)
        t_narrow_end = time.perf_counter()
        print(f"  Narrow facet processing: {t_narrow_end - t_narrow_start:.3f}s")
        
        return narrow_merged_array

    @staticmethod
    def _outline_image(image: np.ndarray) -> np.ndarray:
        return image

    @staticmethod
    def _process_small_facets(image: np.ndarray, min_facet_size: int) -> np.ndarray:
        t0 = time.perf_counter()
        facets_img, facet_sizes, facet_colors = label_facets(image=image)
        t1 = time.perf_counter()
        print(f"    - Labeling facets: {t1 - t0:.3f}s")
        
        t0 = time.perf_counter()
        small_facet_ids = compute_small_facet_ids(
            facet_sizes=facet_sizes,
            min_facet_size=min_facet_size
        )
        t1 = time.perf_counter()
        print(f"    - Finding small facets: {t1 - t0:.3f}s")
        
        t0 = time.perf_counter()
        merged_array = merge_facets(
            image=facets_img,
            facet_sizes=facet_sizes,
            facet_colors=facet_colors,
            merge_facet_ids=small_facet_ids,
        )
        t1 = time.perf_counter()
        print(f"    - Merging small facets: {t1 - t0:.3f}s")
        
        return merged_array

    @staticmethod
    def _process_narrow_facets(image: np.ndarray, narrow_thresh_px: int) -> np.ndarray:
        t0 = time.perf_counter()
        facets_img, facet_sizes, facet_colors = label_facets(image=image)
        t1 = time.perf_counter()
        print(f"    - Labeling facets: {t1 - t0:.3f}s")
        
        t0 = time.perf_counter()
        narrow_facet_ids = compute_narrow_facet_ids(
            image=facets_img,
            facet_sizes=facet_sizes,
            narrow_thresh_px=narrow_thresh_px,
        )
        t1 = time.perf_counter()
        print(f"    - Finding narrow facets: {t1 - t0:.3f}s")
        
        t0 = time.perf_counter()
        merged_array = merge_facets(
            image=facets_img,
            facet_sizes=facet_sizes,
            facet_colors=facet_colors,
            merge_facet_ids=narrow_facet_ids,
        )
        t1 = time.perf_counter()
        print(f"    - Merging narrow facets: {t1 - t0:.3f}s")
        
        return merged_array
