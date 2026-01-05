from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from PIL import Image

from sklearn.cluster import KMeans

from pbn.canvas.utils.merge_facets import (
    label_facets,
    merge_facets,
    compute_small_facet_ids,
    compute_narrow_facet_ids,
)
from pbn.canvas.utils.outline_image import create_outline_mask, create_image_outline
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
    processed_facets: np.ndarray
    outlined_image: np.ndarray

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
        clustered_image = cls._cluster_image(image=prepared_image, n_colors=n_colors)
        processed_image, processed_facets = cls._process_image(image=clustered_image)
        outlined_image = cls._outline_image(image=processed_image)

        return cls(
            input_image=input_image,
            canvas_orientation=canvas_orientation,
            canvas_page_size=canvas_page_size,
            n_colors=n_colors,
            prepared_image=prepared_image,
            clustered_image=clustered_image,
            processed_image=processed_image,
            processed_facets=processed_facets,
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
    def _process_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Process clustered image by:
        1. Finding small facets and merging them
        2. Finding narrow facets and merging them
        3. Reindex facets to start at 0
        """
        small_merged_array, small_facets  = Canvas._process_small_facets(
            image=image,
            min_facet_size=MIN_FACET_PIXELS_SIZE
        )
        narrow_merged_array, narrow_facets = Canvas._process_narrow_facets(
            image=small_merged_array,
            narrow_thresh_px=NARROW_FACET_THRESHOLD_PX
        )

        _, dense_inverse = np.unique(narrow_facets, return_inverse=True)
        
        return narrow_merged_array, dense_inverse.reshape(narrow_facets.shape).astype(np.int32)

    @staticmethod
    def _outline_image(image: np.ndarray) -> np.ndarray:
        outline_mask = create_outline_mask(image=image)
        outline_image = create_image_outline(image=image, outline_mask=outline_mask, outline_color=(0, 0, 0))
        return outline_image

    @staticmethod
    def _process_small_facets(image: np.ndarray, min_facet_size: int) -> tuple[np.ndarray, np.ndarray]:
        facets_img, facet_sizes, facet_colors = label_facets(image=image)
        small_facet_ids = compute_small_facet_ids(
            facet_sizes=facet_sizes,
            min_facet_size=min_facet_size
        )
        merged_array, merged_facets = merge_facets(
            image=facets_img,
            facet_sizes=facet_sizes,
            facet_colors=facet_colors,
            merge_facet_ids=small_facet_ids,
        )
        
        return merged_array, merged_facets

    @staticmethod
    def _process_narrow_facets(image: np.ndarray, narrow_thresh_px: int) -> tuple[np.ndarray, np.ndarray]:
        facets_img, facet_sizes, facet_colors = label_facets(image=image)
        narrow_facet_ids = compute_narrow_facet_ids(
            image=facets_img,
            facet_sizes=facet_sizes,
            narrow_thresh_px=narrow_thresh_px,
        )
        merged_array, merged_facets = merge_facets(
            image=facets_img,
            facet_sizes=facet_sizes,
            facet_colors=facet_colors,
            merge_facet_ids=narrow_facet_ids,
        )
        
        return merged_array, merged_facets
