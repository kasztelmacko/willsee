from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy.ndimage import label

from sklearn.cluster import KMeans

from pbn.canvas.facet import Facet
from pbn.canvas.color_palette import ColorPalette
from pbn.canvas.utils.merge_facets import (
    compute_adjacency_list,
    compute_merge_targets,
    compute_merged_image,
)
from pbn.config.pbn_config import CANVAS_SIZE_CONFIG, MIN_FACET_PIXELS_SIZE

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

    color_palette: ColorPalette

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
    def _cluster_image(image: np.ndarray, n_colors: int) -> tuple[np.ndarray, ColorPalette]:
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
        facets_img, facet_list = Canvas._extract_facets_from_image(clustered_image=image, connectivity=2)

        small_facet_ids = Canvas._find_small_facet_ids(facet_list=facet_list, min_facet_size=MIN_FACET_PIXELS_SIZE)
        image_with_small_facets_removed = Canvas._merge_facets(
            facets_img=facets_img,
            facet_list=facet_list, 
            facet_ids_to_merge=small_facet_ids
        )
        return image_with_small_facets_removed

    @staticmethod
    def _outline_image(image: np.ndarray) -> np.ndarray:
        return image

    @staticmethod
    def _extract_facets_from_image(
        clustered_image: np.ndarray,
        connectivity: int
    ) -> tuple[np.ndarray, list[Facet]]:

        height, width = clustered_image.shape
        facets_img = np.zeros((height, width), dtype=np.int32)
        facet_list: list[Facet] = []
        facet_id = 0
        
        structure = np.ones((3, 3), dtype=bool) if connectivity == 2 else None
        
        for color in np.unique(clustered_image):
            color_mask = clustered_image == color
            labeled_facets, num_facets = label(color_mask, structure=structure)
            
            for i in range(1, num_facets + 1):
                facet_id += 1
                facet_mask = labeled_facets == i
                facet_size_px = int(facet_mask.sum())
                facets_img[facet_mask] = facet_id

                facet = Facet.create_facet(
                    facet_id=facet_id,
                    facet_color_label=int(color),
                    facet_size_px=facet_size_px,
                )
                facet_list.append(facet)
        
        return facets_img, facet_list

    @staticmethod
    def _find_small_facet_ids(
        facet_list: list[Facet], min_facet_size: int
    ) -> list[int]:
        return [facet.facet_id for facet in facet_list if facet.facet_size_px < min_facet_size]


    @staticmethod
    def _merge_facets(
        facets_img: np.ndarray, facet_list: list[Facet], facet_ids_to_merge: list[int]
    ) -> np.ndarray:
        adjacency_list = compute_adjacency_list(image=facets_img, num_facets=len(facet_list))
        merge_target_dict = compute_merge_targets(adjacency_list=adjacency_list, facet_ids_to_merge=facet_ids_to_merge)
        return compute_merged_image(image=facets_img, facet_list=facet_list, merge_targets=merge_target_dict)
        
