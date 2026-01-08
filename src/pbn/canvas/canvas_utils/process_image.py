from collections import defaultdict
from typing import Any, Iterable, Hashable, TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from pbn.canvas.color_palette import ColorPalette

AdjacencyList = list[list[tuple[int, int]]]


def process_image(
    image: np.ndarray,
    min_facet_size: int,
    narrow_thresh_px: int,
) -> np.ndarray:
    """
    Process clustered image by:
    1. Finding small facets and merging them
    2. Finding narrow facets and merging them
    3. Reindexing facet ids to a compact 0..N-1 range
    """
    small_merged_array, _ = process_small_facets(image=image, min_facet_size=min_facet_size)
    narrow_merged_array, _ = process_narrow_facets(
        image=small_merged_array,
        narrow_thresh_px=narrow_thresh_px,
    )

    return narrow_merged_array


def process_small_facets(image: np.ndarray, min_facet_size: int) -> tuple[np.ndarray, np.ndarray]:
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


def process_narrow_facets(image: np.ndarray, narrow_thresh_px: int) -> tuple[np.ndarray, np.ndarray]:
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


def label_facets(image: np.ndarray) -> tuple[np.ndarray, list[int], np.ndarray]:
    """
    Label facets in a clustered RGB image.
    """
    labels, sizes, colors = _label_facets_numba_core(image=image)
    facet_sizes = sizes.tolist()
    return labels, facet_sizes, colors


@njit
def _label_facets_numba_core(image: np.ndarray):
    height, width, _ = image.shape

    labels = -np.ones((height, width), np.int32)

    max_facets = height * width
    facet_sizes = np.zeros(max_facets, np.int32)
    facet_colors = np.zeros((max_facets, 3), np.uint8)

    stack_y = np.empty(height * width, np.int32)
    stack_x = np.empty(height * width, np.int32)

    current_id = 0

    for y in range(height):
        for x in range(width):
            if labels[y, x] != -1:
                continue

            r = image[y, x, 0]
            g = image[y, x, 1]
            b = image[y, x, 2]

            facet_colors[current_id, 0] = r
            facet_colors[current_id, 1] = g
            facet_colors[current_id, 2] = b

            top = 0
            stack_y[top] = y
            stack_x[top] = x
            top += 1

            labels[y, x] = current_id
            size = 0

            while top > 0:
                top -= 1
                cy = stack_y[top]
                cx = stack_x[top]
                size += 1

                ny = cy - 1
                if ny >= 0 and labels[ny, cx] == -1:
                    if (image[ny, cx, 0] == r and
                        image[ny, cx, 1] == g and
                        image[ny, cx, 2] == b):
                        labels[ny, cx] = current_id
                        stack_y[top] = ny
                        stack_x[top] = cx
                        top += 1

                ny = cy + 1
                if ny < height and labels[ny, cx] == -1:
                    if (image[ny, cx, 0] == r and
                        image[ny, cx, 1] == g and
                        image[ny, cx, 2] == b):
                        labels[ny, cx] = current_id
                        stack_y[top] = ny
                        stack_x[top] = cx
                        top += 1

                nx = cx - 1
                if nx >= 0 and labels[cy, nx] == -1:
                    if (image[cy, nx, 0] == r and
                        image[cy, nx, 1] == g and
                        image[cy, nx, 2] == b):
                        labels[cy, nx] = current_id
                        stack_y[top] = cy
                        stack_x[top] = nx
                        top += 1

                nx = cx + 1
                if nx < width and labels[cy, nx] == -1:
                    if (image[cy, nx, 0] == r and
                        image[cy, nx, 1] == g and
                        image[cy, nx, 2] == b):
                        labels[cy, nx] = current_id
                        stack_y[top] = cy
                        stack_x[top] = nx
                        top += 1

            facet_sizes[current_id] = size
            current_id += 1

    return labels, facet_sizes[:current_id], facet_colors[:current_id]


def compute_adjacency_list(
    image: np.ndarray,
    num_facets: int
) -> AdjacencyList:
    """
    Build a region adjacency graph (RAG) where each node is a facet,
    and edge weights are shared boundary lengths between facets.
    """
    height, width = image.shape
    boundary_counts: dict[tuple[int, int], int] = defaultdict[tuple[int, int], int](int)
    
    if width > 1:
        left = image[:, :-1]
        right = image[:, 1:]
        mask = left != right
        pairs = np.stack([left[mask], right[mask]], axis=1)
        pairs = np.sort(pairs, axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        for (a, b), count in zip(unique_pairs, counts):
            boundary_counts[(int(a), int(b))] += int(count)
    
    if height > 1:
        top = image[:-1, :]
        bottom = image[1:, :]
        mask = top != bottom
        pairs = np.stack([top[mask], bottom[mask]], axis=1)
        pairs = np.sort(pairs, axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        for (a, b), count in zip(unique_pairs, counts):
            boundary_counts[(int(a), int(b))] += int(count)
    
    adjacency_list: AdjacencyList = [[] for _ in range(num_facets)]
    for (a, b), length in boundary_counts.items():
        if 0 <= a < num_facets and 0 <= b < num_facets:
            adjacency_list[a].append((b, length))
            adjacency_list[b].append((a, length))
    
    return adjacency_list


def compute_merge_targets(
    facet_sizes: list[int],
    adjacency_list: AdjacencyList,
    facet_ids_to_merge: list[int],
) -> dict[int, int]:
    """
    For each merge-eligible facet, choose a neighbouring facet to merge into,
    based on maximum shared boundary length.
    """
    merge_target: dict[int, int] = {}
    merge_ids_set = set(facet_ids_to_merge)

    for facet_id in range(len(facet_sizes)):
        if facet_id not in merge_ids_set:
            continue

        neighbours_info = adjacency_list[facet_id]
        if not neighbours_info:
            continue

        best_neighbor, best_len = None, -1
        for neighbor_id, boundary_len in neighbours_info:
            if neighbor_id == facet_id:
                continue
            if neighbor_id not in merge_ids_set and boundary_len > best_len:
                best_neighbor, best_len = neighbor_id, boundary_len

        if best_neighbor is None:
            for neighbor_id, boundary_len in neighbours_info:
                if neighbor_id == facet_id:
                    continue
                if boundary_len > best_len:
                    best_neighbor, best_len = neighbor_id, boundary_len

        if best_neighbor is not None:
            merge_target[facet_id] = best_neighbor

    return merge_target


def _apply_merges(
    image: np.ndarray,
    facet_colors: np.ndarray,
    merge_target: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply merge decisions to produce a new RGB image and the updated facet ids.
    """

    def find_root(facet_id: int) -> int:
        seen = set[Any]()
        while facet_id in merge_target and facet_id not in seen:
            seen.add(facet_id)
            facet_id = merge_target[facet_id]
        return facet_id

    num_facets = facet_colors.shape[0]
    final_facet_colors = facet_colors.copy()
    id_map = np.arange(num_facets, dtype=np.int32)
    for facet_id in range(num_facets):
        root = find_root(facet_id)
        id_map[facet_id] = root
        final_facet_colors[facet_id] = facet_colors[root]

    final_labels = id_map[image]
    final_array = final_facet_colors[image]
    return final_array.astype(np.uint8), final_labels.astype(np.int32)


def compute_small_facet_ids(
    min_facet_size: int,
    facet_sizes: list[int],
) -> list[int]:
    """
    Return facet ids whose size is below the given threshold.
    """    
    return [idx for idx, sz in enumerate[int](facet_sizes) if sz < min_facet_size]


def compute_narrow_facet_ids(
    narrow_thresh_px: int,
    image: np.ndarray,
    facet_sizes: list[int],
) -> list[int]:
    """
    Return facet ids that are considered narrow.
    
    A facet is narrow if its average width per row OR average height per column
    is below the threshold.
    """
    return [
        facet_id
        for facet_id in range(len(facet_sizes))
        if facet_sizes[facet_id] > 0
        and (
            (facet_sizes[facet_id] / max(np.count_nonzero(np.any(image == facet_id, axis=1)), 1) < narrow_thresh_px)
            or (facet_sizes[facet_id] / max(np.count_nonzero(np.any(image == facet_id, axis=0)), 1) < narrow_thresh_px)
        )
    ]


def merge_facets(
    merge_facet_ids: list[int],
    image: np.ndarray,
    facet_sizes: list[int],
    facet_colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge connected regions (facets) of equal color into neighbouring facets
    using a simple region adjacency graph (RAG). Returns the merged RGB image
    and the updated facet id map.

    Facets listed in `merge_facet_ids` are merged into neighbouring facets
    chosen by maximum shared boundary length.
    """
    adjacency = compute_adjacency_list(image, num_facets=len(facet_sizes))
    merge_target = compute_merge_targets(facet_sizes, adjacency, merge_facet_ids)
    return _apply_merges(image, facet_colors, merge_target)


def recolor_image_with_palette(
    image: np.ndarray,
    palette: "ColorPalette",
) -> np.ndarray:
    """
    Recolor an image using the palette's current key->RGB mapping.

    Any pixel whose color maps to a palette key is rewritten to that key's
    current RGB value. Unknown colors are left unchanged.
    """
    flat = image.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)

    color_to_key: dict[tuple[int, int, int], Hashable] = palette._color_to_key
    palette_map: dict[Hashable, Iterable[int]] = palette.to_dict()

    new_unique = unique_colors.copy()
    for idx, color_arr in enumerate(unique_colors):
        color_tuple = tuple[int, ...](int(c) for c in color_arr)
        key = color_to_key.get(color_tuple)
        if key is None:
            continue
        target_rgb = palette_map.get(key)
        if target_rgb is None:
            continue
        target_tuple = tuple[int, ...](int(c) for c in target_rgb)
        if target_tuple == color_tuple:
            continue
        new_unique[idx] = target_tuple

    recolored = new_unique[inverse].reshape(image.shape)
    return recolored
