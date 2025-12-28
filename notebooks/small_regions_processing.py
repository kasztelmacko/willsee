from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from collections import deque
from scipy.ndimage import distance_transform_edt

from numba import njit


AdjacencyList = List[List[Tuple[int, int]]]


def _label_facets(clustered_array: np.ndarray) -> Tuple[np.ndarray, List[int], np.ndarray]:
    labels, sizes, colors = _label_facets_numba_core(clustered_array)
    component_sizes = sizes.tolist()
    return labels, component_sizes, colors

@njit
def _label_facets_numba_core(img):
    h, w, _ = img.shape

    labels = -np.ones((h, w), np.int32)

    max_components = h * w
    component_sizes = np.zeros(max_components, np.int32)
    component_colors = np.zeros((max_components, 3), np.uint8)

    stack_y = np.empty(h * w, np.int32)
    stack_x = np.empty(h * w, np.int32)

    current_id = 0

    for y in range(h):
        for x in range(w):
            if labels[y, x] != -1:
                continue

            r = img[y, x, 0]
            g = img[y, x, 1]
            b = img[y, x, 2]

            component_colors[current_id, 0] = r
            component_colors[current_id, 1] = g
            component_colors[current_id, 2] = b

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
                    if (img[ny, cx, 0] == r and
                        img[ny, cx, 1] == g and
                        img[ny, cx, 2] == b):
                        labels[ny, cx] = current_id
                        stack_y[top] = ny
                        stack_x[top] = cx
                        top += 1

                ny = cy + 1
                if ny < h and labels[ny, cx] == -1:
                    if (img[ny, cx, 0] == r and
                        img[ny, cx, 1] == g and
                        img[ny, cx, 2] == b):
                        labels[ny, cx] = current_id
                        stack_y[top] = ny
                        stack_x[top] = cx
                        top += 1

                nx = cx - 1
                if nx >= 0 and labels[cy, nx] == -1:
                    if (img[cy, nx, 0] == r and
                        img[cy, nx, 1] == g and
                        img[cy, nx, 2] == b):
                        labels[cy, nx] = current_id
                        stack_y[top] = cy
                        stack_x[top] = nx
                        top += 1

                nx = cx + 1
                if nx < w and labels[cy, nx] == -1:
                    if (img[cy, nx, 0] == r and
                        img[cy, nx, 1] == g and
                        img[cy, nx, 2] == b):
                        labels[cy, nx] = current_id
                        stack_y[top] = cy
                        stack_x[top] = nx
                        top += 1

            component_sizes[current_id] = size
            current_id += 1

    return labels, component_sizes[:current_id], component_colors[:current_id]

def _build_adjacency(
    image: np.ndarray,
    num_components: int
) -> AdjacencyList:
    """
    Build a region adjacency graph (RAG) where each node is a facet,
    and edge weights are shared boundary lengths between facets.
    
    """
    height, width = image.shape
    boundary_counts: dict[tuple[int, int], int] = defaultdict(int)
    
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
    
    adjacency_list: AdjacencyList = [[] for _ in range(num_components + 1)]
    for (a, b), length in boundary_counts.items():
        if 1 <= a <= num_components and 1 <= b <= num_components:
            adjacency_list[a].append((b, length))
            adjacency_list[b].append((a, length))
    
    return adjacency_list


def _compute_merge_targets(
    component_sizes: List[int],
    adjacency: AdjacencyList,
    merge_ids: set[int],
) -> Dict[int, int]:
    """
    For each merge-eligible component, choose a neighbouring component to merge into,
    based on maximum shared boundary length.
    """
    merge_target: Dict[int, int] = {}
    num_components = len(component_sizes)

    for comp_id in range(num_components):
        if comp_id not in merge_ids:
            continue

        neighbours_info = adjacency[comp_id]
        if not neighbours_info:
            continue

        best_neighbor, best_len = None, -1
        for neighbor_id, boundary_len in neighbours_info:
            if neighbor_id >= comp_id:
                continue
            if boundary_len > best_len:
                best_neighbor, best_len = neighbor_id, boundary_len

        if best_neighbor is not None:
            merge_target[comp_id] = best_neighbor

    return merge_target


def _apply_merges(
    labels_img: np.ndarray,
    component_colors: np.ndarray,
    merge_target: Dict[int, int],
) -> np.ndarray:
    """
    Apply merge decisions to produce a new RGB image.
    """

    def find_root(cid: int) -> int:
        seen = set()
        while cid in merge_target and cid not in seen:
            seen.add(cid)
            cid = merge_target[cid]
        return cid

    num_components = component_colors.shape[0]
    final_component_colors = component_colors.copy()
    for comp_id in range(num_components):
        root = find_root(comp_id)
        final_component_colors[comp_id] = component_colors[root]

    final_array = final_component_colors[labels_img]
    return final_array.astype(np.uint8)


def compute_narrow_component_ids(
    clustered_array: np.ndarray | None = None,
    narrow_thresh_px: int | None = None,
    *,
    labels_img: np.ndarray | None = None,
    component_sizes: List[int] | None = None,
) -> List[int]:
    """
    Return component ids that are considered narrow.
    
    A component is narrow if its average width per row OR average height per column
    is below the threshold.
    """
    if labels_img is None or component_sizes is None:
        labels_img, component_sizes, _ = _label_facets(clustered_array)
    
    return [
        comp_id
        for comp_id in range(len(component_sizes))
        if component_sizes[comp_id] > 0
        and (
            (component_sizes[comp_id] / max(np.count_nonzero(np.any(labels_img == comp_id, axis=1)), 1) < narrow_thresh_px)
            or (component_sizes[comp_id] / max(np.count_nonzero(np.any(labels_img == comp_id, axis=0)), 1) < narrow_thresh_px)
        )
    ]


def compute_small_component_ids(
    clustered_array: np.ndarray | None = None,
    min_facet_size: int | None = None,
    *,
    component_sizes: List[int] | None = None,
) -> List[int]:
    """
    Return component ids whose size is below the given threshold.
    """
    if component_sizes is None:
        _, component_sizes, _ = _label_facets(clustered_array)
    
    return [idx for idx, sz in enumerate(component_sizes) if sz < min_facet_size]


def merge_facets(
    clustered_array: np.ndarray | None = None,
    merge_component_ids: List[int] | None = None,
    *,
    labels_img: np.ndarray | None = None,
    component_sizes: List[int] | None = None,
    component_colors: np.ndarray | None = None,
) -> np.ndarray:
    """
    Merge connected regions (facets) of equal color into neighbouring facets
    using a simple region adjacency graph (RAG).

    Components listed in `merge_component_ids` are merged into neighbouring facets
    chosen by maximum shared boundary length.
    """
    if labels_img is None or component_sizes is None or component_colors is None:
        labels_img, component_sizes, component_colors = _label_facets(clustered_array)
    

    merge_ids_set = set(merge_component_ids)

    adjacency = _build_adjacency(labels_img, num_components=len(component_sizes))
    merge_target = _compute_merge_targets(component_sizes, adjacency, merge_ids_set)
    return _apply_merges(labels_img, component_colors, merge_target)


def compute_narrow_facets_mask(
    clustered_array: np.ndarray, narrow_thresh_px: int
) -> np.ndarray:
    """
    Identify facets whose average thickness is below a pixel threshold.

    For each connected component we compute:
    - average width per occupied row: size / number_of_rows_with_pixels
    - average height per occupied column: size / number_of_columns_with_pixels

    A facet is flagged as narrow if either metric is below `narrow_thresh_px`.

    Returns a (H, W) boolean mask with True for pixels belonging to narrow facets.
    """
    labels_img, component_sizes, _ = _label_facets(clustered_array)
    h, w = labels_img.shape

    narrow_ids = compute_narrow_component_ids(
        labels_img=labels_img,
        component_sizes=component_sizes,
        narrow_thresh_px=narrow_thresh_px,
    )

    narrow_mask = np.zeros((h, w), dtype=bool)
    for comp_id in narrow_ids:
        narrow_mask |= labels_img == comp_id

    return narrow_mask

def compute_component_label_centers(labels_img: np.ndarray) -> List[Tuple[int, int]]:
    """
    Compute an interior label point for each component using a distance transform.

    The selected point corresponds to the center of the maximal inscribed circle,
    keeping the label safely inside the facet. If a component degenerates to a
    zero-distance mask (very thin), we fall back to its centroid.
    """
    h, w = labels_img.shape
    num_components = int(labels_img.max()) + 1
    centers: List[Tuple[int, int]] = []

    for comp_id in range(num_components):
        mask = labels_img == comp_id
        if not np.any(mask):
            centers.append((0, 0))
            continue

        dist = distance_transform_edt(mask)
        max_idx = int(np.argmax(dist))
        max_dist = dist.flat[max_idx]
        y, x = divmod(max_idx, w)

        if max_dist <= 0:
            ys, xs = np.nonzero(mask)
            y = int(np.round(ys.mean()))
            x = int(np.round(xs.mean()))

        centers.append((y, x))

    return centers


def map_palette_indices(component_colors: np.ndarray) -> List[int]:
    """
    Map each component's RGB color to a deterministic palette index (1-based).
    """
    unique_colors = sorted({tuple(color.tolist()) for color in component_colors})
    color_to_idx = {color: idx + 1 for idx, color in enumerate(unique_colors)}
    return [color_to_idx[tuple(color.tolist())] for color in component_colors]


def compute_font_sizes(
    component_sizes: List[int], min_px: int = 10, max_px: int = 30, scale: float = 0.5
) -> List[int]:
    """
    Suggest a font size per component based on sqrt(area) with clamping.
    """
    sizes: List[int] = []
    for area in component_sizes:
        estimated = int(np.round(np.sqrt(max(area, 1)) * scale))
        sizes.append(int(np.clip(estimated, min_px, max_px)))
    return sizes


def compute_facet_label_data(
    clustered_array: np.ndarray,
    min_font_px: int = 10,
    max_font_px: int = 30,
    font_scale: float = 0.5,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[int], List[int], np.ndarray]:
    """
    Compute labeling metadata for a clustered image:
    - labels_img: component ids per pixel
    - centers: interior label positions (y, x) using maximal inscribed centers
    - palette_indices: palette index per component (1-based, deterministic)
    - font_sizes: suggested font size per component
    - component_colors: representative RGB color per component
    """
    labels_img, component_sizes, component_colors = _label_facets(clustered_array)
    centers = compute_component_label_centers(labels_img)
    palette_indices = map_palette_indices(component_colors)
    font_sizes = compute_font_sizes(
        component_sizes, min_px=min_font_px, max_px=max_font_px, scale=font_scale
    )
    return labels_img, centers, palette_indices, font_sizes, component_colors
