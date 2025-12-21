from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


AdjacencyList = List[List[Tuple[int, int]]]


def _label_facets(clustered_array: np.ndarray) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """
    Label connected components (facets) of equal color.

    Returns:
    - labels_img: (H, W) int32 component id per pixel (0..num_components-1)
    - component_sizes: list of sizes per component id
    - component_colors: (num_components, 3) uint8 representative color per component
    """
    h, w, _ = clustered_array.shape

    labels_img = np.full((h, w), fill_value=-1, dtype=np.int32)
    component_sizes: List[int] = []
    component_colors: List[np.ndarray] = []

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    current_id = 0

    for y in range(h):
        for x in range(w):
            if labels_img[y, x] != -1:
                continue

            base_color = clustered_array[y, x]
            stack = [(y, x)]
            labels_img[y, x] = current_id
            size = 0

            while stack:
                cy, cx = stack.pop()
                size += 1

                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and labels_img[ny, nx] == -1:
                        if np.array_equal(clustered_array[ny, nx], base_color):
                            labels_img[ny, nx] = current_id
                            stack.append((ny, nx))

            component_sizes.append(size)
            component_colors.append(base_color.copy())
            current_id += 1

    component_colors_arr = np.stack(component_colors, axis=0).astype(np.uint8)
    return labels_img, component_sizes, component_colors_arr


def _build_adjacency(labels_img: np.ndarray, num_components: int) -> AdjacencyList:
    """
    Build a region adjacency graph (RAG) where each node is a facet/component,
    and edge weights are shared boundary lengths between components.
    """
    h, w = labels_img.shape
    boundary_counts: Dict[Tuple[int, int], int] = defaultdict[Tuple[int, int], int](int)

    for y in range(h):
        for x in range(w):
            id_a = labels_img[y, x]
            if x + 1 < w:
                id_b = labels_img[y, x + 1]
                if id_a != id_b:
                    a, b = sorted((id_a, id_b))
                    boundary_counts[(a, b)] += 1
            if y + 1 < h:
                id_b = labels_img[y + 1, x]
                if id_a != id_b:
                    a, b = sorted((id_a, id_b))
                    boundary_counts[(a, b)] += 1

    adjacency: AdjacencyList = [[] for _ in range(num_components)]
    for (a, b), length in boundary_counts.items():
        adjacency[a].append((b, length))
        adjacency[b].append((a, length))

    return adjacency


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


def compute_narrow_flags(
    labels_img: np.ndarray,
    component_sizes: List[int],
    narrow_thresh_px: int | None,
) -> List[bool] | None:
    """
    Compute which components are narrow based on average thickness.
    """
    if narrow_thresh_px is None:
        return None

    num_components = len(component_sizes)
    narrow_flags: List[bool] = [False] * num_components

    for comp_id in range(num_components):
        comp_mask = labels_img == comp_id
        size = component_sizes[comp_id]
        if size == 0:
            continue

        rows_active = np.count_nonzero(np.any(comp_mask, axis=1))
        cols_active = np.count_nonzero(np.any(comp_mask, axis=0))

        avg_row_width = size / max(rows_active, 1)
        avg_col_height = size / max(cols_active, 1)

        if avg_row_width < narrow_thresh_px or avg_col_height < narrow_thresh_px:
            narrow_flags[comp_id] = True

    return narrow_flags


def compute_narrow_flags_from_array(
    clustered_array: np.ndarray, narrow_thresh_px: int | None
) -> List[bool] | None:
    """
    Convenience helper: compute narrow flags directly from an RGB clustered image.
    """
    labels_img, component_sizes, _ = _label_facets(clustered_array)
    return compute_narrow_flags(labels_img, component_sizes, narrow_thresh_px)


def compute_narrow_component_ids_from_array(
    clustered_array: np.ndarray, narrow_thresh_px: int | None
) -> List[int]:
    """
    Return component ids that are considered narrow.
    """
    labels_img, component_sizes, _ = _label_facets(clustered_array)
    flags = compute_narrow_flags(labels_img, component_sizes, narrow_thresh_px)
    if flags is None:
        return []
    return [idx for idx, is_narrow in enumerate(flags) if is_narrow]


def compute_small_component_ids(
    clustered_array: np.ndarray, min_facet_size: int
) -> List[int]:
    """
    Return component ids whose size is below the given threshold.
    """
    _, component_sizes, _ = _label_facets(clustered_array)
    return [idx for idx, sz in enumerate(component_sizes) if sz < min_facet_size]


def highlight_facets_by_ids(
    clustered_array: np.ndarray,
    component_ids: List[int],
    highlight_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Return a copy of the clustered image with the specified component ids recolored.
    """
    labels_img, _, _ = _label_facets(clustered_array)
    highlight_arr = clustered_array.copy()
    ids_set = set(component_ids)
    mask = np.isin(labels_img, list(ids_set))
    highlight_arr[mask] = np.array(highlight_color, dtype=np.uint8)
    return highlight_arr


def merge_facets(clustered_array: np.ndarray, merge_component_ids: List[int]) -> np.ndarray:
    """
    Merge connected regions (facets) of equal color into neighbouring facets
    using a simple region adjacency graph (RAG).

    Components listed in `merge_component_ids` are merged into neighbouring facets
    chosen by maximum shared boundary length.

    Returns a new (H, W, 3) uint8 array with merged facets recolored.
    """
    labels_img, component_sizes, component_colors = _label_facets(clustered_array)

    merge_ids_set = set(merge_component_ids)
    invalid_ids = [cid for cid in merge_ids_set if cid < 0 or cid >= len(component_sizes)]
    if invalid_ids:
        raise ValueError(f"merge_component_ids contains invalid ids: {invalid_ids}")

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

    narrow_flags = compute_narrow_flags(labels_img, component_sizes, narrow_thresh_px)
    if narrow_flags is None:
        return np.zeros((h, w), dtype=bool)

    narrow_mask = np.zeros((h, w), dtype=bool)
    for comp_id, is_narrow in enumerate(narrow_flags):
        if not is_narrow:
            continue
        narrow_mask |= labels_img == comp_id

    return narrow_mask


def highlight_narrow_facets(
    clustered_array: np.ndarray, narrow_thresh_px: int
) -> np.ndarray:
    """
    Return a copy of the clustered image with narrow facets recolored in red.
    """
    narrow_mask = compute_narrow_facets_mask(clustered_array, narrow_thresh_px)
    highlighted = clustered_array.copy()
    highlighted[narrow_mask] = np.array([255, 0, 0], dtype=np.uint8)
    return highlighted


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
