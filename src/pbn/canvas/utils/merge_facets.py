from collections import defaultdict
from typing import List, Tuple
from pbn.canvas.facet import Facet

import numpy as np

AdjacencyList = List[List[Tuple[int, int]]]


def compute_adjacency_list(
    image: np.ndarray,
    num_facets: int
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
    
    adjacency_list: AdjacencyList = [[] for _ in range(num_facets)]
    for (a, b), length in boundary_counts.items():
        if 0 <= a < num_facets and 0 <= b < num_facets:
            adjacency_list[a].append((b, length))
            adjacency_list[b].append((a, length))
    
    return adjacency_list

def compute_merge_targets(
    adjacency_list: AdjacencyList,
    facet_ids_to_merge: list[int],
) -> dict[int, int]:
    """
    For each merge-eligible facet, choose a neighboring facet to merge into,
    based on maximum shared boundary length.
    
    Only considers neighbors with ID < current facet ID to ensure deterministic merging.
    """
    merge_target_dict: dict[int, int] = {}
    
    for facet_id in range(len(adjacency_list)):
        if facet_id not in facet_ids_to_merge:
            continue
        
        neighbors = adjacency_list[facet_id]
        if not neighbors:
            continue
        
        eligible_neighbors = [
            (neighbor_id, boundary_length)
            for neighbor_id, boundary_length in neighbors
            if neighbor_id < facet_id
        ]
        
        if eligible_neighbors:
            best_neighbor, _ = max(eligible_neighbors, key=lambda x: x[1])
            merge_target_dict[facet_id] = best_neighbor
    
    return merge_target_dict

def compute_merged_image(
    image: np.ndarray, facet_list: list[Facet], merge_targets: dict[int, int]
) -> np.ndarray:
    """
    Apply merge decisions to produce a new facet ID image.
    
    Merged facets are reassigned to their root facet ID (following the merge chain
    to handle transitive merges).
    """
    num_facets = len(facet_list)
    root_map = np.array(
        [_find_root(facet_id=facet_id, merge_targets=merge_targets)
        for facet_id in range(num_facets)], 
        dtype=image.dtype
    )
    
    merged_image = np.clip(image, 0, num_facets - 1)
    
    return root_map[merged_image]

def _find_root(facet_id: int, merge_targets: dict[int, int]) -> int:
    """Find the root facet ID by following the merge chain."""
    seen = set()
    current_id = facet_id
    while current_id in merge_targets and current_id not in seen:
        seen.add(current_id)
        current_id = merge_targets[current_id]
    return current_id