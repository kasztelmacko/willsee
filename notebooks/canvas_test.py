from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import time

from small_regions_processing import (
    merge_facets,
    compute_narrow_component_ids,
    compute_small_component_ids,
    _label_facets,
)


N_COLORS = 20
MIN_FACET_PIXELS_SIZE = 50
NARROW_FACET_THRESHOLD_PX = 10

INPUT_IMAGE_PATH = "data/input_image/2024082.jpg"

A4_PORTRAIT_WIDTH_PX = 1800
A4_PORTRAIT_HEIGHT_PX = int(A4_PORTRAIT_WIDTH_PX * 297 / 210)

A4_LANDSCAPE_WIDTH_PX = 2400
A4_LANDSCAPE_HEIGHT_PX = int(A4_LANDSCAPE_WIDTH_PX * 210 / 297)


def build_clustered_image():
    """Load the image, cluster by color, process facets, and return the final processed image."""
    start_total = time.perf_counter()
    
    # Image loading and preprocessing
    t0 = time.perf_counter()
    with Image.open(fp=INPUT_IMAGE_PATH) as im:
        im = im.convert("RGB")

        # width, height = A4_PORTRAIT_WIDTH_PX, A4_PORTRAIT_HEIGHT_PX
        width, height = A4_LANDSCAPE_WIDTH_PX, A4_LANDSCAPE_HEIGHT_PX

        target_img = im.resize((width, height), resample=Image.LANCZOS)
        target = np.array(target_img, dtype=np.uint8)
        pixels = target.reshape(-1, 3)
    t1 = time.perf_counter()
    print(f"Image loading and preprocessing: {t1 - t0:.3f}s")

    # KMeans clustering
    t0 = time.perf_counter()
    kmeans = KMeans(n_clusters=N_COLORS, random_state=123)
    kmeans.fit(pixels)

    labels = kmeans.labels_
    clustered_pixels = kmeans.cluster_centers_[labels].astype(np.uint8)
    clustered_array = clustered_pixels.reshape(height, width, 3)
    t1 = time.perf_counter()
    print(f"KMeans clustering: {t1 - t0:.3f}s")

    # 1) Remove small facets
    t_small_start = time.perf_counter()
    t0 = time.perf_counter()
    labels_img, component_sizes, component_colors = _label_facets(clustered_array)
    t1 = time.perf_counter()
    print(f"  - Labeling facets: {t1 - t0:.3f}s")
    
    t0 = time.perf_counter()
    small_merge_ids = compute_small_component_ids(
        component_sizes=component_sizes, min_facet_size=MIN_FACET_PIXELS_SIZE
    )
    t1 = time.perf_counter()
    print(f"  - Finding small components: {t1 - t0:.3f}s")
    
    t0 = time.perf_counter()
    small_merged_array = merge_facets(
        labels_img=labels_img,
        component_sizes=component_sizes,
        component_colors=component_colors,
        merge_component_ids=small_merge_ids,
    )
    t1 = time.perf_counter()
    t_small_end = time.perf_counter()
    print(f"  - Merging small facets: {t1 - t0:.3f}s")
    print(f"Small facet removal (total): {t_small_end - t_small_start:.3f}s")

    # 2) Remove narrow facets (on the merged array)
    t_narrow_start = time.perf_counter()
    t0 = time.perf_counter()
    labels_img, component_sizes, component_colors = _label_facets(small_merged_array)
    t1 = time.perf_counter()
    print(f"  - Labeling facets: {t1 - t0:.3f}s")
    
    t0 = time.perf_counter()
    narrow_ids = compute_narrow_component_ids(
        labels_img=labels_img,
        component_sizes=component_sizes,
        narrow_thresh_px=NARROW_FACET_THRESHOLD_PX,
    )
    t1 = time.perf_counter()
    print(f"  - Finding narrow components: {t1 - t0:.3f}s")
    
    t0 = time.perf_counter()
    final_array = merge_facets(
        labels_img=labels_img,
        component_sizes=component_sizes,
        component_colors=component_colors,
        merge_component_ids=narrow_ids,
    )
    t1 = time.perf_counter()
    t_narrow_end = time.perf_counter()
    print(f"  - Merging narrow facets: {t1 - t0:.3f}s")
    print(f"Narrow facet removal (total): {t_narrow_end - t_narrow_start:.3f}s")
    
    t0 = time.perf_counter()
    final_img = Image.fromarray(final_array, mode="RGB")
    t1 = time.perf_counter()
    print(f"Image conversion: {t1 - t0:.3f}s")
    
    end_total = time.perf_counter()
    print(f"\nTotal execution time: {end_total - start_total:.3f}s")
    print("=" * 50)

    return final_img


if __name__ == "__main__":
    final_img = build_clustered_image()
    output_path = "data/input_image/zuza_5_output.png"
    
    t0 = time.perf_counter()
    final_img.save(output_path)
    t1 = time.perf_counter()
    print(f"Image saving: {t1 - t0:.3f}s")
    print(f"Processed image saved to: {output_path}")