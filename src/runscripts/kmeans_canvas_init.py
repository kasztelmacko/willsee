"""
Script to initialize a CanvasPBN from an input image using k-means clustering.
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

from pbn.canvas.canvas_pbn import CanvasPBN


def main():
    image_path = Path("data/input_image/zuza_5.jpg")
    with Image.open(image_path) as im:
        rgb_img = np.array(im.convert("RGB"), dtype=np.uint8)
    
    height, width = rgb_img.shape[:2]
    
    n_clusters = 20
    pixels = rgb_img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    labels = kmeans.labels_
    clustered_pixels = kmeans.cluster_centers_[labels].astype(np.uint8)
    clustered_rgb_img = clustered_pixels.reshape(height, width, 3)
    labels_img = labels.reshape(height, width).astype(np.int32)
    
    canvas = CanvasPBN(
        height=height,
        width=width,
        rgb_img=clustered_rgb_img,
        labels_img=labels_img,
    )
    
    canvas.extract_facets_from_canvas()
    
    print(f"Canvas created: {canvas.width}x{canvas.height}, {len(canvas.facets)} facets, {len(canvas.palette)} colors")


if __name__ == "__main__":
    main()

