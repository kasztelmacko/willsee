import numpy as np
from sklearn.cluster import KMeans


def cluster_image(image: np.ndarray, n_colors: int, random_state: int = 42) -> np.ndarray:
    """
    Cluster an RGB image into `n_colors` using KMeans and return the
    clustered RGB image.
    """
    height, width = image.shape[:2]

    kmeans = KMeans(n_clusters=n_colors, random_state=random_state)
    kmeans.fit(image.reshape(-1, 3))

    clustered_rgb_image = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)
    clustered_rgb_image = clustered_rgb_image.reshape(height, width, 3)

    return clustered_rgb_image

