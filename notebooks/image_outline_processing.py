import numpy as np


def compute_outline_mask(image: np.ndarray) -> np.ndarray:
    """
    Compute a 1‑pixel‑wide outline mask for a clustered image.

    A pixel is marked as outline if at least one of its 4‑connected neighbours
    has a different RGB value.

    Parameters
    ----------
    image:
        (H, W, 3) uint8 array – typically the processed / facet‑merged image.

    Returns
    -------
    mask:
        (H, W) boolean array where True indicates an outline pixel.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")

    h, w, _ = image.shape

    mask = np.zeros((h, w), dtype=bool)

    diff_right = np.any(image[:, :-1] != image[:, 1:], axis=2)
    mask[:, :-1] |= diff_right

    diff_down = np.any(image[:-1, :] != image[1:, :], axis=2)
    mask[:-1, :] |= diff_down

    return mask


