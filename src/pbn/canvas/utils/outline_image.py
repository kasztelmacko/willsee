import numpy as np


def create_outline_mask(image: np.ndarray) -> np.ndarray:
    """
    Compute a 1‑pixel‑wide outline mask for an RGB clustered image.

    A pixel is marked as outline if at least one of its 4‑connected neighbours
    (up/down/left/right) has a different RGB value.
    """
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=bool)

    diff_right = np.any(image[:, 1:, :] != image[:, :-1, :], axis=2)
    mask[:, :-1] |= diff_right

    diff_down = np.any(image[1:, :, :] != image[:-1, :, :], axis=2)
    mask[:-1, :] |= diff_down

    return mask


def create_image_outline(image: np.ndarray, outline_mask: np.ndarray, outline_color: tuple[int, int, int]) -> np.ndarray:
    """
    Render an outline on a white background using the provided boolean mask.
    """
    outlined = np.full_like(image, 255)
    outlined[outline_mask] = np.array(outline_color, dtype=outlined.dtype)
    return outlined