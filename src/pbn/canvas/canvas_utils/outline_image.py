import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

from pbn.canvas.canvas_utils.process_image import label_facets


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



def compute_facets_properties(
    image: np.ndarray,
    min_font_px: int,
    max_font_px: int,
    font_scale: float,
) -> tuple[list[tuple[int, int]], list[int], np.ndarray]:
    """
    Compute per-facet label metadata: centers, suggested font sizes, and colors.

    - Centers: maximal inscribed circle via distance transform (centroid fallback).
    - Font sizes: sqrt(area) * font_scale, clamped to [min_font_px, max_font_px].
    - Colors: representative RGB for each facet from the labeled image.
    """
    labels_img, facet_sizes, facet_colors = label_facets(image=image)
    h, w = labels_img.shape
    num_facets = int(labels_img.max()) + 1

    facet_centers: list[tuple[int, int]] = []
    facet_font_sizes: list[int] = []

    for facet_id in range(num_facets):
        mask = labels_img == facet_id
        if not np.any(mask):
            facet_centers.append((0, 0))
            continue

        dist = distance_transform_edt(mask)
        max_idx = int(np.argmax(dist))
        max_dist = dist.flat[max_idx]
        y, x = divmod(max_idx, w)

        if max_dist <= 0:
            ys, xs = np.nonzero(mask)
            y = int(np.round(ys.mean()))
            x = int(np.round(xs.mean()))

        facet_centers.append((y, x))

    for facet_size in facet_sizes:
        estimated = int(np.round(np.sqrt(max(facet_size, 1)) * font_scale))
        facet_font_sizes.append(int(np.clip(estimated, min_font_px, max_font_px)))

    return facet_centers, facet_font_sizes, facet_colors


def _map_palette_indices(facet_colors: np.ndarray) -> list[int]:
    """
    Map each facet's RGB color to a deterministic palette index (1-based).
    """
    unique_colors = sorted({tuple(color.tolist()) for color in facet_colors})
    color_to_idx = {color: idx + 1 for idx, color in enumerate(unique_colors)}
    return [color_to_idx[tuple(color.tolist())] for color in facet_colors]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Try to load a TrueType font; fall back to Pillow's default bitmap font.
    """
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def create_image_with_color_labels(
    image: np.ndarray,
    outline_image: np.ndarray,
    min_font_px: int,
    max_font_px: int,
    font_scale: float,
    text_color: tuple[int, int, int],
) -> np.ndarray:
    """
    Render palette indices at facet centers onto a provided base image.

    - `image` (RGB) supplies facet geometry for centers and palette ids.
    - `outline_image` (RGB) is the layer to draw on (e.g., outlined image).
    - Palette index is deterministic from facet RGB values (1-based).
    - Returns an RGB numpy array with labels drawn.
    """
    facet_centers, facet_font_sizes, facet_colors = compute_facets_properties(
        image=image,
        min_font_px=min_font_px,
        max_font_px=max_font_px,
        font_scale=font_scale,
    )
    palette_indices = _map_palette_indices(facet_colors)

    labeled_outline_image = outline_image.copy()
    labeled_outline_image = Image.fromarray(labeled_outline_image, mode="RGB")
    draw = ImageDraw.Draw(labeled_outline_image)

    for (y, x), label, font_size in zip(facet_centers, palette_indices, facet_font_sizes):
        font = _load_font(font_size)
        draw.text((int(x), int(y)), str(label), fill=text_color, font=font, anchor="mm")

    return np.array(labeled_outline_image, dtype=outline_image.dtype)