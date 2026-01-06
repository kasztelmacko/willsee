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
    num_facets = len(facet_sizes)

    facet_centers: list[tuple[int, int]] = []
    facet_font_sizes: list[int] = []

    y_min = np.full(num_facets, h, dtype=np.int32)
    y_max = np.full(num_facets, -1, dtype=np.int32)
    x_min = np.full(num_facets, w, dtype=np.int32)
    x_max = np.full(num_facets, -1, dtype=np.int32)

    ys, xs = np.nonzero(labels_img >= 0)
    for y, x in zip(ys, xs):
        fid = labels_img[y, x]
        if 0 <= fid < num_facets:
            if y < y_min[fid]:
                y_min[fid] = y
            if y > y_max[fid]:
                y_max[fid] = y
            if x < x_min[fid]:
                x_min[fid] = x
            if x > x_max[fid]:
                x_max[fid] = x

    for facet_id in range(num_facets):
        if facet_sizes[facet_id] <= 0 or y_max[facet_id] < 0:
            facet_centers.append((0, 0))
            continue

        y0 = max(y_min[facet_id], 0)
        y1 = min(y_max[facet_id] + 1, h)
        x0 = max(x_min[facet_id], 0)
        x1 = min(x_max[facet_id] + 1, w)

        local_mask = labels_img[y0:y1, x0:x1] == facet_id
        padded = np.pad(local_mask, 1, constant_values=False)
        dist = distance_transform_edt(padded)
        max_idx = int(np.argmax(dist))
        max_dist = dist.flat[max_idx]
        py, px = divmod(max_idx, dist.shape[1])
        y = y0 + py - 1
        x = x0 + px - 1

        if max_dist <= 0:
            ys_local, xs_local = np.nonzero(local_mask)
            y = int(np.round(ys_local.mean())) + y0
            x = int(np.round(xs_local.mean())) + x0

        y = int(np.clip(y, 0, h - 1))
        x = int(np.clip(x, 0, w - 1))

        facet_centers.append((y, x))

    for facet_size in facet_sizes:
        estimated = int(np.round(np.sqrt(max(facet_size, 1)) * font_scale))
        facet_font_sizes.append(int(np.clip(estimated, min_font_px, max_font_px)))

    return facet_centers, facet_font_sizes, facet_colors


def _map_palette_indices(facet_colors: np.ndarray) -> tuple[list[int], dict[int, tuple[int, int, int]]]:
    """
    Map each facet's RGB color to a deterministic palette index (1-based) and
    return the palette mapping.
    """
    unique_colors = sorted({tuple(color.tolist()) for color in facet_colors})
    color_palette = {idx + 1: color for idx, color in enumerate(unique_colors)}
    color_to_idx = {color: idx for idx, color in color_palette.items()}
    indices = [color_to_idx[tuple(color.tolist())] for color in facet_colors]
    return indices, color_palette


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
) -> tuple[np.ndarray, dict[int, tuple[int, int, int]]]:
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
    palette_indices, color_palette = _map_palette_indices(facet_colors)

    labeled_outline_image = outline_image.copy()
    labeled_outline_image = Image.fromarray(labeled_outline_image, mode="RGB")
    draw = ImageDraw.Draw(labeled_outline_image)

    h, w, _ = outline_image.shape

    for (y, x), label, font_size in zip(facet_centers, palette_indices, facet_font_sizes):
        margin = max(font_size // 2, 1)
        cy = int(np.clip(y, margin, h - 1 - margin))
        cx = int(np.clip(x, margin, w - 1 - margin))

        font = _load_font(font_size)
        draw.text((cx, cy), str(label), fill=text_color, font=font, anchor="mm")

    return np.array(labeled_outline_image, dtype=outline_image.dtype), color_palette