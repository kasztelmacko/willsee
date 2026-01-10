"""Gradio interface for generating paint-by-number canvases."""
from __future__ import annotations
from pathlib import Path
import sys
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from pbn.canvas.canvas import Canvas
import pbn.config.pbn_config as PBN_CONF


import gradio as gr
from PIL import Image



DEFAULT_PAGE_SIZE = "A4"
DEFAULT_ORIENTATION = "LANDSCAPE"
DEFAULT_N_COLORS = 15
MAX_PALETTE_ENTRIES = 40


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = (int(c) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _parse_color_to_rgb(
    color_value: str | tuple[int, int, int] | list[int],
) -> tuple[int, int, int]:
    def _clamp(value: float) -> int:
        return max(0, min(255, int(round(value))))

    stripped = color_value.strip()
    if stripped.startswith(("rgba(", "rgb(")):
        try:
            r, g, b, *_ = (float(c.strip()) for c in stripped[stripped.index("(") + 1 : stripped.index(")")].split(","))
            return tuple[int, ...](_clamp(c) for c in (r, g, b))
        except Exception:
            pass

    hex_value = stripped.lstrip("#")
    if len(hex_value) == 3:
        hex_value = "".join(ch * 2 for ch in hex_value)
    if len(hex_value) in (6, 8):
        try:
            return tuple[int, ...](_clamp(int(hex_value[i : i + 2], 16)) for i in (0, 2, 4))
        except Exception:
            pass

    raise gr.Error("Invalid color value. Please select a valid color.")


def _palette_component_updates(
    palette_dict: dict,
) -> tuple[list[gr.update], list[gr.update], list[gr.update]]:
    items = list(palette_dict.items())
    row_updates: list[gr.update] = []
    id_updates: list[gr.update] = []
    color_updates: list[gr.update] = []

    for idx in range(MAX_PALETTE_ENTRIES):
        if idx < len(items):
            key, rgb = items[idx]
            row_updates.append(gr.update(visible=True))
            id_updates.append(
                gr.update(
                    value=str(key),
                    label=f"Color ID {key}",
                    visible=True,
                )
            )
            color_updates.append(gr.update(value=_rgb_to_hex(rgb), visible=True))
        else:
            row_updates.append(gr.update(visible=False))
            id_updates.append(gr.update(value="", visible=False))
            color_updates.append(gr.update(value="#ffffff", visible=False))

    return row_updates, id_updates, color_updates


def apply_palette_updates(
    canvas_state: Canvas | None,
    *palette_fields: str,
):
    if canvas_state is None:
        raise gr.Error("Generate a canvas first to edit the palette.")

    ids = palette_fields[:MAX_PALETTE_ENTRIES]
    colors = palette_fields[MAX_PALETTE_ENTRIES:]

    def _parse_palette_key(raw: str) -> str | int | None:
        if raw is None:
            return None
        text = str(raw).strip()
        if text == "":
            return None
        try:
            return int(float(text))
        except Exception:
            return text

    palette = canvas_state.color_palette
    palette_items = palette.to_dict().items()

    for idx, (old_key, _) in enumerate(palette_items):
        if idx >= MAX_PALETTE_ENTRIES:
            break

        new_key_raw = ids[idx] if idx < len(ids) else ""
        new_color_raw = colors[idx] if idx < len(colors) else ""

        parsed_key = _parse_palette_key(new_key_raw)
        target_key = parsed_key if parsed_key is not None else old_key

        if target_key != old_key:
            palette.rename_key(old_key, target_key)

        if new_color_raw:
            try:
                rgb = _parse_color_to_rgb(new_color_raw)
            except Exception:
                continue
            palette.adjust_color(target_key, rgb)

    updated_canvas = canvas_state.render_image_with_replaced_palette(palette)
    palette_dict = updated_canvas.color_palette.to_dict()
    row_updates, id_updates, color_updates = _palette_component_updates(palette_dict)

    return (
        updated_canvas.processed_image,
        updated_canvas.outlined_image,
        updated_canvas,
        *row_updates,
        *id_updates,
        *color_updates,
    )


DEFAULT_MIN_FACET_SIZE = PBN_CONF.MIN_FACET_PIXELS_SIZE
DEFAULT_NARROW_FACET_THRESHOLD = PBN_CONF.NARROW_FACET_THRESHOLD_PX
DEFAULT_MIN_FONT_PX = PBN_CONF.MIN_FONT_PX
DEFAULT_MAX_FONT_PX = PBN_CONF.MAX_FONT_PX
DEFAULT_FONT_SCALE = PBN_CONF.FONT_SCALE
DEFAULT_FACET_LABEL_COLOR_HEX = _rgb_to_hex(PBN_CONF.FACET_LABEL_COLOR)
DEFAULT_FACET_OUTLINE_COLOR_HEX = _rgb_to_hex(PBN_CONF.FACET_OUTLINE_COLOR)


def _orientations_from_config(config: dict[str, dict[str, dict]]) -> list[str]:
    orientations: set[str] = set()
    for size_config in config.values():
        orientations.update(size_config.keys())
    return sorted(orientations)


PAGE_SIZES = sorted(PBN_CONF.CANVAS_SIZE_CONFIG.keys())
ORIENTATIONS = _orientations_from_config(PBN_CONF.CANVAS_SIZE_CONFIG)


def _validate_page_orientation(page_size: str, orientation: str) -> None:
    try:
        PBN_CONF.CANVAS_SIZE_CONFIG[page_size][orientation]
    except KeyError as exc:
        raise gr.Error("Selected page size/orientation is not supported.") from exc


def generate_canvas(
    image: Image.Image,
    canvas_orientation: str = DEFAULT_ORIENTATION,
    canvas_page_size: str = DEFAULT_PAGE_SIZE,
    n_colors: int = DEFAULT_N_COLORS,
    min_facet_size: int = DEFAULT_MIN_FACET_SIZE,
    narrow_thresh_px: int = DEFAULT_NARROW_FACET_THRESHOLD,
    min_font_px: int = DEFAULT_MIN_FONT_PX,
    max_font_px: int = DEFAULT_MAX_FONT_PX,
    font_scale: float = DEFAULT_FONT_SCALE,
    facet_label_color: str | tuple[int, int, int] = DEFAULT_FACET_LABEL_COLOR_HEX,
    facet_outline_color: str | tuple[int, int, int] = DEFAULT_FACET_OUTLINE_COLOR_HEX,
):
    """Run the Canvas pipeline and return processed and outlined images."""
    _validate_page_orientation(canvas_page_size, canvas_orientation)

    min_facet_size = int(min_facet_size)
    narrow_thresh_px = int(narrow_thresh_px)
    min_font_px = int(min_font_px)
    max_font_px = int(max_font_px)
    font_scale = float(font_scale)

    if min_facet_size < 1 or narrow_thresh_px < 1:
        raise gr.Error("Facet thresholds must be positive integers.")
    if min_font_px < 1 or max_font_px < 1:
        raise gr.Error("Font sizes must be positive integers.")
    if min_font_px > max_font_px:
        raise gr.Error("Min font size cannot exceed max font size.")
    if font_scale <= 0:
        raise gr.Error("Font scale must be greater than zero.")

    label_color_rgb = _parse_color_to_rgb(
        color_value=facet_label_color,
    )
    outline_color_rgb = _parse_color_to_rgb(
        color_value=facet_outline_color,
    )

    rgb_image = image.convert("RGB")
    canvas = Canvas.create_canvas(
        input_image=rgb_image,
        canvas_orientation=canvas_orientation,
        canvas_page_size=canvas_page_size,
        n_colors=int(n_colors),
        min_facet_size=min_facet_size,
        narrow_thresh_px=narrow_thresh_px,
        min_font_px=min_font_px,
        max_font_px=max_font_px,
        font_scale=font_scale,
        facet_label_color=label_color_rgb,
        facet_outline_color=outline_color_rgb,
    )
    palette_dict = canvas.color_palette.to_dict()
    row_updates, id_updates, color_updates = _palette_component_updates(palette_dict)
    return (
        canvas.processed_image,
        canvas.outlined_image,
        canvas,
        *row_updates,
        *id_updates,
        *color_updates,
    )


with gr.Blocks(title="Paint-by-Number Canvas") as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil",
                image_mode="RGB",
                label="Input image",
            )
            page_dropdown = gr.Dropdown(
                choices=PAGE_SIZES,
                value=DEFAULT_PAGE_SIZE,
                label="Canvas page size",
            )
            orientation_dropdown = gr.Dropdown(
                choices=ORIENTATIONS,
                value=DEFAULT_ORIENTATION,
                label="Canvas orientation",
            )
            n_colors_slider = gr.Slider(
                minimum=2,
                maximum=40,
                step=1,
                value=DEFAULT_N_COLORS,
                label="Number of colors (clusters)",
            )
            with gr.Accordion("Advanced canvas parameters", open=False):
                min_facet_slider = gr.Slider(
                    minimum=1,
                    maximum=500,
                    step=1,
                    value=DEFAULT_MIN_FACET_SIZE,
                    label="Minimum facet size (pixels)",
                )
                narrow_thresh_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=DEFAULT_NARROW_FACET_THRESHOLD,
                    label="Narrow facet threshold (pixels)",
                )
                min_font_slider = gr.Slider(
                    minimum=4,
                    maximum=100,
                    step=1,
                    value=DEFAULT_MIN_FONT_PX,
                    label="Min label font size (px)",
                )
                max_font_slider = gr.Slider(
                    minimum=6,
                    maximum=200,
                    step=1,
                    value=DEFAULT_MAX_FONT_PX,
                    label="Max label font size (px)",
                )
                font_scale_slider = gr.Slider(
                    minimum=0.1,
                    maximum=3.0,
                    step=0.05,
                    value=DEFAULT_FONT_SCALE,
                    label="Label font scale multiplier",
                )
                label_color_picker = gr.ColorPicker(
                    value=DEFAULT_FACET_LABEL_COLOR_HEX,
                    label="Facet label color",
                )
                outline_color_picker = gr.ColorPicker(
                    value=DEFAULT_FACET_OUTLINE_COLOR_HEX,
                    label="Facet outline color",
                )
            gr.Markdown("### Palette editor")
            palette_rows: list[gr.Row] = []
            palette_id_inputs: list[gr.Textbox] = []
            palette_color_inputs: list[gr.ColorPicker] = []
            for idx in range(MAX_PALETTE_ENTRIES):
                with gr.Row(visible=False) as row:
                    id_box = gr.Textbox(
                        label="Color ID",
                        interactive=True,
                        scale=1,
                    )
                    color_picker = gr.ColorPicker(
                        label="Color",
                        value="#ffffff",
                        scale=2,
                    )
                palette_rows.append(row)
                palette_id_inputs.append(id_box)
                palette_color_inputs.append(color_picker)
            apply_palette_btn = gr.Button("Apply palette changes")
            generate_btn = gr.Button("Generate canvas")
        with gr.Column():
            processed_output = gr.Image(
                type="numpy",
                label="Processed image (facets)",
            )
            outlined_output = gr.Image(
                type="numpy",
                label="Outlined image",
            )
    canvas_state = gr.State()
    generate_btn.click(
        fn=generate_canvas,
        inputs=[
            input_image,
            orientation_dropdown,
            page_dropdown,
            n_colors_slider,
            min_facet_slider,
            narrow_thresh_slider,
            min_font_slider,
            max_font_slider,
            font_scale_slider,
            label_color_picker,
            outline_color_picker,
        ],
        outputs=[
            processed_output,
            outlined_output,
            canvas_state,
            *palette_rows,
            *palette_id_inputs,
            *palette_color_inputs,
        ],
    )
    apply_palette_btn.click(
        fn=apply_palette_updates,
        inputs=[canvas_state, *palette_id_inputs, *palette_color_inputs],
        outputs=[
            processed_output,
            outlined_output,
            canvas_state,
            *palette_rows,
            *palette_id_inputs,
            *palette_color_inputs,
        ],
    )


if __name__ == "__main__":
    demo.launch()

