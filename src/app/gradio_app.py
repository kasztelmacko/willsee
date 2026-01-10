"""Gradio interface for generating paint-by-number canvases."""
from pathlib import Path
import sys
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from pbn.canvas.canvas import Canvas
import pbn.config.pbn_config as PBN_CONF

from __future__ import annotations

import gradio as gr
from PIL import Image



DEFAULT_PAGE_SIZE = "A4"
DEFAULT_ORIENTATION = "LANDSCAPE"
DEFAULT_N_COLORS = 15


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
):
    """Run the Canvas pipeline and return processed and outlined images."""
    _validate_page_orientation(canvas_page_size, canvas_orientation)

    rgb_image = image.convert("RGB")
    canvas = Canvas.create_canvas(
        input_image=rgb_image,
        canvas_orientation=canvas_orientation,
        canvas_page_size=canvas_page_size,
        n_colors=int(n_colors),
    )
    return canvas.processed_image, canvas.outlined_image


with gr.Blocks(title="Paint-by-Number Canvas") as demo:
    gr.Markdown(
        "Upload an image and generate paint-by-number processed and outlined views."
    )
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
    generate_btn.click(
        fn=generate_canvas,
        inputs=[input_image, orientation_dropdown, page_dropdown, n_colors_slider],
        outputs=[processed_output, outlined_output],
    )


if __name__ == "__main__":
    demo.launch()

