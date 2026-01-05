import sys
from pathlib import Path

from PIL import Image
import numpy as np

src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from pbn.canvas.canvas import Canvas  # noqa: E402

N_CLUSTERS = 20
CANVAS_PAGE_SIZE = "A4"
CANVAS_ORIENTATION = "LANDSCAPE"


image_path = Path("data/input_image/2024082.jpg")

with Image.open(image_path) as im:
    input_image = im.convert("RGB")

canvas = Canvas.create_canvas(
    input_image=input_image,
    canvas_orientation=CANVAS_ORIENTATION,
    canvas_page_size=CANVAS_PAGE_SIZE,
    n_colors=N_CLUSTERS
)


processed_facets_img = canvas.processed_image
print(f"\nProcessed image shape: {processed_facets_img.shape}")
print(f"Unique facet ids (processed): {np.unique(canvas.processed_facets).size}")

processed_path = Path("data/input_image/processed_canvas.png")

Image.fromarray(processed_facets_img, mode="RGB").save(processed_path)

Image.fromarray(canvas.outlined_image, mode="RGB").save("data/input_image/processed_canvas_outlined.png")