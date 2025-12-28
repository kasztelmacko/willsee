import sys
from pathlib import Path

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from PIL import Image
import numpy as np
from scipy.ndimage import label
from pbn.canvas.canvas import Canvas
from pbn.canvas.facet import Facet

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

labels_img = canvas.clustered_image
print(f"\nLabels image shape: {labels_img.shape}")
print(f"Unique labels: {len(np.unique(labels_img))}")

processed_facets_img = canvas.processed_image
print(f"\nProcessed image shape: {processed_facets_img.shape}")
print(f"Unique facet IDs in processed image: {len(np.unique(processed_facets_img))}")

processed_path = Path("data/input_image/processed_canvas.png")
clustered_path = Path("data/input_image/clustered.png")

Image.fromarray(processed_facets_img, mode="RGB").save(processed_path)

