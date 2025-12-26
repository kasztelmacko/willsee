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

print("Color Palette:")
print(canvas.color_palette.color_palette_dict)

labels_img = canvas.clustered_image.astype(np.int32)
print(f"\nLabels image shape: {labels_img.shape}")
print(f"Unique labels: {len(np.unique(labels_img))}")

def extract_facets(labels_img: np.ndarray, connectivity: int):
    facets_img = np.zeros_like(labels_img, dtype=np.int32)
    facets_dict = {}
    facet_id = 0

    structure = np.ones((3,3)) if connectivity == 2 else None

    for color in np.unique(labels_img):
        mask = labels_img == color
        labeled_pixels, num = label(mask, structure=structure)

        for i in range(1, num + 1):
            facet_id += 1

            facet_coordinates = np.where(labeled_pixels == i)
            facets_img[facet_coordinates] = facet_id

            facet_mask = np.zeros_like(labeled_pixels, dtype=bool)
            facet_mask[facet_coordinates] = True

            facets_dict[facet_id] = Facet.create_facet(
                facet_id=facet_id,
                facet_color_label=color,
                facet_mask=facet_mask,
            )

    return facets_img, facets_dict

facets_img, facets_dict = extract_facets(labels_img=labels_img, connectivity=2)
print(f"\nTotal facets extracted: {len(facets_dict)}")

print("\nSample facets:")
sample_ids = [2, 5, 751, 1400, 6427, 6434]
for facet_id in sample_ids:
    if facet_id in facets_dict:
        print(f"Facet {facet_id}: {repr(facets_dict[facet_id])}")