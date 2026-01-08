import sys
from pathlib import Path

from PIL import Image

src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from pbn.canvas.canvas import Canvas  # noqa: E402

N_CLUSTERS = 15
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

processed_path = Path("data/input_image/processed_canvas.png")
outlined_path = Path("data/input_image/processed_canvas_outlined.png")

Image.fromarray(processed_facets_img, mode="RGB").save(processed_path)
Image.fromarray(canvas.outlined_image, mode="RGB").save(outlined_path)
print("Original palette:", canvas.color_pallete.to_dict())

palette = canvas.color_pallete
palette.rename_key(1, "A")
palette.adjust_color(3, (0, 0, 0))

updated_canvas = canvas.render_image_with_replaced_palette(palette)

print("Updated palette:", updated_canvas.color_pallete.to_dict())

# Save updated outputs
Image.fromarray(updated_canvas.processed_image, mode="RGB").save("data/input_image/processed_canvas_recolored.png")
Image.fromarray(updated_canvas.outlined_image, mode="RGB").save("data/input_image/processed_canvas_outlined_recolored.png")