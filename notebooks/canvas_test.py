from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import numpy as np
from io import BytesIO
import base64
from flask import Flask, Response

from small_regions_processing import (
    merge_facets,
    highlight_narrow_facets,
    compute_narrow_component_ids_from_array,
    compute_small_component_ids,
    highlight_facets_by_ids,
    compute_facet_label_data,
)
from image_outline_processing import compute_outline_mask


N_COLORS = 20
MIN_FACET_PIXELS_SIZE = 50
NARROW_FACET_THRESHOLD_PX = 10

INPUT_IMAGE_PATH = "data/input_image/zuza_5.jpg"

A4_PORTRAIT_WIDTH_PX = 1800
A4_PORTRAIT_HEIGHT_PX = int(A4_PORTRAIT_WIDTH_PX * 297 / 210)

A4_LANDSCAPE_WIDTH_PX = 2400
A4_LANDSCAPE_HEIGHT_PX = int(A4_LANDSCAPE_WIDTH_PX * 210 / 297)


def build_clustered_image():
    """Load the image, cluster by color, and return:
    - the resized original RGB image,
    - the clustered image,
    - a version with small facets highlighted in red,
    - a version with small facets merged into surrounding regions,
    - a version with narrow facets highlighted in red (after small merge),
    - a final version with narrow facets merged,
    - a 1‑pixel‑wide outline image showing region boundaries (from final version).
    """
    with Image.open(fp=INPUT_IMAGE_PATH) as im:
        im = im.convert("RGB")

        width, height = A4_PORTRAIT_WIDTH_PX, A4_PORTRAIT_HEIGHT_PX
        # width, height = A4_LANDSCAPE_WIDTH_PX, A4_LANDSCAPE_HEIGHT_PX

        target_img = im.resize((width, height), resample=Image.LANCZOS)
        target = np.array(target_img, dtype=np.uint8)
        pixels = target.reshape(-1, 3)

    kmeans = KMeans(n_clusters=N_COLORS, random_state=123)
    kmeans.fit(pixels)

    labels = kmeans.labels_
    clustered_pixels = kmeans.cluster_centers_[labels].astype(np.uint8)
    clustered_array = clustered_pixels.reshape(height, width, 3)

    original_rgb_img = Image.fromarray(target, mode="RGB")
    clustered_img = Image.fromarray(clustered_array, mode="RGB")

    # 1) Highlight and remove small facets.
    small_merge_ids = compute_small_component_ids(clustered_array, MIN_FACET_PIXELS_SIZE)
    small_highlight_array = highlight_facets_by_ids(clustered_array, small_merge_ids)
    small_merged_array = merge_facets(clustered_array, small_merge_ids)
    small_highlight_img = Image.fromarray(small_highlight_array, mode="RGB")
    small_merged_img = Image.fromarray(small_merged_array, mode="RGB")

    # 2) Highlight narrow facets after small-merge (for visualization).
    narrow_highlight_array = highlight_narrow_facets(
        small_merged_array, NARROW_FACET_THRESHOLD_PX
    )
    narrow_highlight_img = Image.fromarray(narrow_highlight_array, mode="RGB")
    # 3) Remove narrow facets (size-based merging disabled in this pass).
    narrow_ids = compute_narrow_component_ids_from_array(
        small_merged_array, NARROW_FACET_THRESHOLD_PX
    )
    final_array = merge_facets(small_merged_array, narrow_ids)
    final_img = Image.fromarray(final_array, mode="RGB")

    outline_mask = compute_outline_mask(final_array)
    outline_rgb = np.full_like(final_array, fill_value=255, dtype=np.uint8)
    outline_rgb[outline_mask] = np.array([0, 0, 0], dtype=np.uint8)
    outline_img = Image.fromarray(outline_rgb, mode="RGB")

    # Compute label positions and render numbered overlay on the fully processed image.
    (
        labels_img,
        centers,
        palette_indices,
        font_sizes,
        _component_colors,
    ) = compute_facet_label_data(
        final_array, min_font_px=8, max_font_px=22, font_scale=0.4
    )

    label_overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(label_overlay)

    def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size=size)
        except OSError:
            return ImageFont.load_default()

    for (y, x), palette_idx, font_size in zip(centers, palette_indices, font_sizes):
        font = _load_font(font_size)
        text = str(palette_idx)
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=0)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pos = (x - tw / 2, y - th / 2)
        draw.text(
            pos,
            text,
            font=font,
            fill=(0, 0, 0, 255),
            stroke_width=0,
        )

    # Save a flattened final image with labels overlaid for offline use.
    final_with_labels = final_img.convert("RGBA")
    final_with_labels.alpha_composite(label_overlay)
    final_with_labels.convert("RGB").save("data/input_image/zuza_5_output.png")

    return (
        original_rgb_img,
        clustered_img,
        small_highlight_img,
        small_merged_img,
        narrow_highlight_img,
        final_img,
        outline_img,
        label_overlay,
        width,
        height,
    )


app = Flask(__name__)

(
    ORIGINAL_IMG,
    CLUSTERED_IMG,
    SMALL_HIGHLIGHT_IMG,
    SMALL_MERGED_IMG,
    NARROW_IMG,
    FINAL_IMG,
    OUTLINE_IMG,
    LABEL_IMG,
    WIDTH,
    HEIGHT,
) = build_clustered_image()


@app.route("/")
def index() -> Response:
    """Serve an HTML page with the original, processed, and outline images plus a pixel inspector."""
    buf_orig = BytesIO()
    ORIGINAL_IMG.save(buf_orig, format="PNG")
    data_orig = base64.b64encode(buf_orig.getvalue()).decode("ascii")

    buf_clustered = BytesIO()
    CLUSTERED_IMG.save(buf_clustered, format="PNG")
    data_clustered = base64.b64encode(buf_clustered.getvalue()).decode("ascii")

    buf_small_high = BytesIO()
    SMALL_HIGHLIGHT_IMG.save(buf_small_high, format="PNG")
    data_small_high = base64.b64encode(buf_small_high.getvalue()).decode("ascii")

    buf_small = BytesIO()
    SMALL_MERGED_IMG.save(buf_small, format="PNG")
    data_small = base64.b64encode(buf_small.getvalue()).decode("ascii")

    buf_narrow = BytesIO()
    NARROW_IMG.save(buf_narrow, format="PNG")
    data_narrow = base64.b64encode(buf_narrow.getvalue()).decode("ascii")

    buf_final = BytesIO()
    FINAL_IMG.save(buf_final, format="PNG")
    data_final = base64.b64encode(buf_final.getvalue()).decode("ascii")

    buf_outline = BytesIO()
    OUTLINE_IMG.save(buf_outline, format="PNG")
    data_outline = base64.b64encode(buf_outline.getvalue()).decode("ascii")

    buf_labels = BytesIO()
    LABEL_IMG.save(buf_labels, format="PNG")
    data_labels = base64.b64encode(buf_labels.getvalue()).decode("ascii")

    html = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Pixel Inspector</title>
    <style>
      body {{
        margin: 16px;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
          sans-serif;
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(2, auto);
        gap: 16px;
        align-items: flex-start;
      }}
      canvas {{
        border: 1px solid #ccc;
        image-rendering: pixelated;
        cursor: crosshair;
        display: block;
      }}
      #info {{
        margin-top: 8px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
          "Liberation Mono", "Courier New", monospace;
      }}
    </style>
  </head>
  <body>
    <div class="grid">
      <div>
        <div>Original (resized)</div>
        <canvas id="origCanvas" width="{WIDTH}" height="{HEIGHT}"></canvas>
      </div>
      <div>
        <div>Clustered</div>
        <canvas id="clusteredCanvas" width="{WIDTH}" height="{HEIGHT}"></canvas>
      </div>
      <div>
        <div>Small facets highlighted</div>
        <canvas id="smallHighCanvas" width="{WIDTH}" height="{HEIGHT}"></canvas>
      </div>
      <div>
        <div>After small merge</div>
        <canvas id="smallCanvas" width="{WIDTH}" height="{HEIGHT}"></canvas>
      </div>
      <div>
        <div>Narrow facets highlighted</div>
        <canvas id="narrowCanvas" width="{WIDTH}" height="{HEIGHT}"></canvas>
      </div>
      <div>
        <div>After narrow merge</div>
        <canvas id="finalCanvas" width="{WIDTH}" height="{HEIGHT}"></canvas>
      </div>
      <div>
        <div>Outline mask</div>
        <canvas id="outlineCanvas" width="{WIDTH}" height="{HEIGHT}"></canvas>
      </div>
      <div>
        <div>Outline + labels</div>
        <canvas id="outline2Canvas" width="{WIDTH}" height="{HEIGHT}"></canvas>
      </div>
      <div>
        <div>Labeled facets</div>
        <canvas id="labelsCanvas" width="{WIDTH}" height="{HEIGHT}"></canvas>
      </div>
    </div>
    <div id="info">
      <div>Image: <span id="which">-</span></div>
      <div>Pos: <span id="pos">(x, y)</span></div>
      <div>RGB: <span id="rgb">(r, g, b)</span></div>
    </div>
    <script>
      const origImg = new Image();
      const clusteredImg = new Image();
      const smallHighImg = new Image();
      const smallImg = new Image();
      const narrowImg = new Image();
      const finalImg = new Image();
      const outlineImg = new Image();
      const labelsImg = new Image();
      origImg.src = "data:image/png;base64,{data_orig}";
      clusteredImg.src = "data:image/png;base64,{data_clustered}";
      smallHighImg.src = "data:image/png;base64,{data_small_high}";
      smallImg.src = "data:image/png;base64,{data_small}";
      narrowImg.src = "data:image/png;base64,{data_narrow}";
      finalImg.src = "data:image/png;base64,{data_final}";
      outlineImg.src = "data:image/png;base64,{data_outline}";
      labelsImg.src = "data:image/png;base64,{data_labels}";

      const origCanvas = document.getElementById("origCanvas");
      const clusteredCanvas = document.getElementById("clusteredCanvas");
      const smallHighCanvas = document.getElementById("smallHighCanvas");
      const smallCanvas = document.getElementById("smallCanvas");
      const narrowCanvas = document.getElementById("narrowCanvas");
      const finalCanvas = document.getElementById("finalCanvas");
      const outlineCanvas = document.getElementById("outlineCanvas");
      const outline2Canvas = document.getElementById("outline2Canvas");
      const labelsCanvas = document.getElementById("labelsCanvas");
      const origCtx = origCanvas.getContext("2d");
      const clusteredCtx = clusteredCanvas.getContext("2d");
      const smallHighCtx = smallHighCanvas.getContext("2d");
      const smallCtx = smallCanvas.getContext("2d");
      const narrowCtx = narrowCanvas.getContext("2d");
      const finalCtx = finalCanvas.getContext("2d");
      const outlineCtx = outlineCanvas.getContext("2d");
      const outline2Ctx = outline2Canvas.getContext("2d");
      const labelsCtx = labelsCanvas.getContext("2d");
      origCtx.imageSmoothingEnabled = false;
      clusteredCtx.imageSmoothingEnabled = false;
      smallHighCtx.imageSmoothingEnabled = false;
      smallCtx.imageSmoothingEnabled = false;
      narrowCtx.imageSmoothingEnabled = false;
      finalCtx.imageSmoothingEnabled = false;
      outlineCtx.imageSmoothingEnabled = false;
      outline2Ctx.imageSmoothingEnabled = false;
      labelsCtx.imageSmoothingEnabled = false;

      const whichEl = document.getElementById("which");
      const posEl = document.getElementById("pos");
      const rgbEl = document.getElementById("rgb");

      function attachInspector(canvas, ctx, whichLabel) {{
        canvas.addEventListener("mousemove", function (e) {{
          const rect = canvas.getBoundingClientRect();
          const x = Math.floor(e.clientX - rect.left);
          const y = Math.floor(e.clientY - rect.top);
          if (x < 0 || x >= canvas.width || y < 0 || y >= canvas.height) return;

          const p = ctx.getImageData(x, y, 1, 1).data;
          whichEl.textContent = whichLabel;
          posEl.textContent = "(" + x + ", " + y + ")";
          rgbEl.textContent = "(" + p[0] + ", " + p[1] + ", " + p[2] + ")";
        }});
      }}

      origImg.onload = function () {{
        origCtx.drawImage(origImg, 0, 0);
        attachInspector(origCanvas, origCtx, "original");
      }};

      clusteredImg.onload = function () {{
        clusteredCtx.drawImage(clusteredImg, 0, 0);
        attachInspector(clusteredCanvas, clusteredCtx, "clustered");
      }};

      smallHighImg.onload = function () {{
        smallHighCtx.drawImage(smallHighImg, 0, 0);
        attachInspector(smallHighCanvas, smallHighCtx, "small-highlight");
      }};

      smallImg.onload = function () {{
        smallCtx.drawImage(smallImg, 0, 0);
        attachInspector(smallCanvas, smallCtx, "small-merged");
      }};

      narrowImg.onload = function () {{
        narrowCtx.drawImage(narrowImg, 0, 0);
        attachInspector(narrowCanvas, narrowCtx, "narrow-highlight");
      }};

      finalImg.onload = function () {{
        finalCtx.drawImage(finalImg, 0, 0);
        attachInspector(finalCanvas, finalCtx, "final");
      }};

      outlineImg.onload = function () {{
        outlineCtx.drawImage(outlineImg, 0, 0);
        attachInspector(outlineCanvas, outlineCtx, "outline");
      }};

      labelsImg.onload = function () {{
        labelsCtx.drawImage(finalImg, 0, 0);
        labelsCtx.drawImage(labelsImg, 0, 0);
        attachInspector(labelsCanvas, labelsCtx, "labeled");
        // Also render labels over the outline for comparison.
        outline2Ctx.drawImage(outlineImg, 0, 0);
        outline2Ctx.drawImage(labelsImg, 0, 0);
        attachInspector(outline2Canvas, outline2Ctx, "outline-labeled");
      }};
    </script>
  </body>
</html>
"""
    return Response(html, mimetype="text/html")


if __name__ == "__main__":
    app.run(debug=True)