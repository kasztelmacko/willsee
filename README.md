## Image Facet Inspector

A small prototype tool for inspecting clustered images. It allows you to click
on a pixel and highlights the connected facet of pixels that have exactly the
same RGB value as the clicked pixel (4-connected neighborhood flood fill).

### How to run

1. Install dependencies (if you use `uv`, from the project root you can run):

```bash
uv sync
```

Or with plain `pip`:

```bash
pip install -e .[all]
```

2. From the project root, launch the inspector:

```bash
python -m inspect_tool.app
```

This starts a local Gradio web app in your browser. Click anywhere on the image
to see the corresponding facet highlighted in red.


