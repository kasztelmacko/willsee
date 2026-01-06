import numpy as np
from PIL import Image


def prepare_image(
    image: Image,
    canvas_orientation: str,
    canvas_page_size: str,
    canvas_size_config: dict
) -> np.ndarray:
    """
    Resize the input image to the configured canvas dimensions and return
    a uint8 RGB array.
    """
    width = canvas_size_config[canvas_page_size][canvas_orientation]["WIDTH"]
    height = canvas_size_config[canvas_page_size][canvas_orientation]["HEIGHT"]

    return np.array(
        image.resize((width, height), resample=Image.LANCZOS),
        dtype=np.uint8,
    )

