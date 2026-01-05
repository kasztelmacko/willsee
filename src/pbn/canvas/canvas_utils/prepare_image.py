import numpy as np
from PIL import Image

from pbn.config.pbn_config import CANVAS_SIZE_CONFIG


def prepare_image(
    image: Image,
    canvas_orientation: str,
    canvas_page_size: str,
) -> np.ndarray:
    """
    Resize the input image to the configured canvas dimensions and return
    a uint8 RGB array.
    """
    width = CANVAS_SIZE_CONFIG[canvas_page_size][canvas_orientation]["WIDTH"]
    height = CANVAS_SIZE_CONFIG[canvas_page_size][canvas_orientation]["HEIGHT"]

    return np.array(
        image.resize((width, height), resample=Image.LANCZOS),
        dtype=np.uint8,
    )

