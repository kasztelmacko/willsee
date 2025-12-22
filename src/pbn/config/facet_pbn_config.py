"""
FACET_SIZE_FONT_SIZE_CONFIG: config to determine facet label point font size based on facet size.
    Config format: {font_size: (min_size, max_size)}, where None in max_size means infinity.
"""

FACET_SIZE_FONT_SIZE_CONFIG = {
    8: (0, 50),
    12: (51, 200),
    16: (200, None),
}