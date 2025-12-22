from dataclasses import dataclass, field


@dataclass(frozen=True)
class PalettePBN:
    """
    Color palette mapping color IDs to RGB values.
    
    Attributes:
        color_map: Dictionary mapping color_id (int) to RGB tuple (int, int, int)
    """
    color_map: dict[int, tuple[int, int, int]]
    _rgb_to_id: dict[tuple[int, int, int], int] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Cache reverse mapping for RGB to color_id lookup."""
        rgb_to_id = {rgb: color_id for color_id, rgb in self.color_map.items()}
        object.__setattr__(self, '_rgb_to_id', rgb_to_id)

    def __getitem__(self, key: int | tuple[int, int, int]) -> tuple[int, int, int] | int:
        """Get RGB value for a color ID, or get color ID for an RGB tuple."""
        if isinstance(key, tuple):
            return self._rgb_to_id[key]
        else:
            return self.color_map[key]

    def __contains__(self, key: int | tuple[int, int, int]) -> bool:
        """Check if a color ID or RGB tuple exists in the palette."""
        if isinstance(key, tuple):
            return key in self._rgb_to_id
        else:
            return key in self.color_map

    def __len__(self) -> int:
        """Return the number of colors in the palette."""
        return len(self.color_map)

    def __iter__(self):
        """Iterate over color IDs."""
        return iter(self.color_map)

    def items(self):
        """Return (color_id, rgb) pairs."""
        return self.color_map.items()

    def keys(self):
        """Return color IDs."""
        return self.color_map.keys()

    def values(self):
        """Return RGB values."""
        return self.color_map.values()

    def get_color_to_id_mapping(self) -> dict[tuple[int, int, int], int]:
        """Return reverse mapping from RGB tuple to color ID."""
        return self._rgb_to_id

