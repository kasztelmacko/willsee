from __future__ import annotations

from typing import Hashable, Iterable, Mapping


def _normalize_color(color: Iterable[int]) -> tuple[int, int, int]:
    """Convert arbitrary iterable (np array, list, tuple) into an RGB tuple of ints."""
    r, g, b = color
    return int(r), int(g), int(b)


class ColorPalette:
    """
    Mutable palette that maps arbitrary keys (labels) to RGB colors and
    provides helpers to remap facet colors to labels.
    """

    def __init__(self, mapping: Mapping[Hashable, Iterable[int]]):
        key_to_rgb = {key: _normalize_color(rgb) for key, rgb in mapping.items()}
        color_to_key = {rgb: key for key, rgb in key_to_rgb.items()}
        self._key_to_rgb: dict[Hashable, tuple[int, int, int]] = key_to_rgb
        self._color_to_key: dict[tuple[int, int, int], Hashable] = color_to_key

    @classmethod
    def from_facet_colors(cls, facet_colors: Iterable[Iterable[int]]) -> "ColorPalette":
        """Create a palette with numeric keys starting at 1, based on unique facet colors."""
        unique_colors = sorted({_normalize_color(color) for color in facet_colors})
        mapping = {idx + 1: color for idx, color in enumerate(unique_colors)}
        return cls(mapping)

    def to_dict(self) -> dict[Hashable, tuple[int, int, int]]:
        """Return a shallow copy of the palette mapping."""
        return dict[Hashable, tuple[int, int, int]](self._key_to_rgb)

    def label_for_color(self, color: Iterable[int]) -> Hashable:
        """Return the palette key for the given color, adding a new numeric key if missing."""
        rgb = _normalize_color(color)
        key = self._color_to_key.get(rgb)
        if key is not None:
            return key

        next_key = self._next_numeric_key()
        self._key_to_rgb[next_key] = rgb
        self._color_to_key[rgb] = next_key
        return next_key

    def labels_for_colors(self, colors: Iterable[Iterable[int]]) -> list[Hashable]:
        """Map a collection of colors to palette keys, expanding the palette as needed."""
        return [self.label_for_color(color) for color in colors]

    def adjust_color(self, key: Hashable, new_rgb: Iterable[int]) -> None:
        """Update the RGB value for a given key. Overwrites existing mapping if present."""
        normalized = _normalize_color(new_rgb)
        self._key_to_rgb[key] = normalized
        self._color_to_key[normalized] = key

    def rename_key(self, old_key: Hashable, new_key: Hashable) -> None:
        """Change the key associated with an existing color."""
        if old_key not in self._key_to_rgb:
            raise KeyError(f"Palette has no key '{old_key}'")
        rgb = self._key_to_rgb.pop(old_key)
        self._key_to_rgb[new_key] = rgb
        for color, key in list[tuple[tuple[int, int, int], Hashable]](self._color_to_key.items()):
            if key == old_key:
                self._color_to_key[color] = new_key

    def _next_numeric_key(self) -> int:
        """Generate the next available positive integer key."""
        numeric_keys = [key for key in self._key_to_rgb if isinstance(key, int)]
        next_id = max(numeric_keys, default=0) + 1
        while next_id in self._key_to_rgb:
            next_id += 1
        return next_id

    def rgb_for_key(self, key: Hashable) -> tuple[int, int, int]:
        """Get the RGB tuple for a given key."""
        return self._key_to_rgb[key]

