from dataclasses import dataclass
from typing import Any

import numpy as np

_LUMA_RED = 0.299
_LUMA_GREEN = 0.587
_LUMA_BLUE = 0.114
_LUMA_MIDPOINT = 128


def _luminance(rgb: tuple[int, int, int]) -> float:
    return _LUMA_RED * rgb[0] + _LUMA_GREEN * rgb[1] + _LUMA_BLUE * rgb[2]


def _bgr_to_rgb(color_bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return (color_bgr[2], color_bgr[1], color_bgr[0])


@dataclass(frozen=True)
class RegionVisual:
    """Per-region first-paint inputs for render_text_skia."""

    cleaned_mask: np.ndarray | None = None
    bubble_color_bgr: tuple[int, int, int] = (255, 255, 255)
    text_color_rgb: tuple[int, int, int] | None = None
    text_background_color: tuple[int, int, int] | None = None
    rotation_deg: float = 0.0
    vertical_stack: bool = False


def build_osb_region(bubble: dict[str, Any]) -> RegionVisual:
    """Derive OSB first-paint colors and optional text background."""
    is_dark_text = bubble.get("is_dark_text", True)
    text_color_rgb = bubble.get("text_color_rgb", None)
    text_background_color = None
    if bubble.get("needs_text_background"):
        if text_color_rgb:
            text_background_color = (
                (255, 255, 255)
                if _luminance(text_color_rgb) < _LUMA_MIDPOINT
                else (0, 0, 0)
            )
        else:
            text_background_color = (0, 0, 0) if is_dark_text else (255, 255, 255)
    return RegionVisual(
        cleaned_mask=None,
        bubble_color_bgr=(50, 50, 50) if is_dark_text else (255, 255, 255),
        text_color_rgb=text_color_rgb,
        text_background_color=text_background_color,
    )


def build_bubble_region(
    bubble: dict[str, Any],
    render_info: dict[str, Any] | None,
) -> RegionVisual:
    """Derive bubble first-paint mask and colors from cleaning output."""
    _ = bubble
    if not render_info:
        return RegionVisual()
    text_color_bgr = render_info.get("text_color_bgr")
    return RegionVisual(
        cleaned_mask=render_info.get("mask"),
        bubble_color_bgr=render_info.get("color", (255, 255, 255)),
        text_color_rgb=_bgr_to_rgb(text_color_bgr) if text_color_bgr else None,
    )
