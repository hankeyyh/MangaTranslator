from dataclasses import replace
from typing import Any

from PIL import Image

from core.config import RenderingConfig
from core.text.region_visual import build_bubble_region, build_osb_region
from core.text.text_renderer import render_text_skia
from utils.exceptions import FontError, RenderingError
from utils.logging import log_message


def generate_test_placeholders(
    sorted_bubble_data: list[dict[str, Any]],
    processed_bubbles_info: list[dict[str, Any]],
    bubble_render_config: RenderingConfig,
    osb_render_config: RenderingConfig,
    verbose: bool = False,
) -> list[str]:
    """
    Generates test placeholder text by probing the rendering engine.
    Finds the largest text string that fits in the bounding box.
    """
    translated_texts = []
    placeholder_long = "Lorem **ipsum** *dolor* sit amet, consectetur adipiscing elit."
    placeholder_short = "Lorem **ipsum** *dolor* sit amet..."
    placeholder_tiny = "Lorem..."
    placeholder_tiers = [
        placeholder_long,
        placeholder_short,
        placeholder_tiny,
    ]

    log_message(
        f"Test mode: generating placeholders for {len(sorted_bubble_data)} bubbles",
        always_print=True,
    )
    # Map for rendering info used in probe
    bubble_render_info_map_probe = {
        tuple(info["bbox"]): {
            "color": info["color"],
            "mask": info.get("mask"),
        }
        for info in processed_bubbles_info
        if "bbox" in info and "color" in info and "mask" in info
    }

    for i, bubble in enumerate(sorted_bubble_data):
        bbox = bubble["bbox"]
        is_outside_text = bubble.get("is_outside_text", False)
        render_config = (
            osb_render_config if is_outside_text else bubble_render_config
        )
        probe_config = replace(render_config, supersampling_factor=1)

        probe_info = bubble_render_info_map_probe.get(tuple(bbox), {})
        region = (
            build_osb_region(bubble)
            if is_outside_text
            else build_bubble_region(bubble, probe_info)
        )

        best_fit = (
            placeholder_tiny.rstrip(".") if is_outside_text else placeholder_tiny
        )  # fallback

        placeholder_tiers_to_use = [
            t.rstrip(".") if is_outside_text else t for t in placeholder_tiers
        ]

        # Use a tiny dummy canvas — layout_only skips all pixel work
        _probe_canvas = Image.new("RGBA", (bbox[2] - bbox[0], bbox[3] - bbox[1]))

        best_font_size = -1

        for text_tier in placeholder_tiers_to_use:
            test_text = text_tier.upper() if is_outside_text else text_tier

            try:
                rendered = render_text_skia(
                    pil_image=_probe_canvas,
                    text=test_text,
                    bbox=bbox,
                    font_dir=probe_config.font_dir,
                    cleaned_mask=region.cleaned_mask,
                    bubble_color_bgr=region.bubble_color_bgr,
                    config=probe_config,
                    verbose=verbose,
                    bubble_id=str(i + 1),
                    raise_on_safe_error=False,
                    layout_only=True,
                    fallback_padding_pixels=(
                        osb_render_config.padding_pixels if is_outside_text else None
                    ),
                )

                font_size = rendered.info.get("font_size", 0)
                if font_size > best_font_size:
                    best_font_size = font_size
                    best_fit = text_tier
                # Longest tier already fits at max size — no shorter tier can beat it
                if best_font_size >= probe_config.max_font_size:
                    break

            except (RenderingError, FontError) as e:
                log_message(
                    f"Probe rendering failed for tier '{text_tier}': {e}",
                    verbose=verbose,
                )
            except Exception as e:
                log_message(
                    f"Probe rendering unexpected error: {e}",
                    always_print=True,
                )

        translated_texts.append(best_fit)

    return translated_texts
