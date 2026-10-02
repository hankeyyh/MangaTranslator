from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from PIL import Image

from core.config import RenderingConfig
from core.image.cleaning import retry_cleaning_with_otsu
from core.text.region_visual import RegionVisual
from core.text.text_renderer import render_text_skia
from utils.exceptions import FontError, ImageProcessingError, RenderingError
from utils.logging import log_message

_UNCHANGED_OCR_BLOCKLIST = {
    "[OCR FAILED]",
    "[Empty response / no content]",
}


@dataclass(frozen=True)
class RenderResult:
    image: Image.Image
    success: bool
    text: str | None = None


@dataclass
class BubbleRetryContext:
    original_cv_image: np.ndarray
    render_info: dict[str, Any] | None
    thresholding_value: int
    roi_shrink_px: int
    processing_scale: float
    classify_colored: bool


def _paint(
    image: Image.Image,
    text: str,
    bbox: tuple[int, int, int, int],
    region: RegionVisual,
    config: RenderingConfig,
    *,
    raise_on_safe_error: bool,
    verbose: bool,
    region_id: str | None,
    rotation_deg: float | None = None,
    vertical_stack: bool | None = None,
    fallback_padding_pixels: float | None = None,
    use_detected_text_color: bool = True,
) -> Image.Image:
    return render_text_skia(
        pil_image=image,
        text=text,
        bbox=bbox,
        font_dir=config.font_dir,
        cleaned_mask=region.cleaned_mask,
        bubble_color_bgr=region.bubble_color_bgr,
        config=config,
        verbose=verbose,
        bubble_id=region_id,
        rotation_deg=region.rotation_deg if rotation_deg is None else rotation_deg,
        vertical_stack=(
            region.vertical_stack if vertical_stack is None else vertical_stack
        ),
        text_color_rgb=region.text_color_rgb if use_detected_text_color else None,
        raise_on_safe_error=raise_on_safe_error,
        text_background_color=region.text_background_color,
        fallback_padding_pixels=fallback_padding_pixels,
    )


def _paste_original_crop(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    original_crop: Image.Image,
) -> Image.Image:
    restored = image.copy()
    restored.paste(original_crop, (bbox[0], bbox[1]))
    return restored


def _restore_osb_patch(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    original_crop: Image.Image | None,
    text: str,
    verbose: bool,
) -> RenderResult:
    if original_crop is None:
        return RenderResult(image, False, text)
    log_message(
        f"Restoring original OSB patch for {bbox}",
        verbose=verbose,
        always_print=True,
    )
    return RenderResult(_paste_original_crop(image, bbox, original_crop), True, text)


def render_outside_text(
    image: Image.Image,
    text: str,
    bbox: tuple[int, int, int, int],
    region: RegionVisual,
    config: RenderingConfig,
    original_crop: Image.Image | None = None,
    ocr_text: str = "",
    *,
    verbose: bool = False,
    region_id: str | None = None,
) -> RenderResult:
    ocr_text = (ocr_text or "").strip()
    if (
        ocr_text
        and ocr_text not in _UNCHANGED_OCR_BLOCKLIST
        and ocr_text == text.strip()
        and original_crop is not None
    ):
        log_message(
            "Restoring original OSB patch because OCR matches translation "
            f"for {bbox}",
            verbose=verbose,
            always_print=True,
        )
        return RenderResult(_paste_original_crop(image, bbox, original_crop), True)

    text = text.upper()
    log_message(
        f"Rendering outside text {bbox}: '{text[:30]}...'",
        verbose=verbose,
    )
    try:
        painted = _paint(
            image,
            text,
            bbox,
            region,
            config,
            raise_on_safe_error=False,
            verbose=verbose,
            region_id=region_id,
            fallback_padding_pixels=config.padding_pixels,
        )
        return RenderResult(painted, True, text)
    except Exception as e:
        log_message(f"Text rendering failed: {e}", verbose=verbose)
        if region.vertical_stack:
            return _restore_osb_patch(image, bbox, original_crop, text, verbose)
        try:
            log_message(
                "OSB render failed, retrying with vertical-stack fallback",
                verbose=verbose,
            )
            painted = _paint(
                image,
                text,
                bbox,
                region,
                config,
                raise_on_safe_error=False,
                verbose=verbose,
                region_id=region_id,
                rotation_deg=0.0,
                vertical_stack=True,
                fallback_padding_pixels=config.padding_pixels,
            )
            log_message("Vertical-stack fallback succeeded", verbose=verbose)
            return RenderResult(painted, True, text)
        except Exception as e2:
            log_message(
                f"Vertical-stack fallback failed: {e2}",
                verbose=verbose,
            )
            return _restore_osb_patch(image, bbox, original_crop, text, verbose)


def render_bubble_text(
    image: Image.Image,
    text: str,
    bbox: tuple[int, int, int, int],
    region: RegionVisual,
    config: RenderingConfig,
    retry: BubbleRetryContext | None = None,
    *,
    verbose: bool = False,
    region_id: str | None = None,
) -> RenderResult:
    log_message(
        f"Rendering bubble {bbox}: '{text[:30]}...'",
        verbose=verbose,
    )
    try:
        painted = _paint(
            image,
            text,
            bbox,
            region,
            config,
            raise_on_safe_error=True,
            verbose=verbose,
            region_id=region_id,
        )
        return RenderResult(painted, True)
    except ImageProcessingError as e:
        safe_area_failed = "Safe area calculation failed" in str(e)
        retry_result = None
        render_info = retry.render_info if retry else None
        base_mask = render_info.get("base_mask") if render_info else None
        if retry is not None and safe_area_failed and base_mask is not None:
            log_message(
                f"Safe area failed for bubble {bbox}, retrying mask with Otsu",
                verbose=verbose,
                always_print=True,
            )
            retry_result = retry_cleaning_with_otsu(
                retry.original_cv_image,
                {
                    "base_mask": base_mask,
                    "bbox": bbox,
                    "is_sam": render_info.get("is_sam", False) if render_info else False,
                    "is_colored": (
                        render_info.get("is_colored", False) if render_info else False
                    ),
                    "text_bbox": render_info.get("text_bbox") if render_info else None,
                    "text_color_bgr": (
                        render_info.get("text_color_bgr") if render_info else None
                    ),
                },
                retry.thresholding_value,
                retry.roi_shrink_px,
                retry.processing_scale,
                verbose=verbose,
                classify_colored=retry.classify_colored,
            )

        rendered_image = image
        success = False
        if retry_result and retry_result.get("mask") is not None:
            region = replace(
                region,
                cleaned_mask=retry_result["mask"],
                bubble_color_bgr=retry_result.get("color", region.bubble_color_bgr),
            )
            if render_info is not None:
                render_info.update(
                    {
                        "mask": region.cleaned_mask,
                        "color": region.bubble_color_bgr,
                        "base_mask": retry_result.get(
                            "base_mask", render_info.get("base_mask")
                        ),
                        "is_colored": retry_result.get(
                            "is_colored",
                            render_info.get("is_colored", False),
                        ),
                        "text_bbox": retry_result.get(
                            "text_bbox", render_info.get("text_bbox")
                        ),
                    }
                )
            try:
                rendered_image = _paint(
                    image,
                    text,
                    bbox,
                    region,
                    config,
                    raise_on_safe_error=False,
                    verbose=verbose,
                    region_id=region_id,
                    use_detected_text_color=False,
                )
                success = True
            except (RenderingError, FontError, ImageProcessingError) as e2:
                log_message(
                    f"Text rendering failed after Otsu retry: {e2}",
                    verbose=verbose,
                )
                rendered_image = image
                success = False

        if not success:
            fallback_msg = (
                f"Safe area calculation failed for {bbox}, using padded bbox fallback"
                if safe_area_failed
                else f"Rendering retry fallback for {bbox}, using padded bbox method"
            )
            log_message(fallback_msg, verbose=verbose)
            try:
                rendered_image = _paint(
                    image,
                    text,
                    bbox,
                    region,
                    config,
                    raise_on_safe_error=False,
                    verbose=verbose,
                    region_id=region_id,
                    use_detected_text_color=False,
                )
                success = True
            except (RenderingError, FontError) as e2:
                log_message(f"Text rendering failed: {e2}", verbose=verbose)
                rendered_image = image
                success = False
        return RenderResult(rendered_image, success)
    except (RenderingError, FontError) as e:
        log_message(f"Text rendering failed: {e}", verbose=verbose)
        return RenderResult(image, False)
