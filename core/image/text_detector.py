"""DBNet raw text mask used by LaMa OSB inpainting.

Inference is limited to already-determined OSB boxes. The full page is not
scanned; box-exterior pixels stay empty and are unused by stroke refine.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from core.ml.model_manager import get_model_manager
from utils.logging import log_message

_PAD_MULTIPLE = 256
_MIN_INFER_SIDE = 256
# Extra context around each OSB box so glyphs on the border are not cropped.
_CROP_PAD_RATIO = 0.1
_CROP_PAD_MIN_PX = 32

Box = tuple[int, int, int, int]


def _to_rgb_uint8(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    image_np = np.asarray(image)
    if image_np.ndim == 2:
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    if image_np.shape[2] == 4:
        return image_np[:, :, :3]
    return image_np


def _clamp_box(box: Box, img_w: int, img_h: int) -> Box | None:
    x0, y0, x1, y1 = (int(v) for v in box)
    x0 = max(0, min(img_w, x0))
    x1 = max(0, min(img_w, x1))
    y0 = max(0, min(img_h, y0))
    y1 = max(0, min(img_h, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _expand_box(box: Box, img_w: int, img_h: int) -> Box | None:
    x0, y0, x1, y1 = box
    pad_x = max(_CROP_PAD_MIN_PX, int(round((x1 - x0) * _CROP_PAD_RATIO)))
    pad_y = max(_CROP_PAD_MIN_PX, int(round((y1 - y0) * _CROP_PAD_RATIO)))
    return _clamp_box((x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y), img_w, img_h)


def _boxes_overlap(a: Box, b: Box) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _union_box(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _merge_overlapping_boxes(boxes: list[Box]) -> list[Box]:
    """Merge overlapping windows so nearby OSB boxes share one DBNet forward."""
    pending = [box for box in boxes if box[2] > box[0] and box[3] > box[1]]
    if not pending:
        return []

    changed = True
    while changed:
        changed = False
        merged: list[Box] = []
        for box in pending:
            placed = False
            for i, existing in enumerate(merged):
                if _boxes_overlap(box, existing):
                    merged[i] = _union_box(box, existing)
                    placed = True
                    changed = True
                    break
            if not placed:
                merged.append(box)
        pending = merged
    return pending


def _inference_windows(
    boxes: list[Box], img_w: int, img_h: int
) -> list[Box]:
    windows: list[Box] = []
    for box in boxes:
        clamped = _clamp_box(box, img_w, img_h)
        if clamped is None:
            continue
        expanded = _expand_box(clamped, img_w, img_h)
        if expanded is not None:
            windows.append(expanded)
    return _merge_overlapping_boxes(windows)


def _resize_for_dbnet(
    image: np.ndarray, detect_size: int
) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    ratio = detect_size / float(max(height, width))
    target_h = max(1, round(height * ratio))
    target_w = max(1, round(width * ratio))
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    pad_h = (
        0
        if target_h % _PAD_MULTIPLE == 0
        else _PAD_MULTIPLE - (target_h % _PAD_MULTIPLE)
    )
    pad_w = (
        0
        if target_w % _PAD_MULTIPLE == 0
        else _PAD_MULTIPLE - (target_w % _PAD_MULTIPLE)
    )
    if pad_h == 0 and pad_w == 0:
        return resized, 0, 0

    canvas = np.zeros((target_h + pad_h, target_w + pad_w, 3), dtype=np.uint8)
    canvas[:target_h, :target_w] = resized
    return canvas, pad_w, pad_h


def _infer_dbnet_mask(
    image_rgb: np.ndarray,
    detect_size: int,
    model,
    device: torch.device,
    verbose: bool = False,
) -> np.ndarray:
    """Run DBNet on one RGB crop and return a uint8 mask at crop size."""
    orig_h, orig_w = image_rgb.shape[:2]
    longest = max(orig_h, orig_w)
    # Downscale large windows; do not upscale past native size except a floor
    # so ResNet still sees a usable feature map.
    infer_size = min(max(256, int(detect_size)), max(longest, _MIN_INFER_SIDE))

    filtered = cv2.bilateralFilter(image_rgb, 17, 80, 80)
    resized, pad_w, pad_h = _resize_for_dbnet(filtered, infer_size)
    batch = torch.from_numpy(resized.astype(np.float32) / 127.5 - 1.0)
    batch = batch.permute(2, 0, 1).unsqueeze(0).to(device)

    log_message(
        f"  - DBNet text mask at {resized.shape[1]}x{resized.shape[0]} "
        f"(crop {orig_w}x{orig_h})",
        verbose=verbose,
    )
    with torch.inference_mode():
        _db, mask = model(batch)
    mask_np = mask[0, 0].detach().float().cpu().numpy()
    mask_np = cv2.resize(
        mask_np,
        (mask_np.shape[1] * 2, mask_np.shape[0] * 2),
        interpolation=cv2.INTER_LINEAR,
    )
    if pad_h > 0:
        mask_np = mask_np[:-pad_h, :]
    if pad_w > 0:
        mask_np = mask_np[:, :-pad_w]
    if mask_np.shape[0] != orig_h or mask_np.shape[1] != orig_w:
        mask_np = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(mask_np * 255.0, 0, 255).astype(np.uint8)


def detect_text_raw_mask(
    image: Image.Image | np.ndarray,
    detect_size: int = 2048,
    verbose: bool = False,
    boxes: list[Box] | None = None,
) -> np.ndarray:
    """Return a uint8 raw text probability mask at the original image size.

    When ``boxes`` is provided (OSB regions already chosen), DBNet runs only on
    padded, overlapping-merged crops of those boxes. Pixels outside the windows
    stay 0. ``boxes is None`` keeps the legacy full-page forward.
    """
    image_rgb = _to_rgb_uint8(image)
    orig_h, orig_w = image_rgb.shape[:2]
    detect_size = max(256, int(detect_size))

    manager = get_model_manager()
    model = manager.load_dbnet(verbose=verbose)
    device = next(model.parameters()).device

    if boxes is None:
        return _infer_dbnet_mask(
            image_rgb, detect_size, model, device, verbose=verbose
        )

    page_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    windows = _inference_windows(boxes, orig_w, orig_h)
    if not windows:
        log_message("  - DBNet skipped: no valid OSB boxes", verbose=verbose)
        return page_mask

    log_message(
        f"  - DBNet on {len(windows)} OSB window(s) "
        f"(from {len(boxes)} box(es), not full page)",
        verbose=verbose,
    )
    for x0, y0, x1, y1 in windows:
        crop = image_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop_mask = _infer_dbnet_mask(
            crop, detect_size, model, device, verbose=verbose
        )
        dest = page_mask[y0:y1, x0:x1]
        np.maximum(dest, crop_mask, out=dest)
    return page_mask
