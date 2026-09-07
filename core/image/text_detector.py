"""Page-level DBNet raw text mask used by LaMa OSB inpainting."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from core.ml.model_manager import get_model_manager
from utils.logging import log_message

_PAD_MULTIPLE = 256


def _to_rgb_uint8(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    image_np = np.asarray(image)
    if image_np.ndim == 2:
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    if image_np.shape[2] == 4:
        return image_np[:, :, :3]
    return image_np


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


def detect_text_raw_mask(
    image: Image.Image | np.ndarray,
    detect_size: int = 2048,
    verbose: bool = False,
) -> np.ndarray:
    """Return a uint8 raw text probability mask at the original image size."""
    image_rgb = _to_rgb_uint8(image)
    orig_h, orig_w = image_rgb.shape[:2]
    detect_size = max(256, int(detect_size))

    filtered = cv2.bilateralFilter(image_rgb, 17, 80, 80)
    resized, pad_w, pad_h = _resize_for_dbnet(filtered, detect_size)
    batch = torch.from_numpy(resized.astype(np.float32) / 127.5 - 1.0)
    batch = batch.permute(2, 0, 1).unsqueeze(0)

    manager = get_model_manager()
    model = manager.load_dbnet(verbose=verbose)
    device = next(model.parameters()).device
    batch = batch.to(device)

    log_message(
        f"  - DBNet text mask at {resized.shape[1]}x{resized.shape[0]}",
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
