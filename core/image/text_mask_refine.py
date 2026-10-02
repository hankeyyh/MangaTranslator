"""Fit DBNet raw masks to text strokes, optionally with DenseCRF."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils.logging import log_message

_CRF_AVAILABLE = None
_MAX_GLYPH_FOR_DILATE = 48
_DEFAULT_MAX_DILATE = 15
_GRABCUT_BAND_RADIUS = 4
# After the spatial clip, only revert if the result still looks like a filled box.
_GRABCUT_MAX_GROW = 3.0
_GRABCUT_MAX_BOX_FILL = 0.5


def _try_import_crf():
    global _CRF_AVAILABLE
    if _CRF_AVAILABLE is not None:
        return _CRF_AVAILABLE
    try:
        import pydensecrf.densecrf as dcrf  # noqa: F401
        from pydensecrf.utils import unary_from_softmax  # noqa: F401

        _CRF_AVAILABLE = True
    except Exception:
        _CRF_AVAILABLE = False
    return _CRF_AVAILABLE


def _odd(value: int, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    if value % 2 == 0:
        value -= 1
    return max(value, minimum)


def _as_u8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def estimate_glyph_size(mask_u8: np.ndarray) -> int:
    """Typical connected-component size (min side), not the OSB box."""
    binary = (_as_u8_mask(mask_u8) > 0).astype(np.uint8)
    if not np.any(binary):
        return 0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    sizes = []
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] <= 9:
            continue
        sizes.append(
            min(stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT])
        )
    if not sizes:
        return 0
    return int(np.median(sizes))


def estimate_stroke_width(mask_u8: np.ndarray) -> int:
    """Stroke width from the distance-transform ridge of a binary mask."""
    binary = (_as_u8_mask(mask_u8) > 0).astype(np.uint8)
    if not np.any(binary):
        return 0
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    radii = dist[binary > 0]
    if radii.size == 0:
        return 0
    half = float(np.percentile(radii, 90))
    return max(1, round(half * 2))


def dilate_kernel_size(
    mask_u8: np.ndarray,
    dilation_offset: int = 15,
    max_dilate: int = _DEFAULT_MAX_DILATE,
) -> int:
    """MIT-style kernel, but sized from glyphs/strokes and hard-capped.

    Large OSB boxes must not drive dilation: a 300px box used to yield a ~95px
    kernel and turn stroke holes back into blobs.
    """
    glyph = estimate_glyph_size(mask_u8)
    mask2d = _as_u8_mask(mask_u8)
    if glyph <= 0:
        if not np.any(mask2d > 0):
            return 3
        glyph = min(mask2d.shape[0], mask2d.shape[1])
    glyph = min(max(glyph, 1), _MAX_GLYPH_FOR_DILATE)
    size = _odd((int((glyph + max(int(dilation_offset), 0)) * 0.3) // 2) * 2 + 1)

    stroke = estimate_stroke_width(mask_u8)
    if stroke > 0:
        size = min(size, _odd(stroke * 2 + 1, minimum=5))

    return min(size, _odd(max_dilate, minimum=3))


def refine_mask_crf(rgbimg: np.ndarray, rawmask: np.ndarray) -> np.ndarray:
    """DenseCRF snap from manga-image-translator, when pydensecrf is installed."""
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    if rawmask.ndim == 2:
        rawmask = rawmask[:, :, None]
    mask_softmax = np.concatenate(
        [cv2.bitwise_not(rawmask)[:, :, None], rawmask], axis=2
    )
    mask_softmax = mask_softmax.astype(np.float32) / 255.0
    feat_first = mask_softmax.transpose((2, 0, 1)).reshape((2, -1))
    unary = np.ascontiguousarray(unary_from_softmax(feat_first))

    d = dcrf.DenseCRF2D(rgbimg.shape[1], rgbimg.shape[0], 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(
        sxy=1, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NO_NORMALIZATION
    )
    d.addPairwiseBilateral(
        sxy=23,
        srgb=7,
        rgbim=np.ascontiguousarray(rgbimg),
        compat=20,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NO_NORMALIZATION,
    )
    result = np.argmax(d.inference(5), axis=0).reshape(
        (rgbimg.shape[0], rgbimg.shape[1])
    )
    return (result * 255).astype(np.uint8)


def _seed_mask(rawmask: np.ndarray) -> np.ndarray:
    return ((_as_u8_mask(rawmask) >= 32).astype(np.uint8)) * 255


def _neighborhood(seed_u8: np.ndarray, radius: int) -> np.ndarray:
    seed_bin = (_as_u8_mask(seed_u8) > 0).astype(np.uint8)
    if not np.any(seed_bin):
        return seed_bin
    kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd(radius * 2 + 1, minimum=3),) * 2
    )
    return cv2.dilate(seed_bin, kern)


def limit_snap_to_seed(
    seed_u8: np.ndarray,
    snapped_u8: np.ndarray,
    band_radius: int = _GRABCUT_BAND_RADIUS,
    max_grow: float = _GRABCUT_MAX_GROW,
    max_box_fill: float = _GRABCUT_MAX_BOX_FILL,
) -> tuple[np.ndarray, str]:
    """Keep a snap inside the raw-stroke neighborhood; revert box-filling leftovers.

    Returns (mask, status) where status is ``ok``, ``clipped``, or ``reverted``.
    """
    seed = _as_u8_mask(seed_u8) > 0
    snapped = _as_u8_mask(snapped_u8) > 0
    if not np.any(seed):
        return (seed.astype(np.uint8) * 255), "ok"

    band = _neighborhood(seed.astype(np.uint8) * 255, band_radius) > 0
    clipped = (snapped & band) | seed
    seed_n = int(np.count_nonzero(seed))
    clip_n = int(np.count_nonzero(clipped))
    box_n = int(clipped.size)
    # Band clip is the hard cap. Revert only if the seed itself was already a
    # blob (band ~= whole crop) and the snap still fills the OSB box.
    if clip_n > max_grow * max(seed_n, 1) and clip_n > max_box_fill * box_n:
        return (seed.astype(np.uint8) * 255), "reverted"
    if int(np.count_nonzero(snapped & ~band)):
        return (clipped.astype(np.uint8) * 255), "clipped"
    return (clipped.astype(np.uint8) * 255), "ok"


def refine_mask_grabcut(
    rgbimg: np.ndarray,
    rawmask: np.ndarray,
    band_radius: int = _GRABCUT_BAND_RADIUS,
) -> np.ndarray:
    """Color-GMM fallback, confined to a thin band around the DBNet seed."""
    seed = _as_u8_mask(rawmask)
    height, width = seed.shape[:2]
    seed_bin = seed >= 32
    if height < 8 or width < 8 or not np.any(seed_bin):
        return seed_bin.astype(np.uint8) * 255

    band = _neighborhood(seed_bin.astype(np.uint8) * 255, band_radius) > 0
    gc_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[band] = cv2.GC_PR_BGD
    gc_mask[seed_bin] = cv2.GC_PR_FGD
    gc_mask[seed >= 160] = cv2.GC_FGD

    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(
            np.ascontiguousarray(rgbimg),
            gc_mask,
            None,
            bgd,
            fgd,
            3,
            cv2.GC_INIT_WITH_MASK,
        )
        grabbed = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
    except Exception:
        grabbed = seed_bin

    snapped, _ = limit_snap_to_seed(
        seed_bin.astype(np.uint8) * 255,
        grabbed.astype(np.uint8) * 255,
        band_radius=band_radius,
    )
    return snapped


def _fit_box_stroke(
    crop_rgb: np.ndarray,
    crop_raw: np.ndarray,
    use_crf: bool,
    dilation_offset: int,
    min_cover: float,
    max_cover: float,
    max_dilate: int,
) -> tuple[np.ndarray, int, str]:
    if crop_raw.size == 0:
        return crop_raw, 0, "ok"

    seed = _seed_mask(crop_raw)
    snap_status = "ok"
    if use_crf:
        if _try_import_crf():
            snapped = refine_mask_crf(crop_rgb, crop_raw)
            snapped, snap_status = limit_snap_to_seed(seed, snapped)
        else:
            snapped = refine_mask_grabcut(crop_rgb, crop_raw)
            snapped, snap_status = limit_snap_to_seed(seed, snapped)
    else:
        snapped = seed

    # Size the kernel from the seed, not a GrabCut blob that already filled the box.
    dilate_size = dilate_kernel_size(
        seed, dilation_offset=dilation_offset, max_dilate=max_dilate
    )
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    refined = cv2.dilate(snapped, kern)

    cover = float(np.count_nonzero(refined)) / float(refined.size)
    if cover < min_cover:
        return np.full_like(refined, 255), dilate_size, snap_status
    if cover > max_cover:
        stricter = (crop_raw >= 128).astype(np.uint8) * 255
        stricter = cv2.dilate(stricter, kern)
        if np.count_nonzero(stricter) > 0:
            return stricter, dilate_size, snap_status
    return refined, dilate_size, snap_status


def dump_stroke_debug(
    image_rgb: np.ndarray,
    raw_mask: np.ndarray,
    stroke_mask: np.ndarray,
    dump_dir: str | Path,
    stem: str,
) -> Path:
    """Write raw / final / overlay PNGs so hole size can be inspected."""
    out_dir = Path(dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_u8 = _as_u8_mask(raw_mask)
    if raw_u8.dtype != np.uint8:
        raw_u8 = np.clip(raw_u8, 0, 255).astype(np.uint8)
    stroke_bool = np.asarray(stroke_mask) > 0
    if stroke_bool.ndim == 3:
        stroke_bool = stroke_bool[:, :, 0]

    cv2.imwrite(str(out_dir / f"{stem}_dbnet_raw.png"), raw_u8)
    cv2.imwrite(
        str(out_dir / f"{stem}_stroke_mask.png"),
        (stroke_bool.astype(np.uint8) * 255),
    )

    overlay = np.asarray(image_rgb).copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    elif overlay.shape[2] == 4:
        overlay = overlay[:, :, :3]
    overlay = overlay.astype(np.float32)
    raw_bin = raw_u8 >= 32
    if np.any(raw_bin):
        overlay[raw_bin] = (
            overlay[raw_bin] * 0.45 + np.array([0.0, 200.0, 255.0]) * 0.55
        )
    if np.any(stroke_bool):
        overlay[stroke_bool] = (
            overlay[stroke_bool] * 0.35 + np.array([255.0, 40.0, 40.0]) * 0.65
        )
    cv2.imwrite(
        str(out_dir / f"{stem}_stroke_overlay.png"),
        cv2.cvtColor(np.clip(overlay, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
    )
    return out_dir


def refine_box_masks(
    image_rgb: np.ndarray,
    raw_mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    dilation_offset: int = 15,
    kernel_size: int = 3,
    use_crf: bool = True,
    min_cover: float = 0.01,
    max_cover: float = 0.85,
    max_dilate: int = _DEFAULT_MAX_DILATE,
    verbose: bool = False,
) -> np.ndarray:
    """Build a page-sized stroke mask from DBNet raw output clipped to OSB boxes."""
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
        image_rgb = image_rgb[:, :, :3]
    height, width = image_rgb.shape[:2]
    if raw_mask.shape[:2] != (height, width):
        raw_mask = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_LINEAR)

    if use_crf:
        if _try_import_crf():
            snap_name = "DenseCRF"
        else:
            snap_name = (
                f"GrabCut band={_GRABCUT_BAND_RADIUS}px (pydensecrf not installed)"
            )
        log_message(f"  - Stroke snap: {snap_name}", verbose=verbose)

    final = np.zeros((height, width), dtype=np.uint8)
    used_rect_fallback = 0
    dilate_sizes: list[int] = []
    snap_clipped = 0
    snap_reverted = 0
    for box in boxes:
        x0, y0, x1, y1 = (int(v) for v in box)
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        refined, dilate_size, snap_status = _fit_box_stroke(
            image_rgb[y0:y1, x0:x1],
            raw_mask[y0:y1, x0:x1],
            use_crf=use_crf,
            dilation_offset=dilation_offset,
            min_cover=min_cover,
            max_cover=max_cover,
            max_dilate=max_dilate,
        )
        dilate_sizes.append(dilate_size)
        if snap_status == "clipped":
            snap_clipped += 1
        elif snap_status == "reverted":
            snap_reverted += 1
        if np.all(refined == 255):
            used_rect_fallback += 1
        final[y0:y1, x0:x1] = cv2.bitwise_or(final[y0:y1, x0:x1], refined)

    kern_size = max(1, int(kernel_size))
    if kern_size > 1:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_size, kern_size))
        final = cv2.dilate(final, kern)

    if dilate_sizes:
        log_message(
            "  - Stroke dilate kernels "
            f"{sorted(set(dilate_sizes))} (cap={_odd(max_dilate)})",
            verbose=verbose,
        )
    if snap_clipped or snap_reverted:
        log_message(
            "  - Stroke snap limited "
            f"(clipped={snap_clipped}, reverted={snap_reverted})",
            verbose=verbose,
        )
    if used_rect_fallback:
        log_message(
            f"  - Stroke refine fell back to rectangle for {used_rect_fallback} box(es)",
            verbose=verbose,
        )
    return final > 0
