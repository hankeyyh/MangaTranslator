"""Map the public job config JSON onto MangaTranslatorConfig."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from core.config import (
    CleaningConfig,
    DetectionConfig,
    MangaTranslatorConfig,
    OutputConfig,
    OutsideTextConfig,
    PreprocessingConfig,
    RenderingConfig,
    TranslationConfig,
)
from core.llm_defaults import get_provider_sampling_defaults
from core.validation import autodetect_yolo_model_path
from deploy.api.fonts import resolve_font_dir
from deploy.modal_config import DEFAULT_FONT_NAME, FONTS_VOLUME_PATH, MODEL_MOUNT_PATH
from utils.logging import log_message


def build_mt_config(
    config_in: Mapping[str, Any],
    models_dir: Path | None = None,
    fonts_root: Path | None = None,
) -> MangaTranslatorConfig:
    models_dir = models_dir or Path(MODEL_MOUNT_PATH)
    fonts_root = fonts_root or Path(FONTS_VOLUME_PATH)

    provider = str(config_in.get("provider") or "DeepSeek")
    sampling = get_provider_sampling_defaults(provider)
    detection_in = config_in.get("detection") or {}
    bubble_detector = str(detection_in.get("bubble_detector_model") or "yolo_2")
    use_panel_sorting = bool(detection_in.get("use_panel_sorting", True))
    rendering_in = config_in.get("rendering") or {}
    rtl = bool(rendering_in.get("rtl", True))
    output_format = str(config_in.get("output_format") or "webp")
    font_name = str(config_in.get("font_name") or DEFAULT_FONT_NAME)
    font_dir = resolve_font_dir(font_name, fonts_root)
    log_message(f"Resolved font_dir={font_dir} from font_name={font_name}", always_print=True)
    yolo_path = autodetect_yolo_model_path(models_dir, bubble_detector)

    outside_in = config_in.get("outside_text") or {}
    outside_enabled = bool(outside_in.get("enabled", True))
    inpainting_method = str(outside_in.get("inpainting_method") or "lama_large")
    lama_use_crf = bool(outside_in.get("lama_use_crf", False))
    lama_detect_size = int(outside_in.get("lama_detect_size") or 1024)
    lama_inpainting_size = int(outside_in.get("lama_inpainting_size") or 1024)
    if inpainting_method in {"flux_klein_9b", "flux_klein_4b", "flux_kontext"}:
        inpainting_method = "lama_large"
    test_mode = bool(config_in.get("test_mode", False))

    return MangaTranslatorConfig(
        yolo_model_path=str(yolo_path),
        detection=DetectionConfig(
            bubble_detector_model=bubble_detector,
            # True：额外跑 YOLOv12x，把偏紧的气泡框扩到盖住漏字。现网 OSB 不走该模型，仅用于撑框成本过高，故关掉。
            use_osb_text_verification=False,
            use_panel_sorting=use_panel_sorting,
        ),
        cleaning=CleaningConfig(),
        translation=TranslationConfig(
            provider=provider,
            model_name=str(config_in.get("model_name") or "deepseek-v4-flash"),
            input_language=str(config_in.get("input_language") or "Japanese"),
            output_language=str(config_in.get("output_language") or "English"),
            reading_direction="rtl" if rtl else "ltr",
            translation_mode=str(config_in.get("translation_mode") or "one-step"),
            ocr_method=str(config_in.get("ocr_method") or "LLM"),
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            top_k=int(sampling["top_k"]),
            upscale_method="lanczos",
        ),
        rendering=RenderingConfig(font_dir=str(font_dir)),
        output=OutputConfig(output_format=output_format, upscale_final_image=False),
        outside_text=OutsideTextConfig(
            enabled=outside_enabled,
            inpainting_method=inpainting_method,
            # True：OSB 用 RT-DETR 的 text_free（旁白/字幕块），不跑 YOLOv12x。
            # False：YOLOv12x 全页检字后去掉气泡内结果；模型失败才回退 text_free。
            osb_text_free_only=True,
            huggingface_token=os.environ.get("HF_TOKEN") or "",
            lama_use_crf=lama_use_crf,
            lama_detect_size=lama_detect_size,  # DBNet：OSB crop 笔画 mask 推理最长边上限
            lama_inpainting_size=lama_inpainting_size,  # LaMa：OSB 擦字 patch 推理最长边上限
        ),
        preprocessing=PreprocessingConfig(enabled=False),
        verbose=True,
        parallel_requests=1,
        test_mode=test_mode,
    )
