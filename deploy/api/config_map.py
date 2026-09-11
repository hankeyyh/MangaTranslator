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
from deploy.modal_config import FONTS_VOLUME_PATH, MODEL_MOUNT_PATH


def build_mt_config(
    config_in: Mapping[str, Any],
    models_dir: Path | None = None,
    fonts_root: Path | None = None,
) -> MangaTranslatorConfig:
    models_dir = models_dir or Path(MODEL_MOUNT_PATH)
    fonts_root = fonts_root or Path(FONTS_VOLUME_PATH)

    provider = str(config_in.get("provider") or "DeepSeek")
    sampling = get_provider_sampling_defaults(provider)
    bubble_detector = str(
        (config_in.get("detection") or {}).get("bubble_detector_model") or "yolo_2"
    )
    rendering_in = config_in.get("rendering") or {}
    rtl = bool(rendering_in.get("rtl", True))
    output_format = str(config_in.get("output_format") or "webp")
    font_dir = resolve_font_dir(str(config_in.get("font_name") or "Anime Ace 3.0"), fonts_root)
    yolo_path = autodetect_yolo_model_path(models_dir, bubble_detector)

    outside_in = config_in.get("outside_text") or {}
    outside_enabled = bool(outside_in.get("enabled", True))
    inpainting_method = str(outside_in.get("inpainting_method") or "lama_large")
    if inpainting_method in {"flux_klein_9b", "flux_klein_4b", "flux_kontext"}:
        inpainting_method = "lama_large"

    return MangaTranslatorConfig(
        yolo_model_path=str(yolo_path),
        detection=DetectionConfig(bubble_detector_model=bubble_detector),
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
            osb_text_free_only=True,
            huggingface_token=os.environ.get("HF_TOKEN") or "",
        ),
        preprocessing=PreprocessingConfig(enabled=False),
        verbose=True,
        parallel_requests=1,
    )
