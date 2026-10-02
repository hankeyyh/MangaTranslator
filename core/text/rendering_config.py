from core.config import MangaTranslatorConfig, RenderingConfig
from core.scaling import scale_font_size, scale_scalar
from core.text.text_processing import supports_long_word_breaking


def _shared_typography_kwargs(config: MangaTranslatorConfig) -> dict:
    should_hyphenate = bool(config.rendering.hyphenate_before_scaling)
    if not supports_long_word_breaking(config.translation.output_language):
        should_hyphenate = False
    return {
        "hyphenate_before_scaling": should_hyphenate,
        "hyphen_penalty": config.rendering.hyphen_penalty,
        "hyphenation_min_word_length": config.rendering.hyphenation_min_word_length,
        "badness_exponent": config.rendering.badness_exponent,
        "supersampling_factor": config.rendering.supersampling_factor,
        "detach_trailing_punctuation": config.rendering.detach_trailing_punctuation,
    }


def build_bubble_rendering_config(
    config: MangaTranslatorConfig,
    processing_scale: float,
) -> RenderingConfig:
    min_font_size = scale_font_size(
        config.rendering.min_font_size, processing_scale, minimum=4, maximum=256
    )
    max_font_size = scale_font_size(
        config.rendering.max_font_size,
        processing_scale,
        minimum=min_font_size,
        maximum=384,
    )
    return RenderingConfig(
        font_dir=config.rendering.font_dir,
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        line_spacing_mult=config.rendering.line_spacing_mult,
        use_subpixel_rendering=config.rendering.use_subpixel_rendering,
        font_hinting=config.rendering.font_hinting,
        use_ligatures=config.rendering.use_ligatures,
        padding_pixels=scale_scalar(
            config.rendering.padding_pixels,
            processing_scale,
            minimum=1.0,
            maximum=80.0,
        ),
        outline_width=0.0,
        auto_vertical_text=config.rendering.auto_vertical_text,
        vertical_line_spacing_mult=config.rendering.vertical_line_spacing_mult,
        vertical_font_size_mult=config.rendering.vertical_font_size_mult,
        **_shared_typography_kwargs(config),
    )


def build_osb_rendering_config(
    config: MangaTranslatorConfig,
    processing_scale: float,
) -> RenderingConfig:
    min_font_size = scale_font_size(
        config.outside_text.osb_min_font_size,
        processing_scale,
        minimum=4,
        maximum=512,
    )
    max_font_size = scale_font_size(
        config.outside_text.osb_max_font_size,
        processing_scale,
        minimum=min_font_size,
        maximum=640,
    )
    return RenderingConfig(
        font_dir=(
            config.outside_text.osb_font_dir
            if config.outside_text.osb_font_dir
            else config.rendering.font_dir
        ),
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        line_spacing_mult=config.outside_text.osb_line_spacing,
        use_subpixel_rendering=config.outside_text.osb_use_subpixel_rendering,
        font_hinting=config.outside_text.osb_font_hinting,
        use_ligatures=config.outside_text.osb_use_ligatures,
        padding_pixels=scale_scalar(
            config.outside_text.osb_padding_pixels,
            processing_scale,
            minimum=1.0,
            maximum=80.0,
        ),
        outline_width=scale_scalar(
            config.outside_text.osb_outline_width,
            processing_scale,
            minimum=0.0,
            maximum=24.0,
        ),
        auto_vertical_text=config.outside_text.osb_auto_vertical_text,
        vertical_line_spacing_mult=config.outside_text.osb_vertical_line_spacing_mult,
        vertical_font_size_mult=config.outside_text.osb_vertical_font_size_mult,
        **_shared_typography_kwargs(config),
    )
