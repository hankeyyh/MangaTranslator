"""DeepL text translation endpoint.

DeepL is a traditional MT API (no vision). Callers must OCR first and pass
plain text. Language names match this project's UI labels and are mapped to
DeepL codes, following manga-image-translator's deepl translator.
"""

from __future__ import annotations

import time
from typing import Any

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message

# Target-language DeepL codes. Source English/Chinese/Portuguese use the
# unsuffixed codes in `_to_deepl_source_lang`.
_DEEPL_TARGET_LANG: dict[str, str] = {
    "Arabic": "AR",
    "Bulgarian": "BG",
    "Chinese (Simplified)": "ZH-HANS",
    "Chinese (Traditional)": "ZH-HANT",
    "Czech": "CS",
    "Danish": "DA",
    "Dutch": "NL",
    "English": "EN-US",
    "Estonian": "ET",
    "Finnish": "FI",
    "French": "FR",
    "German": "DE",
    "Greek": "EL",
    "Hebrew": "HE",
    "Hungarian": "HU",
    "Indonesian": "ID",
    "Italian": "IT",
    "Japanese": "JA",
    "Korean": "KO",
    "Latvian": "LV",
    "Lithuanian": "LT",
    "Norwegian": "NB",
    "Polish": "PL",
    "Portuguese": "PT-BR",
    "Romanian": "RO",
    "Russian": "RU",
    "Slovak": "SK",
    "Slovenian": "SL",
    "Spanish": "ES",
    "Swedish": "SV",
    "Thai": "TH",
    "Turkish": "TR",
    "Ukrainian": "UK",
    "Vietnamese": "VI",
}

_OCR_FAILED = "[OCR FAILED]"
_MAX_TEXTS_PER_REQUEST = 50


def _to_deepl_source_lang(language_name: str) -> str:
    if language_name == "English":
        return "EN"
    if language_name in ("Chinese (Simplified)", "Chinese (Traditional)"):
        return "ZH"
    if language_name == "Portuguese":
        return "PT"
    code = _DEEPL_TARGET_LANG.get(language_name)
    if not code:
        supported = ", ".join(sorted(_DEEPL_TARGET_LANG))
        raise TranslationError(
            f"DeepL does not support source language '{language_name}'. "
            f"Supported languages: {supported}"
        )
    return code.split("-")[0]


def _to_deepl_target_lang(language_name: str) -> str:
    code = _DEEPL_TARGET_LANG.get(language_name)
    if not code:
        supported = ", ".join(sorted(_DEEPL_TARGET_LANG))
        raise TranslationError(
            f"DeepL does not support target language '{language_name}'. "
            f"Supported languages: {supported}"
        )
    return code


def _import_deepl():
    try:
        import deepl
    except ImportError as e:
        raise TranslationError(
            "The 'deepl' package is required for the DeepL translator. "
            "Install it with: pip install deepl"
        ) from e
    return deepl


def call_deepl_endpoint(
    api_key: str,
    texts: list[str],
    source_language: str,
    target_language: str,
    context: str | None = None,
    debug: bool = False,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> list[str]:
    """Translate OCR strings with the DeepL API.

    Args:
        api_key: DeepL auth key (Free keys typically end with ``:fx``).
        texts: Source strings in reading order. ``[OCR FAILED]`` is preserved.
        source_language: UI language name (e.g. ``Japanese``).
        target_language: UI language name (e.g. ``English``).
        context: Optional extra context that is not translated (DeepL ``context``).
        debug: Verbose logging.
        max_retries: Retries for rate-limit / transient errors.
        base_delay: Initial retry delay in seconds.

    Returns:
        Translations aligned with ``texts``.
    """
    if not api_key:
        raise ValidationError("API key is required for DeepL endpoint")
    if not texts:
        return []

    deepl = _import_deepl()
    source_lang = _to_deepl_source_lang(source_language)
    target_lang = _to_deepl_target_lang(target_language)
    translator = deepl.Translator(api_key)

    results = list(texts)
    pending_indices = [
        i
        for i, text in enumerate(texts)
        if (text or "").strip() and (text or "").strip() != _OCR_FAILED
    ]
    if not pending_indices:
        return results

    extra_kwargs: dict[str, Any] = {}
    if context and context.strip():
        extra_kwargs["context"] = context.strip()

    log_message(
        f"DeepL translating {len(pending_indices)} segment(s) "
        f"{source_lang} → {target_lang}",
        verbose=debug,
    )

    for chunk_start in range(0, len(pending_indices), _MAX_TEXTS_PER_REQUEST):
        chunk_indices = pending_indices[
            chunk_start : chunk_start + _MAX_TEXTS_PER_REQUEST
        ]
        chunk_texts = [texts[i] for i in chunk_indices]
        translated = _translate_chunk(
            translator,
            chunk_texts,
            source_lang=source_lang,
            target_lang=target_lang,
            extra_kwargs=extra_kwargs,
            debug=debug,
            max_retries=max_retries,
            base_delay=base_delay,
        )
        if len(translated) != len(chunk_indices):
            raise TranslationError(
                f"DeepL returned {len(translated)} translations for "
                f"{len(chunk_indices)} input segments."
            )
        for index, translated_text in zip(chunk_indices, translated, strict=True):
            results[index] = translated_text

    return results


def _translate_chunk(
    translator: Any,
    chunk_texts: list[str],
    source_lang: str,
    target_lang: str,
    extra_kwargs: dict[str, Any],
    debug: bool,
    max_retries: int,
    base_delay: float,
) -> list[str]:
    deepl = _import_deepl()
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)
        try:
            log_message(
                f"DeepL API request (attempt {attempt + 1}/{max_retries + 1})",
                verbose=debug,
            )
            response = translator.translate_text(
                chunk_texts,
                source_lang=source_lang,
                target_lang=target_lang,
                **extra_kwargs,
            )
            if not isinstance(response, list):
                response = [response]
            return [item.text for item in response]
        except deepl.AuthorizationException as e:
            raise TranslationError(
                "DeepL API authorization failed. Check the API key "
                "(DEEPL_API_KEY / DEEPL_AUTH_KEY)."
            ) from e
        except deepl.QuotaExceededException as e:
            raise TranslationError(
                "DeepL API quota exceeded. Check your DeepL account usage."
            ) from e
        except deepl.TooManyRequestsException as e:
            last_error = e
            if attempt < max_retries:
                log_message(
                    f"DeepL rate limited, retrying in {current_delay:.1f}s",
                    verbose=debug,
                )
                time.sleep(current_delay)
                continue
            raise TranslationError(
                f"DeepL API rate limited after {max_retries + 1} attempts."
            ) from e
        except deepl.DeepLException as e:
            last_error = e
            message = str(e)
            transient = "503" in message or "429" in message or "timeout" in message.lower()
            if transient and attempt < max_retries:
                log_message(
                    f"DeepL transient error, retrying in {current_delay:.1f}s: {message}",
                    verbose=debug,
                )
                time.sleep(current_delay)
                continue
            raise TranslationError(f"DeepL API error: {message}") from e

    raise TranslationError(
        f"Failed to get response from DeepL API after {max_retries + 1} attempts: "
        f"{last_error!s}"
    )
