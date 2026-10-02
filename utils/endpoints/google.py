import json
import time
from typing import Any

import requests

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message


def _parse_gemini_generate_content_result(
    result: dict[str, Any],
    debug: bool = False,
) -> str | None:
    """Extract text from a Gemini generateContent JSON body.

    Gemini often returns ``"content": null`` (SAFETY, MAX_TOKENS, RECITATION)
    instead of omitting the key. ``dict.get("content", {})`` still yields
    ``None`` in that case, so callers must not chain ``.get`` onto it.
    """
    prompt_feedback = result.get("promptFeedback")
    if prompt_feedback and prompt_feedback.get("blockReason"):
        return None

    candidates = result.get("candidates") or []
    if not candidates:
        block_reason = (
            prompt_feedback.get("blockReason") if prompt_feedback else "unknown"
        )
        log_message(
            f"No candidates in Google response. Block reason: {block_reason}",
            always_print=True,
        )
        log_message(
            f"Full response: {json.dumps(result, indent=2)}",
            verbose=debug,
        )
        return None

    candidate = candidates[0] if isinstance(candidates[0], dict) else {}
    content = candidate.get("content") or {}
    if not isinstance(content, dict):
        content = {}
    content_parts = content.get("parts") or []
    if not isinstance(content_parts, list):
        content_parts = []

    for part in content_parts:
        if (
            isinstance(part, dict)
            and "text" in part
            and not part.get("thought", False)
        ):
            return (part.get("text") or "").strip()

    if (
        content_parts
        and isinstance(content_parts[0], dict)
        and "text" in content_parts[0]
    ):
        return (content_parts[0].get("text") or "").strip()

    finish_reason = candidate.get("finishReason") or "unknown"
    log_message(
        f"No text content in Google response. Finish reason: {finish_reason}",
        always_print=True,
    )
    log_message(
        f"Full response: {json.dumps(result, indent=2)}",
        verbose=debug,
    )
    return ""


def call_gemini_endpoint(
    api_key: str,
    model_name: str,
    parts: list[dict[str, Any]],
    generation_config: dict[str, Any],
    system_prompt: str | None = None,
    debug: bool = False,
    timeout: int = 120,
    max_retries: int = 3,
    base_delay: float = 1.0,
    enable_web_search: bool = False,
    enable_code_execution: bool = False,
) -> str | None:
    """
    Calls the Google API endpoint with the provided data and handles retries.

    Args:
        api_key (str): Google API key.
        model_name (str): Google model to use.
        parts (List[Dict[str, Any]]): List of content parts (text, images).
        generation_config (Dict[str, Any]): Configuration for generation (temp, top_p, etc.).
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.
        enable_web_search (bool): Enable web search (Google Search) for up-to-date information.
        enable_code_execution (bool): Enable Gemini's code execution tool for image zoom/inspection.

    Returns:
        Optional[str]: The raw text content from the API response if successful,
                       None if blocked by safety settings or if no content is found after retries.

    Raises:
        ValueError: If API key is missing.
        RuntimeError: If API call fails after retries for non-rate-limited HTTP errors,
                      connection errors, or response processing fails.
    """
    if not api_key:
        raise ValidationError("API key is required for Google endpoint")

    # Detect Gemini 3 models - they require v1alpha API for per-part media_resolution
    is_gemini_3 = "gemini-3" in (model_name or "").lower()
    api_version = "v1alpha" if is_gemini_3 else "v1beta"
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent?key={api_key}"

    safety_settings_config = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": generation_config,
        "safetySettings": safety_settings_config,
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    tools = []
    if enable_web_search:
        tools.append({"googleSearch": {}})
    if enable_code_execution:
        tools.append({"code_execution": {}})
    if tools:
        payload["tools"] = tools

    for attempt in range(max_retries + 1):
        current_delay = min(
            base_delay * (2**attempt), 16.0
        )  # Exponential backoff, max 16s
        try:
            log_message(
                f"Google API request (attempt {attempt + 1}/{max_retries + 1})",
                verbose=debug,
            )

            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()

            log_message("Processing Google response", verbose=debug)
            try:
                result = response.json()
                return _parse_gemini_generate_content_result(result, debug=debug)

            except (
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
                AttributeError,
            ) as e:
                raise TranslationError(
                    f"Error processing successful Google API response: {e!s}"
                ) from e

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_text = e.response.text[:500]  # Limit error text length

            if status_code == 429 and attempt < max_retries:
                log_message(
                    f"Rate limited, retrying in {current_delay:.1f}s", verbose=debug
                )
                time.sleep(current_delay)
                continue
            else:
                error_reason = f"Status {status_code}: {error_text}"
                if status_code == 429 and attempt == max_retries:
                    error_reason = (
                        f"Rate limited after {max_retries + 1} attempts: {error_text}"
                    )

                raise TranslationError(f"Google API HTTP Error: {error_reason}") from e

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                log_message(
                    f"Connection error, retrying in {current_delay:.1f}s: {e!s}",
                    verbose=debug,
                )
                time.sleep(current_delay)
                continue
            else:
                raise TranslationError(
                    f"Google API Connection Error after retries: {e!s}"
                ) from e

    raise TranslationError(
        f"Failed to get response from Google API after {max_retries + 1} attempts."
    )
