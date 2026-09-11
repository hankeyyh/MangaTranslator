"""Upload translated pages to Supabase Storage."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import httpx

_CONTENT_TYPES = {
    ".webp": "image/webp",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _supabase_env() -> tuple[str, str]:
    base = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or ""
    if not base or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
    return base, key


def upload_supabase_object(bucket: str, object_path: str, file_path: Path) -> str:
    base, key = _supabase_env()
    suffix = file_path.suffix.lower()
    content_type = _CONTENT_TYPES.get(suffix, "application/octet-stream")
    encoded = quote(object_path.lstrip("/"), safe="/")
    url = f"{base}/storage/v1/object/{bucket}/{encoded}"
    headers = {
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    data = file_path.read_bytes()
    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, headers=headers, content=data)
        if response.status_code not in {200, 201}:
            raise RuntimeError(
                f"supabase upload failed ({response.status_code}): {response.text[:500]}"
            )
    return object_path
