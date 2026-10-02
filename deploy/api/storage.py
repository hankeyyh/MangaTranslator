"""Upload translated pages to Supabase Storage or Cloudflare R2."""

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


def upload_r2_object(bucket: str, object_path: str, file_path: Path) -> str:
    """Upload to the caller's R2 key; return only after the PUT succeeds."""
    endpoint = (os.environ.get("R2_ENDPOINT") or "").rstrip("/")
    access_key = os.environ.get("R2_ACCESS_KEY_ID") or ""
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY") or ""
    if not endpoint or not access_key or not secret_key:
        raise RuntimeError(
            "R2_ENDPOINT, R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY are required"
        )

    # R2 is optional; volume / supabase don't need the S3 SDK at import time.
    import boto3
    from botocore.config import Config

    content_type = _CONTENT_TYPES.get(
        file_path.suffix.lower(), "application/octet-stream"
    )
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=10,
            read_timeout=120,
            retries={"mode": "standard", "max_attempts": 3},
        ),
    )
    try:
        with file_path.open("rb") as body:
            client.put_object(
                Bucket=bucket,
                Key=object_path,
                Body=body,
                ContentType=content_type,
            )
    finally:
        client.close()
    return object_path
