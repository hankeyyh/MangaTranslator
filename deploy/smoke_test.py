"""Smoke test for the MangaTranslator Modal gateway.

Usage:
    python deploy/smoke_test.py --url https://USER--manga-translator-mt-web.modal.run
    python deploy/smoke_test.py --url URL --preset fast --image page.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import requests
from PIL import Image, ImageDraw

TERMINAL = {"job_completed", "job_failed"}


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def create_test_image(path: Path) -> None:
    image = Image.new("RGB", (768, 1024), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.ellipse((180, 220, 588, 520), fill="white", outline="black", width=6)
    draw.rectangle((80, 700, 680, 900), fill="white", outline="black", width=6)
    draw.text((250, 340), "テスト", fill="black")
    draw.text((120, 770), "こんにちは", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def preset_config(name: str) -> dict:
    if name == "precise":
        return {
            "input_language": "Japanese",
            "output_language": "English",
            "provider": "deepl",
            "model_name": "deepl",
            "translation_mode": "two-step",
            "ocr_method": "manga-ocr",
            "font_name": "Anime Ace 3.0",
            "detection": {"bubble_detector_model": "yolo_2"},
            "outside_text": {"enabled": True, "inpainting_method": "lama_large"},
            "rendering": {"rtl": True},
            "output_format": "webp",
        }
    return {
        "input_language": "Japanese",
        "output_language": "English",
        "provider": "deepseek",
        "model_name": "deepseek-v4-flash",
        "translation_mode": "one-step",
        "ocr_method": "LLM",
        "font_name": "Anime Ace 3.0",
        "detection": {"bubble_detector_model": "yolo_2"},
        "outside_text": {"enabled": True, "inpainting_method": "lama_large"},
        "rendering": {"rtl": True},
        "output_format": "webp",
    }


def parse_sse_line_block(block: str) -> tuple[str | None, dict | None]:
    event_name = None
    data_raw = None
    for line in block.splitlines():
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_raw = line[len("data:") :].strip()
    if not event_name:
        return None, None
    if event_name == "heartbeat":
        return event_name, {}
    payload = json.loads(data_raw) if data_raw else {}
    return event_name, payload


def consume_sse(url: str, api_key: str, timeout: int) -> list[dict]:
    events: list[dict] = []
    with requests.get(
        url,
        headers=_headers(api_key),
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        buffer = ""
        started = time.time()
        for raw in response.iter_content(chunk_size=None, decode_unicode=True):
            if raw:
                buffer += raw
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                name, payload = parse_sse_line_block(block)
                if not name:
                    continue
                print(f"  sse {name}: {payload}")
                if name != "heartbeat":
                    events.append({"event": name, **(payload or {})})
                if name in TERMINAL:
                    return events
            if time.time() - started > timeout:
                raise TimeoutError(f"SSE timed out after {timeout}s")
    return events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Gateway base URL")
    parser.add_argument("--api-key", default=os.environ.get("MT_API_KEY", ""))
    parser.add_argument("--preset", choices=["fast", "precise", "health"], default="fast")
    parser.add_argument("--image", action="append", default=[], help="Local image path")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--job-id", default="")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    print(f"health check {base}/health")
    health = requests.get(f"{base}/health", timeout=30)
    health.raise_for_status()
    print(f"  {health.json()}")
    if args.preset == "health":
        return 0

    if not args.api_key:
        print("MT_API_KEY is required", file=sys.stderr)
        return 2

    images = [Path(p) for p in args.image]
    tmp_image = None
    if not images:
        tmp_image = Path("/tmp/mt-smoke.png")
        create_test_image(tmp_image)
        images = [tmp_image]

    job_id = args.job_id or f"smoke-{args.preset}-{uuid.uuid4().hex[:8]}"
    payload = {
        "job_id": job_id,
        "images": [
            {"image_id": f"img-{i}", "index": i} for i, _ in enumerate(images)
        ],
        "output": {"type": "volume", "paths": [f"{job_id}/{i}.webp" for i in range(len(images))]},
        "config": preset_config(args.preset),
    }

    files = [("images", (path.name, path.read_bytes(), "image/png")) for path in images]
    print(f"submit {job_id} preset={args.preset} images={len(images)}")
    submit = requests.post(
        f"{base}/v1/jobs",
        headers=_headers(args.api_key),
        data={"payload": json.dumps(payload)},
        files=files,
        timeout=60,
    )
    print(f"  HTTP {submit.status_code} {submit.text}")
    submit.raise_for_status()

    events = consume_sse(
        f"{base}/v1/jobs/{job_id}/events",
        args.api_key,
        timeout=args.timeout,
    )
    terminal = events[-1]["event"] if events else None
    print(f"terminal event: {terminal}")
    if terminal != "job_completed":
        return 1

    first_id = payload["images"][0]["image_id"]
    output = requests.get(
        f"{base}/v1/jobs/{job_id}/output/{first_id}",
        headers=_headers(args.api_key),
        timeout=60,
    )
    if output.status_code == 200:
        dest = Path(f"/tmp/{job_id}-{first_id}.webp")
        dest.write_bytes(output.content)
        print(f"saved output {dest} ({len(output.content)} bytes)")
    else:
        print(f"output download skipped: HTTP {output.status_code} {output.text[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
