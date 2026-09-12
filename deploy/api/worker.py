"""GPU worker: download → translate_and_render → upload → append events."""

from __future__ import annotations

import os
import shutil
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from deploy.api.config_map import build_mt_config
from deploy.api.jobs import JobStore
from deploy.api.storage import upload_supabase_object
from deploy.modal_config import (
    FONTS_VOLUME_PATH,
    MODEL_MOUNT_PATH,
    SCRATCH_MOUNT_PATH,
)

SCRATCH_SCHEME = "scratch:"


def _worker_id() -> str:
    return (
        os.environ.get("MODAL_TASK_ID")
        or os.environ.get("HOSTNAME")
        or "mt-worker"
    )


def _sorted_images(request: dict[str, Any]) -> list[dict[str, Any]]:
    images = list(request.get("images") or [])
    return sorted(images, key=lambda item: int(item.get("index") or 0))


def _output_path_for(request: dict[str, Any], image: dict[str, Any], index: int) -> str | None:
    output = request.get("output") or {}
    paths = list(output.get("paths") or [])
    if 0 <= index < len(paths) and paths[index]:
        return paths[index]
    image_id = image.get("image_id")
    job_id = request.get("job_id") or "job"
    fmt = str((request.get("config") or {}).get("output_format") or "webp")
    return f"{job_id}/{image_id}.{fmt}"


def _download_image(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith(SCRATCH_SCHEME):
        source = Path(SCRATCH_MOUNT_PATH) / url[len(SCRATCH_SCHEME) :]
        if not source.is_file():
            raise FileNotFoundError(f"scratch image not found: {source}")
        shutil.copyfile(source, dest)
        return

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported image url: {url}")

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        dest.write_bytes(response.content)


def _guess_suffix(url: str, default: str = ".png") -> str:
    if url.startswith(SCRATCH_SCHEME):
        suffix = Path(url).suffix
        return suffix if suffix else default
    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return suffix
    return default


def _output_type(request: dict[str, Any]) -> str:
    return str((request.get("output") or {}).get("type") or "none")


def _persist_output(request: dict[str, Any]) -> bool:
    """volume / supabase 才落最终图；none 只跑翻译，不写盘、不上传。"""
    return _output_type(request) != "none"


def _write_output(
    request: dict[str, Any],
    image: dict[str, Any],
    index: int,
    local_path: Path,
) -> str | None:
    if not _persist_output(request):
        return None

    output = request.get("output") or {}
    output_type = _output_type(request)
    object_path = _output_path_for(request, image, index) or local_path.name

    if output_type == "supabase":
        bucket = output.get("bucket") or "translation_storage"
        return upload_supabase_object(bucket, object_path, local_path)

    if output_type == "volume":
        dest = Path(SCRATCH_MOUNT_PATH) / "results" / object_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, dest)
        return object_path

    return object_path


def run_job(job_id: str, job_backend: Any, scratch_volume: Any | None = None) -> None:
    os.chdir("/app")
    if scratch_volume is not None:
        try:
            scratch_volume.reload()
        except Exception as exc:
            print(f"scratch volume reload skipped: {exc}")
    store = JobStore(job_backend)
    job = store.get(job_id)
    if job is None:
        raise RuntimeError(f"job not found: {job_id}")

    request = dict(job.get("request") or {})
    request["job_id"] = job_id
    images = _sorted_images(request)
    persist = _persist_output(request)
    work_dir = Path(SCRATCH_MOUNT_PATH) / "work" / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        store.append_event(job, "started", {"worker_id": _worker_id()})
        from core.pipeline import translate_and_render

        mt_config = build_mt_config(
            request.get("config") or {},
            models_dir=Path(MODEL_MOUNT_PATH),
            fonts_root=Path(FONTS_VOLUME_PATH),
        )

        success_count = 0
        error_count = 0
        previous_texts: list[list[str]] = []

        for order, image in enumerate(images):
            image_id = str(image.get("image_id"))
            index = int(image.get("index") or order)
            job = store.get(job_id) or job
            store.append_event(
                job,
                "image_progress",
                {
                    "image_id": image_id,
                    "index": index,
                    "stage": "start",
                    "progress": 0.0,
                },
            )
            try:
                suffix = _guess_suffix(str(image.get("url") or ""))
                source_path = work_dir / f"{index}_{image_id}{suffix}"
                output_ext = "." + str(
                    (request.get("config") or {}).get("output_format") or "webp"
                )
                result_path = (
                    work_dir / f"{index}_{image_id}_out{output_ext}" if persist else None
                )
                _download_image(str(image.get("url") or ""), source_path)

                ocr_out: list[str] = []
                translate_and_render(
                    image_path=source_path,
                    config=mt_config,
                    output_path=result_path,
                    previous_context_texts=previous_texts or None,
                    ocr_texts_out=ocr_out,
                )
                if ocr_out:
                    previous_texts.append(list(ocr_out))

                if persist:
                    if result_path is None or not result_path.is_file():
                        raise RuntimeError("translator did not write an output file")
                    output_path = _write_output(request, image, order, result_path)
                else:
                    output_path = None

                completed: dict[str, Any] = {
                    "image_id": image_id,
                    "index": index,
                }
                if output_path:
                    completed["output_path"] = output_path
                job = store.get(job_id) or job
                store.append_event(job, "image_completed", completed)
                success_count += 1
            except Exception as exc:
                error_count += 1
                job = store.get(job_id) or job
                store.append_event(
                    job,
                    "image_failed",
                    {
                        "image_id": image_id,
                        "index": index,
                        "error": str(exc),
                    },
                )
                print(f"image {image_id} failed: {exc}")
                traceback.print_exc()

        job = store.get(job_id) or job
        store.append_event(
            job,
            "job_completed",
            {"success_count": success_count, "error_count": error_count},
        )
        if persist and scratch_volume is not None:
            scratch_volume.commit()
    except Exception as exc:
        job = store.get(job_id) or job
        store.append_event(job, "job_failed", {"error": str(exc)})
        traceback.print_exc()
        if persist and scratch_volume is not None:
            scratch_volume.commit()
        raise
    finally:
        if not persist:
            shutil.rmtree(work_dir, ignore_errors=True)
