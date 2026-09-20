"""GPU worker: download → translate_and_render → upload → append events."""

from __future__ import annotations

import os
import shutil
import threading
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from PIL import Image

from deploy.api.config_map import build_mt_config
from deploy.api.jobs import JobStore
from deploy.api.storage import upload_supabase_object
from deploy.modal_config import (
    FONTS_VOLUME_PATH,
    MODEL_MOUNT_PATH,
    SCRATCH_MOUNT_PATH,
    TORCH_INDUCTOR_CACHE,
    TORCH_KERNEL_CACHE,
    TRITON_CACHE,
    WORK_DIR_ROOT,
)

SCRATCH_SCHEME = "scratch:"
WORK_ROOT = Path(WORK_DIR_ROOT)
# process_job is @modal.concurrent: reload/commit/read/write share one Volume mount.
_VOLUME_LOCK = threading.Lock()


def warmup_imports() -> None:
    """CPU work for @modal.enter(snap=True): imports + local kernel-cache dirs.

    Must not touch the GPU. CPU memory snapshots cannot capture CUDA state.
    """
    for cache_dir in (TORCH_KERNEL_CACHE, TORCH_INDUCTOR_CACHE, TRITON_CACHE):
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    import cv2  # noqa: F401
    import numpy  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import ultralytics  # noqa: F401
    from core.pipeline import translate_and_render  # noqa: F401

    print(
        f"Worker CPU stack imported (kernel_cache={TORCH_KERNEL_CACHE})",
        flush=True,
    )


def warmup_cuda() -> None:
    """GPU work for @modal.enter(snap=False): recreate CUDA context after restore."""
    import torch

    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")
    print(f"Worker CUDA ready (cuda={torch.cuda.is_available()})", flush=True)


def _worker_id() -> str:
    return os.environ.get("MODAL_TASK_ID") or os.environ.get("HOSTNAME") or "mt-worker"


def _sorted_images(request: dict[str, Any]) -> list[dict[str, Any]]:
    images = list(request.get("images") or [])
    return sorted(images, key=lambda item: int(item.get("index") or 0))


def _output_path_for(
    request: dict[str, Any], image: dict[str, Any], index: int
) -> str | None:
    output = request.get("output") or {}
    paths = list(output.get("paths") or [])
    if 0 <= index < len(paths) and paths[index]:
        return paths[index]
    image_id = image.get("image_id")
    job_id = request.get("job_id") or "job"
    fmt = str((request.get("config") or {}).get("output_format") or "webp")
    return f"{job_id}/{image_id}.{fmt}"


def _copy_from_scratch(url: str, dest: Path) -> None:
    source = Path(SCRATCH_MOUNT_PATH) / url[len(SCRATCH_SCHEME) :]
    if not source.is_file():
        raise FileNotFoundError(f"scratch image not found: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


def _download_image(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith(SCRATCH_SCHEME):
        with _VOLUME_LOCK:
            _copy_from_scratch(url, dest)
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


def _compress_for_upload(path: Path, quality: int = 80) -> None:
    image = Image.open(path)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        image.save(path, format="JPEG", quality=quality, optimize=True)
        return
    if suffix == ".png":
        image.save(path, format="PNG", optimize=True, compress_level=6)
        return
    if image.mode == "P":
        image = image.convert("RGBA")
    image.save(path, format="WEBP", quality=quality, method=6)


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
        _compress_for_upload(local_path)
        return upload_supabase_object(bucket, object_path, local_path)

    if output_type == "volume":
        dest = Path(SCRATCH_MOUNT_PATH) / "results" / object_path
        with _VOLUME_LOCK:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_path, dest)
        return object_path

    return object_path


def _reload_scratch_volume(scratch_volume: Any) -> None:
    try:
        scratch_volume.reload()
    except Exception as exc:
        print(f"scratch volume reload skipped: {exc}")


def _commit_scratch_volume(scratch_volume: Any) -> None:
    with _VOLUME_LOCK:
        scratch_volume.commit()


def _maybe_reload_scratch(
    images: list[dict[str, Any]], scratch_volume: Any | None
) -> None:
    if scratch_volume is None:
        return
    if not any(
        str(image.get("url") or "").startswith(SCRATCH_SCHEME) for image in images
    ):
        return
    with _VOLUME_LOCK:
        _reload_scratch_volume(scratch_volume)


def run_job(job_id: str, job_backend: Any, scratch_volume: Any | None = None) -> None:
    if Path("/app").is_dir():
        os.chdir("/app")
    store = JobStore(job_backend)
    job = store.get(job_id)
    if job is None:
        raise RuntimeError(f"job not found: {job_id}")

    request = dict(job.get("request") or {})
    request["job_id"] = job_id
    images = _sorted_images(request)
    persist = _persist_output(request)
    wrote_volume = persist and _output_type(request) == "volume"
    work_dir = WORK_ROOT / job_id

    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        # _maybe_reload_scratch：让 GPU 容器看见网关刚 commit 进 Volume 的上传图。
        # Modal Volume 不是 NFS 实时共享盘。每个容器挂载的是 某个时间点的快照
        # GPU worker 经常是热容器（上一单跑完没立刻销毁）。它挂着的 /scratch 还是容器启动时那份旧快照，里面没有这一单的 uploads。
        _maybe_reload_scratch(images, scratch_volume)
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
                url = str(image.get("url") or "")
                suffix = _guess_suffix(url)
                source_path = work_dir / f"{index}_{image_id}{suffix}"
                output_ext = "." + str(
                    (request.get("config") or {}).get("output_format") or "webp"
                )
                result_path = (
                    work_dir / f"{index}_{image_id}_out{output_ext}"
                    if persist
                    else None
                )
                _download_image(url, source_path)

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
        if wrote_volume and scratch_volume is not None:
            _commit_scratch_volume(scratch_volume)
    except Exception as exc:
        job = store.get(job_id) or job
        store.append_event(job, "job_failed", {"error": str(exc)})
        traceback.print_exc()
        if wrote_volume and scratch_volume is not None:
            _commit_scratch_volume(scratch_volume)
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
