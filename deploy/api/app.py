"""FastAPI gateway: POST /v1/jobs and GET /v1/jobs/{id}/events."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from deploy.api.jobs import JobStore
from deploy.api.schemas import CreateJobRequest, CreateJobResponse
from deploy.modal_config import (
    SCRATCH_MOUNT_PATH,
    SSE_HEARTBEAT_SECONDS,
    SSE_POLL_SECONDS,
    TERMINAL_EVENTS,
)

SpawnJob = Callable[[str], Any]


async def _invoke(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a Modal method via .aio when present so FastAPI does not block."""
    aio = getattr(fn, "aio", None)
    if callable(aio):
        return await aio(*args, **kwargs)
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _require_api_key(authorization: str | None) -> None:
    expected = os.environ.get("MT_API_KEY") or ""
    if not expected:
        raise HTTPException(status_code=500, detail="MT_API_KEY is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="invalid api key")


def _save_multipart_files(
    job_id: str,
    body: CreateJobRequest,
    files: list[tuple[str, bytes]],
) -> CreateJobRequest:
    if len(files) != len(body.images):
        raise HTTPException(
            status_code=400,
            detail="multipart file count must match images",
        )
    upload_root = Path(SCRATCH_MOUNT_PATH) / "uploads" / job_id
    upload_root.mkdir(parents=True, exist_ok=True)
    updated = []
    for image, (filename, content) in zip(body.images, files, strict=True):
        suffix = Path(filename).suffix or ".png"
        dest = upload_root / f"{image.image_id}{suffix}"
        dest.write_bytes(content)
        relative = dest.relative_to(SCRATCH_MOUNT_PATH).as_posix()
        updated.append(image.model_copy(update={"url": f"scratch:{relative}"}))
    return body.model_copy(update={"images": updated})


async def _parse_job_request(
    request: Request,
) -> tuple[CreateJobRequest, list[tuple[str, bytes]]]:
    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in content_type:
        form = await request.form()
        payload = form.get("payload")
        if not payload or not isinstance(payload, str):
            raise HTTPException(status_code=400, detail="multipart field 'payload' is required")
        try:
            body = CreateJobRequest.model_validate_json(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        uploads: list[tuple[str, bytes]] = []
        for key, value in form.multi_items():
            if key == "payload":
                continue
            filename = getattr(value, "filename", None)
            read = getattr(value, "read", None)
            if filename and read:
                uploads.append((filename, await read()))
        return body, uploads
    try:
        payload = await request.json()
        body = CreateJobRequest.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return body, []


def create_app(
    job_backend: Any,
    spawn_job: SpawnJob,
    scratch_volume: Any | None = None,
) -> FastAPI:
    store = JobStore(job_backend)
    app = FastAPI(title="MangaTranslator MT", version="v1")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "manga-translator-mt"}

    @app.post("/v1/jobs", response_model=CreateJobResponse, status_code=202)
    async def create_job(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> Response:
        _require_api_key(authorization)
        body, uploads = await _parse_job_request(request)

        job_id = (body.job_id or "").strip() or str(uuid.uuid4())
        if uploads:
            body = _save_multipart_files(job_id, body, uploads)
            if scratch_volume is not None:
                await _invoke(scratch_volume.commit)

        missing_urls = [item.image_id for item in body.images if not item.url]
        if missing_urls:
            raise HTTPException(
                status_code=400,
                detail=f"image url is required for: {', '.join(missing_urls)}",
            )

        request_payload = body.model_dump()
        request_payload["job_id"] = job_id
        job, created = await store.create_queued_async(
            job_id,
            request_payload,
            len(body.images),
            replace_failed=True,
        )
        if created:
            await _invoke(spawn_job, job_id)

        response = CreateJobResponse(
            job_id=job_id,
            status=str(job.get("status") or "queued"),
            image_count=int(job.get("image_count") or len(body.images)),
        )
        return Response(
            content=response.model_dump_json(),
            status_code=202 if created else 200,
            media_type="application/json",
        )

    async def _event_stream(job_id: str, after: int) -> AsyncIterator[str]:
        last_seq = after
        last_heartbeat = 0.0
        elapsed = 0.0
        while True:
            job = await store.get_async(job_id)
            if job is None:
                yield (
                    "event: job_failed\n"
                    f"data: {json.dumps({'seq': last_seq + 1, 'job_id': job_id, 'error': 'job not found'})}\n\n"
                )
                return

            for event in store.events_after(job, last_seq):
                seq = int(event.get("seq") or 0)
                name = str(event.get("event") or "message")
                yield f"id: {seq}\nevent: {name}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                last_seq = seq
                if name in TERMINAL_EVENTS:
                    return

            await asyncio.sleep(SSE_POLL_SECONDS)
            elapsed += SSE_POLL_SECONDS
            if elapsed - last_heartbeat >= SSE_HEARTBEAT_SECONDS:
                last_heartbeat = elapsed
                yield "event: heartbeat\ndata: {}\n\n"

    @app.get("/v1/jobs/{job_id}/events")
    async def job_events(
        job_id: str,
        after: int = 0,
        authorization: str | None = Header(default=None),
    ) -> StreamingResponse:
        _require_api_key(authorization)
        if await store.get_async(job_id) is None:
            raise HTTPException(status_code=404, detail="job not found")
        return StreamingResponse(
            _event_stream(job_id, after),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/v1/jobs/{job_id}/output/{image_id}")
    def job_output(
        job_id: str,
        image_id: str,
        authorization: str | None = Header(default=None),
    ) -> Response:
        """Debug helper for output.type=volume. Not part of the public API contract."""
        _require_api_key(authorization)
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        request = job.get("request") or {}
        images = list(request.get("images") or [])
        try:
            index = next(
                i for i, item in enumerate(sorted(images, key=lambda x: int(x.get("index") or 0)))
                if item.get("image_id") == image_id
            )
        except StopIteration:
            raise HTTPException(status_code=404, detail="image not found") from None
        output = request.get("output") or {}
        paths = list(output.get("paths") or [])
        relative = paths[index] if index < len(paths) and paths[index] else None
        if relative is None:
            fmt = str((request.get("config") or {}).get("output_format") or "webp")
            relative = f"{job_id}/{image_id}.{fmt}"
        path = Path(SCRATCH_MOUNT_PATH) / "results" / relative
        if scratch_volume is not None:
            try:
                scratch_volume.reload()
            except Exception:
                pass
        if not path.is_file():
            raise HTTPException(status_code=404, detail="output file not found")
        media = {
            ".webp": "image/webp",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(path.suffix.lower(), "application/octet-stream")
        return Response(content=path.read_bytes(), media_type=media)

    return app
