"""Job metadata + ordered event log stored in a Modal Dict."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _aio(method: Any) -> Any | None:
    aio = getattr(method, "aio", None)
    return aio if callable(aio) else None


class JobStore:
    def __init__(self, backend: Any):
        self._backend = backend

    def get(self, job_id: str) -> dict[str, Any] | None:
        getter = getattr(self._backend, "get", None)
        if callable(getter):
            value = getter(job_id)
        else:
            try:
                value = self._backend[job_id]
            except KeyError:
                return None
        if value is None:
            return None
        return value

    async def get_async(self, job_id: str) -> dict[str, Any] | None:
        getter = getattr(self._backend, "get", None)
        if callable(getter):
            aio = _aio(getter)
            value = await aio(job_id) if aio is not None else getter(job_id)
        else:
            getitem = getattr(self._backend, "__getitem__", None)
            aio = _aio(getitem)
            try:
                value = await aio(job_id) if aio is not None else self._backend[job_id]
            except KeyError:
                return None
        if value is None:
            return None
        return value

    def put(self, job_id: str, job: dict[str, Any]) -> None:
        putter = getattr(self._backend, "put", None)
        if callable(putter):
            putter(job_id, job)
            return
        self._backend[job_id] = job

    async def put_async(self, job_id: str, job: dict[str, Any]) -> None:
        putter = getattr(self._backend, "put", None)
        if callable(putter):
            aio = _aio(putter)
            if aio is not None:
                await aio(job_id, job)
            else:
                putter(job_id, job)
            return
        setitem = getattr(self._backend, "__setitem__", None)
        aio = _aio(setitem)
        if aio is not None:
            await aio(job_id, job)
            return
        self._backend[job_id] = job

    def create_queued(
        self,
        job_id: str,
        request: Mapping[str, Any],
        image_count: int,
        *,
        replace_failed: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        existing = self.get(job_id)
        reused = self._reuse_existing(existing, replace_failed)
        if reused is not None:
            return reused, False
        job = self._new_queued_job(job_id, request, image_count)
        return self.append_event(job, "queued", {"position": 0}), True

    async def create_queued_async(
        self,
        job_id: str,
        request: Mapping[str, Any],
        image_count: int,
        *,
        replace_failed: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        existing = await self.get_async(job_id)
        reused = self._reuse_existing(existing, replace_failed)
        if reused is not None:
            return reused, False
        job = self._new_queued_job(job_id, request, image_count)
        return await self.append_event_async(job, "queued", {"position": 0}), True

    def append_event(
        self,
        job: dict[str, Any],
        event: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        job = self._apply_event(job, event, payload)
        self.put(job["job_id"], job)
        return job

    async def append_event_async(
        self,
        job: dict[str, Any],
        event: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        job = self._apply_event(job, event, payload)
        await self.put_async(job["job_id"], job)
        return job

    def events_after(self, job: Mapping[str, Any], after: int) -> list[dict[str, Any]]:
        events = list(job.get("events") or [])
        return [event for event in events if int(event.get("seq") or 0) > after]

    @staticmethod
    def _reuse_existing(
        existing: dict[str, Any] | None,
        replace_failed: bool,
    ) -> dict[str, Any] | None:
        if existing is None:
            return None
        status = existing.get("status")
        if status in {"queued", "running", "completed"}:
            return existing
        if not replace_failed:
            return existing
        return None

    @staticmethod
    def _new_queued_job(
        job_id: str,
        request: Mapping[str, Any],
        image_count: int,
    ) -> dict[str, Any]:
        return {
            "job_id": job_id,
            "status": "queued",
            "image_count": image_count,
            "created_at": utc_now(),
            "request": dict(request),
            "events": [],
            "next_seq": 1,
        }

    @staticmethod
    def _apply_event(
        job: dict[str, Any],
        event: str,
        payload: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        payload = dict(payload or {})
        seq = int(job.get("next_seq") or 1)
        record = {
            "seq": seq,
            "event": event,
            "job_id": job["job_id"],
            **payload,
        }
        events = list(job.get("events") or [])
        events.append(record)
        job["events"] = events
        job["next_seq"] = seq + 1
        job["updated_at"] = utc_now()

        if event == "started":
            job["status"] = "running"
        elif event == "job_completed":
            job["status"] = "completed"
        elif event == "job_failed":
            job["status"] = "failed"

        return job
