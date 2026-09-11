"""Job metadata + ordered event log stored in a Modal Dict."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping, Protocol


class DictLike(Protocol):
    def __contains__(self, key: object) -> bool: ...
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_mapping(store: DictLike) -> MutableMapping[str, Any]:
    return store  # type: ignore[return-value]


class JobStore:
    def __init__(self, backend: DictLike):
        self._backend = _as_mapping(backend)

    def get(self, job_id: str) -> dict[str, Any] | None:
        try:
            value = self._backend[job_id]
        except KeyError:
            return None
        if value is None:
            return None
        return value

    def put(self, job_id: str, job: dict[str, Any]) -> None:
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
        if existing is not None:
            status = existing.get("status")
            if status in {"queued", "running", "completed"}:
                return existing, False
            if not replace_failed:
                return existing, False

        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "queued",
            "image_count": image_count,
            "created_at": utc_now(),
            "request": dict(request),
            "events": [],
            "next_seq": 1,
        }
        job = self.append_event(job, "queued", {"position": 0})
        return job, True

    def append_event(
        self,
        job: dict[str, Any],
        event: str,
        payload: Mapping[str, Any] | None = None,
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

        self.put(job["job_id"], job)
        return job

    def events_after(self, job: Mapping[str, Any], after: int) -> list[dict[str, Any]]:
        events = list(job.get("events") or [])
        return [event for event in events if int(event.get("seq") or 0) > after]
