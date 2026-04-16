"""Request correlation and stage timing helpers for API observability."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


def new_request_id() -> str:
    return str(uuid.uuid4())


@contextmanager
def stage_timer(
    log: logging.Logger,
    *,
    request_id: str,
    pipeline_version: str,
    stage: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, float]]:
    """
    Context manager that records duration_ms and logs a structured line.

    Yields a dict with key 'start' (perf_counter); on exit sets 'duration_ms'.
    """
    ctx: Dict[str, float] = {"start": time.perf_counter()}
    try:
        yield ctx
    finally:
        duration_ms = (time.perf_counter() - ctx["start"]) * 1000.0
        ctx["duration_ms"] = duration_ms
        payload = {
            "request_id": request_id,
            "pipeline_version": pipeline_version,
            "stage": stage,
            "duration_ms": round(duration_ms, 3),
        }
        if extra:
            payload.update(extra)
        log.info("stage_timing %s", payload)


def log_request_summary(
    log: logging.Logger,
    *,
    request_id: str,
    pipeline_version: str,
    path: str,
    total_ms: float,
    status: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "request_id": request_id,
        "pipeline_version": pipeline_version,
        "path": path,
        "total_ms": round(total_ms, 3),
        "status": status,
    }
    if extra:
        payload.update(extra)
    log.info("request_summary %s", payload)
