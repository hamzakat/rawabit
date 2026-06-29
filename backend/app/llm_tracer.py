"""LLM call tracing for ingestion pipeline.

Records every LLM API call (completion and embedding) during ingestion
to the ``llm_call_trace`` table for debugging, cost tracking, and
thesis evaluation.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from .db import get_connection
from .settings import Settings


def _coerce_int(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_usage(response: object) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage", None) if response is not None else None
    if usage is None:
        return None, None, None
    return (
        _coerce_int(getattr(usage, "prompt_tokens", None)),
        _coerce_int(getattr(usage, "completion_tokens", None)),
        _coerce_int(getattr(usage, "total_tokens", None)),
    )


class LLMTracer:
    def __init__(self, settings: Settings, case_id: str, job_id: str) -> None:
        self._settings = settings
        self._case_id = case_id
        self._job_id = job_id

    def record(
        self,
        *,
        stage: str,
        model: str,
        provider: str = "openai",
        started: float,
        response: object | None = None,
        error: str | None = None,
    ) -> None:
        latency_ms = round((time.monotonic() - started) * 1000, 2)
        prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
        error_text = (error or "")[:2000] if error else None
        now = datetime.now(timezone.utc).isoformat()
        try:
            with get_connection(self._settings) as conn:
                conn.execute(
                    "INSERT INTO llm_call_trace "
                    "(job_id, case_id, stage, model, provider, "
                    "request_summary, latency_ms, prompt_tokens, completion_tokens, "
                    "total_tokens, error, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        self._job_id,
                        self._case_id,
                        stage,
                        model,
                        provider,
                        f"stage={stage} model={model}",
                        latency_ms,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        error_text,
                        now,
                    ),
                )
                conn.commit()
        except Exception:
            pass
