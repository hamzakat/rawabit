from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from typing import Any

from .analysis_service import AnalysisService
from .db import get_connection
from .settings import Settings
from .utils import utc_now_iso

logger = logging.getLogger(__name__)


class AnalysisRunner:
    """Processes persisted analyzer generation and chart-repair jobs."""

    def __init__(
        self,
        settings: Settings,
        analysis_service: AnalysisService | None = None,
    ) -> None:
        self._settings = settings
        self._analysis_service = analysis_service or AnalysisService(settings)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._recover_inflight_analyses()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="rawabit-analysis",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None

    def _run(self) -> None:
        poll_interval = max(self._settings.ingestion_poll_interval_seconds, 0.25)
        while not self._stop_event.is_set():
            try:
                analysis_id = self._next_queued_analysis_id()
            except sqlite3.OperationalError:
                if self._stop_event.is_set():
                    return
                self._stop_event.wait(poll_interval)
                continue
            if analysis_id:
                self._process_analysis(analysis_id)
            else:
                self._stop_event.wait(poll_interval)

    def _recover_inflight_analyses(self) -> None:
        now = utc_now_iso()
        with get_connection(self._settings) as connection:
            connection.execute(
                "UPDATE analysis SET status = 'queued', updated_at = ? "
                "WHERE status = 'generating'",
                (now,),
            )
            connection.execute(
                "UPDATE analysis SET status = 'repair_queued', updated_at = ? "
                "WHERE status = 'repairing'",
                (now,),
            )

    def _next_queued_analysis_id(self) -> str | None:
        with get_connection(self._settings) as connection:
            row = connection.execute(
                "SELECT id FROM analysis "
                "WHERE status IN ('queued', 'repair_queued') "
                "ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
        return str(row["id"]) if row else None

    def _process_analysis(self, analysis_id: str) -> None:
        try:
            with get_connection(self._settings) as connection:
                row = connection.execute(
                    "SELECT id, case_id, analysis_type, prompt, status, rag_answer, "
                    "charts_json, subgraph_json, pending_repair_json "
                    "FROM analysis WHERE id = ?",
                    (analysis_id,),
                ).fetchone()
            if not row:
                return
            if row["status"] == "queued":
                self._process_generation(dict(row))
            elif row["status"] == "repair_queued":
                self._process_repair(dict(row))
        except Exception as exc:
            logger.exception(
                "Unexpected analysis worker failure for %s (%s).",
                analysis_id,
                type(exc).__name__,
            )
            self._mark_failed(analysis_id, self._user_error(exc))

    def _process_generation(self, row: dict[str, Any]) -> None:
        analysis_id = str(row["id"])
        now = utc_now_iso()
        with get_connection(self._settings) as connection:
            cursor = connection.execute(
                "UPDATE analysis SET status = 'generating', error = NULL, updated_at = ? "
                "WHERE id = ? AND status = 'queued'",
                (now, analysis_id),
            )
            if cursor.rowcount == 0:
                return
            case = connection.execute(
                'SELECT case_slug FROM "case" WHERE id = ?',
                (row["case_id"],),
            ).fetchone()
            documents = connection.execute(
                "SELECT id, case_id, original_filename, stored_file_path, confidence_code, tags "
                "FROM document WHERE case_id = ?",
                (row["case_id"],),
            ).fetchall()
        if not case:
            self._mark_failed(analysis_id, "The case no longer exists.")
            return
        try:
            generated = asyncio.run(
                self._analysis_service.generate_analysis(
                    case_id=str(row["case_id"]),
                    case_root=self._settings.cases_root / str(case["case_slug"]),
                    prompt=str(row["prompt"]),
                    analysis_type=str(row["analysis_type"]),
                    case_documents=[dict(item) for item in documents],
                )
            )
        except Exception as exc:
            logger.error(
                "Analysis generation failed for %s (%s): %s",
                analysis_id,
                type(exc).__name__,
                str(exc)[:500],
            )
            self._mark_failed(analysis_id, self._user_error(exc))
            return

        with get_connection(self._settings) as connection:
            connection.execute(
                "UPDATE analysis SET status = 'complete', rag_answer = ?, summary_text = ?, "
                "charts_json = ?, highlight_json = ?, subgraph_json = ?, references_json = ?, "
                "chunks_json = ?, model_name = ?, error = NULL, pending_repair_json = NULL, "
                "updated_at = ? WHERE id = ? AND status = 'generating'",
                (
                    generated.get("rag_answer"),
                    generated.get("summary_text"),
                    self._json(generated.get("charts") or []),
                    self._json(generated.get("highlight") or {}),
                    self._json(generated.get("subgraph") or {}),
                    self._json(generated.get("references") or []),
                    self._json(generated.get("chunks") or []),
                    generated.get("model_name") or self._settings.rag_llm_model,
                    utc_now_iso(),
                    analysis_id,
                ),
            )

    def _process_repair(self, row: dict[str, Any]) -> None:
        analysis_id = str(row["id"])
        pending = self._decode(row.get("pending_repair_json"))
        charts = self._decode(row.get("charts_json")) or []
        if not isinstance(pending, dict):
            self._mark_failed(analysis_id, "The queued chart repair is invalid.")
            return
        chart_index = next(
            (
                index
                for index, chart in enumerate(charts)
                if isinstance(chart, dict) and chart.get("id") == pending.get("chart_id")
            ),
            None,
        )
        if chart_index is None:
            self._mark_failed(analysis_id, "The chart queued for repair no longer exists.")
            return

        with get_connection(self._settings) as connection:
            cursor = connection.execute(
                "UPDATE analysis SET status = 'repairing', error = NULL, updated_at = ? "
                "WHERE id = ? AND status = 'repair_queued'",
                (utc_now_iso(), analysis_id),
            )
            if cursor.rowcount == 0:
                return

        chart = dict(charts[chart_index])
        deadline = time.monotonic() + max(
            0.0, float(self._settings.rag_network_retry_window_seconds)
        )
        try:
            repaired = asyncio.run(
                self._analysis_service.repair_chart(
                    analysis_type=str(row["analysis_type"]),
                    chart=chart,
                    user_prompt=str(row["prompt"]),
                    rag_answer=str(row.get("rag_answer") or ""),
                    graph=self._decode(row.get("subgraph_json"))
                    or {"nodes": [], "edges": []},
                    error=str(pending.get("error") or ""),
                    mermaid_code=str(
                        pending.get("mermaid_code")
                        or chart.get("mermaid_code")
                        or ""
                    ),
                    deadline=deadline,
                )
            )
        except Exception as exc:
            logger.error(
                "Analysis chart repair failed for %s (%s): %s",
                analysis_id,
                type(exc).__name__,
                str(exc)[:500],
            )
            self._mark_failed(analysis_id, self._user_error(exc))
            return

        repaired["repair_attempts"] = int(chart.get("repair_attempts") or 0) + 1
        charts[chart_index] = repaired
        with get_connection(self._settings) as connection:
            connection.execute(
                "UPDATE analysis SET status = 'complete', charts_json = ?, error = NULL, "
                "pending_repair_json = NULL, updated_at = ? "
                "WHERE id = ? AND status = 'repairing'",
                (self._json(charts), utc_now_iso(), analysis_id),
            )

    def _mark_failed(self, analysis_id: str, error: str) -> None:
        concise = " ".join(str(error or "").split()).strip()[:1000]
        with get_connection(self._settings) as connection:
            connection.execute(
                "UPDATE analysis SET status = 'failed', error = ?, updated_at = ? "
                "WHERE id = ?",
                (concise or "Analysis generation failed.", utc_now_iso(), analysis_id),
            )

    @staticmethod
    def _user_error(exc: Exception) -> str:
        message = " ".join(str(exc).split()).strip()
        if not message:
            return "Analysis generation failed."
        return message[:1000]

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @staticmethod
    def _decode(value: Any) -> Any:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
