from __future__ import annotations

import sqlite3
import threading

from .db import get_connection
from .ingestion_pipeline import IngestionPipeline
from .job_logs import append_job_log
from .settings import Settings
from .utils import utc_now_iso


class JobRunner:
    def __init__(
        self, settings: Settings, pipeline: IngestionPipeline | None = None
    ) -> None:
        self._settings = settings
        self._pipeline = pipeline or IngestionPipeline(settings)
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._state_lock = threading.Lock()
        self._active_case_ids: set[str] = set()

    def start(self) -> None:
        if any(thread.is_alive() for thread in self._threads):
            return
        self._recover_inflight_jobs()
        self._stop_event.clear()
        worker_count = max(1, int(self._settings.ingestion_worker_concurrency))
        self._threads = [
            threading.Thread(
                target=self._run, daemon=True, name=f"rawabit-ingest-{idx + 1}"
            )
            for idx in range(worker_count)
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads = []

    def _run(self) -> None:
        poll_interval = max(self._settings.ingestion_poll_interval_seconds, 0.25)
        while not self._stop_event.is_set():
            try:
                next_job = self._next_queued_job()
            except sqlite3.OperationalError:
                if self._stop_event.is_set():
                    return
                self._stop_event.wait(poll_interval)
                continue
            if next_job:
                self._process_job(*next_job)
            else:
                self._stop_event.wait(poll_interval)

    def _next_queued_job(self) -> tuple[str, str] | None:
        with self._state_lock:
            with get_connection(self._settings) as connection:
                rows = connection.execute(
                    "SELECT id, case_id FROM ingestion_job WHERE status = ? "
                    "ORDER BY "
                    "CASE queue_priority WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END ASC, "
                    "started_at ASC",
                    ("queued",),
                ).fetchall()
            for row in rows:
                case_id = str(row["case_id"] or "").strip()
                job_id = str(row["id"] or "").strip()
                if not case_id or not job_id:
                    continue
                if case_id in self._active_case_ids:
                    continue
                self._active_case_ids.add(case_id)
                return job_id, case_id
        return None

    def _recover_inflight_jobs(self) -> None:
        with get_connection(self._settings) as connection:
            rows = connection.execute(
                "SELECT id, case_id FROM ingestion_job WHERE status IN (?, ?, ?)",
                ("parsing", "inserting", "indexing"),
            ).fetchall()
        for row in rows:
            append_job_log(
                self._settings,
                case_id=row["case_id"],
                job_id=row["id"],
                level="error",
                message=(
                    "Worker restart detected while job was in-flight. "
                    "Marking job as failed."
                ),
            )
            self._fail_job(
                row["id"],
                "Ingestion worker restarted before completion. Re-ingest the document to retry.",
            )

    def _process_job(self, job_id: str, reserved_case_id: str) -> None:
        now = utc_now_iso()
        try:
            with get_connection(self._settings) as connection:
                cursor = connection.execute(
                    "UPDATE ingestion_job SET status = ?, progress = ?, started_at = ?, error = NULL, finished_at = NULL "
                    "WHERE id = ? AND status = ?",
                    ("parsing", 5, now, job_id, "queued"),
                )
                if cursor.rowcount == 0:
                    return
                connection.execute(
                    "UPDATE document SET ingestion_status = ?, ingestion_error = NULL, updated_at = ? "
                    "WHERE id = (SELECT document_id FROM ingestion_job WHERE id = ?) ",
                    ("parsing", now, job_id),
                )
                case_row = connection.execute(
                    "SELECT case_id FROM ingestion_job WHERE id = ?",
                    (job_id,),
                ).fetchone()
            if case_row:
                append_job_log(
                    self._settings,
                    case_id=case_row["case_id"],
                    job_id=job_id,
                    message="Worker picked up queued job and started ingestion.",
                )

            error_holder: dict[str, Exception | None] = {"error": None}

            def _target() -> None:
                try:
                    self._pipeline.process_job(job_id)
                except (
                    Exception
                ) as exc:  # pragma: no cover - exercised via outer assertions
                    error_holder["error"] = exc

            worker = threading.Thread(target=_target, daemon=True)
            worker.start()
            worker.join()

            if error_holder["error"] is not None:
                case_id = self._case_id_for_job(job_id)
                if case_id:
                    append_job_log(
                        self._settings,
                        case_id=case_id,
                        job_id=job_id,
                        level="error",
                        message=f"Pipeline raised exception: {error_holder['error']}",
                    )
                self._fail_job(job_id, str(error_holder["error"]))
                return

            status = self._get_job_status(job_id)
            if status not in {"complete", "completed_with_warnings", "failed"}:
                self._fail_job(
                    job_id,
                    f"Ingestion ended in non-terminal status '{status}'.",
                )
        finally:
            with self._state_lock:
                self._active_case_ids.discard(reserved_case_id)

    def _get_job_status(self, job_id: str) -> str | None:
        try:
            with get_connection(self._settings) as connection:
                row = connection.execute(
                    "SELECT status FROM ingestion_job WHERE id = ?",
                    (job_id,),
                ).fetchone()
        except sqlite3.OperationalError:
            return None
        if not row:
            return None
        return row["status"]

    def _case_id_for_job(self, job_id: str) -> str | None:
        try:
            with get_connection(self._settings) as connection:
                row = connection.execute(
                    "SELECT case_id FROM ingestion_job WHERE id = ?",
                    (job_id,),
                ).fetchone()
        except sqlite3.OperationalError:
            return None
        if not row:
            return None
        return row["case_id"]

    def _fail_job(self, job_id: str, error: str) -> None:
        finished_at = utc_now_iso()
        trimmed_error = error.strip()
        if len(trimmed_error) > 1000:
            trimmed_error = trimmed_error[:1000] + "..."
        case_id = self._case_id_for_job(job_id)
        with get_connection(self._settings) as connection:
            connection.execute(
                "UPDATE ingestion_job SET status = ?, progress = ?, finished_at = ?, error = ? WHERE id = ?",
                ("failed", None, finished_at, trimmed_error, job_id),
            )
            connection.execute(
                "UPDATE document SET ingestion_status = ?, ingestion_error = ?, updated_at = ? "
                "WHERE id = (SELECT document_id FROM ingestion_job WHERE id = ?)",
                ("failed", trimmed_error, finished_at, job_id),
            )
        if case_id:
            append_job_log(
                self._settings,
                case_id=case_id,
                job_id=job_id,
                level="error",
                message=f"Job failed: {trimmed_error}",
            )
