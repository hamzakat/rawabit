from __future__ import annotations

import os
from pathlib import Path
import shutil
import time
import uuid

from backend.app.db import get_connection, init_db
from backend.app.job_runner import JobRunner
from backend.app.settings import get_settings
from backend.app.utils import utc_now_iso


def _configure_env(temp_dir: Path) -> None:
    os.environ["RAWABIT_DB_PATH"] = str(temp_dir / "db.sqlite")
    os.environ["RAWABIT_CASES_ROOT"] = str(temp_dir / "cases")


def _seed_case_document_job(settings, job_id: str) -> tuple[str, str]:
    case_id = "case-1"
    document_id = "doc-1"
    now = utc_now_iso()
    with get_connection(settings) as connection:
        connection.execute(
            "INSERT INTO \"case\" (id, name, description, status, case_slug, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (case_id, "Test Case", None, "active", "test-case", now, now),
        )
        connection.execute(
            "INSERT INTO document (id, case_id, original_filename, stored_file_path, mime_type, size_bytes, "
            "confidence_source_reliability, confidence_information_validity, confidence_code, tags, notes, "
            "ingestion_status, ingestion_error, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                document_id,
                case_id,
                "doc.txt",
                "raw/doc.txt",
                "text/plain",
                10,
                "A",
                "1",
                "A1",
                None,
                None,
                "queued",
                None,
                now,
                now,
            ),
        )
        connection.execute(
            "INSERT INTO ingestion_job (id, case_id, document_id, status, progress, started_at, finished_at, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, case_id, document_id, "queued", None, now, None, None),
        )
    return case_id, document_id


class _SuccessPipeline:
    def __init__(self, settings) -> None:
        self._settings = settings

    def process_job(self, job_id: str) -> None:
        now = utc_now_iso()
        with get_connection(self._settings) as connection:
            connection.execute(
                "UPDATE ingestion_job SET status = ?, progress = ?, finished_at = ?, error = NULL WHERE id = ?",
                ("complete", 100, now, job_id),
            )
            connection.execute(
                "UPDATE document SET ingestion_status = ?, ingestion_error = NULL, updated_at = ? "
                "WHERE id = (SELECT document_id FROM ingestion_job WHERE id = ?)",
                ("complete", now, job_id),
            )


class _FailingPipeline:
    def process_job(self, _: str) -> None:
        raise RuntimeError("pipeline exploded")


class _NonTerminalPipeline:
    def process_job(self, _: str) -> None:
        # Returns without setting terminal status on purpose.
        return


class _SlowSuccessPipeline:
    def __init__(self, settings) -> None:
        self._settings = settings

    def process_job(self, job_id: str) -> None:
        time.sleep(0.2)
        now = utc_now_iso()
        with get_connection(self._settings) as connection:
            connection.execute(
                "UPDATE ingestion_job SET status = ?, progress = ?, finished_at = ?, error = NULL WHERE id = ?",
                ("complete", 100, now, job_id),
            )
            connection.execute(
                "UPDATE document SET ingestion_status = ?, ingestion_error = NULL, updated_at = ? "
                "WHERE id = (SELECT document_id FROM ingestion_job WHERE id = ?)",
                ("complete", now, job_id),
            )


def test_job_runner_marks_failed_when_pipeline_raises() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"jobrunner-fail-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        init_db(settings)
        _seed_case_document_job(settings, job_id="job-1")

        runner = JobRunner(settings=settings, pipeline=_FailingPipeline())
        runner._process_job("job-1")  # noqa: SLF001 - intentional targeted unit test

        with get_connection(settings) as connection:
            job_row = connection.execute(
                "SELECT status, progress, error, finished_at FROM ingestion_job WHERE id = ?",
                ("job-1",),
            ).fetchone()
            doc_row = connection.execute(
                "SELECT ingestion_status, ingestion_error FROM document WHERE id = ?",
                ("doc-1",),
            ).fetchone()
            log_rows = connection.execute(
                "SELECT level, message FROM ingestion_job_log WHERE job_id = ? ORDER BY id ASC",
                ("job-1",),
            ).fetchall()

        assert job_row is not None
        assert job_row["status"] == "failed"
        assert job_row["progress"] is None
        assert "pipeline exploded" in job_row["error"]
        assert job_row["finished_at"] is not None

        assert doc_row is not None
        assert doc_row["ingestion_status"] == "failed"
        assert "pipeline exploded" in doc_row["ingestion_error"]
        assert any("pipeline exploded" in (row["message"] or "") for row in log_rows)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_job_runner_fails_when_pipeline_returns_non_terminal_status() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"jobrunner-nonterminal-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        init_db(settings)
        _seed_case_document_job(settings, job_id="job-nonterminal")

        runner = JobRunner(settings=settings, pipeline=_NonTerminalPipeline())
        runner._process_job("job-nonterminal")  # noqa: SLF001 - targeted unit test

        with get_connection(settings) as connection:
            job_row = connection.execute(
                "SELECT status, error FROM ingestion_job WHERE id = ?",
                ("job-nonterminal",),
            ).fetchone()

        assert job_row is not None
        assert job_row["status"] == "failed"
        assert "non-terminal status" in (job_row["error"] or "")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_job_runner_waits_for_slow_pipeline_without_watchdog_timeout() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"jobrunner-slow-success-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        init_db(settings)
        _seed_case_document_job(settings, job_id="job-slow-success")
        runner = JobRunner(settings=settings, pipeline=_SlowSuccessPipeline(settings))
        runner._process_job("job-slow-success")  # noqa: SLF001 - targeted unit test

        with get_connection(settings) as connection:
            job_row = connection.execute(
                "SELECT status, error FROM ingestion_job WHERE id = ?",
                ("job-slow-success",),
            ).fetchone()
        assert job_row is not None
        assert job_row["status"] == "complete"
        assert job_row["error"] is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_job_runner_keeps_success_status_from_pipeline() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"jobrunner-success-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        init_db(settings)
        _seed_case_document_job(settings, job_id="job-2")

        runner = JobRunner(settings=settings, pipeline=_SuccessPipeline(settings))
        runner._process_job("job-2")  # noqa: SLF001 - intentional targeted unit test

        with get_connection(settings) as connection:
            job_row = connection.execute(
                "SELECT status, progress, error, finished_at FROM ingestion_job WHERE id = ?",
                ("job-2",),
            ).fetchone()
            doc_row = connection.execute(
                "SELECT ingestion_status, ingestion_error FROM document WHERE id = ?",
                ("doc-1",),
            ).fetchone()

        assert job_row is not None
        assert job_row["status"] == "complete"
        assert job_row["progress"] == 100
        assert job_row["error"] is None
        assert job_row["finished_at"] is not None

        assert doc_row is not None
        assert doc_row["ingestion_status"] == "complete"
        assert doc_row["ingestion_error"] is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_job_runner_start_recovers_inflight_jobs() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"jobrunner-recover-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        init_db(settings)
        _seed_case_document_job(settings, job_id="job-recover")

        now = utc_now_iso()
        with get_connection(settings) as connection:
            connection.execute(
                "UPDATE ingestion_job SET status = ?, progress = ?, started_at = ?, finished_at = NULL, error = NULL WHERE id = ?",
                ("parsing", 15, now, "job-recover"),
            )
            connection.execute(
                "UPDATE document SET ingestion_status = ?, ingestion_error = NULL, updated_at = ? WHERE id = ?",
                ("parsing", now, "doc-1"),
            )

        runner = JobRunner(settings=settings, pipeline=_SuccessPipeline(settings))
        runner.start()
        runner.stop()

        with get_connection(settings) as connection:
            job_row = connection.execute(
                "SELECT status, progress, error, finished_at FROM ingestion_job WHERE id = ?",
                ("job-recover",),
            ).fetchone()
            doc_row = connection.execute(
                "SELECT ingestion_status, ingestion_error FROM document WHERE id = ?",
                ("doc-1",),
            ).fetchone()

        assert job_row is not None
        assert job_row["status"] == "failed"
        assert job_row["progress"] is None
        assert "restarted before completion" in (job_row["error"] or "").lower()
        assert job_row["finished_at"] is not None

        assert doc_row is not None
        assert doc_row["ingestion_status"] == "failed"
        assert "restarted before completion" in (doc_row["ingestion_error"] or "").lower()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_job_runner_fails_fast_when_openrouter_key_missing() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"jobrunner-openrouter-missing-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)

    old_rawabit_key = os.environ.get("RAWABIT_LLM_PROVIDER_API_KEY")
    old_openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    os.environ["RAWABIT_LLM_PROVIDER_API_KEY"] = ""
    os.environ["OPENROUTER_API_KEY"] = ""
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        init_db(settings)
        _seed_case_document_job(settings, job_id="job-3")

        raw_dir = temp_dir / "cases" / "test-case" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "doc.txt").write_text("evidence", encoding="utf-8")

        runner = JobRunner(settings=settings)
        runner._process_job("job-3")  # noqa: SLF001 - targeted failure-mode test

        with get_connection(settings) as connection:
            job_row = connection.execute(
                "SELECT status, error FROM ingestion_job WHERE id = ?",
                ("job-3",),
            ).fetchone()
            doc_row = connection.execute(
                "SELECT ingestion_status, ingestion_error FROM document WHERE id = ?",
                ("doc-1",),
            ).fetchone()

        assert job_row is not None
        assert job_row["status"] == "failed"
        assert "LLM provider API key is not configured" in (job_row["error"] or "")

        assert doc_row is not None
        assert doc_row["ingestion_status"] == "failed"
        assert "LLM provider API key is not configured" in (doc_row["ingestion_error"] or "")
    finally:
        if old_rawabit_key is None:
            os.environ.pop("RAWABIT_LLM_PROVIDER_API_KEY", None)
        else:
            os.environ["RAWABIT_LLM_PROVIDER_API_KEY"] = old_rawabit_key
        if old_openrouter_key is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = old_openrouter_key
        shutil.rmtree(temp_dir, ignore_errors=True)
