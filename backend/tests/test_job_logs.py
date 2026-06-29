from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import uuid

from backend.app.db import get_connection, init_db
from backend.app.job_logs import (
    append_job_log,
    clear_active_job_log_context,
    install_job_log_capture,
    list_job_logs,
    set_active_job_log_context,
)
from backend.app.settings import get_settings
from backend.app.utils import utc_now_iso


def _configure_env(temp_dir: Path) -> None:
    os.environ["RAWABIT_DB_PATH"] = str(temp_dir / "db.sqlite")
    os.environ["RAWABIT_CASES_ROOT"] = str(temp_dir / "cases")


def _seed_case_document_job(settings, job_id: str) -> tuple[str, str]:
    case_id = "case-logs-1"
    document_id = "doc-logs-1"
    now = utc_now_iso()
    with get_connection(settings) as connection:
        connection.execute(
            "INSERT INTO \"case\" (id, name, description, status, case_slug, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (case_id, "Log Capture Case", None, "active", "log-capture-case", now, now),
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
                16,
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
            (job_id, case_id, document_id, "parsing", 10, now, None, None),
        )
    return case_id, document_id


def test_job_log_capture_persists_ingestion_logs_and_skips_http_access_logs() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"job-logs-capture-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)

    root_logger = logging.getLogger()
    previous_level = root_logger.level
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        init_db(settings)
        job_id = "job-log-capture-1"
        case_id, _ = _seed_case_document_job(settings, job_id=job_id)
        install_job_log_capture(settings)
        root_logger.setLevel(logging.INFO)

        clear_active_job_log_context()
        set_active_job_log_context(case_id=case_id, job_id=job_id)
        logging.getLogger("backend.app.ingestion_pipeline").info("Starting parser stage.")
        logging.getLogger("uvicorn.access").info(
            '127.0.0.1:55555 - "GET /api/cases/example/jobs HTTP/1.1" 200 OK'
        )
        try:
            raise RuntimeError("simulated parsing failure")
        except RuntimeError:
            logging.getLogger("raganything.parser").exception("Parser crashed")
        clear_active_job_log_context()

        rows = list_job_logs(settings, case_id=case_id, job_id=job_id, limit=1000)
        messages = [str(row["message"]) for row in rows]
        assert any("backend.app.ingestion_pipeline: Starting parser stage." in msg for msg in messages)
        assert not any("GET /api/cases/example/jobs HTTP/1.1" in msg for msg in messages)
        assert any("raganything.parser: Parser crashed" in msg for msg in messages)
        assert any("Traceback (most recent call last):" in msg for msg in messages)
        assert any("RuntimeError: simulated parsing failure" in msg for msg in messages)
    finally:
        clear_active_job_log_context()
        root_logger.setLevel(previous_level)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_job_log_capture_collects_non_propagating_logger_records() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"job-logs-non-propagating-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)

    root_logger = logging.getLogger()
    previous_level = root_logger.level
    non_propagating_logger = logging.getLogger("lightrag")
    previous_propagate = non_propagating_logger.propagate
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        init_db(settings)
        job_id = "job-log-capture-3"
        case_id, _ = _seed_case_document_job(settings, job_id=job_id)
        non_propagating_logger.propagate = False
        install_job_log_capture(settings)
        root_logger.setLevel(logging.INFO)

        set_active_job_log_context(case_id=case_id, job_id=job_id)
        non_propagating_logger.info("LightRAG init progress row")
        clear_active_job_log_context()

        rows = list_job_logs(settings, case_id=case_id, job_id=job_id, limit=1000)
        messages = [str(row["message"]) for row in rows]
        assert any("lightrag: LightRAG init progress row" in msg for msg in messages)
    finally:
        clear_active_job_log_context()
        root_logger.setLevel(previous_level)
        non_propagating_logger.propagate = previous_propagate
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_append_job_log_does_not_truncate_messages() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"job-logs-no-truncation-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        init_db(settings)
        job_id = "job-log-capture-2"
        case_id, _ = _seed_case_document_job(settings, job_id=job_id)
        long_message = "X" * 12000

        append_job_log(
            settings,
            case_id=case_id,
            job_id=job_id,
            message=long_message,
            level="info",
        )

        rows = list_job_logs(settings, case_id=case_id, job_id=job_id, limit=10)
        assert rows
        assert rows[-1]["message"] == long_message
    finally:
        clear_active_job_log_context()
        shutil.rmtree(temp_dir, ignore_errors=True)
