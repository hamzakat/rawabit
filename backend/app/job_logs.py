from __future__ import annotations

from contextvars import ContextVar
import logging
import traceback
from typing import Any

from .db import get_connection
from .settings import Settings
from .utils import utc_now_iso

_active_job_log_context: ContextVar[dict[str, str] | None] = ContextVar(
    "rawabit_active_job_log_context",
    default=None,
)
_HTTP_ACCESS_LOGGERS = {"uvicorn.access"}
_CAPTURE_PREF_LOGGERS = ("lightrag", "nano-vectordb", "raganything")


def _persist_job_log_row(
    settings: Settings,
    *,
    case_id: str,
    job_id: str,
    level: str,
    message: str,
) -> None:
    with get_connection(settings) as connection:
        connection.execute(
            "INSERT INTO ingestion_job_log (job_id, case_id, created_at, level, message) "
            "VALUES (?, ?, ?, ?, ?)",
            (job_id, case_id, utc_now_iso(), level, message),
        )


def set_active_job_log_context(case_id: str, job_id: str) -> None:
    _active_job_log_context.set({"case_id": case_id, "job_id": job_id})


def clear_active_job_log_context() -> None:
    _active_job_log_context.set(None)


def install_job_log_capture(settings: Settings) -> None:
    root_logger = logging.getLogger()
    manager = root_logger.manager

    for logger in _iter_all_loggers(manager):
        for handler in list(logger.handlers):
            if isinstance(handler, JobContextLogCaptureHandler):
                logger.removeHandler(handler)
                handler.close()

    for handler in list(root_logger.handlers):
        if isinstance(handler, JobContextLogCaptureHandler):
            root_logger.removeHandler(handler)
            handler.close()

    capture_handler = JobContextLogCaptureHandler(settings)
    root_logger.addHandler(capture_handler)

    for logger in _iter_all_loggers(manager):
        _attach_handler_for_non_propagating_logger(logger, capture_handler)

    for logger_name in _CAPTURE_PREF_LOGGERS:
        _attach_handler_for_non_propagating_logger(
            logging.getLogger(logger_name),
            capture_handler,
        )


def _iter_all_loggers(manager: Any) -> list[logging.Logger]:
    loggers: list[logging.Logger] = []
    logger_dict = getattr(manager, "loggerDict", {})
    for value in logger_dict.values():
        if isinstance(value, logging.Logger):
            loggers.append(value)
    return loggers


def _attach_handler_for_non_propagating_logger(
    logger: logging.Logger,
    handler: logging.Handler,
) -> None:
    if logger.name in _HTTP_ACCESS_LOGGERS:
        return
    if logger.name.startswith("backend.app.job_logs"):
        return
    if logger.propagate:
        return
    if any(existing is handler for existing in logger.handlers):
        return
    logger.addHandler(handler)


class JobContextLogCaptureHandler(logging.Handler):
    def __init__(self, settings: Settings) -> None:
        super().__init__(level=logging.NOTSET)
        self._settings = settings

    def emit(self, record: logging.LogRecord) -> None:
        if record.name in _HTTP_ACCESS_LOGGERS:
            return
        if record.name.startswith("backend.app.job_logs"):
            return
        context = _active_job_log_context.get()
        if not context:
            return
        message = self._format_record_message(record)
        if not message:
            return
        level = self._normalize_level(record.levelname)
        try:
            _persist_job_log_row(
                self._settings,
                case_id=context["case_id"],
                job_id=context["job_id"],
                level=level,
                message=message,
            )
        except Exception:
            # Keep log capture best-effort. Never recurse to logger here.
            return

    @staticmethod
    def _normalize_level(level_name: str) -> str:
        normalized = (level_name or "info").strip().lower() or "info"
        if normalized == "warn":
            return "warning"
        return normalized

    @staticmethod
    def _format_record_message(record: logging.LogRecord) -> str:
        source = (record.name or "").strip()
        message = (record.getMessage() or "").strip()
        exc_text = ""
        if record.exc_info:
            exc_text = "".join(traceback.format_exception(*record.exc_info)).strip()
        elif isinstance(record.exc_text, str):
            exc_text = record.exc_text.strip()
        if exc_text:
            message = f"{message}\n{exc_text}".strip() if message else exc_text
        if not message and not source:
            return ""
        if source and message:
            return f"{source}: {message}"
        return source or message


def append_job_log(
    settings: Settings,
    case_id: str,
    job_id: str,
    message: str,
    level: str = "info",
) -> None:
    cleaned = (message or "").strip()
    if not cleaned:
        return
    normalized_level = (level or "info").strip().lower() or "info"

    try:
        _persist_job_log_row(
            settings,
            case_id=case_id,
            job_id=job_id,
            level=normalized_level,
            message=cleaned,
        )
    except Exception:
        # Keep ingestion flow resilient if job log persistence fails.
        return


def list_job_logs(
    settings: Settings,
    case_id: str,
    job_id: str,
    *,
    after_id: int = 0,
    limit: int = 500,
) -> list[dict[str, Any]]:
    normalized_after = max(0, int(after_id))
    normalized_limit = max(1, min(int(limit), 2000))
    with get_connection(settings) as connection:
        rows = connection.execute(
            "SELECT id, job_id, case_id, created_at, level, message "
            "FROM ingestion_job_log "
            "WHERE case_id = ? AND job_id = ? AND id > ? "
            "ORDER BY id ASC LIMIT ?",
            (case_id, job_id, normalized_after, normalized_limit),
        ).fetchall()
    return [dict(row) for row in rows]
