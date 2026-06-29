from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from .settings import Settings


SCHEMA_STATEMENTS: Iterable[str] = (
    """
    CREATE TABLE IF NOT EXISTS "case" (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        status TEXT NOT NULL,
        case_slug TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS case_summary_cache (
        case_id TEXT PRIMARY KEY,
        summary_json TEXT NOT NULL,
        source_document_count INTEGER NOT NULL,
        source_completed_job_id TEXT,
        last_refreshed_at TEXT NOT NULL,
        FOREIGN KEY (case_id) REFERENCES "case"(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS document (
        id TEXT PRIMARY KEY,
        case_id TEXT NOT NULL,
        original_filename TEXT NOT NULL,
        stored_file_path TEXT NOT NULL,
        content_hash_sha256 TEXT,
        mime_type TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        confidence_source_reliability TEXT NOT NULL,
        confidence_information_validity TEXT NOT NULL,
        confidence_code TEXT NOT NULL,
        tags TEXT,
        notes TEXT,
        ingest_model_name TEXT,
        ingestion_status TEXT NOT NULL,
        ingestion_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (case_id) REFERENCES "case"(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS chat (
        id TEXT PRIMARY KEY,
        case_id TEXT NOT NULL,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (case_id) REFERENCES "case"(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS message (
        id TEXT PRIMARY KEY,
        chat_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        rag_metadata_json TEXT,
        FOREIGN KEY (chat_id) REFERENCES chat(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS analysis (
        id TEXT PRIMARY KEY,
        case_id TEXT NOT NULL,
        analysis_type TEXT NOT NULL,
        prompt TEXT NOT NULL,
        title TEXT NOT NULL,
        status TEXT NOT NULL,
        rag_answer TEXT,
        summary_text TEXT,
        charts_json TEXT,
        highlight_json TEXT,
        subgraph_json TEXT,
        references_json TEXT,
        chunks_json TEXT,
        model_name TEXT,
        error TEXT,
        pending_repair_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (case_id) REFERENCES "case"(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analysis_case_updated
    ON analysis(case_id, updated_at DESC, created_at DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_job (
        id TEXT PRIMARY KEY,
        case_id TEXT NOT NULL,
        document_id TEXT NOT NULL,
        ingest_profile TEXT NOT NULL DEFAULT 'balanced_fast_intel',
        processing_mode TEXT NOT NULL DEFAULT 'multimodal',
        advanced_overrides_json TEXT,
        preflight_json TEXT,
        effective_config_json TEXT,
        complexity_class TEXT,
        eta_seconds INTEGER,
        queue_priority TEXT NOT NULL DEFAULT 'normal',
        route_type TEXT,
        status TEXT NOT NULL,
        progress INTEGER,
        started_at TEXT,
        finished_at TEXT,
        parse_duration_s REAL,
        insert_duration_s REAL,
        finalize_duration_s REAL,
        current_stage TEXT,
        error TEXT,
        FOREIGN KEY (case_id) REFERENCES "case"(id) ON DELETE CASCADE,
        FOREIGN KEY (document_id) REFERENCES document(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_job_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        case_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        level TEXT NOT NULL,
        message TEXT NOT NULL,
        FOREIGN KEY (job_id) REFERENCES ingestion_job(id) ON DELETE CASCADE,
        FOREIGN KEY (case_id) REFERENCES "case"(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_ingestion_job_log_job_id_id
    ON ingestion_job_log(job_id, id);
    """,
    """
    CREATE TABLE IF NOT EXISTS llm_call_trace (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        case_id TEXT NOT NULL,
        stage TEXT NOT NULL,
        model TEXT NOT NULL,
        provider TEXT NOT NULL DEFAULT 'openai',
        request_summary TEXT,
        latency_ms REAL,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        total_tokens INTEGER,
        error TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (job_id) REFERENCES ingestion_job(id) ON DELETE CASCADE,
        FOREIGN KEY (case_id) REFERENCES "case"(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_llm_call_trace_job
    ON llm_call_trace(job_id, id);
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS document_search_fts USING fts5(
        case_id UNINDEXED,
        document_id UNINDEXED,
        source_kind UNINDEXED,
        segment_key UNINDEXED,
        original_filename,
        stored_file_path UNINDEXED,
        confidence_code UNINDEXED,
        content,
        tokenize='unicode61'
    );
    """,
)


def _ensure_document_columns(connection: sqlite3.Connection) -> None:
    columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(document)").fetchall()
    }
    migrations: list[str] = []
    if "content_hash_sha256" not in columns:
        migrations.append("ALTER TABLE document ADD COLUMN content_hash_sha256 TEXT")
    if "ingest_model_name" not in columns:
        migrations.append("ALTER TABLE document ADD COLUMN ingest_model_name TEXT")
    for statement in migrations:
        connection.execute(statement)
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_case_hash "
        "ON document(case_id, content_hash_sha256)"
    )


def _ensure_ingestion_job_columns(connection: sqlite3.Connection) -> None:
    columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(ingestion_job)").fetchall()
    }
    migrations: list[str] = []
    if "ingest_profile" not in columns:
        migrations.append(
            "ALTER TABLE ingestion_job ADD COLUMN ingest_profile TEXT NOT NULL DEFAULT 'balanced_fast_intel'"
        )
    if "processing_mode" not in columns:
        migrations.append(
            "ALTER TABLE ingestion_job ADD COLUMN processing_mode TEXT NOT NULL DEFAULT 'multimodal'"
        )
    if "route_type" not in columns:
        migrations.append("ALTER TABLE ingestion_job ADD COLUMN route_type TEXT")
    if "advanced_overrides_json" not in columns:
        migrations.append("ALTER TABLE ingestion_job ADD COLUMN advanced_overrides_json TEXT")
    if "preflight_json" not in columns:
        migrations.append("ALTER TABLE ingestion_job ADD COLUMN preflight_json TEXT")
    if "effective_config_json" not in columns:
        migrations.append("ALTER TABLE ingestion_job ADD COLUMN effective_config_json TEXT")
    if "complexity_class" not in columns:
        migrations.append("ALTER TABLE ingestion_job ADD COLUMN complexity_class TEXT")
    if "eta_seconds" not in columns:
        migrations.append("ALTER TABLE ingestion_job ADD COLUMN eta_seconds INTEGER")
    if "queue_priority" not in columns:
        migrations.append(
            "ALTER TABLE ingestion_job ADD COLUMN queue_priority TEXT NOT NULL DEFAULT 'normal'"
        )
    if "parse_duration_s" not in columns:
        migrations.append("ALTER TABLE ingestion_job ADD COLUMN parse_duration_s REAL")
    if "insert_duration_s" not in columns:
        migrations.append("ALTER TABLE ingestion_job ADD COLUMN insert_duration_s REAL")
    if "finalize_duration_s" not in columns:
        migrations.append("ALTER TABLE ingestion_job ADD COLUMN finalize_duration_s REAL")
    if "current_stage" not in columns:
        migrations.append("ALTER TABLE ingestion_job ADD COLUMN current_stage TEXT")
    for statement in migrations:
        connection.execute(statement)


def _ensure_analysis_schema(connection: sqlite3.Connection) -> None:
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(analysis)").fetchall()
    }
    if not columns:
        return
    if "charts_json" in columns and "diagram_html" not in columns:
        if "error" not in columns:
            connection.execute("ALTER TABLE analysis ADD COLUMN error TEXT")
        if "pending_repair_json" not in columns:
            connection.execute(
                "ALTER TABLE analysis ADD COLUMN pending_repair_json TEXT"
            )
        return
    connection.execute("DROP TABLE analysis")
    connection.execute(
        """
        CREATE TABLE analysis (
            id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            analysis_type TEXT NOT NULL,
            prompt TEXT NOT NULL,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            rag_answer TEXT,
            summary_text TEXT,
            charts_json TEXT,
            highlight_json TEXT,
            subgraph_json TEXT,
            references_json TEXT,
            chunks_json TEXT,
            model_name TEXT,
            error TEXT,
            pending_repair_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES "case"(id) ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        "CREATE INDEX idx_analysis_case_updated "
        "ON analysis(case_id, updated_at DESC, created_at DESC)"
    )


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def init_db(settings: Settings) -> None:
    connection = _connect(settings.db_path)
    try:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        _ensure_document_columns(connection)
        _ensure_ingestion_job_columns(connection)
        _ensure_analysis_schema(connection)
        connection.commit()
    finally:
        connection.close()


@contextmanager
def get_connection(settings: Settings):
    connection = _connect(settings.db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
