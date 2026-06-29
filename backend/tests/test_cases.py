from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid

from fastapi.testclient import TestClient

from backend.app.db import get_connection
from backend.app.settings import get_settings


def _configure_env(temp_dir: Path) -> None:
    os.environ["RAWABIT_DB_PATH"] = str(temp_dir / "db.sqlite")
    os.environ["RAWABIT_CASES_ROOT"] = str(temp_dir / "cases")


def _create_case(client: TestClient, name: str, description: str | None = None) -> dict:
    payload: dict[str, str] = {"name": name}
    if description is not None:
        payload["description"] = description
    response = client.post("/api/cases", json=payload)
    assert response.status_code == 200
    return response.json()["data"]


def test_case_create_and_delete() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"case-test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            response = client.post("/api/cases", json={"name": "Test Case"})
            assert response.status_code == 200
            payload = response.json()
            assert payload["status"] == "success"
            case = payload["data"]
            case_id = case["id"]
            case_slug = case["case_slug"]
            cases_root = temp_dir / "cases"
            assert (cases_root / case_slug).exists()

            files = {"file": ("case-evidence.txt", b"evidence", "text/plain")}
            data = {
                "confidence_source_reliability": "A",
                "confidence_information_validity": "1",
            }
            upload_response = client.post(
                f"/api/cases/{case_id}/documents", data=data, files=files
            )
            assert upload_response.status_code == 200
            document_id = upload_response.json()["data"]["document_id"]

            case_root = cases_root / case_slug
            processed_dir = case_root / "processed" / document_id
            processed_dir.mkdir(parents=True, exist_ok=True)
            (processed_dir / "artifact.json").write_text("{}", encoding="utf-8")
            (case_root / "lightrag" / "scratch.tmp").write_text(
                "temp", encoding="utf-8"
            )

            delete_response = client.delete(f"/api/cases/{case_id}")
            assert delete_response.status_code == 200
            assert delete_response.json()["data"]["deleted"] is True
            assert not (cases_root / case_slug).exists()

            settings = get_settings()
            with get_connection(settings) as connection:
                case_row = connection.execute(
                    "SELECT id FROM \"case\" WHERE id = ?", (case_id,)
                ).fetchone()
                assert case_row is None
                doc_rows = connection.execute(
                    "SELECT COUNT(*) as c FROM document WHERE case_id = ?", (case_id,)
                ).fetchone()
                assert doc_rows["c"] == 0
                job_rows = connection.execute(
                    "SELECT COUNT(*) as c FROM ingestion_job WHERE case_id = ?",
                    (case_id,),
                ).fetchone()
                assert job_rows["c"] == 0

            get_response = client.get(f"/api/cases/{case_id}")
            assert get_response.status_code == 404
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_case_update_supports_rename() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"case-rename-test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _create_case(client, "Original Name", "Original description")
            response = client.patch(
                f"/api/cases/{case['id']}",
                json={"name": "Renamed Case", "description": "Updated description"},
            )
            assert response.status_code == 200
            updated = response.json()["data"]
            assert updated["name"] == "Renamed Case"
            assert updated["description"] == "Updated description"
            assert updated["case_slug"] == case["case_slug"]

            fetched = client.get(f"/api/cases/{case['id']}")
            assert fetched.status_code == 200
            fetched_case = fetched.json()["data"]
            assert fetched_case["name"] == "Renamed Case"
            assert fetched_case["description"] == "Updated description"
            assert fetched_case["case_slug"] == case["case_slug"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_case_list_includes_active_job_counts() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"case-jobs-test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            active_case = _create_case(client, "Active Jobs")
            idle_case = _create_case(client, "Idle Case")
            now = "2026-04-18T12:00:00Z"

            settings = get_settings()
            with get_connection(settings) as connection:
                connection.execute(
                    "INSERT INTO document (id, case_id, original_filename, stored_file_path, mime_type, size_bytes, "
                    "confidence_source_reliability, confidence_information_validity, confidence_code, ingestion_status, "
                    "ingestion_error, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "doc-active",
                        active_case["id"],
                        "active.txt",
                        "raw/active.txt",
                        "text/plain",
                        12,
                        "A",
                        "1",
                        "A1",
                        "queued",
                        None,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    "INSERT INTO document (id, case_id, original_filename, stored_file_path, mime_type, size_bytes, "
                    "confidence_source_reliability, confidence_information_validity, confidence_code, ingestion_status, "
                    "ingestion_error, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "doc-idle",
                        idle_case["id"],
                        "idle.txt",
                        "raw/idle.txt",
                        "text/plain",
                        12,
                        "A",
                        "1",
                        "A1",
                        "complete",
                        None,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    "INSERT INTO ingestion_job (id, case_id, document_id, ingest_profile, processing_mode, queue_priority, status, "
                    "progress, started_at, finished_at, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "job-active",
                        active_case["id"],
                        "doc-active",
                        "balanced_fast_intel",
                        "multimodal",
                        "normal",
                        "parsing",
                        35,
                        now,
                        None,
                        None,
                    ),
                )
                connection.execute(
                    "INSERT INTO ingestion_job (id, case_id, document_id, ingest_profile, processing_mode, queue_priority, status, "
                    "progress, started_at, finished_at, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "job-idle",
                        idle_case["id"],
                        "doc-idle",
                        "balanced_fast_intel",
                        "multimodal",
                        "normal",
                        "complete",
                        100,
                        now,
                        now,
                        None,
                    ),
                )

            response = client.get("/api/cases")
            assert response.status_code == 200
            rows = {row["id"]: row for row in response.json()["data"]}
            assert rows[active_case["id"]]["active_job_count"] == 1
            assert rows[idle_case["id"]]["active_job_count"] == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
