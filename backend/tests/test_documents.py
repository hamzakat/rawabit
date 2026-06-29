from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import uuid
import json

from fastapi.testclient import TestClient

import numpy as np
from nano_vectordb import NanoVectorDB

from backend.app.cleanup import cleanup_orphan_lightrag_documents
from backend.app.db import get_connection
from backend.app.fs import ensure_case_lightrag_dir, resolve_case_lightrag_dir
from backend.app.settings import get_settings


def _configure_env(temp_dir: Path) -> None:
    os.environ["RAWABIT_DB_PATH"] = str(temp_dir / "db.sqlite")
    os.environ["RAWABIT_CASES_ROOT"] = str(temp_dir / "cases")


def _create_case(client: TestClient) -> dict:
    response = client.post("/api/cases", json={"name": "Test Case"})
    assert response.status_code == 200
    return response.json()["data"]


def _create_case_named(client: TestClient, name: str) -> dict:
    response = client.post("/api/cases", json={"name": name})
    assert response.status_code == 200
    return response.json()["data"]


def _upload_document(
    client: TestClient,
    case_id: str,
    filename: str,
    content: bytes,
    ingest_profile: str | None = None,
    processing_mode: str | None = None,
    notes: str | None = None,
) -> dict:
    files = {"file": (filename, content, "text/plain")}
    data = {
        "confidence_source_reliability": "A",
        "confidence_information_validity": "1",
    }
    if ingest_profile:
        data["ingest_profile"] = ingest_profile
    if processing_mode:
        data["processing_mode"] = processing_mode
    if notes:
        data["notes"] = notes
    response = client.post(f"/api/cases/{case_id}/documents", data=data, files=files)
    assert response.status_code == 200
    return response.json()["data"]


def test_document_get_download_reingest_delete() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"docs-test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _create_case(client)
            payload = _upload_document(
                client,
                case["id"],
                "alpha.txt",
                b"alpha evidence",
                ingest_profile="balanced_fast",
                notes="Analyst says the receipt may be incomplete.",
            )
            document_id = payload["document_id"]
            job_id = payload["job_id"]
            assert job_id
            assert payload["ingest_profile"] == "balanced_fast"
            assert payload["processing_mode"] == "multimodal"
            assert (
                payload["content_hash_sha256"]
                == hashlib.sha256(b"alpha evidence").hexdigest()
            )
            assert payload["preflight"]["complexity_class"] in {
                "small",
                "medium",
                "large",
                "very_large",
            }
            assert payload["preflight"]["eta_seconds"] > 0

            get_response = client.get(
                f"/api/cases/{case['id']}/documents/{document_id}"
            )
            assert get_response.status_code == 200
            document = get_response.json()["data"]
            assert document["original_filename"] == "alpha.txt"
            assert (
                document["content_hash_sha256"]
                == hashlib.sha256(b"alpha evidence").hexdigest()
            )
            assert document["notes"] == "Analyst says the receipt may be incomplete."
            assert document["ingest_model_name"] is None

            download_response = client.get(
                f"/api/cases/{case['id']}/documents/{document_id}/download"
            )
            assert download_response.status_code == 200
            assert download_response.content == b"alpha evidence"
            assert download_response.headers["content-type"].startswith("text/plain")

            reference_preview_response = client.get(
                f"/api/cases/{case['id']}/documents/{document_id}/reference-preview",
                params={"reference_id": "ref-alpha", "q": "alpha"},
            )
            assert reference_preview_response.status_code == 200
            reference_preview = reference_preview_response.json()["data"]
            assert reference_preview["document_id"] == document_id
            assert reference_preview["segment_key"] == "ref-alpha"
            assert "alpha evidence" in reference_preview["content"]

            case_root = Path(os.environ["RAWABIT_CASES_ROOT"]) / case["case_slug"]
            stored_path = case_root / document["stored_file_path"]
            assert stored_path.exists()
            processed_path = case_root / "processed" / document_id
            processed_path.mkdir(parents=True, exist_ok=True)
            (processed_path / "stale.txt").write_text("stale", encoding="utf-8")

            # Re-ingest should clear previous processed artifacts for this document.
            reingest_response = client.post(
                f"/api/cases/{case['id']}/documents/{document_id}/reingest?ingest_profile=full_enrichment"
            )
            assert reingest_response.status_code == 200
            job_id = reingest_response.json()["data"]["job_id"]
            assert (
                reingest_response.json()["data"]["ingest_profile"] == "full_enrichment"
            )
            assert reingest_response.json()["data"]["processing_mode"] == "multimodal"
            assert reingest_response.json()["data"]["preflight"]["eta_seconds"] > 0
            assert not processed_path.exists()

            settings = get_settings()
            with get_connection(settings) as connection:
                created_job = connection.execute(
                    "SELECT id, status, ingest_profile, processing_mode FROM ingestion_job WHERE id = ?",
                    (payload["job_id"],),
                ).fetchone()
                assert created_job is not None
                assert created_job["status"] == "queued"
                assert created_job["ingest_profile"] == "balanced_fast"
                assert created_job["processing_mode"] == "multimodal"
                job = connection.execute(
                    "SELECT id, status, ingest_profile, processing_mode FROM ingestion_job WHERE id = ?",
                    (job_id,),
                ).fetchone()
                assert job is not None
                assert job["status"] == "queued"
                assert job["ingest_profile"] == "full_enrichment"
                assert job["processing_mode"] == "multimodal"
                status = connection.execute(
                    "SELECT ingestion_status FROM document WHERE id = ?",
                    (document_id,),
                ).fetchone()
                assert status["ingestion_status"] == "queued"

            jobs_response = client.get(f"/api/cases/{case['id']}/jobs")
            assert jobs_response.status_code == 200
            jobs = jobs_response.json()["data"]
            assert isinstance(jobs, list)
            assert any(job["id"] == payload["job_id"] for job in jobs)
            assert any(job["ingest_profile"] == "balanced_fast" for job in jobs)
            assert any(job["processing_mode"] == "multimodal" for job in jobs)

            job_response = client.get(
                f"/api/cases/{case['id']}/jobs/{payload['job_id']}"
            )
            assert job_response.status_code == 200
            job_data = job_response.json()["data"]
            assert job_data["id"] == payload["job_id"]
            assert job_data["ingest_profile"] == "balanced_fast"
            assert job_data["processing_mode"] == "multimodal"

            logs_response = client.get(
                f"/api/cases/{case['id']}/jobs/{payload['job_id']}/logs"
            )
            assert logs_response.status_code == 200
            logs = logs_response.json()["data"]
            assert isinstance(logs, list)
            assert any("queued" in (entry["message"] or "").lower() for entry in logs)

            processed_path.mkdir(parents=True, exist_ok=True)
            (processed_path / "stale.txt").write_text("stale", encoding="utf-8")

            delete_response = client.delete(
                f"/api/cases/{case['id']}/documents/{document_id}"
            )
            assert delete_response.status_code == 200
            assert delete_response.json()["data"]["deleted"] is True
            assert not stored_path.exists()
            assert not processed_path.exists()

            with get_connection(settings) as connection:
                doc_row = connection.execute(
                    "SELECT id FROM document WHERE id = ?", (document_id,)
                ).fetchone()
                assert doc_row is None
                jobs_left = connection.execute(
                    "SELECT COUNT(*) as c FROM ingestion_job WHERE document_id = ?",
                    (document_id,),
                ).fetchone()
                assert jobs_left["c"] == 0

            missing_response = client.get(
                f"/api/cases/{case['id']}/documents/{document_id}"
            )
            assert missing_response.status_code == 404
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_document_case_scoped_access() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"docs-scope-test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case_a = _create_case_named(client, "Case A")
            case_b = _create_case_named(client, "Case B")
            payload = _upload_document(
                client, case_a["id"], "bravo.txt", b"bravo evidence"
            )
            document_id = payload["document_id"]
            job_id = payload["job_id"]

            get_response = client.get(
                f"/api/cases/{case_b['id']}/documents/{document_id}"
            )
            assert get_response.status_code == 404

            download_response = client.get(
                f"/api/cases/{case_b['id']}/documents/{document_id}/download"
            )
            assert download_response.status_code == 404

            reingest_response = client.post(
                f"/api/cases/{case_b['id']}/documents/{document_id}/reingest"
            )
            assert reingest_response.status_code == 404

            delete_response = client.delete(
                f"/api/cases/{case_b['id']}/documents/{document_id}"
            )
            assert delete_response.status_code == 404

            jobs_response = client.get(f"/api/cases/{case_b['id']}/jobs")
            assert jobs_response.status_code == 200
            jobs = jobs_response.json()["data"]
            assert all(job["id"] != job_id for job in jobs)

            job_response = client.get(f"/api/cases/{case_b['id']}/jobs/{job_id}")
            assert job_response.status_code == 404

            logs_response = client.get(f"/api/cases/{case_b['id']}/jobs/{job_id}/logs")
            assert logs_response.status_code == 404
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_document_rejects_invalid_ingest_profile() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"docs-invalid-profile-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _create_case(client)
            files = {"file": ("alpha.txt", b"alpha evidence", "text/plain")}
            data = {
                "confidence_source_reliability": "A",
                "confidence_information_validity": "1",
                "ingest_profile": "not-real",
            }
            upload_response = client.post(
                f"/api/cases/{case['id']}/documents",
                data=data,
                files=files,
            )
            assert upload_response.status_code == 400
            assert "ingest_profile" in upload_response.json()["message"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_document_rejects_invalid_processing_mode() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"docs-invalid-processing-mode-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _create_case(client)
            files = {"file": ("alpha.txt", b"alpha evidence", "text/plain")}
            data = {
                "confidence_source_reliability": "A",
                "confidence_information_validity": "1",
                "processing_mode": "invalid-mode",
            }
            upload_response = client.post(
                f"/api/cases/{case['id']}/documents",
                data=data,
                files=files,
            )
            assert upload_response.status_code == 400
            assert "processing_mode" in upload_response.json()["message"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_document_upload_and_reingest_support_text_first_processing_mode() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"docs-text-first-processing-mode-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _create_case(client)
            payload = _upload_document(
                client,
                case["id"],
                "text-first.txt",
                b"text first evidence",
                ingest_profile="balanced_fast_intel",
                processing_mode="text_first",
            )
            assert payload["processing_mode"] == "text_first"

            reingest_response = client.post(
                (
                    f"/api/cases/{case['id']}/documents/{payload['document_id']}/reingest"
                    "?ingest_profile=balanced_fast_intel&processing_mode=text_first"
                )
            )
            assert reingest_response.status_code == 200
            assert reingest_response.json()["data"]["processing_mode"] == "text_first"

            settings = get_settings()
            with get_connection(settings) as connection:
                jobs = connection.execute(
                    "SELECT id, processing_mode FROM ingestion_job WHERE document_id = ? ORDER BY started_at ASC",
                    (payload["document_id"],),
                ).fetchall()
                assert len(jobs) == 2
                assert jobs[0]["processing_mode"] == "text_first"
                assert jobs[1]["processing_mode"] == "text_first"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_document_upload_defaults_to_balanced_fast_intel_profile() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"docs-default-profile-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    previous_default_profile = os.environ.get("RAWABIT_INGEST_PROFILE_DEFAULT")
    os.environ["RAWABIT_INGEST_PROFILE_DEFAULT"] = "balanced_fast_intel"
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _create_case(client)
            files = {"file": ("omega.txt", b"omega evidence", "text/plain")}
            data = {
                "confidence_source_reliability": "A",
                "confidence_information_validity": "1",
            }
            upload_response = client.post(
                f"/api/cases/{case['id']}/documents",
                data=data,
                files=files,
            )
            assert upload_response.status_code == 200
            payload = upload_response.json()["data"]
            assert payload["ingest_profile"] == "balanced_fast_intel"
            assert payload["processing_mode"] == "multimodal"
    finally:
        if previous_default_profile is None:
            os.environ.pop("RAWABIT_INGEST_PROFILE_DEFAULT", None)
        else:
            os.environ["RAWABIT_INGEST_PROFILE_DEFAULT"] = previous_default_profile
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_document_duplicate_check_and_duplicate_upload_flow() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"docs-duplicate-check-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _create_case(client)
            content = b"duplicate evidence"
            content_hash = hashlib.sha256(content).hexdigest()

            first = _upload_document(client, case["id"], "duplicate-a.txt", content)
            assert first["content_hash_sha256"] == content_hash

            duplicate_check = client.post(
                f"/api/cases/{case['id']}/documents/duplicates/check",
                json={
                    "files": [
                        {
                            "client_id": "file-1",
                            "original_filename": "duplicate-b.txt",
                            "size_bytes": len(content),
                            "content_hash_sha256": content_hash,
                        }
                    ]
                },
            )
            assert duplicate_check.status_code == 200
            rows = duplicate_check.json()["data"]
            assert len(rows) == 1
            assert rows[0]["client_id"] == "file-1"
            assert rows[0]["content_hash_sha256"] == content_hash
            assert len(rows[0]["matches"]) == 1
            assert rows[0]["matches"][0]["id"] == first["document_id"]

            duplicate_upload = client.post(
                f"/api/cases/{case['id']}/documents",
                data={
                    "confidence_source_reliability": "A",
                    "confidence_information_validity": "1",
                    "content_hash_sha256": content_hash,
                },
                files={"file": ("duplicate-b.txt", content, "text/plain")},
            )
            assert duplicate_upload.status_code == 409
            assert (
                duplicate_upload.json()["message"]
                == "Duplicate evidence already exists in this case."
            )
            assert (
                duplicate_upload.json()["data"]["content_hash_sha256"] == content_hash
            )
            assert len(duplicate_upload.json()["data"]["matches"]) == 1

            allowed_duplicate_upload = client.post(
                f"/api/cases/{case['id']}/documents",
                data={
                    "confidence_source_reliability": "A",
                    "confidence_information_validity": "1",
                    "content_hash_sha256": content_hash,
                    "allow_duplicate": "true",
                },
                files={"file": ("duplicate-c.txt", content, "text/plain")},
            )
            assert allowed_duplicate_upload.status_code == 200
            second = allowed_duplicate_upload.json()["data"]
            assert second["document_id"] != first["document_id"]
            assert second["content_hash_sha256"] == content_hash

            documents = client.get(f"/api/cases/{case['id']}/documents").json()["data"]
            assert len(documents) == 2
            assert all(doc["content_hash_sha256"] == content_hash for doc in documents)

            case_detail = client.get(f"/api/cases/{case['id']}").json()["data"]
            case_root = (
                Path(os.environ["RAWABIT_CASES_ROOT"]) / case_detail["case_slug"]
            )
            stored_paths = [case_root / doc["stored_file_path"] for doc in documents]
            assert stored_paths[0] != stored_paths[1]
            assert all(path.exists() for path in stored_paths)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_document_delete_falls_back_when_lightrag_delete_not_allowed(
    monkeypatch,
) -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"docs-delete-fallback-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app
        from backend.app import cleanup as cleanup_module

        with TestClient(create_app()) as client:
            case = _create_case(client)
            payload = _upload_document(
                client,
                case["id"],
                "gamma.txt",
                b"gamma evidence",
                ingest_profile="balanced_fast",
            )
            document_id = payload["document_id"]

            with get_connection(get_settings()) as connection:
                doc = connection.execute(
                    "SELECT original_filename FROM document WHERE id = ?",
                    (document_id,),
                ).fetchone()
            assert doc is not None

            case_root = Path(os.environ["RAWABIT_CASES_ROOT"]) / case["case_slug"]
            lightrag_dir = case_root / "lightrag"
            lightrag_dir.mkdir(parents=True, exist_ok=True)

            chunk_1 = "chunk-doc-fallback-1"
            chunk_2 = "chunk-doc-fallback-2"

            (lightrag_dir / "kv_store_doc_status.json").write_text(
                json.dumps(
                    {
                        document_id: {
                            "status": "processed",
                            "chunks_list": [chunk_1, chunk_2],
                            "file_path": doc["original_filename"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            (lightrag_dir / "kv_store_full_docs.json").write_text(
                json.dumps({document_id: {"file_path": doc["original_filename"]}}),
                encoding="utf-8",
            )
            (lightrag_dir / "kv_store_full_entities.json").write_text(
                json.dumps({document_id: {"entity_names": ["EntityA"]}}),
                encoding="utf-8",
            )
            (lightrag_dir / "kv_store_full_relations.json").write_text(
                json.dumps({document_id: {"relation_pairs": [["EntityA", "EntityB"]]}}),
                encoding="utf-8",
            )
            (lightrag_dir / "kv_store_text_chunks.json").write_text(
                json.dumps(
                    {
                        chunk_1: {
                            "_id": chunk_1,
                            "full_doc_id": document_id,
                            "file_path": doc["original_filename"],
                        },
                        chunk_2: {
                            "_id": chunk_2,
                            "full_doc_id": document_id,
                            "file_path": doc["original_filename"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (lightrag_dir / "kv_store_entity_chunks.json").write_text(
                json.dumps({"EntityA": {"chunk_ids": [chunk_1, chunk_2]}}),
                encoding="utf-8",
            )
            (lightrag_dir / "kv_store_relation_chunks.json").write_text(
                json.dumps({"EntityA<SEP>EntityB": {"chunk_ids": [chunk_1]}}),
                encoding="utf-8",
            )

            chunks_db = NanoVectorDB(
                embedding_dim=4, storage_file=str(lightrag_dir / "vdb_chunks.json")
            )
            chunks_db.upsert(
                [
                    {
                        "__id__": chunk_1,
                        "__vector__": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        "full_doc_id": document_id,
                    },
                    {
                        "__id__": chunk_2,
                        "__vector__": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                        "full_doc_id": document_id,
                    },
                ]
            )
            chunks_db.save()

            entities_db = NanoVectorDB(
                embedding_dim=4, storage_file=str(lightrag_dir / "vdb_entities.json")
            )
            entities_db.upsert(
                [
                    {
                        "__id__": "ent-fallback-1",
                        "__vector__": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
                        "source_id": chunk_1,
                    }
                ]
            )
            entities_db.save()

            rel_db = NanoVectorDB(
                embedding_dim=4,
                storage_file=str(lightrag_dir / "vdb_relationships.json"),
            )
            rel_db.upsert(
                [
                    {
                        "__id__": "rel-fallback-1",
                        "__vector__": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                        "source_id": chunk_2,
                    }
                ]
            )
            rel_db.save()

            async def _fake_lightrag_delete(*_, **__):
                return (
                    "not_allowed",
                    "Deletion not allowed: current job 'Single document deletion' is not a document deletion job",
                )

            monkeypatch.setattr(
                cleanup_module,
                "_delete_lightrag_doc",
                _fake_lightrag_delete,
            )

            delete_response = client.delete(
                f"/api/cases/{case['id']}/documents/{document_id}"
            )
            assert delete_response.status_code == 200
            assert delete_response.json()["data"]["deleted"] is True

            if not lightrag_dir.exists():
                return

            doc_status = json.loads(
                (lightrag_dir / "kv_store_doc_status.json").read_text(encoding="utf-8")
            )
            assert document_id not in doc_status
            full_docs = json.loads(
                (lightrag_dir / "kv_store_full_docs.json").read_text(encoding="utf-8")
            )
            assert document_id not in full_docs
            text_chunks = json.loads(
                (lightrag_dir / "kv_store_text_chunks.json").read_text(encoding="utf-8")
            )
            assert chunk_1 not in text_chunks
            assert chunk_2 not in text_chunks

            vdb_chunks = json.loads(
                (lightrag_dir / "vdb_chunks.json").read_text(encoding="utf-8")
            )
            vdb_entities = json.loads(
                (lightrag_dir / "vdb_entities.json").read_text(encoding="utf-8")
            )
            vdb_relationships = json.loads(
                (lightrag_dir / "vdb_relationships.json").read_text(encoding="utf-8")
            )
            assert vdb_chunks["data"] == []
            assert vdb_entities["data"] == []
            assert vdb_relationships["data"] == []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_delete_last_document_resets_case_ingestion_workspace() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"docs-reset-case-workspace-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _create_case(client)
            payload = _upload_document(
                client,
                case["id"],
                "delta.txt",
                b"delta evidence",
                ingest_profile="balanced_fast",
            )
            document_id = payload["document_id"]

            case_root = Path(os.environ["RAWABIT_CASES_ROOT"]) / case["case_slug"]
            lightrag_dir = case_root / "lightrag"
            lightrag_dir.mkdir(parents=True, exist_ok=True)
            (lightrag_dir / "kv_store_llm_response_cache.json").write_text(
                json.dumps({"dummy": {"value": 1}}),
                encoding="utf-8",
            )
            processed_other = case_root / "processed" / "stale-dir"
            processed_other.mkdir(parents=True, exist_ok=True)
            (processed_other / "orphan.txt").write_text("orphan", encoding="utf-8")

            delete_response = client.delete(
                f"/api/cases/{case['id']}/documents/{document_id}"
            )
            assert delete_response.status_code == 200

            assert not (case_root / "processed").exists()
            if lightrag_dir.exists():
                marker_files = [entry.name for entry in lightrag_dir.iterdir()]
                assert "kv_store_llm_response_cache.json" not in marker_files
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_delete_document_keeps_case_workspace_when_other_docs_exist() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"docs-keep-case-workspace-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _create_case(client)
            payload_a = _upload_document(
                client,
                case["id"],
                "one.txt",
                b"one",
                ingest_profile="balanced_fast",
            )
            _upload_document(
                client,
                case["id"],
                "two.txt",
                b"two",
                ingest_profile="balanced_fast",
            )

            case_root = Path(os.environ["RAWABIT_CASES_ROOT"]) / case["case_slug"]
            lightrag_dir = case_root / "lightrag"
            lightrag_dir.mkdir(parents=True, exist_ok=True)
            marker = lightrag_dir / "marker.keep"
            marker.write_text("keep", encoding="utf-8")

            delete_response = client.delete(
                f"/api/cases/{case['id']}/documents/{payload_a['document_id']}"
            )
            assert delete_response.status_code == 200
            assert marker.exists()
            assert lightrag_dir.exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cleanup_orphan_lightrag_documents_removes_stale_doc_state() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"docs-clean-orphans-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        case_root = temp_dir / "cases" / "demo"
        lightrag_dir = case_root / "lightrag"
        lightrag_dir.mkdir(parents=True, exist_ok=True)

        active_doc = "doc-active"
        orphan_doc = "doc-orphan"
        (lightrag_dir / "kv_store_doc_status.json").write_text(
            json.dumps(
                {
                    active_doc: {
                        "status": "processed",
                        "chunks_list": ["chunk-active"],
                    },
                    orphan_doc: {
                        "status": "processed",
                        "chunks_list": ["chunk-orphan"],
                    },
                }
            ),
            encoding="utf-8",
        )
        (lightrag_dir / "kv_store_full_docs.json").write_text(
            json.dumps(
                {
                    active_doc: {"file_path": f"raw/{active_doc}.txt"},
                    orphan_doc: {"file_path": f"raw/{orphan_doc}.txt"},
                }
            ),
            encoding="utf-8",
        )
        (lightrag_dir / "kv_store_text_chunks.json").write_text(
            json.dumps(
                {
                    "chunk-active": {"_id": "chunk-active", "full_doc_id": active_doc},
                    "chunk-orphan": {"_id": "chunk-orphan", "full_doc_id": orphan_doc},
                }
            ),
            encoding="utf-8",
        )

        chunks_db = NanoVectorDB(
            embedding_dim=4, storage_file=str(lightrag_dir / "vdb_chunks.json")
        )
        chunks_db.upsert(
            [
                {
                    "__id__": "chunk-active",
                    "__vector__": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    "full_doc_id": active_doc,
                },
                {
                    "__id__": "chunk-orphan",
                    "__vector__": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                    "full_doc_id": orphan_doc,
                },
            ]
        )
        chunks_db.save()

        removed = cleanup_orphan_lightrag_documents(lightrag_dir, {active_doc})
        assert removed == [orphan_doc]

        doc_status = json.loads(
            (lightrag_dir / "kv_store_doc_status.json").read_text(encoding="utf-8")
        )
        assert active_doc in doc_status
        assert orphan_doc not in doc_status
        text_chunks = json.loads(
            (lightrag_dir / "kv_store_text_chunks.json").read_text(encoding="utf-8")
        )
        assert "chunk-active" in text_chunks
        assert "chunk-orphan" not in text_chunks
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_case_lightrag_workspace_isolated_for_fresh_case_and_legacy_safe() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"docs-lightrag-workspace-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        fresh_case_root = temp_dir / "cases" / "fresh"
        (fresh_case_root / "lightrag").mkdir(parents=True, exist_ok=True)
        fresh_dir = ensure_case_lightrag_dir(fresh_case_root, "case-fresh")
        assert fresh_dir == fresh_case_root / "lightrag" / "case-fresh"
        assert fresh_dir.exists()
        assert resolve_case_lightrag_dir(fresh_case_root, "case-fresh") == fresh_dir

        legacy_case_root = temp_dir / "cases" / "legacy"
        legacy_lightrag = legacy_case_root / "lightrag"
        legacy_lightrag.mkdir(parents=True, exist_ok=True)
        (legacy_lightrag / "kv_store_doc_status.json").write_text(
            "{}", encoding="utf-8"
        )
        legacy_dir = ensure_case_lightrag_dir(legacy_case_root, "case-legacy")
        assert legacy_dir == legacy_lightrag
        assert (
            resolve_case_lightrag_dir(legacy_case_root, "case-legacy")
            == legacy_lightrag
        )
        assert not (legacy_lightrag / "case-legacy").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
