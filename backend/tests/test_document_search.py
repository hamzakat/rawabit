from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.document_search import DocumentSearchService
from backend.app.fs import ensure_case_lightrag_dir
from backend.app.settings import get_settings


def _configure_env(temp_dir: Path) -> None:
    os.environ["RAWABIT_DB_PATH"] = str(temp_dir / "db.sqlite")
    os.environ["RAWABIT_CASES_ROOT"] = str(temp_dir / "cases")


def _create_case(client: TestClient, name: str = "Search Case") -> dict:
    response = client.post("/api/cases", json={"name": name})
    assert response.status_code == 200
    return response.json()["data"]


def _upload_text(client: TestClient, case_id: str, filename: str, text: str) -> dict:
    response = client.post(
        f"/api/cases/{case_id}/documents",
        data={
            "confidence_source_reliability": "A",
            "confidence_information_validity": "1",
        },
        files={"file": (filename, text.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200
    return response.json()["data"]


def test_document_search_indexes_raw_text_and_filters_sources() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"doc-search-raw-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _create_case(client)
            _upload_text(
                client,
                case["id"],
                "ledger-note.txt",
                "Alpha ledger transfer between shell companies.",
            )

            raw_response = client.get(
                f"/api/cases/{case['id']}/documents/search",
                params={"q": "ledger", "source": "raw"},
            )
            assert raw_response.status_code == 200
            raw_results = raw_response.json()["data"]
            assert len(raw_results) == 1
            assert raw_results[0]["source_kind"] == "raw"
            assert raw_results[0]["original_filename"] == "ledger-note.txt"
            assert raw_results[0]["confidence_code"] == "A1"
            assert "ledger" in raw_results[0]["snippet"].lower()
            assert any(
                part["match"] and "ledger" in part["text"].lower()
                for part in raw_results[0]["snippet_parts"]
            )

            preview_response = client.get(
                f"/api/cases/{case['id']}/documents/{raw_results[0]['document_id']}/search-preview",
                params={"q": "ledger", "source_kind": "raw", "segment_key": "raw"},
            )
            assert preview_response.status_code == 200
            preview = preview_response.json()["data"]
            assert "Alpha ledger transfer" in preview["content"]
            assert preview["match_ranges"]
            matched = preview["content"][
                preview["match_ranges"][0]["start"] : preview["match_ranges"][0]["end"]
            ]
            assert matched.lower().startswith("ledger")

            processed_response = client.get(
                f"/api/cases/{case['id']}/documents/search",
                params={"q": "ledger", "source": "processed"},
            )
            assert processed_response.status_code == 200
            assert processed_response.json()["data"] == []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_document_search_indexes_processed_lightrag_chunks() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"doc-search-processed-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _create_case(client)
            upload = _upload_text(client, case["id"], "raw.txt", "Plain raw evidence.")
            document_id = upload["document_id"]
            case_root = temp_dir / "cases" / case["case_slug"]
            lightrag_dir = case_root / "lightrag"
            lightrag_dir.mkdir(parents=True, exist_ok=True)
            (lightrag_dir / "kv_store_text_chunks.json").write_text(
                json.dumps(
                    {
                        "chunk-1": {
                            "full_doc_id": document_id,
                            "content": "Processed analysis mentions Orion Holdings and a wire transfer.",
                        }
                    }
                ),
                encoding="utf-8",
            )

            settings = get_settings()
            DocumentSearchService(settings).index_processed_document(
                case_id=case["id"],
                document_id=document_id,
                case_root=case_root,
                original_filename="raw.txt",
                stored_file_path=f"raw/{document_id}_raw.txt",
                confidence_code="A1",
            )

            response = client.get(
                f"/api/cases/{case['id']}/documents/search",
                params={"q": "orion", "source": "processed"},
            )
            assert response.status_code == 200
            results = response.json()["data"]
            assert len(results) == 1
            assert results[0]["source_kind"] == "processed"
            assert results[0]["segment_key"] == "chunk-1"
            assert "orion" in results[0]["snippet"].lower()
            assert any(
                part["match"] and "orion" in part["text"].lower()
                for part in results[0]["snippet_parts"]
            )

            preview_response = client.get(
                f"/api/cases/{case['id']}/documents/{document_id}/search-preview",
                params={
                    "q": "orion",
                    "source_kind": "processed",
                    "segment_key": "chunk-1",
                },
            )
            assert preview_response.status_code == 200
            preview = preview_response.json()["data"]
            assert preview["source_kind"] == "processed"
            assert preview["segment_key"] == "chunk-1"
            assert "Orion Holdings" in preview["content"]
            assert preview["match_ranges"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_document_search_reads_isolated_lightrag_workspace_chunks() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"doc-search-workspace-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _create_case(client)
            upload = _upload_text(client, case["id"], "raw.txt", "Plain raw evidence.")
            document_id = upload["document_id"]
            case_root = temp_dir / "cases" / case["case_slug"]
            lightrag_dir = ensure_case_lightrag_dir(case_root, case["id"])
            (lightrag_dir / "kv_store_text_chunks.json").write_text(
                json.dumps(
                    {
                        "chunk-1": {
                            "full_doc_id": document_id,
                            "content": "Workspace chunk mentions Polaris Logistics and a transfer.",
                        }
                    }
                ),
                encoding="utf-8",
            )

            settings = get_settings()
            DocumentSearchService(settings).index_processed_document(
                case_id=case["id"],
                document_id=document_id,
                case_root=case_root,
                original_filename="raw.txt",
                stored_file_path=f"raw/{document_id}_raw.txt",
                confidence_code="A1",
            )

            response = client.get(
                f"/api/cases/{case['id']}/documents/search",
                params={"q": "polaris", "source": "processed"},
            )
            assert response.status_code == 200
            results = response.json()["data"]
            assert len(results) == 1
            assert results[0]["segment_key"] == "chunk-1"
            assert "polaris" in results[0]["snippet"].lower()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_document_search_is_case_scoped() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"doc-search-scope-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case_a = _create_case(client, "Case A")
            case_b = _create_case(client, "Case B")
            _upload_text(
                client, case_a["id"], "alpha.txt", "Case-only keyword: zephyr."
            )

            own_response = client.get(
                f"/api/cases/{case_a['id']}/documents/search",
                params={"q": "zephyr"},
            )
            assert own_response.status_code == 200
            assert len(own_response.json()["data"]) == 1

            other_response = client.get(
                f"/api/cases/{case_b['id']}/documents/search",
                params={"q": "zephyr"},
            )
            assert other_response.status_code == 200
            assert other_response.json()["data"] == []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
