from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.db import get_connection


def _configure_env(temp_dir: Path) -> None:
    os.environ["RAWABIT_DB_PATH"] = str(temp_dir / "db.sqlite")
    os.environ["RAWABIT_CASES_ROOT"] = str(temp_dir / "cases")


def _write_graph_artifacts(case_root: Path, document_id: str, original_filename: str) -> None:
    lightrag_dir = case_root / "lightrag"
    lightrag_dir.mkdir(parents=True, exist_ok=True)
    graphml = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="node" attr.name="entity_type" attr.type="string"/>
  <key id="d1" for="node" attr.name="description" attr.type="string"/>
  <key id="d2" for="edge" attr.name="source_id" attr.type="string"/>
  <key id="d3" for="edge" attr.name="keywords" attr.type="string"/>
  <key id="d4" for="edge" attr.name="description" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="ALICE"><data key="d0">Person</data><data key="d1">Primary actor</data></node>
    <node id="BOB"><data key="d0">Person</data><data key="d1">Counterparty</data></node>
    <edge source="ALICE" target="BOB"><data key="d2">chunk-1</data><data key="d3">COMMUNICATED_WITH</data><data key="d4">Alice called Bob</data></edge>
  </graph>
</graphml>
"""
    (lightrag_dir / "graph_chunk_entity_relation.graphml").write_text(graphml, encoding="utf-8")
    (lightrag_dir / "kv_store_text_chunks.json").write_text(
        json.dumps(
            {
                "chunk-1": {
                    "_id": "chunk-1",
                    "reference_id": "ref-1",
                    "document_id": document_id,
                    "full_doc_id": document_id,
                    "confidence_code": "A1",
                    "file_path": original_filename,
                    "content": "Alice called Bob about the shipment.",
                }
            }
        ),
        encoding="utf-8",
    )


def test_case_summary_endpoint_and_case_payload(monkeypatch) -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"case-summary-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.case_summary import CaseSummaryService
        from backend.app.main import create_app

        summary_payload = {
            "intelligence_summary": "Known communication link between key actors.",
            "investigation_summary": "Validate intent, timeline, and corroborating documents.",
            "five_w_one_h": {
                "who": "ALICE and BOB",
                "what": "A communication exchange",
                "when": None,
                "where": None,
                "why": None,
                "how": "Direct call",
            },
            "unknowns": ["Exact date is not explicit."],
            "summary_text": "Known communication link between key actors.\n\nValidate intent, timeline, and corroborating documents.",
        }
        monkeypatch.setattr(
            CaseSummaryService,
            "_generate_summary_with_llm",
            lambda self, case, base: summary_payload,
        )

        with TestClient(create_app()) as client:
            case = client.post("/api/cases", json={"name": "Summary Case"}).json()["data"]
            case_id = case["id"]
            upload_response = client.post(
                f"/api/cases/{case_id}/documents",
                data={
                    "confidence_source_reliability": "A",
                    "confidence_information_validity": "1",
                },
                files={"file": ("alpha.txt", b"alpha evidence", "text/plain")},
            )
            assert upload_response.status_code == 200
            document_id = upload_response.json()["data"]["document_id"]
            document = client.get(f"/api/cases/{case_id}/documents/{document_id}").json()["data"]

            case_root = temp_dir / "cases" / case["case_slug"]
            _write_graph_artifacts(
                case_root=case_root,
                document_id=document_id,
                original_filename=document["original_filename"],
            )

            summary_response = client.get(f"/api/cases/{case_id}/summary")
            assert summary_response.status_code == 200
            summary = summary_response.json()["data"]
            assert summary["case_id"] == case_id
            assert summary["evidence_count"] == 1
            assert summary["entity_count"] >= 2
            assert summary["relationship_count"] >= 1
            assert summary["intelligence_summary"]
            assert summary["investigation_summary"]
            assert isinstance(summary["five_w_one_h"], dict)

            case_response = client.get(f"/api/cases/{case_id}")
            assert case_response.status_code == 200
            assert case_response.json()["data"]["summary"]["case_id"] == case_id

            list_response = client.get("/api/cases")
            assert list_response.status_code == 200
            case_rows = list_response.json()["data"]
            row = next(item for item in case_rows if item["id"] == case_id)
            assert isinstance(row.get("summary_snippet"), str)
            assert row["summary_snippet"]

            summary_payload = {
                "intelligence_summary": "Manual refresh found updated actor context.",
                "investigation_summary": "Manual refresh should overwrite the cached summary.",
                "five_w_one_h": {
                    "who": "ALICE and BOB",
                    "what": "Updated communication context",
                    "when": None,
                    "where": None,
                    "why": None,
                    "how": "Manual refresh",
                },
                "unknowns": [],
                "summary_text": "Manual refresh found updated actor context.\n\nManual refresh should overwrite the cached summary.",
            }
            refresh_response = client.post(f"/api/cases/{case_id}/summary/refresh")
            assert refresh_response.status_code == 200
            refreshed = refresh_response.json()["data"]
            assert refreshed["case_id"] == case_id
            assert refreshed["intelligence_summary"] == "Manual refresh found updated actor context."
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_case_summary_refresh_keeps_previous_on_llm_failure(monkeypatch) -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"case-summary-fallback-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.case_summary import CaseSummaryService
        from backend.app.main import create_app
        from backend.app.settings import get_settings

        with TestClient(create_app()) as client:
            case = client.post("/api/cases", json={"name": "Fallback Case"}).json()["data"]
            case_id = case["id"]
            upload_response = client.post(
                f"/api/cases/{case_id}/documents",
                data={
                    "confidence_source_reliability": "A",
                    "confidence_information_validity": "1",
                },
                files={"file": ("alpha.txt", b"alpha evidence", "text/plain")},
            )
            assert upload_response.status_code == 200
            document_id = upload_response.json()["data"]["document_id"]
            document = client.get(f"/api/cases/{case_id}/documents/{document_id}").json()["data"]
            case_root = temp_dir / "cases" / case["case_slug"]
            _write_graph_artifacts(
                case_root=case_root,
                document_id=document_id,
                original_filename=document["original_filename"],
            )

        service = CaseSummaryService(get_settings())
        monkeypatch.setattr(
            CaseSummaryService,
            "_generate_summary_with_llm",
            lambda self, case, base: {
                "intelligence_summary": "Initial intelligence summary.",
                "investigation_summary": "Initial investigation summary.",
                "five_w_one_h": {
                    "who": "ALICE and BOB",
                    "what": "Communication",
                    "when": None,
                    "where": None,
                    "why": None,
                    "how": "Call",
                },
                "unknowns": [],
                "summary_text": "Initial intelligence summary.\n\nInitial investigation summary.",
            },
        )
        first = service.refresh_case_summary(case_id, source_job_id="job-1")
        assert first["intelligence_summary"] == "Initial intelligence summary."

        with TestClient(create_app()) as client:
            upload_response = client.post(
                f"/api/cases/{case_id}/documents",
                data={
                    "confidence_source_reliability": "A",
                    "confidence_information_validity": "1",
                },
                files={"file": ("beta.txt", b"beta evidence", "text/plain")},
            )
            assert upload_response.status_code == 200
            second_document_id = upload_response.json()["data"]["document_id"]

        with get_connection(get_settings()) as connection:
            connection.execute(
                "UPDATE document SET ingestion_status = ?, updated_at = ? WHERE id IN (?, ?)",
                (
                    "complete",
                    "2026-05-07T15:00:00+00:00",
                    document_id,
                    second_document_id,
                ),
            )

        monkeypatch.setattr(
            CaseSummaryService,
            "_generate_summary_with_llm",
            lambda self, case, base: None,
        )
        second = service.refresh_case_summary(case_id, source_job_id="job-2")
        assert second["intelligence_summary"] == "Initial intelligence summary."
        assert second["source_job_id"] == "job-2"
        assert second["document_count"] == 2
        assert second["completed_document_count"] == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
