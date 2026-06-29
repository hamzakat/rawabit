from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from backend.app.analysis_runner import AnalysisRunner
from backend.app.analysis_service import AnalysisService, MAX_ANALYSIS_REPAIR_ATTEMPTS
from backend.app.chat_service import ChatQueryResult, ChatService
from backend.app.db import get_connection, init_db
from backend.app.settings import get_settings


VALID_RELATIONSHIP_CODE = """flowchart LR
    %% source-id: ALICE
    ALICE((\"Alice\"))
    %% source-id: BOB
    BOB((\"Bob\"))
    ALICE --- BOB
    class ALICE,BOB person"""

VALID_HIERARCHY_CODE = """flowchart TD
    %% source-id: ALICE
    ALICE((\"Alice\"))
    %% source-id: BOB
    BOB((\"Bob\"))
    ALICE -- \"coordinates\" --> BOB
    class ALICE,BOB person"""

VALID_BUNDLE = {
    "summary_text": "Alice is the central participant connected to Bob.",
    "charts": [
        {
            "id": "relationship-map",
            "kind": "relationship_map",
            "title": "Relationship map",
            "item_ids": ["ALICE", "BOB"],
            "mermaid_code": VALID_RELATIONSHIP_CODE,
        },
        {
            "id": "operational-hierarchy",
            "kind": "operational_hierarchy",
            "title": "Operational hierarchy",
            "item_ids": ["ALICE", "BOB"],
            "mermaid_code": VALID_HIERARCHY_CODE,
        },
    ]
}


def _configure_env(temp_dir: Path) -> None:
    os.environ["RAWABIT_DB_PATH"] = str(temp_dir / "db.sqlite")
    os.environ["RAWABIT_CASES_ROOT"] = str(temp_dir / "cases")
    os.environ["RAWABIT_INGEST_POLL_SECONDS"] = "0.05"


def _create_case(client: TestClient, name: str = "Analysis Case") -> dict[str, Any]:
    response = client.post("/api/cases", json={"name": name})
    assert response.status_code == 200
    return response.json()["data"]


def _upload_document(client: TestClient, case_id: str) -> dict[str, Any]:
    response = client.post(
        f"/api/cases/{case_id}/documents",
        data={
            "confidence_source_reliability": "A",
            "confidence_information_validity": "1",
        },
        files={"file": ("analysis-evidence.txt", b"Alice paid Bob.", "text/plain")},
    )
    assert response.status_code == 200
    return response.json()["data"]


def _write_minimal_graph(case_root: Path) -> None:
    working_dir = case_root / "lightrag"
    working_dir.mkdir(parents=True, exist_ok=True)
    (working_dir / "vdb_entities.json").write_text(
        json.dumps(
            [
                {"entity_name": "ALICE", "label": "[PERSON] Alice", "entity_type": "person"},
                {"entity_name": "BOB", "label": "[PERSON] Bob", "entity_type": "person"},
                {
                    "entity_name": "PAYMENT_EVENT",
                    "label": "[EVENT] Payment to Bob",
                    "entity_type": "event",
                },
            ]
        ),
        encoding="utf-8",
    )
    (working_dir / "vdb_relationships.json").write_text(
        json.dumps(
            [
                {
                    "src_id": "ALICE",
                    "tgt_id": "BOB",
                    "relation_type": "ASSOCIATED_WITH",
                    "description": "Alice and Bob are associated.",
                }
            ]
        ),
        encoding="utf-8",
    )


def _fake_rag_result(prompt: str, mode: str) -> ChatQueryResult:
    references = [{"reference_id": "ref-1", "file_path": "raw/analysis-evidence.txt"}]
    chunks = [
        {
            "reference_id": "ref-1",
            "file_path": "raw/analysis-evidence.txt",
            "snippet": "Alice paid Bob.",
        }
    ]
    return ChatQueryResult(
        assistant_content=f"Retrieved answer for {prompt}.",
        highlight={
            "highlight_entities": ["ALICE", "BOB"],
            "highlight_relationships": [{"src_id": "ALICE", "tgt_id": "BOB"}],
            "references": references,
            "supporting_chunks": chunks,
        },
        retrieval_eval={"mode": mode},
        references=references,
        chunks=chunks,
        metadata={"status": "success", "mode": mode, "model_name": "test-rag-model"},
    )


async def _fake_query_case_message(
    self: ChatService,
    *,
    case_root: Path,
    case_id: str,
    user_content: str,
    mode: str,
    conversation_history: list[dict[str, str]],
    case_documents: list[dict[str, object]],
    options: dict[str, object] | None = None,
    allow_mode_fallback: bool = True,
) -> ChatQueryResult:
    assert case_root.exists()
    assert case_id and case_documents and options
    assert conversation_history == []
    assert mode == "hybrid"
    assert allow_mode_fallback is False
    return _fake_rag_result(user_content, mode)


async def _fake_call_llm(
    self: AnalysisService,
    prompt_text: str,
    *,
    deadline: float | None = None,
    operation: str = "",
) -> str:
    assert deadline is not None
    if "repair one Mermaid" in prompt_text:
        return VALID_RELATIONSHIP_CODE
    return f"```json\n{json.dumps(VALID_BUNDLE)}\n```"


def _setup_analysis_case(client: TestClient, temp_dir: Path) -> dict[str, Any]:
    case = _create_case(client)
    _upload_document(client, case["id"])
    _write_minimal_graph(temp_dir / "cases" / case["case_slug"])
    return case


def _wait_for_analysis(
    client: TestClient,
    case_id: str,
    analysis_id: str,
    *,
    status: str,
    timeout: float = 8.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/cases/{case_id}/analyses/{analysis_id}")
        assert response.status_code == 200
        latest = response.json()["data"]
        if latest["status"] == status:
            return latest
        time.sleep(0.05)
    raise AssertionError(f"Analysis did not reach {status}; latest={latest}")


def test_analysis_create_list_get_repair_and_delete(monkeypatch) -> None:
    temp_dir = Path.cwd() / "data" / "pytest-tmp" / f"analysis-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True)
    try:
        _configure_env(temp_dir)
        monkeypatch.setattr(ChatService, "query_case_message", _fake_query_case_message)
        monkeypatch.setattr(AnalysisService, "_call_llm", _fake_call_llm)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _setup_analysis_case(client, temp_dir)
            response = client.post(
                f"/api/cases/{case['id']}/analyses",
                json={"prompt": "Show the Alice-Bob link", "analysis_type": "link"},
            )
            assert response.status_code == 200
            queued = response.json()["data"]
            assert queued["status"] == "queued"
            assert queued["charts"] == []
            created = _wait_for_analysis(
                client, case["id"], queued["id"], status="complete"
            )
            assert [chart["kind"] for chart in created["charts"]] == [
                "relationship_map",
                "operational_hierarchy",
            ]
            assert all(chart["repair_attempts"] == 0 for chart in created["charts"])
            assert created["summary_text"] == "Alice is the central participant connected to Bob."
            assert created["highlight"]["highlight_entities"] == ["ALICE", "BOB"]

            listed = client.get(f"/api/cases/{case['id']}/analyses").json()["data"]
            assert [item["id"] for item in listed] == [created["id"]]
            assert client.get(
                f"/api/cases/{case['id']}/analyses/{created['id']}"
            ).status_code == 200

            repaired_response = client.post(
                f"/api/cases/{case['id']}/analyses/{created['id']}/repair",
                json={
                    "chart_id": "relationship-map",
                    "error": "Parse error",
                    "mermaid_code": "flowchart LR\nA -->",
                },
            )
            assert repaired_response.status_code == 200
            assert repaired_response.json()["data"]["status"] == "repair_queued"
            repaired = _wait_for_analysis(
                client, case["id"], created["id"], status="complete"
            )
            repaired_chart = next(chart for chart in repaired["charts"] if chart["id"] == "relationship-map")
            untouched_chart = next(chart for chart in repaired["charts"] if chart["id"] == "operational-hierarchy")
            assert repaired_chart["repair_attempts"] == 1
            assert repaired_chart["mermaid_code"] == VALID_RELATIONSHIP_CODE
            assert untouched_chart["repair_attempts"] == 0

            assert client.delete(
                f"/api/cases/{case['id']}/analyses/{created['id']}"
            ).status_code == 200
            assert client.get(f"/api/cases/{case['id']}/analyses").json()["data"] == []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_analysis_requires_case_documents() -> None:
    temp_dir = Path.cwd() / "data" / "pytest-tmp" / f"analysis-no-docs-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _create_case(client)
            response = client.post(
                f"/api/cases/{case['id']}/analyses",
                json={"prompt": "Show the links", "analysis_type": "link"},
            )
            assert response.status_code == 409
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_bundle_validator_rejects_duplicates_unsafe_directives_and_wrong_kinds() -> None:
    duplicate = {
        "summary_text": "Summary.",
        "charts": [
            {
                "id": "bad-chart",
                "kind": "commodity_flow",
                "title": "Bad chart",
                "item_ids": ["ALICE", "ALICE"],
                "mermaid_code": """flowchart LR
%% source-id: ALICE
A((\"Alice\"))
%% source-id: ALICE
B((\"Alice\"))
click A href \"https://example.com\"""",
            }
        ]
    }
    errors = AnalysisService._validate_bundle(duplicate, "link")
    assert any("Unsupported link chart kind" in error for error in errors)
    assert any("duplicate item_ids" in error for error in errors)
    assert any("forbidden Mermaid" in error for error in errors)
    assert any("repeats source-id" in error for error in errors)
    assert any("duplicate normalized labels" in error for error in errors)

    too_many = {
        "summary_text": "Summary.",
        "charts": [VALID_BUNDLE["charts"][0]] * 4,
    }
    assert AnalysisService._validate_bundle(too_many, "link") == [
        "Bundle may contain at most 3 charts."
    ]


def test_invalid_bundle_is_repaired_before_persistence(monkeypatch) -> None:
    temp_dir = Path.cwd() / "data" / "pytest-tmp" / f"analysis-repair-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True)
    try:
        _configure_env(temp_dir)
        monkeypatch.setattr(ChatService, "query_case_message", _fake_query_case_message)
        calls = 0

        async def fake_repair(
            self: AnalysisService,
            prompt_text: str,
            *,
            deadline: float | None = None,
            operation: str = "",
        ) -> str:
            nonlocal calls
            calls += 1
            if "Rejected bundle" in prompt_text:
                return json.dumps(VALID_BUNDLE)
            return "not json"

        monkeypatch.setattr(AnalysisService, "_call_llm", fake_repair)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _setup_analysis_case(client, temp_dir)
            response = client.post(
                f"/api/cases/{case['id']}/analyses",
                json={"prompt": "Show links", "analysis_type": "link"},
            )
            assert response.status_code == 200
            queued = response.json()["data"]
            completed = _wait_for_analysis(
                client, case["id"], queued["id"], status="complete"
            )
            assert len(completed["charts"]) == 2
            assert calls == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_bundle_repair_limit(monkeypatch) -> None:
    temp_dir = Path.cwd() / "data" / "pytest-tmp" / f"analysis-limit-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True)
    try:
        _configure_env(temp_dir)
        monkeypatch.setattr(ChatService, "query_case_message", _fake_query_case_message)
        calls = 0

        async def always_invalid(
            self: AnalysisService,
            prompt_text: str,
            *,
            deadline: float | None = None,
            operation: str = "",
        ) -> str:
            nonlocal calls
            calls += 1
            return "{}"

        monkeypatch.setattr(AnalysisService, "_call_llm", always_invalid)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _setup_analysis_case(client, temp_dir)
            response = client.post(
                f"/api/cases/{case['id']}/analyses",
                json={"prompt": "Show links", "analysis_type": "link"},
            )
            assert response.status_code == 200
            queued = response.json()["data"]
            failed = _wait_for_analysis(
                client, case["id"], queued["id"], status="failed"
            )
            assert "Unable to generate a valid analysis chart bundle" in failed["error"]
            assert calls == MAX_ANALYSIS_REPAIR_ATTEMPTS + 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_chart_repair_limit(monkeypatch) -> None:
    temp_dir = Path.cwd() / "data" / "pytest-tmp" / f"chart-limit-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True)
    try:
        _configure_env(temp_dir)
        monkeypatch.setattr(ChatService, "query_case_message", _fake_query_case_message)
        monkeypatch.setattr(AnalysisService, "_call_llm", _fake_call_llm)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _setup_analysis_case(client, temp_dir)
            queued = client.post(
                f"/api/cases/{case['id']}/analyses",
                json={"prompt": "Show links", "analysis_type": "link"},
            ).json()["data"]
            analysis = _wait_for_analysis(
                client, case["id"], queued["id"], status="complete"
            )
            for attempt in range(MAX_ANALYSIS_REPAIR_ATTEMPTS):
                response = client.post(
                    f"/api/cases/{case['id']}/analyses/{analysis['id']}/repair",
                    json={"chart_id": "relationship-map", "error": f"error {attempt}"},
                )
                assert response.status_code == 200
                analysis = _wait_for_analysis(
                    client, case["id"], analysis["id"], status="complete"
                )
            capped = client.post(
                f"/api/cases/{case['id']}/analyses/{analysis['id']}/repair",
                json={"chart_id": "relationship-map", "error": "one too many"},
            )
            assert capped.status_code == 409
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_html_analysis_schema_is_recreated_empty() -> None:
    temp_dir = Path.cwd() / "data" / "pytest-tmp" / f"analysis-schema-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True)
    try:
        _configure_env(temp_dir)
        db_path = temp_dir / "db.sqlite"
        with sqlite3.connect(db_path) as connection:
            connection.execute(
                'CREATE TABLE "case" (id TEXT PRIMARY KEY, name TEXT, description TEXT, '
                "status TEXT, case_slug TEXT UNIQUE, created_at TEXT, updated_at TEXT)"
            )
            connection.execute(
                "CREATE TABLE analysis (id TEXT PRIMARY KEY, case_id TEXT, "
                "analysis_type TEXT, prompt TEXT, title TEXT, status TEXT, "
                "diagram_html TEXT, created_at TEXT, updated_at TEXT)"
            )
            connection.execute(
                "INSERT INTO analysis VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("old", "case", "link", "old", "old", "complete", "<html></html>", "now", "now"),
            )
        from backend.app.main import create_app

        with TestClient(create_app()):
            with get_connection(get_settings()) as connection:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(analysis)").fetchall()
                }
                count = connection.execute("SELECT COUNT(*) FROM analysis").fetchone()[0]
            assert "charts_json" in columns
            assert "diagram_html" not in columns
            assert {"error", "pending_repair_json"}.issubset(columns)
            assert count == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_failed_analysis_retries_without_duplicate_history(monkeypatch) -> None:
    temp_dir = Path.cwd() / "data" / "pytest-tmp" / f"analysis-retry-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True)
    try:
        _configure_env(temp_dir)
        calls = 0

        async def fake_generate(self: AnalysisService, **kwargs: Any) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("Temporary provider outage.")
            return {
                "rag_answer": "Retrieved answer.",
                "summary_text": VALID_BUNDLE["summary_text"],
                "charts": [
                    {**chart, "repair_attempts": 0}
                    for chart in VALID_BUNDLE["charts"]
                ],
                "highlight": {},
                "subgraph": {"nodes": [], "edges": []},
                "references": [],
                "chunks": [],
                "model_name": "test-model",
            }

        monkeypatch.setattr(AnalysisService, "generate_analysis", fake_generate)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _setup_analysis_case(client, temp_dir)
            queued = client.post(
                f"/api/cases/{case['id']}/analyses",
                json={"prompt": "Show links", "analysis_type": "link"},
            ).json()["data"]
            failed = _wait_for_analysis(
                client, case["id"], queued["id"], status="failed"
            )
            retried = client.post(
                f"/api/cases/{case['id']}/analyses/{failed['id']}/retry"
            )
            assert retried.status_code == 200
            assert retried.json()["data"]["id"] == failed["id"]
            completed = _wait_for_analysis(
                client, case["id"], failed["id"], status="complete"
            )
            assert completed["id"] == failed["id"]
            history = client.get(
                f"/api/cases/{case['id']}/analyses"
            ).json()["data"]
            assert [item["id"] for item in history] == [failed["id"]]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_analysis_llm_network_timeout_retries_with_shared_deadline(monkeypatch) -> None:
    settings = get_settings()
    settings.rag_network_retry_window_seconds = 10
    service = AnalysisService(settings)
    calls = 0

    async def fake_request(prompt_text: str, remaining_seconds: float) -> str:
        nonlocal calls
        calls += 1
        assert prompt_text == "prompt"
        assert remaining_seconds > 0
        if calls == 1:
            raise asyncio.TimeoutError()
        return "ok"

    async def no_delay(_: float) -> None:
        return None

    monkeypatch.setattr(service, "_request_llm_once", fake_request)
    monkeypatch.setattr("backend.app.analysis_service.asyncio.sleep", no_delay)
    assert asyncio.run(service._call_llm("prompt")) == "ok"
    assert calls == 2


def test_analysis_runner_recovers_interrupted_statuses() -> None:
    temp_dir = Path.cwd() / "data" / "pytest-tmp" / f"analysis-recovery-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        init_db(settings)
        now = "2026-06-22T00:00:00Z"
        with get_connection(settings) as connection:
            connection.execute(
                'INSERT INTO "case" (id, name, description, status, case_slug, created_at, updated_at) '
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("case", "Case", None, "active", "case", now, now),
            )
            for analysis_id, status in (
                ("generation", "generating"),
                ("repair", "repairing"),
            ):
                connection.execute(
                    "INSERT INTO analysis "
                    "(id, case_id, analysis_type, prompt, title, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        analysis_id,
                        "case",
                        "link",
                        "Question",
                        "Question",
                        status,
                        now,
                        now,
                    ),
                )
        AnalysisRunner(settings)._recover_inflight_analyses()
        with get_connection(settings) as connection:
            statuses = {
                row["id"]: row["status"]
                for row in connection.execute(
                    "SELECT id, status FROM analysis ORDER BY id"
                ).fetchall()
            }
        assert statuses == {
            "generation": "queued",
            "repair": "repair_queued",
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
