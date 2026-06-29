from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.chat_service import ChatQueryResult, ChatService
from backend.app.db import get_connection
from backend.app.settings import get_settings
from backend.app.utils import utc_now_iso


def _configure_env(temp_dir: Path) -> None:
    os.environ["RAWABIT_DB_PATH"] = str(temp_dir / "db.sqlite")
    os.environ["RAWABIT_CASES_ROOT"] = str(temp_dir / "cases")


def _create_case(client: TestClient, name: str = "Chat Case") -> dict:
    response = client.post("/api/cases", json={"name": name})
    assert response.status_code == 200
    return response.json()["data"]


def _create_chat(client: TestClient, case_id: str, title: str | None = None) -> dict:
    payload = {"title": title} if title is not None else {}
    response = client.post(f"/api/cases/{case_id}/chats", json=payload)
    assert response.status_code == 200
    return response.json()["data"]


def _upload_document(
    client: TestClient,
    case_id: str,
    filename: str = "chat-evidence.txt",
    content: bytes = b"chat evidence",
) -> dict:
    files = {"file": (filename, content, "text/plain")}
    data = {
        "confidence_source_reliability": "A",
        "confidence_information_validity": "1",
    }
    response = client.post(f"/api/cases/{case_id}/documents", data=data, files=files)
    assert response.status_code == 200
    return response.json()["data"]


def _write_minimal_graph(case_root: Path) -> None:
    working_dir = case_root / "lightrag"
    working_dir.mkdir(parents=True, exist_ok=True)
    (working_dir / "vdb_entities.json").write_text(
        json.dumps(
            [
                {"entity_name": "ALICE", "entity_type": "person"},
                {"entity_name": "BOB", "entity_type": "person"},
                {"entity_name": "TRANSFER_EVENT", "entity_type": "event"},
            ]
        ),
        encoding="utf-8",
    )
    (working_dir / "vdb_relationships.json").write_text(
        json.dumps(
            [
                {"src_id": "ALICE", "tgt_id": "BOB", "relation_type": "KNOWS"},
                {"src_id": "ALICE", "tgt_id": "TRANSFER_EVENT", "relation_type": "PARTICIPATED_IN"},
            ]
        ),
        encoding="utf-8",
    )


def test_chat_query_options_honor_requested_top_k() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"chat-topk-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        service = ChatService(settings)

        options = service._build_query_options(  # noqa: SLF001
            {"top_k": 20, "chunk_top_k": 20},
        )

        assert options["top_k"] == 20
        assert options["chunk_top_k"] == 20
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_chat_crud_and_case_scoping(monkeypatch) -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"chat-test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)

        async def _fake_query_case_message(
            self,
            *,
            case_root: Path,
            user_content: str,
            mode: str,
            conversation_history: list[dict[str, str]],
            case_documents: list[dict[str, object]],
            options: dict[str, object] | None = None,
        ) -> ChatQueryResult:
            assert case_root.exists()
            assert mode == "mix"
            assert user_content == "Who is linked to Alice?"
            file_path = (
                str(case_documents[0]["stored_file_path"])
                if case_documents
                else "raw/chat-evidence.txt"
            )
            references = [{"reference_id": "ref-1", "file_path": file_path}]
            chunks = [
                {
                    "reference_id": "ref-1",
                    "file_path": file_path,
                    "snippet": "Alice and Bob exchanged messages.",
                }
            ]
            highlight = {
                "highlight_entities": ["ALICE", "BOB"],
                "highlight_relationships": [{"src_id": "ALICE", "tgt_id": "BOB"}],
                "references": references,
                "supporting_chunks": chunks,
            }
            retrieval_eval = {
                "mode": mode,
                "top_k": 5,
                "retrieved_entity_ids_topk": ["ALICE", "BOB"],
                "retrieved_entity_types_topk": ["person", "person"],
                "retrieved_relation_ids_topk": ["ALICE__KNOWS__BOB"],
                "retrieved_relation_types_topk": ["KNOWS"],
                "quality_flags": {
                    "entities_present": True,
                    "relations_present": True,
                    "non_empty_payload": True,
                },
            }
            return ChatQueryResult(
                assistant_content="Alice is linked to Bob.",
                highlight=highlight,
                retrieval_eval=retrieval_eval,
                references=references,
                chunks=chunks,
                metadata={
                    "status": "success",
                    "mode": mode,
                    "model_name": "test-model",
                    "options": options or {},
                    "highlight": highlight,
                    "retrieval_eval": retrieval_eval,
                    "references": references,
                    "chunks": chunks,
                    "history_size": len(conversation_history),
                },
            )

        monkeypatch.setattr(ChatService, "query_case_message", _fake_query_case_message)

        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case_a = _create_case(client, "Case A")
            case_b = _create_case(client, "Case B")
            _upload_document(client, case_a["id"])

            chat = _create_chat(client, case_a["id"], "Test")
            assert chat["case_id"] == case_a["id"]
            assert chat["title"] == "Test"

            list_response = client.get(f"/api/cases/{case_a['id']}/chats")
            assert list_response.status_code == 200
            chats = list_response.json()["data"]
            assert len(chats) == 1
            assert chats[0]["id"] == chat["id"]
            assert chats[0]["case_id"] == case_a["id"]

            message_response = client.post(
                f"/api/cases/{case_a['id']}/chats/{chat['id']}/messages",
                json={
                    "content": "Who is linked to Alice?",
                    "mode": "mix",
                    "options": {"top_k": 5},
                },
            )
            assert message_response.status_code == 200
            response_data = message_response.json()["data"]
            assert response_data["message"]["role"] == "assistant"
            assert response_data["message"]["content"] == "Alice is linked to Bob."
            assert response_data["highlight"]["highlight_entities"] == ["ALICE", "BOB"]
            assert response_data["highlight"]["highlight_relationships"] == [
                {"src_id": "ALICE", "tgt_id": "BOB"}
            ]
            assert len(response_data["references"]) == 1
            reference_path = response_data["references"][0]["file_path"]
            assert reference_path.startswith("raw/")
            assert response_data["references"] == [
                {"reference_id": "ref-1", "file_path": reference_path}
            ]
            assert response_data["chunks"] == [
                {
                    "reference_id": "ref-1",
                    "file_path": reference_path,
                    "snippet": "Alice and Bob exchanged messages.",
                }
            ]
            assert response_data["retrieval_eval"]["mode"] == "mix"
            assert response_data["retrieval_eval"]["retrieved_entity_ids_topk"] == [
                "ALICE",
                "BOB",
            ]
            assert response_data["retrieval_eval"]["retrieved_relation_types_topk"] == [
                "KNOWS"
            ]
            assert "analysis_view" not in response_data
            assert "highlight_views" not in response_data
            assert response_data["model_name"] == "test-model"

            chat_response = client.get(f"/api/cases/{case_a['id']}/chats/{chat['id']}")
            assert chat_response.status_code == 200
            chat_data = chat_response.json()["data"]
            assert chat_data["id"] == chat["id"]
            assert chat_data["case_id"] == case_a["id"]
            assert len(chat_data["messages"]) == 2
            assert [row["role"] for row in chat_data["messages"]] == ["user", "assistant"]
            assistant_message = chat_data["messages"][-1]
            assert assistant_message["rag_metadata"]["mode"] == "mix"
            assert assistant_message["rag_metadata"]["model_name"] == "test-model"
            assert "analysis_view" not in assistant_message["rag_metadata"]
            assert "highlight_views" not in assistant_message["rag_metadata"]
            assert assistant_message["rag_metadata"]["references"] == [
                {"reference_id": "ref-1", "file_path": reference_path}
            ]
            assert (
                assistant_message["rag_metadata"]["highlight"]["highlight_entities"]
                == ["ALICE", "BOB"]
            )
            assert assistant_message["rag_metadata"]["retrieval_eval"]["mode"] == "mix"
            assert assistant_message["rag_metadata"]["retrieval_eval"][
                "retrieved_relation_ids_topk"
            ] == ["ALICE__KNOWS__BOB"]

            cross_case_get = client.get(f"/api/cases/{case_b['id']}/chats/{chat['id']}")
            assert cross_case_get.status_code == 404
            cross_case_send = client.post(
                f"/api/cases/{case_b['id']}/chats/{chat['id']}/messages",
                json={"content": "cross", "mode": "mix"},
            )
            assert cross_case_send.status_code == 404

            settings = get_settings()
            with get_connection(settings) as connection:
                rows = connection.execute(
                    "SELECT role, content, rag_metadata_json FROM message WHERE chat_id = ? "
                    "ORDER BY created_at ASC",
                    (chat["id"],),
                ).fetchall()
                assert len(rows) == 2
                assert rows[0]["role"] == "user"
                assert rows[1]["role"] == "assistant"
                assert rows[1]["rag_metadata_json"]
                metadata = json.loads(rows[1]["rag_metadata_json"])
                assert metadata["mode"] == "mix"
                assert metadata["model_name"] == "test-model"
                assert "analysis_view" not in metadata
                assert "highlight_views" not in metadata
                assert metadata["highlight"]["highlight_relationships"] == [
                    {"src_id": "ALICE", "tgt_id": "BOB"}
                ]
                assert metadata["retrieval_eval"]["quality_flags"]["non_empty_payload"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_chat_message_query_failure_stores_empty_highlight(monkeypatch) -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"chat-failure-test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)

        async def _failing_query(*args, **kwargs):
            raise RuntimeError("Synthetic query failure")

        monkeypatch.setattr(ChatService, "query_case_message", _failing_query)

        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case_data = _create_case(client, "Failure Case")
            _upload_document(client, case_data["id"])
            chat = _create_chat(client, case_data["id"])

            response = client.post(
                f"/api/cases/{case_data['id']}/chats/{chat['id']}/messages",
                json={"content": "test failure", "mode": "hybrid"},
            )
            assert response.status_code == 200
            data = response.json()["data"]
            assert data["message"]["role"] == "assistant"
            assert data["highlight"]["highlight_entities"] == []
            assert data["highlight"]["highlight_relationships"] == []
            assert data["highlight"]["references"] == []
            assert data["highlight"]["supporting_chunks"] == []
            assert data["references"] == []
            assert data["chunks"] == []
            assert data["retrieval_eval"]["mode"] == "hybrid"
            assert data["retrieval_eval"]["retrieved_entity_ids_topk"] == []
            assert data["retrieval_eval"]["quality_flags"]["non_empty_payload"] is False

            chat_response = client.get(f"/api/cases/{case_data['id']}/chats/{chat['id']}")
            assert chat_response.status_code == 200
            chat_data = chat_response.json()["data"]
            assert len(chat_data["messages"]) == 2
            assert [row["role"] for row in chat_data["messages"]] == ["user", "assistant"]
            assistant = chat_data["messages"][-1]
            assert assistant["rag_metadata"]["status"] == "failed"
            assert assistant["rag_metadata"]["mode"] == "hybrid"
            assert assistant["rag_metadata"]["model_name"] == get_settings().rag_llm_model
            assert assistant["rag_metadata"]["highlight"]["highlight_entities"] == []
            assert assistant["rag_metadata"]["retrieval_eval"]["quality_flags"] == {
                "entities_present": False,
                "relations_present": False,
                "non_empty_payload": False,
            }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_chat_send_requires_evidence() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"chat-no-evidence-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case_data = _create_case(client, "No Evidence Case")
            chat = _create_chat(client, case_data["id"])
            response = client.post(
                f"/api/cases/{case_data['id']}/chats/{chat['id']}/messages",
                json={"content": "Can I ask now?", "mode": "hybrid"},
            )
            assert response.status_code == 409
            assert response.json()["message"] == "Upload evidence before sending chat messages."
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_chat_send_requires_previous_question_to_be_answered() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"chat-pending-user-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case_data = _create_case(client, "Pending Question Case")
            _upload_document(client, case_data["id"])
            chat = _create_chat(client, case_data["id"])

            settings = get_settings()
            with get_connection(settings) as connection:
                connection.execute(
                    "INSERT INTO message (id, chat_id, role, content, created_at, rag_metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), chat["id"], "user", "pending", utc_now_iso(), None),
                )

            response = client.post(
                f"/api/cases/{case_data['id']}/chats/{chat['id']}/messages",
                json={"content": "Second question", "mode": "hybrid"},
            )
            assert response.status_code == 409
            assert response.json()["message"] == "Wait for the current question to be answered before sending another message."
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_chat_delete_cascades_messages_and_is_case_scoped() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"chat-delete-test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)

        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case_a = _create_case(client, "Delete Case A")
            case_b = _create_case(client, "Delete Case B")
            chat = _create_chat(client, case_a["id"], "Delete Me")

            settings = get_settings()
            with get_connection(settings) as connection:
                now = utc_now_iso()
                connection.execute(
                    "INSERT INTO message (id, chat_id, role, content, created_at, rag_metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), chat["id"], "user", "hello", now, None),
                )
                connection.execute(
                    "INSERT INTO message (id, chat_id, role, content, created_at, rag_metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), chat["id"], "assistant", "reply", now, "{}"),
                )

            delete_response = client.delete(f"/api/cases/{case_a['id']}/chats/{chat['id']}")
            assert delete_response.status_code == 200
            assert delete_response.json()["data"]["deleted"] is True

            with get_connection(settings) as connection:
                chat_row = connection.execute(
                    "SELECT id FROM chat WHERE id = ?",
                    (chat["id"],),
                ).fetchone()
                assert chat_row is None
                message_count = connection.execute(
                    "SELECT COUNT(*) AS c FROM message WHERE chat_id = ?",
                    (chat["id"],),
                ).fetchone()
                assert int(message_count["c"]) == 0

            list_response = client.get(f"/api/cases/{case_a['id']}/chats")
            assert list_response.status_code == 200
            assert list_response.json()["data"] == []

            protected_chat = _create_chat(client, case_a["id"], "Protected")
            cross_case_delete = client.delete(
                f"/api/cases/{case_b['id']}/chats/{protected_chat['id']}"
            )
            assert cross_case_delete.status_code == 404

            not_found_delete = client.delete(
                f"/api/cases/{case_a['id']}/chats/{uuid.uuid4()}"
            )
            assert not_found_delete.status_code == 404

            remaining_chats = client.get(f"/api/cases/{case_a['id']}/chats")
            assert remaining_chats.status_code == 200
            remaining_data = remaining_chats.json()["data"]
            assert len(remaining_data) == 1
            assert remaining_data[0]["id"] == protected_chat["id"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_chat_service_runs_queries_on_runtime_loop() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"chat-runtime-loop-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        service = ChatService(settings)
        case_root = temp_dir / "cases" / "demo"
        case_root.mkdir(parents=True, exist_ok=True)
        _write_minimal_graph(case_root)

        class _FakeLightRag:
            async def aquery_llm(self, *_args, **_kwargs):
                return {
                    "status": "success",
                    "llm_response": {"content": "runtime-loop-answer", "model": "runtime-model"},
                    "data": {
                        "entities": [{"entity_name": "ALICE", "entity_type": "person"}],
                        "relationships": [
                            {
                                "src_id": "ALICE",
                                "tgt_id": "BOB",
                                "relation_type": "KNOWS",
                            }
                        ],
                        "references": [{"reference_id": "ref-1", "file_path": "chat-evidence.txt"}],
                        "chunks": [
                            {
                                "reference_id": "ref-1",
                                "file_path": "chat-evidence.txt",
                                "content": "snippet",
                            }
                        ],
                    },
                    "metadata": {},
                }

        class _FakeRag:
            def __init__(self) -> None:
                self.lightrag = _FakeLightRag()

        class _FakePipeline:
            def __init__(self) -> None:
                self.runtime_called = False
                self.finalized = False

            async def initialize_rag_for_query(self, **_kwargs):
                return _FakeRag()

            async def finalize_rag_runtime(self, _rag):
                self.finalized = True

            async def run_in_runtime_loop(self, coro):
                self.runtime_called = True
                return await coro

        fake_pipeline = _FakePipeline()
        service._ingestion_pipeline = fake_pipeline  # type: ignore[attr-defined]

        result = asyncio.run(
            service.query_case_message(
                case_root=case_root,
                user_content="question",
                mode="mix",
                conversation_history=[],
                case_documents=[
                    {
                        "id": "doc-1",
                        "original_filename": "chat-evidence.txt",
                        "stored_file_path": "raw/chat-evidence.txt",
                        "confidence_code": "A1",
                    }
                ],
                options={},
            )
        )
        assert fake_pipeline.runtime_called is True
        assert fake_pipeline.finalized is True
        assert result.assistant_content == "runtime-loop-answer"
        assert result.metadata["model_name"] == "runtime-model"
        assert "analysis_view" not in result.metadata
        assert "highlight_views" not in result.metadata
        assert result.highlight["highlight_entities"] == ["ALICE"]
        assert len(result.highlight["highlight_relationships"]) == 1
        assert result.highlight["highlight_relationships"][0]["src_id"] == "ALICE"
        assert result.highlight["highlight_relationships"][0]["tgt_id"] == "BOB"
        assert result.highlight["highlight_relationships"][0]["edge_id"]
        assert result.retrieval_eval["retrieved_entity_ids_topk"] == ["ALICE"]
        assert result.retrieval_eval["retrieved_relation_types_topk"] == ["KNOWS"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_chat_service_naive_mode_supports_string_entity_rows() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"chat-naive-rows-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        service = ChatService(settings)
        case_root = temp_dir / "cases" / "demo"
        case_root.mkdir(parents=True, exist_ok=True)
        _write_minimal_graph(case_root)

        class _FakeLightRag:
            async def aquery_llm(self, *_args, **_kwargs):
                return {
                    "status": "success",
                    "llm_response": {"content": "naive-answer"},
                    "data": {
                        "entities": ["ALICE", "BOB"],
                        "relationships": [{"src_id": "ALICE", "tgt_id": "BOB"}],
                        "references": [{"reference_id": "ref-1", "file_path": "chat-evidence.txt"}],
                        "chunks": [
                            {
                                "reference_id": "ref-1",
                                "file_path": "chat-evidence.txt",
                                "content": "snippet",
                            }
                        ],
                    },
                    "metadata": {},
                }

        class _FakeRag:
            def __init__(self) -> None:
                self.lightrag = _FakeLightRag()

        class _FakePipeline:
            async def initialize_rag_for_query(self, **_kwargs):
                return _FakeRag()

            async def finalize_rag_runtime(self, _rag):
                return None

            async def run_in_runtime_loop(self, coro):
                return await coro

        service._ingestion_pipeline = _FakePipeline()  # type: ignore[attr-defined]

        result = asyncio.run(
            service.query_case_message(
                case_root=case_root,
                user_content="question",
                mode="naive",
                conversation_history=[],
                case_documents=[
                    {
                        "id": "doc-1",
                        "original_filename": "chat-evidence.txt",
                        "stored_file_path": "raw/chat-evidence.txt",
                        "confidence_code": "A1",
                    }
                ],
                options={"top_k": 5},
            )
        )
        assert result.retrieval_eval["mode"] == "naive"
        assert result.retrieval_eval["retrieved_entity_ids_topk"] == ["ALICE", "BOB"]
        assert result.highlight["highlight_entities"] == ["ALICE", "BOB"]
        assert "analysis_view" not in result.metadata
        assert "highlight_views" not in result.metadata
        assert result.retrieval_eval["quality_flags"]["non_empty_payload"] is True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_chat_service_highlights_entity_only_matches() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"chat-entity-highlight-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        service = ChatService(settings)
        case_root = temp_dir / "cases" / "demo"
        case_root.mkdir(parents=True, exist_ok=True)
        _write_minimal_graph(case_root)

        class _FakeLightRag:
            async def aquery_llm(self, *_args, **_kwargs):
                return {
                    "status": "success",
                    "llm_response": {"content": "alice-answer"},
                    "data": {
                        "entities": ["ALICE"],
                        "relationships": [],
                        "references": [{"reference_id": "ref-1", "file_path": "chat-evidence.txt"}],
                        "chunks": [
                            {
                                "reference_id": "ref-1",
                                "file_path": "chat-evidence.txt",
                                "content": "Alice appears in the retrieved evidence.",
                            }
                        ],
                    },
                    "metadata": {},
                }

        class _FakeRag:
            def __init__(self) -> None:
                self.lightrag = _FakeLightRag()

        class _FakePipeline:
            async def initialize_rag_for_query(self, **_kwargs):
                return _FakeRag()

            async def finalize_rag_runtime(self, _rag):
                return None

            async def run_in_runtime_loop(self, coro):
                return await coro

        service._ingestion_pipeline = _FakePipeline()  # type: ignore[attr-defined]

        result = asyncio.run(
            service.query_case_message(
                case_root=case_root,
                user_content="question",
                mode="mix",
                conversation_history=[],
                case_documents=[
                    {
                        "id": "doc-1",
                        "original_filename": "chat-evidence.txt",
                        "stored_file_path": "raw/chat-evidence.txt",
                        "confidence_code": "A1",
                    }
                ],
                options={},
            )
        )
        assert "analysis_view" not in result.metadata
        assert "highlight_views" not in result.metadata
        assert result.highlight["highlight_entities"] == ["ALICE"]
        assert result.highlight["highlight_relationships"] == []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_chat_service_retries_oom_with_lighter_mode() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"chat-oom-fallback-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        service = ChatService(settings)
        case_root = temp_dir / "cases" / "demo"
        case_root.mkdir(parents=True, exist_ok=True)
        _write_minimal_graph(case_root)

        class _FakeLightRag:
            def __init__(self) -> None:
                self.calls: list[tuple[str, object]] = []

            async def aquery_llm(self, _query, *, param, system_prompt=None):
                self.calls.append((param.mode, param))
                if len(self.calls) == 1:
                    return {
                        "status": "failure",
                        "message": "Query failed: Error code: Out of Memory",
                        "data": {},
                        "metadata": {},
                        "llm_response": {
                            "content": None,
                            "response_iterator": None,
                            "is_streaming": False,
                        },
                    }
                return {
                    "status": "success",
                    "llm_response": {"content": "fallback-answer", "model": "runtime-model"},
                    "data": {
                        "entities": [{"entity_name": "ALICE", "entity_type": "person"}],
                        "relationships": [
                            {
                                "src_id": "ALICE",
                                "tgt_id": "BOB",
                                "relation_type": "KNOWS",
                            }
                        ],
                        "references": [{"reference_id": "ref-1", "file_path": "chat-evidence.txt"}],
                        "chunks": [
                            {
                                "reference_id": "ref-1",
                                "file_path": "chat-evidence.txt",
                                "content": "snippet",
                            }
                        ],
                    },
                    "metadata": {},
                }

        class _FakeRag:
            def __init__(self) -> None:
                self.lightrag = _FakeLightRag()

        class _FakePipeline:
            def __init__(self) -> None:
                self.rag = _FakeRag()

            async def initialize_rag_for_query(self, **_kwargs):
                return self.rag

            async def finalize_rag_runtime(self, _rag):
                return None

            async def run_in_runtime_loop(self, coro):
                return await coro

        fake_pipeline = _FakePipeline()
        service._ingestion_pipeline = fake_pipeline  # type: ignore[attr-defined]

        result = asyncio.run(
            service.query_case_message(
                case_root=case_root,
                user_content="question",
                mode="hybrid",
                conversation_history=[],
                case_documents=[
                    {
                        "id": "doc-1",
                        "original_filename": "chat-evidence.txt",
                        "stored_file_path": "raw/chat-evidence.txt",
                        "confidence_code": "A1",
                    }
                ],
                options={},
            )
        )
        assert result.assistant_content == "fallback-answer"
        assert result.metadata["mode"] == "local"
        assert result.metadata["requested_mode"] == "hybrid"
        assert result.metadata["fallback_applied"] is True
        assert "analysis_view" not in result.metadata
        assert "highlight_views" not in result.metadata
        assert [call[0] for call in fake_pipeline.rag.lightrag.calls] == ["hybrid", "local"]

        strict_service = ChatService(settings)
        strict_pipeline = _FakePipeline()
        strict_service._ingestion_pipeline = strict_pipeline  # type: ignore[attr-defined]
        strict_result = asyncio.run(
            strict_service.query_case_message(
                case_root=case_root,
                user_content="question",
                mode="hybrid",
                conversation_history=[],
                case_documents=[
                    {
                        "id": "doc-1",
                        "original_filename": "chat-evidence.txt",
                        "stored_file_path": "raw/chat-evidence.txt",
                        "confidence_code": "A1",
                    }
                ],
                options={},
                allow_mode_fallback=False,
            )
        )
        assert strict_result.metadata["mode"] == "hybrid"
        assert strict_result.metadata["requested_mode"] == "hybrid"
        assert "fallback_applied" not in strict_result.metadata
        assert [call[0] for call in strict_pipeline.rag.lightrag.calls] == ["hybrid"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_chat_service_infers_retrieval_payload_from_chunk_snippets() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"chat-infer-snippets-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        service = ChatService(settings)
        case_root = temp_dir / "cases" / "demo"
        case_root.mkdir(parents=True, exist_ok=True)
        _write_minimal_graph(case_root)

        class _FakeLightRag:
            async def aquery_llm(self, *_args, **_kwargs):
                return {
                    "status": "success",
                    "llm_response": {"content": "naive-answer"},
                    "data": {
                        "entities": [],
                        "relationships": [],
                        "references": [{"reference_id": "ref-1", "file_path": "chat-evidence.txt"}],
                        "chunks": [
                            {
                                "reference_id": "ref-1",
                                "file_path": "chat-evidence.txt",
                                "content": "ALICE,123,shareholder of,BOB HOLDINGS LTD.,456",
                            }
                        ],
                    },
                    "metadata": {},
                }

        class _FakeRag:
            def __init__(self) -> None:
                self.lightrag = _FakeLightRag()

        class _FakePipeline:
            async def initialize_rag_for_query(self, **_kwargs):
                return _FakeRag()

            async def finalize_rag_runtime(self, _rag):
                return None

            async def run_in_runtime_loop(self, coro):
                return await coro

        service._ingestion_pipeline = _FakePipeline()  # type: ignore[attr-defined]

        result = asyncio.run(
            service.query_case_message(
                case_root=case_root,
                user_content="question",
                mode="naive",
                conversation_history=[],
                case_documents=[
                    {
                        "id": "doc-1",
                        "original_filename": "chat-evidence.txt",
                        "stored_file_path": "raw/chat-evidence.txt",
                        "confidence_code": "A1",
                    }
                ],
                options={"top_k": 5},
            )
        )
        assert result.highlight["highlight_entities"] == []
        assert result.retrieval_eval["retrieved_entity_ids_topk"][:2] == [
            "ALICE",
            "BOB HOLDINGS LTD",
        ]
        assert result.retrieval_eval["retrieved_relation_ids_topk"] == [
            "ALICE__MEMBER_OF__BOB HOLDINGS LTD"
        ]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
