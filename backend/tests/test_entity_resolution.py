from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.entity_resolution import EntityResolutionService
from backend.app.settings import get_settings


def _configure_env(temp_dir: Path) -> None:
    os.environ["RAWABIT_DB_PATH"] = str(temp_dir / "db.sqlite")
    os.environ["RAWABIT_CASES_ROOT"] = str(temp_dir / "cases")


def _create_case(client: TestClient, name: str = "ER Case") -> dict:
    response = client.post("/api/cases", json={"name": name})
    assert response.status_code == 200
    return response.json()["data"]


def _upload_document(client: TestClient, case_id: str) -> dict:
    files = {"file": ("er-source.txt", b"evidence", "text/plain")}
    data = {
        "confidence_source_reliability": "A",
        "confidence_information_validity": "1",
    }
    response = client.post(f"/api/cases/{case_id}/documents", data=data, files=files)
    assert response.status_code == 200
    return response.json()["data"]


def _write_entity_resolution_graph(
    *,
    case_root: Path,
    document_id: str,
    original_filename: str,
) -> None:
    lightrag_dir = case_root / "lightrag"
    lightrag_dir.mkdir(parents=True, exist_ok=True)

    graphml = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="node" attr.name="entity_type" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="DHS"><data key="d0">organization</data></node>
    <node id="Department of Homeland Security"><data key="d0">organization</data></node>
    <node id="Departement of Homeland Security"><data key="d0">organization</data></node>
    <node id="DHS Office of Inspector General Report"><data key="d0">organization</data></node>
    <node id="ICE"><data key="d0">organization</data></node>
    <edge source="DHS" target="ICE"></edge>
    <edge source="Department of Homeland Security" target="ICE"></edge>
  </graph>
</graphml>
"""
    (lightrag_dir / "graph_chunk_entity_relation.graphml").write_text(graphml, encoding="utf-8")

    (lightrag_dir / "kv_store_full_entities.json").write_text(
        (
            "{"
            f"\"{document_id}\": "
            "{"
            "\"entity_names\": ["
            "\"DHS\","
            "\"Department of Homeland Security\","
            "\"Departement of Homeland Security\","
            "\"DHS Office of Inspector General Report\","
            "\"ICE\""
            "]"
            "}"
            "}"
        ),
        encoding="utf-8",
    )
    (lightrag_dir / "kv_store_text_chunks.json").write_text(
        (
            "{"
            "\"chunk-1\": {"
            "\"_id\": \"chunk-1\","
            f"\"document_id\": \"{document_id}\","
            f"\"full_doc_id\": \"{document_id}\","
            "\"reference_id\": \"ref-er-1\","
            f"\"file_path\": \"{original_filename}\""
            "}"
            "}"
        ),
        encoding="utf-8",
    )


def _build_case_with_graph(temp_dir: Path) -> tuple[str, Path]:
    _configure_env(temp_dir)
    from backend.app.main import create_app

    with TestClient(create_app()) as client:
        case = _create_case(client)
        upload = _upload_document(client, case["id"])
        document = client.get(
            f"/api/cases/{case['id']}/documents/{upload['document_id']}"
        ).json()["data"]

        case_root = temp_dir / "cases" / case["case_slug"]
        _write_entity_resolution_graph(
            case_root=case_root,
            document_id=upload["document_id"],
            original_filename=document["original_filename"],
        )
        return case["id"], case_root


def test_entity_resolution_run_generic_resolution(monkeypatch) -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"entity-resolution-run-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        case_id, _case_root = _build_case_with_graph(temp_dir)
        service = EntityResolutionService(get_settings())
        merge_calls: list[tuple[tuple[str, ...], str]] = []

        async def _fake_merge(
            *,
            case_id: str,
            source_entities: list[str],
            target_entity: str,
        ) -> dict[str, str]:
            _ = case_id
            merge_calls.append((tuple(source_entities), target_entity))
            return {"status": "ok"}

        async def _fake_llm_call(client, prompt_text):  # noqa: ANN001
            _ = client, prompt_text
            # Simulate LLM returning one merge proposal
            return '{"merges": [{"keep": "DHS", "merge_into": "Department of Homeland Security", "confidence": 0.95, "reasoning": "acronym match"}]}'

        monkeypatch.setattr(service, "_amerge_lightrag_entities", _fake_merge)
        monkeypatch.setattr(service, "_call_llm", _fake_llm_call)

        result = asyncio.run(
            service.arun_generic_resolution(
                case_id=case_id,
                client=None,
            )
        )
        assert result["merged_count"] >= 1
        assert merge_calls
        assert any(target == "Department of Homeland Security" for _, target in merge_calls)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_entity_resolution_endpoints(monkeypatch) -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"entity-resolution-api-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        async def _fake_merge(
            self,
            *,
            case_id: str,
            source_entities: list[str],
            target_entity: str,
        ) -> dict[str, str]:
            _ = self
            _ = case_id
            _ = source_entities
            _ = target_entity
            return {"status": "ok"}

        monkeypatch.setattr(
            EntityResolutionService,
            "_amerge_lightrag_entities",
            _fake_merge,
        )

        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = _create_case(client, name="ER API Case")
            upload = _upload_document(client, case["id"])
            document = client.get(
                f"/api/cases/{case['id']}/documents/{upload['document_id']}"
            ).json()["data"]
            case_root = temp_dir / "cases" / case["case_slug"]
            _write_entity_resolution_graph(
                case_root=case_root,
                document_id=upload["document_id"],
                original_filename=document["original_filename"],
            )

            suggestions_response = client.get(
                f"/api/cases/{case['id']}/graph/entity-resolution/suggestions?min_confidence=medium&limit=20"
            )
            assert suggestions_response.status_code == 200
            suggestions_data = suggestions_response.json()["data"]
            assert any(
                {item["entity_a"], item["entity_b"]}
                == {"DHS", "Department of Homeland Security"}
                for item in suggestions_data
            )

            run_response = client.post(
                f"/api/cases/{case['id']}/graph/entity-resolution/run",
                json={
                    "auto_merge_high_confidence": False,
                    "max_auto_merges": 5,
                    "suggestion_limit": 20,
                },
            )
            assert run_response.status_code == 200
            run_data = run_response.json()["data"]
            assert run_data["detected_candidates"] >= 1
            assert run_data["auto_merged_count"] == 0

            merge_response = client.post(
                f"/api/cases/{case['id']}/graph/entity-resolution/merge",
                json={
                    "source_entities": ["DHS"],
                    "target_entity": "Department of Homeland Security",
                    "reason": "manual analyst merge",
                },
            )
            assert merge_response.status_code == 200
            merge_data = merge_response.json()["data"]
            assert merge_data["source_entities"] == ["DHS"]
            assert merge_data["target_entity"] == "Department of Homeland Security"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_entity_resolution_normalizes_merged_target_type_metadata() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"entity-resolution-type-normalize-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        storage_dir = temp_dir / "lightrag"
        storage_dir.mkdir(parents=True, exist_ok=True)
        graphml = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="node" attr.name="entity_name" attr.type="string"/>
  <key id="d1" for="node" attr.name="entity_type" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="[ORGANIZATION] ACME LTD">
      <data key="d0">[ORGANIZATION] ACME LTD</data>
      <data key="d1">person</data>
    </node>
  </graph>
</graphml>
"""
        (storage_dir / "graph_chunk_entity_relation.graphml").write_text(
            graphml, encoding="utf-8"
        )
        (storage_dir / "vdb_entities.json").write_text(
            '{"embedding_dim":1024,"data":[{"entity_name":"[ORGANIZATION] ACME LTD","entity_type":"person","content":"[ORGANIZATION] ACME LTD\\nDescription"}]}',
            encoding="utf-8",
        )

        EntityResolutionService._normalize_merged_entity_type_metadata(
            storage_dir, "[ORGANIZATION] ACME LTD"
        )

        updated_graphml = (storage_dir / "graph_chunk_entity_relation.graphml").read_text(
            encoding="utf-8"
        )
        updated_vdb = (storage_dir / "vdb_entities.json").read_text(encoding="utf-8")
        assert ">organization<" in updated_graphml
        assert '"entity_type": "organization"' in updated_vdb
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
