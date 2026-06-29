from __future__ import annotations

import csv
import io
import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.fs import ensure_case_lightrag_dir


def _configure_env(temp_dir: Path) -> None:
    os.environ["RAWABIT_DB_PATH"] = str(temp_dir / "db.sqlite")
    os.environ["RAWABIT_CASES_ROOT"] = str(temp_dir / "cases")
    os.environ["RAWABIT_OPENROUTER_API_KEY"] = ""


def _upload_text_document(client: TestClient, case_id: str) -> dict:
    response = client.post(
        f"/api/cases/{case_id}/documents",
        data={
            "confidence_source_reliability": "A",
            "confidence_information_validity": "1",
        },
        files={"file": ("graph-source.txt", b"Agent A deployed a PepperBall launcher in Chicago.", "text/plain")},
    )
    assert response.status_code == 200
    return response.json()["data"]


def test_relationships_reports_projection_warning_when_actor_projection_empty() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"graph-api-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = client.post(
                "/api/cases", json={"name": "Graph Projection Case"}
            ).json()["data"]
            case_root = temp_dir / "cases" / case["case_slug"]
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
    <node id="ORG_A"><data key="d0">Organization</data><data key="d1">Primary org</data></node>
    <node id="OTHER_B"><data key="d0">Other</data><data key="d1">Non-actor node</data></node>
    <edge source="ORG_A" target="OTHER_B">
      <data key="d2">chunk-1</data>
      <data key="d3">ASSOCIATED_WITH</data>
      <data key="d4">Organization linked to non-actor node</data>
    </edge>
  </graph>
</graphml>
"""
            (lightrag_dir / "graph_chunk_entity_relation.graphml").write_text(
                graphml, encoding="utf-8"
            )
            (lightrag_dir / "kv_store_text_chunks.json").write_text(
                json.dumps(
                    {
                        "chunk-1": {
                            "_id": "chunk-1",
                            "reference_id": "ref-1",
                            "document_id": "doc-1",
                            "full_doc_id": "doc-1",
                            "confidence_code": "A1",
                            "file_path": "graph.txt",
                            "content": "Organization linked to non-actor node.",
                        }
                    }
                ),
                encoding="utf-8",
            )

            graph_response = client.get(
                f"/api/cases/{case['id']}/graph?view=link&limit=50"
            )
            assert graph_response.status_code == 200
            graph_payload = graph_response.json()["data"]
            assert len(graph_payload["nodes"]) == 2
            assert len(graph_payload["edges"]) == 1

            rel_response = client.get(f"/api/cases/{case['id']}/relationships?limit=50")
            assert rel_response.status_code == 200
            rel_payload = rel_response.json()["data"]
            assert rel_payload["relationships"] == []
            assert rel_payload["actorNodeCount"] == 1
            assert rel_payload["actorEdgeCount"] == 0
            assert rel_payload["rawNodeCount"] == 2
            assert rel_payload["rawEdgeCount"] == 1
            assert "projectionWarning" in rel_payload
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_graph_view_derives_labels_for_placeholder_event_entities() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"graph-api-placeholder-label-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = client.post(
                "/api/cases", json={"name": "Placeholder Graph Case"}
            ).json()["data"]
            case_root = temp_dir / "cases" / case["case_slug"]
            lightrag_dir = case_root / "lightrag"
            lightrag_dir.mkdir(parents=True, exist_ok=True)

            graphml = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="node" attr.name="entity_type" attr.type="string"/>
  <key id="d1" for="node" attr.name="description" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="event_1"><data key="d0">event</data><data key="d1">A scuffle in Chicago.</data></node>
    <node id="ORG_A"><data key="d0">Organization</data><data key="d1">Primary org</data></node>
    <edge source="ORG_A" target="event_1" />
  </graph>
</graphml>
"""
            (lightrag_dir / "graph_chunk_entity_relation.graphml").write_text(
                graphml, encoding="utf-8"
            )

            response = client.get(f"/api/cases/{case['id']}/graph?view=event&limit=50")
            assert response.status_code == 200
            payload = response.json()["data"]
            placeholder_node = next(
                node for node in payload["nodes"] if node["id"] == "event_1"
            )
            assert placeholder_node["label"] == "Scuffle in Chicago"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_graph_export_entities_and_relations_csv() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"graph-api-export-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = client.post("/api/cases", json={"name": "Graph Export Case"}).json()[
                "data"
            ]
            document = _upload_text_document(client, case["id"])
            document_id = document["document_id"]
            case_root = temp_dir / "cases" / case["case_slug"]
            lightrag_dir = ensure_case_lightrag_dir(case_root, case["id"])

            graphml = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="node" attr.name="entity_type" attr.type="string"/>
  <key id="d1" for="node" attr.name="description" attr.type="string"/>
  <key id="d2" for="node" attr.name="source_id" attr.type="string"/>
  <key id="d3" for="edge" attr.name="source_id" attr.type="string"/>
  <key id="d4" for="edge" attr.name="keywords" attr.type="string"/>
  <key id="d5" for="edge" attr.name="description" attr.type="string"/>
  <key id="d6" for="edge" attr.name="timestamp" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="Agent A"><data key="d0">person</data><data key="d1">Field agent in Chicago.</data><data key="d2">chunk-1</data></node>
    <node id="Chicago"><data key="d0">location</data><data key="d1">Deployment city.</data><data key="d2">chunk-1</data></node>
    <edge source="Agent A" target="Chicago"><data key="d3">chunk-1</data><data key="d4">OCCURRED_AT</data><data key="d5">Agent A deployed equipment in Chicago.</data><data key="d6">2026-01-02</data></edge>
  </graph>
</graphml>
"""
            (lightrag_dir / "graph_chunk_entity_relation.graphml").write_text(
                graphml, encoding="utf-8"
            )
            (lightrag_dir / "kv_store_text_chunks.json").write_text(
                json.dumps(
                    {
                        "chunk-1": {
                            "_id": "chunk-1",
                            "reference_id": "chunk-1",
                            "document_id": document_id,
                            "full_doc_id": document_id,
                            "confidence_code": "A1",
                            "file_path": "graph-source.txt",
                            "content": "Agent A deployed a PepperBall launcher in Chicago.",
                        }
                    }
                ),
                encoding="utf-8",
            )

            entities_response = client.get(
                f"/api/cases/{case['id']}/graph/export/entities.csv"
            )
            assert entities_response.status_code == 200
            assert entities_response.headers["content-type"] == "text/csv; charset=utf-8"
            assert (
                entities_response.headers["content-disposition"]
                == f'attachment; filename="{case["case_slug"]}-entities.csv"'
            )
            entity_rows = list(csv.DictReader(io.StringIO(entities_response.text)))
            assert entity_rows[0].keys() == {
                "id",
                "name",
                "type",
                "description",
                "sources",
            }
            agent_row = next(row for row in entity_rows if row["id"] == "Agent A")
            assert agent_row["name"] == "Agent A"
            assert agent_row["type"] == "person"
            assert agent_row["description"] == "Field agent in Chicago."
            assert "#chunk-1" in agent_row["sources"]

            relations_response = client.get(
                f"/api/cases/{case['id']}/graph/export/relations.csv"
            )
            assert relations_response.status_code == 200
            assert (
                relations_response.headers["content-disposition"]
                == f'attachment; filename="{case["case_slug"]}-relations.csv"'
            )
            relation_rows = list(csv.DictReader(io.StringIO(relations_response.text)))
            assert relation_rows[0].keys() == {
                "id",
                "source_id",
                "source_name",
                "source_type",
                "target_id",
                "target_name",
                "target_type",
                "relation_type",
                "description",
                "timestamp",
                "confidence_score",
                "confidence_band",
                "sources",
            }
            relation_row = relation_rows[0]
            assert relation_row["source_id"] == "Agent A"
            assert relation_row["source_name"] == "Agent A"
            assert relation_row["source_type"] == "person"
            assert relation_row["target_id"] == "Chicago"
            assert relation_row["target_name"] == "Chicago"
            assert relation_row["target_type"] == "location"
            assert relation_row["relation_type"] == "OCCURRED_AT"
            assert relation_row["description"] == "Agent A deployed equipment in Chicago."
            assert relation_row["timestamp"] == "2026-01-02"
            assert "#chunk-1" in relation_row["sources"]

            missing_response = client.get(
                f"/api/cases/{uuid.uuid4()}/graph/export/entities.csv"
            )
            assert missing_response.status_code == 404
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_graph_entity_details_flag_noise_and_synthesize_description() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"graph-api-entity-details-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = client.post(
                "/api/cases", json={"name": "Entity Detail Case"}
            ).json()["data"]
            case_root = temp_dir / "cases" / case["case_slug"]
            lightrag_dir = ensure_case_lightrag_dir(case_root, case["id"])

            graphml = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="node" attr.name="entity_type" attr.type="string"/>
  <key id="d1" for="node" attr.name="description" attr.type="string"/>
  <key id="d2" for="edge" attr.name="keywords" attr.type="string"/>
  <key id="d3" for="edge" attr.name="description" attr.type="string"/>
  <key id="d4" for="edge" attr.name="source_id" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="Newsletter"><data key="d0">communication</data><data key="d1">Section title.&lt;SEP&gt;A regularly distributed publication.</data></node>
    <node id="Bellingcat"><data key="d0">organization</data><data key="d1">Investigative newsroom documenting incidents.</data></node>
    <edge source="Bellingcat" target="Newsletter"><data key="d2">REFERENCES</data><data key="d3">Bellingcat references the newsletter section.</data><data key="d4">chunk-1</data></edge>
  </graph>
</graphml>
"""
            (lightrag_dir / "graph_chunk_entity_relation.graphml").write_text(
                graphml, encoding="utf-8"
            )
            (lightrag_dir / "kv_store_text_chunks.json").write_text(
                json.dumps(
                    {
                        "chunk-1": {
                            "_id": "chunk-1",
                            "reference_id": "chunk-1",
                            "document_id": "doc-1",
                            "full_doc_id": "doc-1",
                            "confidence_code": "A1",
                            "file_path": "raw/demo.txt",
                            "content": "Newsletter and Bellingcat are mentioned in the same article footer.",
                        }
                    }
                ),
                encoding="utf-8",
            )

            response = client.get(f"/api/cases/{case['id']}/graph/entity/Newsletter")
            assert response.status_code == 200
            payload = response.json()["data"]
            assert payload["noise_candidate"] is True
            assert "low-value presentation" in payload["description"].lower()
            assert payload["related_relationships"][0]["entity_label"] == "Bellingcat"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_event_and_flow_views_project_grounded_edges_with_evidence() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"graph-api-event-flow-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = client.post("/api/cases", json={"name": "Event Flow Case"}).json()["data"]
            document_id = "doc-1"
            case_root = temp_dir / "cases" / case["case_slug"]
            lightrag_dir = ensure_case_lightrag_dir(case_root, case["id"])

            graphml = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="node" attr.name="entity_type" attr.type="string"/>
  <key id="d1" for="node" attr.name="description" attr.type="string"/>
  <key id="d2" for="edge" attr.name="keywords" attr.type="string"/>
  <key id="d3" for="edge" attr.name="description" attr.type="string"/>
  <key id="d4" for="edge" attr.name="source_id" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="Deployment in Chicago"><data key="d0">event</data><data key="d1">Agent A deployed a PepperBall launcher in Chicago on Oct. 23.</data></node>
    <node id="Agent A"><data key="d0">person</data><data key="d1">Agent A.</data></node>
    <node id="Chicago"><data key="d0">location</data><data key="d1">Chicago.</data></node>
    <node id="PepperBall launcher"><data key="d0">asset</data><data key="d1">PepperBall launcher.</data></node>
    <edge source="Agent A" target="PepperBall launcher"><data key="d2">USED</data><data key="d3">Agent A deployed a PepperBall launcher in Chicago.</data><data key="d4">chunk-1</data></edge>
  </graph>
</graphml>
"""
            (lightrag_dir / "graph_chunk_entity_relation.graphml").write_text(
                graphml, encoding="utf-8"
            )
            (lightrag_dir / "vdb_entities.json").write_text(
                json.dumps(
                    {
                        "event": {
                            "entity_name": "Deployment in Chicago",
                            "entity_type": "event",
                            "description": "Agent A deployed a PepperBall launcher in Chicago on Oct. 23.",
                            "source_id": "chunk-1",
                        },
                        "agent": {
                            "entity_name": "Agent A",
                            "entity_type": "person",
                            "description": "Agent A.",
                            "source_id": "chunk-1",
                        },
                        "location": {
                            "entity_name": "Chicago",
                            "entity_type": "location",
                            "description": "Chicago.",
                            "source_id": "chunk-1",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (lightrag_dir / "kv_store_text_chunks.json").write_text(
                json.dumps(
                    {
                        "chunk-1": {
                            "_id": "chunk-1",
                            "reference_id": "chunk-1",
                            "document_id": document_id,
                            "full_doc_id": document_id,
                            "confidence_code": "A1",
                            "content": "Agent A deployed a PepperBall launcher in Chicago on Oct. 23.",
                        }
                    }
                ),
                encoding="utf-8",
            )
            from backend.app.graph_api import (
                ANALYSIS_PROMPT_VERSION,
                ANALYSIS_SCHEMA_VERSION,
                GraphStore,
            )

            store = GraphStore(
                case_root=case_root,
                case_id=case["id"],
                documents=[],
            )
            source_hash = store.analysis_source_hash()
            analysis_dir = case_root / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            (analysis_dir / "event-analysis.json").write_text(
                json.dumps(
                    {
                        "schema_version": ANALYSIS_SCHEMA_VERSION,
                        "analysis_type": "event",
                        "source_graph_hash": source_hash,
                        "prompt_version": ANALYSIS_PROMPT_VERSION,
                        "method": "UNODC event charting test artifact",
                        "nodes": [
                            {"id": "Deployment in Chicago", "event_status": "confirmed"},
                            {"id": "Agent A"},
                        ],
                        "edges": [
                            {
                                "src_id": "Deployment in Chicago",
                                "tgt_id": "Agent A",
                                "relation_type": "INVOLVES",
                                "label": "Deployment in Chicago involved Agent A.",
                                "confidence_score": 0.82,
                                "confidence_band": "high",
                                "evidence_refs": ["chunk-1"],
                            }
                        ],
                        "unknowns": [],
                    }
                ),
                encoding="utf-8",
            )
            (analysis_dir / "flow-analysis.json").write_text(
                json.dumps(
                    {
                        "schema_version": ANALYSIS_SCHEMA_VERSION,
                        "analysis_type": "flow",
                        "source_graph_hash": source_hash,
                        "prompt_version": ANALYSIS_PROMPT_VERSION,
                        "method": "UNODC flow analysis test artifact",
                        "nodes": [
                            {"id": "Agent A", "role": "source"},
                            {"id": "PepperBall launcher", "role": "destination"},
                        ],
                        "edges": [
                            {
                                "src_id": "Agent A",
                                "tgt_id": "PepperBall launcher",
                                "relation_type": "FLOW",
                                "label": "Agent A directed the deployment of the PepperBall launcher.",
                                "flow_object": "PepperBall launcher deployment",
                                "direction": "src_to_tgt",
                                "confidence_score": 0.76,
                                "confidence_band": "high",
                                "evidence_refs": ["chunk-1"],
                            }
                        ],
                        "unknowns": [],
                    }
                ),
                encoding="utf-8",
            )

            event_response = client.get(f"/api/cases/{case['id']}/graph?view=event&limit=50")
            assert event_response.status_code == 200
            event_payload = event_response.json()["data"]
            assert event_payload["analysis_meta"]["source"] == "artifact"
            assert any(
                edge["src_id"] == "Deployment in Chicago" and edge["tgt_id"] == "Agent A"
                for edge in event_payload["edges"]
            )
            projected = next(
                edge
                for edge in event_payload["edges"]
                if edge["src_id"] == "Deployment in Chicago" and edge["tgt_id"] == "Agent A"
            )
            assert projected["evidence"][0]["source_id"] == "chunk-1"
            assert "PepperBall launcher" in projected["evidence"][0]["snippet"]

            flow_response = client.get(f"/api/cases/{case['id']}/graph?view=flow&limit=50")
            assert flow_response.status_code == 200
            flow_payload = flow_response.json()["data"]
            assert flow_payload["analysis_meta"]["source"] == "artifact"
            assert flow_payload["edges"][0]["relation_type"] == "FLOW"
            assert flow_payload["edges"][0]["evidence"][0]["source_id"] == "chunk-1"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_flow_analysis_rejects_generic_association_artifact_edges() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"graph-api-flow-validation-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        from backend.app.main import create_app

        with TestClient(create_app()) as client:
            case = client.post("/api/cases", json={"name": "Flow Validation Case"}).json()["data"]
            case_root = temp_dir / "cases" / case["case_slug"]
            lightrag_dir = ensure_case_lightrag_dir(case_root, case["id"])

            graphml = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="node" attr.name="entity_type" attr.type="string"/>
  <key id="d1" for="node" attr.name="description" attr.type="string"/>
  <key id="d2" for="edge" attr.name="keywords" attr.type="string"/>
  <key id="d3" for="edge" attr.name="description" attr.type="string"/>
  <key id="d4" for="edge" attr.name="source_id" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="Person A"><data key="d0">person</data><data key="d1">Person A.</data></node>
    <node id="Person B"><data key="d0">person</data><data key="d1">Person B.</data></node>
    <edge source="Person A" target="Person B"><data key="d2">ASSOCIATED_WITH</data><data key="d3">Person A is associated with Person B.</data><data key="d4">chunk-1</data></edge>
  </graph>
</graphml>
"""
            (lightrag_dir / "graph_chunk_entity_relation.graphml").write_text(
                graphml, encoding="utf-8"
            )
            (lightrag_dir / "kv_store_text_chunks.json").write_text(
                json.dumps({"chunk-1": {"_id": "chunk-1", "content": "Person A is associated with Person B."}}),
                encoding="utf-8",
            )
            from backend.app.graph_api import (
                ANALYSIS_PROMPT_VERSION,
                ANALYSIS_SCHEMA_VERSION,
                GraphStore,
            )

            source_hash = GraphStore(case_root=case_root, case_id=case["id"], documents=[]).analysis_source_hash()
            analysis_dir = case_root / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            (analysis_dir / "flow-analysis.json").write_text(
                json.dumps(
                    {
                        "schema_version": ANALYSIS_SCHEMA_VERSION,
                        "analysis_type": "flow",
                        "source_graph_hash": source_hash,
                        "prompt_version": ANALYSIS_PROMPT_VERSION,
                        "nodes": [{"id": "Person A"}, {"id": "Person B"}],
                        "edges": [
                            {
                                "src_id": "Person A",
                                "tgt_id": "Person B",
                                "relation_type": "ASSOCIATED_WITH",
                                "label": "Generic association is not a directed flow.",
                                "flow_object": "unknown",
                                "evidence_refs": ["chunk-1"],
                            }
                        ],
                        "unknowns": [],
                    }
                ),
                encoding="utf-8",
            )

            response = client.get(f"/api/cases/{case['id']}/graph?view=flow&limit=50")
            assert response.status_code == 200
            payload = response.json()["data"]
            assert payload["analysis_meta"]["source"] == "fallback"
            assert payload["nodes"] == []
            assert payload["edges"] == []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
