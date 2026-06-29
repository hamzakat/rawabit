from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from backend.app.prompt_catalog import (
    RUNTIME_REQUIRED_PROMPT_KEYS,
    get_prompt_catalog,
)


def _catalog_payload(
    prompt_rows: dict[str, dict],
    rules: list[dict] | None = None,
    library_prompt_catalog: dict[str, dict[str, dict]] | None = None,
) -> dict:
    payload = {
        "schema_version": "1.0",
        "metadata": {"owner": "tests", "updated_at": "2026-02-21T00:00:00Z"},
        "prompts": prompt_rows,
        "external_overrides": {"rules": rules or []},
    }
    if library_prompt_catalog is not None:
        payload["library_prompt_catalog"] = library_prompt_catalog
    return payload


def _prompt_row(text: str, placeholders: list[str] | None = None) -> dict:
    return {
        "text": text,
        "description": "test prompt",
        "placeholders": placeholders or [],
        "domain": "tests",
        "default_model_hint": None,
        "default_params": {},
        "enabled": True,
    }


def test_prompt_catalog_validate_required_keys() -> None:
    temp_dir = Path.cwd() / "data" / "pytest-tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "prompt-catalog-required.json"
    payload = _catalog_payload(
        {
            "ingestion.preinsert_summary": _prompt_row(
                "Summary: {source_text}", ["source_text"]
            ),
            "ingestion.evidence_normalization": _prompt_row(
                "Normalize {source_kind}: {source_text}",
                ["source_kind", "source_text"],
            ),
            "ingestion.image_caption.with_visible_text": _prompt_row("with-visible"),
            "ingestion.image_caption.description_only": _prompt_row("description-only"),
            "ingestion.embedding.dimension_probe_input": _prompt_row(
                "{probe_text}", ["probe_text"]
            ),
            "analysis.link_projection": _prompt_row(
                "Link {graph_json}", ["graph_json"]
            ),
            "analysis.event_projection": _prompt_row(
                "Event {graph_json}", ["graph_json"]
            ),
            "analysis.flow_projection": _prompt_row(
                "Flow {graph_json}", ["graph_json"]
            ),
            "analysis.narrative_summary": _prompt_row(
                "Summary {graph_json} {charts_json}",
                ["graph_json", "charts_json"],
            ),
            "analysis.bundle_repair": _prompt_row(
                "Repair {rejected_bundle} {validation_errors}",
                ["rejected_bundle", "validation_errors"],
            ),
            "analysis.mermaid_repair": _prompt_row(
                "Repair {mermaid_code} {render_error}",
                ["mermaid_code", "render_error"],
            ),
            "summary.entity_detail": _prompt_row(
                "Entity {entity_label}",
                [
                    "entity_label",
                    "entity_type",
                    "entity_summary",
                    "evidence_json",
                    "related_relationships_json",
                ],
            ),
            "summary.relationship_detail": _prompt_row(
                "Relationship {src_label} {relation_type} {tgt_label}",
                [
                    "src_label",
                    "tgt_label",
                    "relation_type",
                    "existing_description",
                    "evidence_json",
                ],
            ),
        }
    )
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    catalog = get_prompt_catalog(path, auto_reload=True)
    catalog.validate_required_keys(RUNTIME_REQUIRED_PROMPT_KEYS)
    assert (
        catalog.render("ingestion.preinsert_summary", {"source_text": "doc body"})
        == "Summary: doc body"
    )


def test_prompt_catalog_hot_reload_on_file_change() -> None:
    temp_dir = Path.cwd() / "data" / "pytest-tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "prompt-catalog-reload.json"
    path.write_text(
        json.dumps(
            _catalog_payload({"diagnostic.test": _prompt_row("v1:{value}", ["value"])}),
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    catalog = get_prompt_catalog(path, auto_reload=True)
    assert catalog.render("diagnostic.test", {"value": "x"}) == "v1:x"

    time.sleep(1.1)
    path.write_text(
        json.dumps(
            _catalog_payload({"diagnostic.test": _prompt_row("v2:{value}", ["value"])}),
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    assert catalog.render("diagnostic.test", {"value": "x"}) == "v2:x"


def test_prompt_catalog_invalid_reload_keeps_previous_snapshot() -> None:
    temp_dir = Path.cwd() / "data" / "pytest-tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "prompt-catalog-invalid-reload.json"
    path.write_text(
        json.dumps(
            _catalog_payload(
                {"diagnostic.test": _prompt_row("stable:{value}", ["value"])}
            ),
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    catalog = get_prompt_catalog(path, auto_reload=True)
    assert catalog.render("diagnostic.test", {"value": "x"}) == "stable:x"

    time.sleep(1.1)
    path.write_text("{invalid json", encoding="utf-8")
    assert catalog.render("diagnostic.test", {"value": "x"}) == "stable:x"


def test_prompt_catalog_external_overrides_apply_to_messages() -> None:
    temp_dir = Path.cwd() / "data" / "pytest-tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "prompt-catalog-overrides.json"
    payload = _catalog_payload(
        {"diagnostic.test": _prompt_row("{value}", ["value"])},
        rules=[
            {
                "id": "replace_message_text",
                "enabled": True,
                "target": "messages",
                "match_type": "contains",
                "pattern": "old",
                "replace": "new",
                "priority": 1,
            }
        ],
    )
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    catalog = get_prompt_catalog(path, auto_reload=True)
    _, _, messages = catalog.apply_external_overrides(
        messages=[{"role": "user", "content": [{"type": "text", "text": "old value"}]}]
    )
    assert messages is not None
    assert messages[0]["content"][0]["text"] == "new value"


def test_prompt_catalog_render_rejects_unknown_placeholder() -> None:
    temp_dir = Path.cwd() / "data" / "pytest-tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "prompt-catalog-placeholders.json"
    path.write_text(
        json.dumps(
            _catalog_payload({"diagnostic.test": _prompt_row("{value}", ["value"])}),
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    catalog = get_prompt_catalog(path, auto_reload=True)
    with pytest.raises(RuntimeError):
        catalog.render("diagnostic.test", {"value": "ok", "extra": "bad"})


def test_library_prompt_overrides_only_apply_when_value_present() -> None:
    temp_dir = Path.cwd() / "data" / "pytest-tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "prompt-catalog-library-overrides.json"
    path.write_text(
        json.dumps(
            _catalog_payload(
                {"diagnostic.test": _prompt_row("{value}", ["value"])},
                library_prompt_catalog={
                    "lightrag": {
                        "entity_extraction_system_prompt": {
                            "description": "LightRAG system prompt for entity and relation extraction.",
                            "value": "override-system",
                        },
                        "entity_extraction_examples": {
                            "description": "Few-shot extraction examples used in LightRAG extraction flow.",
                            "value": None,
                        },
                    },
                    "raganything": {
                        "vision_prompt": {
                            "description": "RAGAnything multimodal image analysis prompt.",
                        }
                    },
                },
            ),
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    catalog = get_prompt_catalog(path, auto_reload=True)
    assert catalog.get_library_prompt_overrides("lightrag") == {
        "entity_extraction_system_prompt": "override-system"
    }
    assert catalog.get_library_prompt_overrides("raganything") == {}
    assert catalog.get_library_prompt_descriptions("lightrag") == {
        "entity_extraction_system_prompt": "LightRAG system prompt for entity and relation extraction.",
        "entity_extraction_examples": "Few-shot extraction examples used in LightRAG extraction flow.",
    }


def test_library_prompt_catalog_rejects_invalid_value_type() -> None:
    temp_dir = Path.cwd() / "data" / "pytest-tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "prompt-catalog-library-invalid-type.json"
    path.write_text(
        json.dumps(
            _catalog_payload(
                {"diagnostic.test": _prompt_row("{value}", ["value"])},
                library_prompt_catalog={
                    "lightrag": {
                        "entity_extraction_system_prompt": {
                            "description": "desc",
                            "value": {"invalid": True},
                        }
                    }
                },
            ),
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    catalog = get_prompt_catalog(path, auto_reload=True)
    with pytest.raises(RuntimeError):
        catalog.get_library_prompt_overrides("lightrag")


def test_default_pole_prompts_preserve_organizations_and_reference_context() -> None:
    path = Path("backend/config/prompts.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload["prompts"]
    lightrag = payload["library_prompt_catalog"]["lightrag"]

    normalization_prompt = prompts["ingestion.evidence_normalization"]["text"]
    system_prompt = lightrag["entity_extraction_system_prompt"]["value"]
    user_prompt = lightrag["entity_extraction_user_prompt"]["value"]
    examples = "\n".join(lightrag["entity_extraction_examples"]["value"])

    assert "unresolved reference context" in normalization_prompt
    assert "[PERSON], [ORGANIZATION], [OBJECT], [LOCATION], [EVENT]" in system_prompt
    assert "Organizations must use the [ORGANIZATION] prefix" in system_prompt
    assert "source-system IDs" in user_prompt
    assert "[PERSON] ACME Imports" not in examples
    assert "[ORGANIZATION] ACME Imports" in examples


def test_analysis_prompts_use_mermaid_bundle_contract_and_unodc_rules() -> None:
    payload = json.loads(Path("backend/config/prompts.json").read_text(encoding="utf-8"))
    prompts = payload["prompts"]
    combined = "\n".join(
        prompts[key]["text"]
        for key in (
            "analysis.link_projection",
            "analysis.event_projection",
            "analysis.flow_projection",
            "analysis.bundle_repair",
            "analysis.mermaid_repair",
        )
    )

    assert "self-contained HTML5 document" not in combined
    assert "strict JSON" in combined
    assert "%% source-id:" in combined
    assert "relationship_map" in prompts["analysis.link_projection"]["text"]
    assert "operational_hierarchy" in prompts["analysis.link_projection"]["text"]
    assert "affiliation_structure" in prompts["analysis.link_projection"]["text"]
    assert "Exclude Event nodes" in prompts["analysis.link_projection"]["text"]
    assert "chronological_timeline" in prompts["analysis.event_projection"]["text"]
    assert "event_dependencies" in prompts["analysis.event_projection"]["text"]
    assert "actor_event_matrix" in prompts["analysis.event_projection"]["text"]
    assert "commodity_flow" in prompts["analysis.flow_projection"]["text"]
    assert "activity_flow" in prompts["analysis.flow_projection"]["text"]
    assert "event_flow" in prompts["analysis.flow_projection"]["text"]
    assert "criminal hierarchy" not in combined.lower()
    assert "kingpin" not in combined.lower()
    from backend.app.analysis_service import AnalysisService

    combined_contract = AnalysisService._combined_bundle_instruction()
    assert "`charts`" in combined_contract
    assert "`summary_text`" in combined_contract
    assert "no second narrative call" in combined_contract
