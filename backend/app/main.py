from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from .analysis_runner import AnalysisRunner
from .analysis_service import AnalysisService, MAX_ANALYSIS_REPAIR_ATTEMPTS
from .case_summary import CaseSummaryService
from .cleanup import cleanup_case_ingestion_artifacts, cleanup_document_artifacts
from .chat_service import ChatService
from .db import get_connection, init_db
from .document_search import DocumentSearchService, SEARCH_SOURCE_VALUES
from .entity_resolution import EntityResolutionService
from .fs import create_case_workspace, delete_case_workspace
from .graph_api import GraphInsightService, GraphStore, parse_csv_query
from .ingestion_preflight import compute_ingestion_preflight
from .ingestion_pipeline import IngestionPipeline
from .job_runner import JobRunner
from .job_logs import append_job_log, install_job_log_capture, list_job_logs
from .prompt_catalog import (
    RUNTIME_REQUIRED_PROMPT_KEYS,
    get_prompt_catalog,
)
from .schemas import (
    AnalysisCreate,
    AnalysisRepairRequest,
    CaseCreate,
    CaseUpdate,
    ChatCreate,
    ChatMessageCreate,
    DocumentDuplicateCheckRequest,
    EntityResolutionMergeRequest,
)
from .settings import get_settings, save_settings_overrides, load_settings_overrides, USER_MUTABLE_SETTINGS
from .utils import ensure_unique_slug, slugify, utc_now_iso


def create_app() -> FastAPI:
    settings = get_settings()
    install_job_log_capture(settings)
    prompt_catalog = get_prompt_catalog(
        path=settings.prompt_catalog_path,
        auto_reload=settings.prompt_catalog_auto_reload,
    )
    prompt_catalog.validate_required_keys(RUNTIME_REQUIRED_PROMPT_KEYS)
    confidence_source_values = {"A", "B", "C", "X"}
    confidence_validity_values = {"1", "2", "3", "4"}
    allowed_ingest_profiles = {
        "balanced_fast",
        "balanced_fast_intel",
        "full_enrichment",
    }
    allowed_processing_modes = {"multimodal", "text_first"}
    allowed_parse_method_overrides = {
        "auto",
        "txt",
        "ocr",
        "native",
        "vlm-first",
        "transcript-first",
    }
    case_summary_service = CaseSummaryService(settings)
    ingestion_pipeline = IngestionPipeline(
        settings,
        case_summary_service=case_summary_service,
    )
    job_runner = JobRunner(settings, pipeline=ingestion_pipeline)
    chat_service = ChatService(settings)
    analysis_service = AnalysisService(settings, chat_service=chat_service)
    analysis_runner = AnalysisRunner(settings, analysis_service=analysis_service)
    entity_resolution_service = EntityResolutionService(settings)
    document_search_service = DocumentSearchService(settings)
    graph_insight_service = GraphInsightService(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        init_db(settings)
        settings.cases_root.mkdir(parents=True, exist_ok=True)
        if settings.ingestion_enabled:
            job_runner.start()
        analysis_runner.start()
        yield
        analysis_runner.stop()
        if settings.ingestion_enabled:
            job_runner.stop()

    app = FastAPI(title="Rawabit API", version="0.1.0", lifespan=lifespan)

    @app.get("/api/settings")
    def get_runtime_settings() -> JSONResponse:
        current_overrides = load_settings_overrides()
        defaults: dict[str, object] = {
            "rag_llm_model": settings.rag_llm_model,
            "rag_vlm_model": settings.rag_vlm_model,
            "rag_embedding_model": settings.rag_embedding_model,
            "rag_embedding_dim_hint": settings.rag_embedding_dim_hint,
            "rag_llm_max_tokens": settings.rag_llm_max_tokens,
            "rag_llm_temperature": settings.rag_llm_temperature,
            "rag_llm_timeout_seconds": settings.rag_llm_timeout_seconds,
            "rag_llm_max_async": settings.rag_llm_max_async,
            "rag_embedding_max_async": settings.rag_embedding_max_async,
            "rag_cosine_threshold": settings.rag_cosine_threshold,
            "rag_default_top_k": settings.rag_default_top_k,
            "rag_default_chunk_top_k": settings.rag_default_chunk_top_k,
            "ingestion_worker_concurrency": settings.ingestion_worker_concurrency,
            "rag_lightrag_max_parallel_insert": settings.rag_lightrag_max_parallel_insert,
        }
        return _envelope({
            "overrides": current_overrides or {},
            "effective": defaults,
            "mutable_fields": sorted(USER_MUTABLE_SETTINGS),
        })

    @app.patch("/api/settings")
    async def update_runtime_settings(request: Request) -> JSONResponse:
        body = await request.json()
        payload: dict[str, object] = body if isinstance(body, dict) else {}
        cleaned: dict[str, object] = {}
        for key, value in (payload or {}).items():
            if key in USER_MUTABLE_SETTINGS:
                cleaned[key] = value
        save_settings_overrides(cleaned)
        settings.reload_overrides()
        return _envelope({"saved_overrides": cleaned}, message="Settings saved and applied immediately.")

    def _case_doc_counts(connection) -> dict[str, int]:
        rows = connection.execute(
            "SELECT case_id, COUNT(*) as doc_count FROM document GROUP BY case_id"
        ).fetchall()
        return {row["case_id"]: row["doc_count"] for row in rows}

    def _case_active_job_counts(connection) -> dict[str, int]:
        rows = connection.execute(
            "SELECT case_id, COUNT(*) as active_job_count "
            "FROM ingestion_job "
            "WHERE status IN ('queued', 'parsing', 'inserting', 'indexing') "
            "GROUP BY case_id"
        ).fetchall()
        return {row["case_id"]: row["active_job_count"] for row in rows}

    def _fetch_case_or_404(connection, case_id: str) -> dict[str, Any]:
        row = connection.execute(
            "SELECT id, name, description, status, case_slug, created_at, updated_at "
            'FROM "case" WHERE id = ?',
            (case_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Case not found")
        return dict(row)

    def _get_case_summary_or_none(case_id: str) -> dict[str, Any] | None:
        try:
            return case_summary_service.get_case_summary(case_id)
        except ValueError:
            return None
        except Exception:
            return None

    def _document_select_sql() -> str:
        return (
            "SELECT id, case_id, original_filename, stored_file_path, content_hash_sha256, mime_type, size_bytes, "
            "confidence_source_reliability, confidence_information_validity, confidence_code, "
            "tags, notes, ingest_model_name, ingestion_status, ingestion_error, created_at, updated_at, "
            "(SELECT j.processing_mode FROM ingestion_job j "
            "WHERE j.document_id = document.id AND j.case_id = document.case_id AND j.status = 'complete' "
            "ORDER BY COALESCE(j.finished_at, j.started_at, '') DESC LIMIT 1) AS latest_processing_mode "
            "FROM document"
        )

    def _list_matching_documents_by_hash(
        connection, case_id: str, content_hash_sha256: str
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            (
                f"{_document_select_sql()} "
                "WHERE case_id = ? AND content_hash_sha256 = ? "
                "ORDER BY created_at DESC"
            ),
            (case_id, content_hash_sha256),
        ).fetchall()
        return [dict(row) for row in rows]

    def _fetch_document_or_404(
        connection, case_id: str, document_id: str
    ) -> dict[str, Any]:
        row = connection.execute(
            f"{_document_select_sql()} WHERE id = ? AND case_id = ?",
            (document_id, case_id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        return dict(row)

    def _fetch_chat_or_404(connection, case_id: str, chat_id: str) -> dict[str, Any]:
        row = connection.execute(
            "SELECT id, case_id, title, created_at, updated_at "
            "FROM chat WHERE id = ? AND case_id = ?",
            (chat_id, case_id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Chat not found")
        return dict(row)

    def _fetch_analysis_or_404(
        connection, case_id: str, analysis_id: str
    ) -> dict[str, Any]:
        row = connection.execute(
            "SELECT id, case_id, analysis_type, prompt, title, status, "
            "rag_answer, summary_text, charts_json, highlight_json, subgraph_json, "
            "references_json, chunks_json, model_name, error, pending_repair_json, "
            "created_at, updated_at "
            "FROM analysis WHERE id = ? AND case_id = ?",
            (analysis_id, case_id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return dict(row)

    def _list_chat_messages(connection, chat_id: str) -> list[dict[str, Any]]:
        rows = connection.execute(
            "SELECT id, chat_id, role, content, created_at, rag_metadata_json "
            "FROM message WHERE chat_id = ? ORDER BY created_at ASC",
            (chat_id,),
        ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            payload["rag_metadata"] = _decode_json_text(
                payload.pop("rag_metadata_json", None)
            )
            output.append(payload)
        return output

    def _list_case_documents_for_graph(
        connection, case_id: str
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            "SELECT id, case_id, original_filename, stored_file_path, confidence_code, tags "
            "FROM document WHERE case_id = ?",
            (case_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def _fetch_job_or_404(connection, case_id: str, job_id: str) -> dict[str, Any]:
        row = connection.execute(
            "SELECT id, case_id, document_id, status FROM ingestion_job WHERE id = ? AND case_id = ?",
            (job_id, case_id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(row)

    def _normalize_chat_title(value: str | None) -> str:
        normalized = " ".join((value or "").strip().split())
        if not normalized:
            return "New chat"
        if len(normalized) > 200:
            normalized = normalized[:200].rstrip()
        return normalized or "New chat"

    def _chat_title_from_message(content: str) -> str:
        normalized = " ".join(content.strip().split())
        if not normalized:
            return "New chat"
        if len(normalized) > 72:
            normalized = f"{normalized[:69].rstrip()}..."
        return _normalize_chat_title(normalized)

    def _normalize_highlight_payload(
        payload: dict[str, Any] | None,
        *,
        references: list[dict[str, str]],
        chunks: list[dict[str, str]],
    ) -> dict[str, Any]:
        source = payload if isinstance(payload, dict) else {}
        entities = [
            str(entity).strip()
            for entity in source.get("highlight_entities", [])
            if isinstance(entity, str) and str(entity).strip()
        ]
        seen_entities: set[str] = set()
        highlight_entities: list[str] = []
        for entity in entities:
            if entity in seen_entities:
                continue
            seen_entities.add(entity)
            highlight_entities.append(entity)

        highlight_relationships: list[dict[str, str]] = []
        seen_relationships: set[tuple[str, str, str, str]] = set()
        raw_relationships = source.get("highlight_relationships", [])
        if isinstance(raw_relationships, list):
            for relationship in raw_relationships:
                if not isinstance(relationship, dict):
                    continue
                src_id = str(relationship.get("src_id") or "").strip()
                tgt_id = str(relationship.get("tgt_id") or "").strip()
                edge_id = str(relationship.get("edge_id") or "").strip()
                relation_type = str(relationship.get("relation_type") or "").strip()
                if not src_id or not tgt_id:
                    continue
                key = (src_id, tgt_id, edge_id, relation_type)
                if key in seen_relationships:
                    continue
                seen_relationships.add(key)
                row: dict[str, str] = {"src_id": src_id, "tgt_id": tgt_id}
                if edge_id:
                    row["edge_id"] = edge_id
                if relation_type:
                    row["relation_type"] = relation_type
                highlight_relationships.append(row)

        def _normalize_reference_rows(raw: Any) -> list[dict[str, str]]:
            output: list[dict[str, str]] = []
            seen_refs: set[tuple[str, str]] = set()
            if not isinstance(raw, list):
                return output
            for item in raw:
                if not isinstance(item, dict):
                    continue
                reference_id = str(item.get("reference_id") or "").strip()
                file_path = str(item.get("file_path") or "").strip()
                if not reference_id or not file_path:
                    continue
                key = (reference_id, file_path)
                if key in seen_refs:
                    continue
                seen_refs.add(key)
                output.append({"reference_id": reference_id, "file_path": file_path})
            return output

        def _normalize_chunk_rows(raw: Any) -> list[dict[str, str]]:
            output: list[dict[str, str]] = []
            seen_chunks: set[tuple[str, str, str]] = set()
            if not isinstance(raw, list):
                return output
            for item in raw:
                if not isinstance(item, dict):
                    continue
                reference_id = str(item.get("reference_id") or "").strip()
                file_path = str(item.get("file_path") or "").strip()
                if not reference_id or not file_path:
                    continue
                snippet = str(item.get("snippet") or "").strip()
                full_text = str(item.get("full_text") or "").strip()
                key = (reference_id, file_path, snippet)
                if key in seen_chunks:
                    continue
                seen_chunks.add(key)
                row = {"reference_id": reference_id, "file_path": file_path}
                if snippet:
                    row["snippet"] = snippet
                if full_text:
                    row["full_text"] = full_text
                output.append(row)
            return output

        highlight_references = _normalize_reference_rows(source.get("references"))
        normalized_references = highlight_references or references
        highlight_chunks = _normalize_chunk_rows(source.get("supporting_chunks"))
        normalized_chunks = highlight_chunks or chunks
        return {
            "highlight_entities": highlight_entities,
            "highlight_relationships": highlight_relationships,
            "supporting_chunks": normalized_chunks,
            "references": normalized_references,
        }

    def _normalize_retrieval_eval_payload(
        payload: dict[str, Any] | None,
        *,
        mode: str,
    ) -> dict[str, Any]:
        source = payload if isinstance(payload, dict) else {}

        def _normalize_str_list(
            raw: Any, *, upper: bool = False, lower: bool = False
        ) -> list[str]:
            output: list[str] = []
            seen: set[str] = set()
            if not isinstance(raw, list):
                return output
            for item in raw:
                value = str(item).strip()
                if not value:
                    continue
                if upper:
                    value = value.upper()
                elif lower:
                    value = value.lower()
                if value in seen:
                    continue
                seen.add(value)
                output.append(value)
            return output

        top_k_raw = source.get("top_k")
        if isinstance(top_k_raw, bool):
            top_k = 5
        elif isinstance(top_k_raw, (int, float)):
            top_k = int(top_k_raw)
        elif isinstance(top_k_raw, str):
            try:
                top_k = int(float(top_k_raw.strip()))
            except ValueError:
                top_k = 5
        else:
            top_k = 5
        if top_k <= 0:
            top_k = 5
        if top_k > 100:
            top_k = 100

        entity_ids = _normalize_str_list(source.get("retrieved_entity_ids_topk"))
        entity_types = _normalize_str_list(
            source.get("retrieved_entity_types_topk"),
            lower=True,
        )
        relation_ids = _normalize_str_list(source.get("retrieved_relation_ids_topk"))
        relation_types = _normalize_str_list(
            source.get("retrieved_relation_types_topk"),
            upper=True,
        )

        quality_flags_raw = source.get("quality_flags")
        entities_present = bool(entity_ids)
        relations_present = bool(relation_ids)
        non_empty_payload = bool(entity_ids or relation_ids)
        if isinstance(quality_flags_raw, dict):
            entities_present = bool(
                quality_flags_raw.get("entities_present", entities_present)
            )
            relations_present = bool(
                quality_flags_raw.get("relations_present", relations_present)
            )
            non_empty_payload = bool(
                quality_flags_raw.get("non_empty_payload", non_empty_payload)
            )

        return {
            "mode": str(source.get("mode") or mode).strip().lower() or mode,
            "top_k": top_k,
            "retrieved_entity_ids_topk": entity_ids[:top_k],
            "retrieved_entity_types_topk": entity_types[:top_k],
            "retrieved_relation_ids_topk": relation_ids[:top_k],
            "retrieved_relation_types_topk": relation_types[:top_k],
            "quality_flags": {
                "entities_present": entities_present,
                "relations_present": relations_present,
                "non_empty_payload": non_empty_payload,
            },
        }

    def _load_case_graph_store(case_id: str) -> GraphStore:
        with get_connection(settings) as connection:
            case = _fetch_case_or_404(connection, case_id)
            documents = _list_case_documents_for_graph(connection, case_id)
        case_root = settings.cases_root / case["case_slug"]
        return GraphStore(case_root=case_root, case_id=case_id, documents=documents)

    def _load_case_graph_export_context(
        case_id: str,
    ) -> tuple[dict[str, Any], GraphStore]:
        with get_connection(settings) as connection:
            case = _fetch_case_or_404(connection, case_id)
            documents = _list_case_documents_for_graph(connection, case_id)
        case_root = settings.cases_root / case["case_slug"]
        store = GraphStore(case_root=case_root, case_id=case_id, documents=documents)
        return case, store

    def _csv_download_response(content: str, *, filename: str) -> Response:
        return Response(
            content=content,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    def _normalize_confidence(
        source_reliability: str, information_validity: str
    ) -> tuple[str, str, str]:
        source = source_reliability.strip().upper()
        validity = information_validity.strip()
        if (
            source not in confidence_source_values
            or validity not in confidence_validity_values
        ):
            raise HTTPException(status_code=400, detail="Invalid confidence selection")
        return source, validity, f"{source}{validity}"

    def _safe_filename(value: str) -> str:
        base = Path(value).name or "document"
        cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", base).strip("._")
        return cleaned or "document"

    def _normalize_ingest_profile(value: str | None) -> str:
        profile = (value or settings.ingest_profile_default).strip().lower()
        if profile not in allowed_ingest_profiles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ingest_profile. Allowed: {', '.join(sorted(allowed_ingest_profiles))}",
            )
        return profile

    def _normalize_processing_mode(value: str | None) -> str:
        mode = (value or "multimodal").strip().lower()
        if mode not in allowed_processing_modes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid processing_mode. Allowed: {', '.join(sorted(allowed_processing_modes))}",
            )
        return mode

    def _normalize_content_hash_sha256(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        if not re.fullmatch(r"[0-9a-f]{64}", normalized):
            raise HTTPException(status_code=400, detail="Invalid content_hash_sha256")
        return normalized

    def _normalize_allow_duplicate(value: str | None) -> bool:
        if value is None:
            return False
        normalized = _to_bool(value)
        if normalized is None:
            raise HTTPException(status_code=400, detail="Invalid allow_duplicate flag")
        return normalized

    def _to_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return None

    def _to_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                try:
                    return int(float(stripped))
                except ValueError:
                    return None
        return None

    def _to_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                try:
                    return float(stripped)
                except ValueError:
                    return None
        return None

    def _normalize_advanced_overrides(raw_value: str | None) -> dict[str, Any] | None:
        if raw_value is None:
            return None
        stripped = raw_value.strip()
        if not stripped:
            return None
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid advanced_overrides JSON: {exc.msg}",
            ) from exc
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400,
                detail="advanced_overrides must be a JSON object.",
            )

        normalized: dict[str, Any] = {}
        parse_method = payload.get("parse_method")
        if isinstance(parse_method, str) and parse_method.strip():
            candidate = parse_method.strip().lower()
            if candidate not in allowed_parse_method_overrides:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "advanced_overrides.parse_method is invalid. Allowed: "
                        + ", ".join(sorted(allowed_parse_method_overrides))
                    ),
                )
            normalized["parse_method"] = candidate

        ocr_mode = payload.get("ocr_mode")
        if isinstance(ocr_mode, str) and ocr_mode.strip():
            candidate = ocr_mode.strip().lower()
            if candidate not in {"off", "auto", "force"}:
                raise HTTPException(
                    status_code=400,
                    detail="advanced_overrides.ocr_mode must be one of: off, auto, force.",
                )
            normalized["ocr_mode"] = candidate

        enable_vlm = _to_bool(payload.get("enable_vlm"))
        if enable_vlm is not None:
            normalized["enable_vlm"] = enable_vlm
        enable_vlm_visible_text = _to_bool(payload.get("enable_vlm_visible_text"))
        if enable_vlm_visible_text is not None:
            normalized["enable_vlm_visible_text"] = enable_vlm_visible_text
        enable_preinsert_summary = _to_bool(payload.get("enable_preinsert_summary"))
        if enable_preinsert_summary is not None:
            normalized["enable_preinsert_summary"] = enable_preinsert_summary

        vlm_parallelism = _to_int(payload.get("vlm_parallelism"))
        if vlm_parallelism is not None:
            normalized["vlm_parallelism"] = max(1, min(vlm_parallelism, 12))
        max_parallel_insert = _to_int(payload.get("max_parallel_insert"))
        if max_parallel_insert is not None:
            normalized["max_parallel_insert"] = max(1, min(max_parallel_insert, 16))
        summary_max_tokens = _to_int(payload.get("summary_max_tokens"))
        if summary_max_tokens is not None:
            normalized["summary_max_tokens"] = max(80, min(summary_max_tokens, 1200))

        queue_priority = payload.get("queue_priority")
        if isinstance(queue_priority, str) and queue_priority.strip():
            candidate = queue_priority.strip().lower()
            if candidate not in {"low", "normal", "high"}:
                raise HTTPException(
                    status_code=400,
                    detail="advanced_overrides.queue_priority must be one of: low, normal, high.",
                )
            normalized["queue_priority"] = candidate

        return normalized or None

    def _to_json_text(payload: dict[str, Any] | None) -> str | None:
        if not payload:
            return None
        return json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )

    def _to_json_blob(payload: Any) -> str:
        return json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )

    def _decode_json_text(payload: Any) -> Any:
        if not isinstance(payload, str) or not payload.strip():
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def _serialize_job_row(row: dict[str, Any]) -> dict[str, Any]:
        data = dict(row)
        data["advanced_overrides"] = _decode_json_text(
            data.pop("advanced_overrides_json", None)
        )
        data["preflight"] = _decode_json_text(data.pop("preflight_json", None))
        data["effective_config"] = _decode_json_text(
            data.pop("effective_config_json", None)
        )
        return data

    def _serialize_analysis_row(row: dict[str, Any]) -> dict[str, Any]:
        data = dict(row)
        data["charts"] = _decode_json_text(data.pop("charts_json", None)) or []
        data["highlight"] = _decode_json_text(data.pop("highlight_json", None)) or {
            "highlight_entities": [],
            "highlight_relationships": [],
            "references": [],
        }
        data["subgraph"] = _decode_json_text(data.pop("subgraph_json", None)) or {
            "nodes": [],
            "edges": [],
        }
        data["references"] = _decode_json_text(data.pop("references_json", None)) or []
        data["chunks"] = _decode_json_text(data.pop("chunks_json", None)) or []
        data.pop("pending_repair_json", None)
        return data

    def _resolve_case_file(case_root: Path, stored_path: str) -> Path:
        candidate = Path(stored_path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (case_root / candidate).resolve()
        case_root_resolved = case_root.resolve()
        if (
            resolved != case_root_resolved
            and case_root_resolved not in resolved.parents
        ):
            raise HTTPException(status_code=400, detail="Invalid document path")
        return resolved

    async def _write_upload_and_hash(
        upload: UploadFile,
        target_path: Path,
    ) -> tuple[int, str]:
        hasher = hashlib.sha256()
        size_bytes = 0
        with target_path.open("wb") as target:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                hasher.update(chunk)
                size_bytes += len(chunk)
        await upload.close()
        return size_bytes, hasher.hexdigest()

    def _envelope(data: Any, message: str = "ok") -> JSONResponse:
        return JSONResponse(
            {
                "status": "success",
                "message": message,
                "data": data,
            }
        )

    def _error_envelope(
        message: str,
        *,
        data: Any = None,
        status_code: int = 400,
    ) -> JSONResponse:
        return JSONResponse(
            {
                "status": "error",
                "message": message,
                "data": data,
            },
            status_code=status_code,
        )

    @app.get("/api/cases")
    def list_cases() -> JSONResponse:
        with get_connection(settings) as connection:
            rows = connection.execute(
                "SELECT id, name, description, status, case_slug, updated_at "
                'FROM "case" ORDER BY updated_at DESC'
            ).fetchall()
            doc_counts = _case_doc_counts(connection)
            active_job_counts = _case_active_job_counts(connection)
            summary_snippets = CaseSummaryService.list_summary_snippets(connection)

        data: list[dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            row_dict["doc_count"] = doc_counts.get(row["id"], 0)
            row_dict["active_job_count"] = active_job_counts.get(row["id"], 0)
            row_dict["summary_snippet"] = summary_snippets.get(row["id"])
            data.append(row_dict)
        return _envelope(data)

    @app.post("/api/cases")
    def create_case(payload: CaseCreate) -> JSONResponse:
        now = utc_now_iso()
        case_id = str(uuid.uuid4())
        base_slug = slugify(payload.name)

        with get_connection(settings) as connection:
            existing_slugs = [
                row["case_slug"]
                for row in connection.execute('SELECT case_slug FROM "case"').fetchall()
            ]
            case_slug = ensure_unique_slug(base_slug, existing_slugs)

        try:
            create_case_workspace(settings.cases_root, case_slug)
        except FileExistsError:
            raise HTTPException(status_code=409, detail="Case workspace already exists")

        try:
            with get_connection(settings) as connection:
                connection.execute(
                    'INSERT INTO "case" (id, name, description, status, case_slug, created_at, updated_at) '
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        case_id,
                        payload.name.strip(),
                        payload.description,
                        "active",
                        case_slug,
                        now,
                        now,
                    ),
                )
        except Exception:
            delete_case_workspace(settings.cases_root, case_slug)
            raise

        data = {
            "id": case_id,
            "name": payload.name.strip(),
            "description": payload.description,
            "status": "active",
            "case_slug": case_slug,
            "created_at": now,
            "updated_at": now,
        }
        return _envelope(data, message="case created")

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str) -> JSONResponse:
        with get_connection(settings) as connection:
            row = _fetch_case_or_404(connection, case_id)
        row["summary"] = _get_case_summary_or_none(case_id)
        return _envelope(row)

    @app.get("/api/cases/{case_id}/summary")
    def get_case_summary(case_id: str) -> JSONResponse:
        try:
            data = case_summary_service.get_case_summary(case_id)
        except ValueError as exc:
            detail = str(exc)
            if "Case not found" in detail:
                raise HTTPException(status_code=404, detail=detail) from exc
            raise HTTPException(status_code=400, detail=detail) from exc
        return _envelope(data)

    @app.post("/api/cases/{case_id}/summary/refresh")
    def refresh_case_summary(case_id: str) -> JSONResponse:
        try:
            data = case_summary_service.refresh_case_summary(case_id)
        except ValueError as exc:
            detail = str(exc)
            if "Case not found" in detail:
                raise HTTPException(status_code=404, detail=detail) from exc
            raise HTTPException(status_code=400, detail=detail) from exc
        return _envelope(data, message="case summary refreshed")

    @app.patch("/api/cases/{case_id}")
    def update_case(case_id: str, payload: CaseUpdate) -> JSONResponse:
        with get_connection(settings) as connection:
            existing = _fetch_case_or_404(connection, case_id)
            name = payload.name.strip() if payload.name else existing["name"]
            description = (
                payload.description
                if payload.description is not None
                else existing["description"]
            )
            status = payload.status if payload.status else existing["status"]
            now = utc_now_iso()
            connection.execute(
                'UPDATE "case" SET name = ?, description = ?, status = ?, updated_at = ? WHERE id = ?',
                (name, description, status, now, case_id),
            )
        data = {
            "id": case_id,
            "name": name,
            "description": description,
            "status": status,
            "case_slug": existing["case_slug"],
            "updated_at": now,
        }
        return _envelope(data, message="case updated")

    @app.delete("/api/cases/{case_id}")
    def delete_case(case_id: str) -> JSONResponse:
        with get_connection(settings) as connection:
            existing = _fetch_case_or_404(connection, case_id)

        try:
            delete_case_workspace(settings.cases_root, existing["case_slug"])
        except (PermissionError, OSError):
            raise HTTPException(
                status_code=409,
                detail="Case workspace is in use. Stop ingestion and retry.",
            )

        with get_connection(settings) as connection:
            connection.execute(
                "DELETE FROM ingestion_job WHERE case_id = ?", (case_id,)
            )
            connection.execute("DELETE FROM document WHERE case_id = ?", (case_id,))
            connection.execute(
                "DELETE FROM chat WHERE case_id = ?",
                (case_id,),
            )
            connection.execute('DELETE FROM "case" WHERE id = ?', (case_id,))
        return _envelope({"deleted": True}, message="case deleted")

    @app.get("/api/cases/{case_id}/chats")
    def list_chats(case_id: str) -> JSONResponse:
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            rows = connection.execute(
                "SELECT id, case_id, title, created_at, updated_at "
                "FROM chat WHERE case_id = ? ORDER BY updated_at DESC, created_at DESC",
                (case_id,),
            ).fetchall()
        return _envelope([dict(row) for row in rows])

    @app.post("/api/cases/{case_id}/chats")
    def create_chat(case_id: str, payload: ChatCreate | None = None) -> JSONResponse:
        now = utc_now_iso()
        chat_id = str(uuid.uuid4())
        title = _normalize_chat_title(payload.title if payload else None)
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            connection.execute(
                "INSERT INTO chat (id, case_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (chat_id, case_id, title, now, now),
            )
        return _envelope(
            {
                "id": chat_id,
                "case_id": case_id,
                "title": title,
                "created_at": now,
                "updated_at": now,
            },
            message="chat created",
        )

    @app.get("/api/cases/{case_id}/chats/{chat_id}")
    def get_chat(case_id: str, chat_id: str) -> JSONResponse:
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            chat = _fetch_chat_or_404(connection, case_id, chat_id)
            messages = _list_chat_messages(connection, chat_id)
        chat["messages"] = messages
        return _envelope(chat)

    @app.delete("/api/cases/{case_id}/chats/{chat_id}")
    def delete_chat(case_id: str, chat_id: str) -> JSONResponse:
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            _fetch_chat_or_404(connection, case_id, chat_id)
            connection.execute(
                "DELETE FROM chat WHERE id = ? AND case_id = ?",
                (chat_id, case_id),
            )
        return _envelope({"deleted": True}, message="chat deleted")

    @app.get("/api/cases/{case_id}/analyses")
    def list_analyses(case_id: str) -> JSONResponse:
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            rows = connection.execute(
                "SELECT id, case_id, analysis_type, prompt, title, status, "
                "rag_answer, summary_text, charts_json, highlight_json, subgraph_json, "
                "references_json, chunks_json, model_name, error, pending_repair_json, "
                "created_at, updated_at "
                "FROM analysis WHERE case_id = ? "
                "ORDER BY updated_at DESC, created_at DESC",
                (case_id,),
            ).fetchall()
        return _envelope([_serialize_analysis_row(dict(row)) for row in rows])

    @app.post("/api/cases/{case_id}/analyses")
    def create_analysis(case_id: str, payload: AnalysisCreate) -> JSONResponse:
        prompt = payload.prompt.strip()
        analysis_type = payload.analysis_type.strip().lower()
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            case_documents = _list_case_documents_for_graph(connection, case_id)
            if len(case_documents) == 0:
                return _error_envelope(
                    "Upload evidence before creating analyses.",
                    status_code=409,
                )
        if analysis_type not in {"link", "event", "flow"}:
            return _error_envelope(
                "Analysis type must be one of: link, event, flow.",
                status_code=400,
            )

        analysis_id = str(uuid.uuid4())
        now = utc_now_iso()
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            connection.execute(
                "INSERT INTO analysis ("
                "id, case_id, analysis_type, prompt, title, status, rag_answer, "
                "summary_text, charts_json, highlight_json, subgraph_json, "
                "references_json, chunks_json, model_name, error, pending_repair_json, "
                "created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    analysis_id,
                    case_id,
                    analysis_type,
                    prompt,
                    analysis_service.build_title(prompt),
                    "queued",
                    None,
                    None,
                    _to_json_blob([]),
                    _to_json_blob({}),
                    _to_json_blob({}),
                    _to_json_blob([]),
                    _to_json_blob([]),
                    settings.rag_llm_model,
                    None,
                    None,
                    now,
                    now,
                ),
            )
            row = _fetch_analysis_or_404(connection, case_id, analysis_id)
        return _envelope(_serialize_analysis_row(row), message="analysis queued")

    @app.get("/api/cases/{case_id}/analyses/{analysis_id}")
    def get_analysis(case_id: str, analysis_id: str) -> JSONResponse:
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            row = _fetch_analysis_or_404(connection, case_id, analysis_id)
        return _envelope(_serialize_analysis_row(row))

    @app.delete("/api/cases/{case_id}/analyses/{analysis_id}")
    def delete_analysis(case_id: str, analysis_id: str) -> JSONResponse:
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            _fetch_analysis_or_404(connection, case_id, analysis_id)
            connection.execute(
                "DELETE FROM analysis WHERE id = ? AND case_id = ?",
                (analysis_id, case_id),
            )
        return _envelope({"deleted": True}, message="analysis deleted")

    @app.post("/api/cases/{case_id}/analyses/{analysis_id}/repair")
    def repair_analysis_chart(
        case_id: str,
        analysis_id: str,
        payload: AnalysisRepairRequest,
    ) -> JSONResponse:
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            row = _fetch_analysis_or_404(connection, case_id, analysis_id)
        if row.get("status") != "complete":
            return _error_envelope(
                "Only a completed analysis can queue a chart repair.",
                status_code=409,
            )
        charts = _decode_json_text(row.get("charts_json")) or []
        chart_index = next(
            (
                index
                for index, chart in enumerate(charts)
                if isinstance(chart, dict) and chart.get("id") == payload.chart_id
            ),
            None,
        )
        if chart_index is None:
            return _error_envelope("Analysis chart not found.", status_code=404)
        chart = dict(charts[chart_index])
        attempts = int(chart.get("repair_attempts") or 0)
        if attempts >= MAX_ANALYSIS_REPAIR_ATTEMPTS:
            return _error_envelope(
                "Mermaid repair limit reached for this chart.",
                status_code=409,
            )
        current_code = (payload.mermaid_code or chart.get("mermaid_code") or "").strip()
        if not current_code:
            return _error_envelope("No Mermaid code is available to repair.")
        pending_repair = {
            "chart_id": payload.chart_id,
            "error": payload.error.strip(),
            "mermaid_code": current_code,
        }
        now = utc_now_iso()
        with get_connection(settings) as connection:
            _fetch_analysis_or_404(connection, case_id, analysis_id)
            connection.execute(
                "UPDATE analysis SET status = 'repair_queued', error = NULL, "
                "pending_repair_json = ?, updated_at = ? "
                "WHERE id = ? AND case_id = ? AND status = 'complete'",
                (_to_json_blob(pending_repair), now, analysis_id, case_id),
            )
            updated = _fetch_analysis_or_404(connection, case_id, analysis_id)
        return _envelope(_serialize_analysis_row(updated), message="analysis chart repair queued")

    @app.post("/api/cases/{case_id}/analyses/{analysis_id}/retry")
    def retry_analysis(case_id: str, analysis_id: str) -> JSONResponse:
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            row = _fetch_analysis_or_404(connection, case_id, analysis_id)
            if row.get("status") != "failed":
                return _error_envelope(
                    "Only a failed analysis can be retried.",
                    status_code=409,
                )
            next_status = (
                "repair_queued"
                if _decode_json_text(row.get("pending_repair_json"))
                else "queued"
            )
            connection.execute(
                "UPDATE analysis SET status = ?, error = NULL, updated_at = ? "
                "WHERE id = ? AND case_id = ? AND status = 'failed'",
                (next_status, utc_now_iso(), analysis_id, case_id),
            )
            updated = _fetch_analysis_or_404(connection, case_id, analysis_id)
        return _envelope(_serialize_analysis_row(updated), message="analysis requeued")

    @app.post("/api/cases/{case_id}/chats/{chat_id}/messages")
    async def create_chat_message(
        case_id: str,
        chat_id: str,
        payload: ChatMessageCreate,
    ) -> JSONResponse:
        user_content = payload.content.strip()
        if not user_content:
            raise HTTPException(
                status_code=400, detail="Message content cannot be empty"
            )

        with get_connection(settings) as connection:
            case = _fetch_case_or_404(connection, case_id)
            chat = _fetch_chat_or_404(connection, case_id, chat_id)
            existing_messages = _list_chat_messages(connection, chat_id)
            case_documents = _list_case_documents_for_graph(connection, case_id)
            if len(case_documents) == 0:
                return _error_envelope(
                    "Upload evidence before sending chat messages.",
                    status_code=409,
                )
            if (
                existing_messages
                and str(existing_messages[-1].get("role") or "").strip() == "user"
            ):
                return _error_envelope(
                    "Wait for the current question to be answered before sending another message.",
                    status_code=409,
                )

            user_message_created_at = utc_now_iso()
            connection.execute(
                "INSERT INTO message (id, chat_id, role, content, created_at, rag_metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    chat_id,
                    "user",
                    user_content,
                    user_message_created_at,
                    None,
                ),
            )
            chat_title = str(chat.get("title") or "")
            has_existing_messages = len(existing_messages) > 0
            if (
                not has_existing_messages
                and _normalize_chat_title(chat_title) == "New chat"
            ):
                connection.execute(
                    "UPDATE chat SET title = ?, updated_at = ? WHERE id = ? AND case_id = ?",
                    (
                        _chat_title_from_message(user_content),
                        user_message_created_at,
                        chat_id,
                        case_id,
                    ),
                )
            else:
                connection.execute(
                    "UPDATE chat SET updated_at = ? WHERE id = ? AND case_id = ?",
                    (user_message_created_at, chat_id, case_id),
                )

        conversation_history = [
            {
                "role": str(item.get("role") or "").strip(),
                "content": str(item.get("content") or "").strip(),
            }
            for item in existing_messages
            if str(item.get("role") or "").strip() in {"user", "assistant"}
            and str(item.get("content") or "").strip()
        ]
        if len(conversation_history) > 24:
            conversation_history = conversation_history[-24:]

        case_root = settings.cases_root / str(case["case_slug"])
        try:
            result = await chat_service.query_case_message(
                case_root=case_root,
                user_content=user_content,
                mode=payload.mode,
                conversation_history=conversation_history,
                case_documents=case_documents,
                options=payload.options,
            )
        except Exception as exc:
            result = chat_service.build_failure_result(
                mode=payload.mode,
                error_message=str(exc),
            )

        normalized_highlight = _normalize_highlight_payload(
            result.highlight,
            references=result.references,
            chunks=result.chunks,
        )
        resolved_mode = (
            str(result.metadata.get("mode") or "").strip()
            if isinstance(result.metadata, dict)
            else ""
        ) or payload.mode
        normalized_retrieval_eval = _normalize_retrieval_eval_payload(
            result.retrieval_eval,
            mode=resolved_mode,
        )
        response_references = [
            dict(item)
            for item in normalized_highlight.get("references", [])
            if isinstance(item, dict)
        ]
        response_chunks = [
            dict(item)
            for item in normalized_highlight.get("supporting_chunks", [])
            if isinstance(item, dict)
        ]
        assistant_metadata: dict[str, Any] = (
            dict(result.metadata) if isinstance(result.metadata, dict) else {}
        )
        assistant_metadata["mode"] = resolved_mode
        assistant_metadata["requested_mode"] = payload.mode
        assistant_metadata["references"] = response_references
        assistant_metadata["chunks"] = response_chunks
        assistant_metadata["highlight"] = normalized_highlight
        assistant_metadata["retrieval_eval"] = normalized_retrieval_eval
        if "model_name" not in assistant_metadata:
            assistant_metadata["model_name"] = settings.rag_llm_model

        assistant_message_id = str(uuid.uuid4())
        assistant_created_at = utc_now_iso()
        assistant_content = str(result.assistant_content or "").strip()

        with get_connection(settings) as connection:
            _fetch_chat_or_404(connection, case_id, chat_id)
            connection.execute(
                "INSERT INTO message (id, chat_id, role, content, created_at, rag_metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    assistant_message_id,
                    chat_id,
                    "assistant",
                    assistant_content,
                    assistant_created_at,
                    _to_json_text(assistant_metadata),
                ),
            )
            connection.execute(
                "UPDATE chat SET updated_at = ? WHERE id = ? AND case_id = ?",
                (assistant_created_at, chat_id, case_id),
            )

        return _envelope(
            {
                "message": {
                    "id": assistant_message_id,
                    "role": "assistant",
                    "content": assistant_content,
                    "created_at": assistant_created_at,
                },
                "highlight": normalized_highlight,
                "retrieval_eval": normalized_retrieval_eval,
                "references": response_references,
                "chunks": response_chunks,
                "model_name": assistant_metadata.get("model_name"),
            }
        )

    @app.get("/api/cases/{case_id}/documents")
    def list_documents(case_id: str) -> JSONResponse:
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            rows = connection.execute(
                f"{_document_select_sql()} WHERE case_id = ? ORDER BY created_at DESC",
                (case_id,),
            ).fetchall()
        data = [dict(row) for row in rows]
        return _envelope(data)

    @app.post("/api/cases/{case_id}/documents/duplicates/check")
    def check_document_duplicates(
        case_id: str,
        payload: DocumentDuplicateCheckRequest,
    ) -> JSONResponse:
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            data = [
                {
                    "client_id": item.client_id,
                    "original_filename": item.original_filename,
                    "size_bytes": item.size_bytes,
                    "content_hash_sha256": _normalize_content_hash_sha256(
                        item.content_hash_sha256
                    ),
                    "matches": _list_matching_documents_by_hash(
                        connection,
                        case_id,
                        _normalize_content_hash_sha256(item.content_hash_sha256) or "",
                    ),
                }
                for item in payload.files
            ]
        return _envelope(data)

    @app.post("/api/cases/{case_id}/documents")
    async def upload_document(
        case_id: str,
        file: UploadFile = File(...),
        confidence_source_reliability: str = Form(...),
        confidence_information_validity: str = Form(...),
        ingest_profile: str | None = Form(None),
        processing_mode: str | None = Form(None),
        advanced_overrides: str | None = Form(None),
        content_hash_sha256: str | None = Form(None),
        allow_duplicate: str | None = Form(None),
        tags: str | None = Form(None),
        notes: str | None = Form(None),
    ) -> JSONResponse:
        source, validity, code = _normalize_confidence(
            confidence_source_reliability, confidence_information_validity
        )
        normalized_profile = _normalize_ingest_profile(ingest_profile)
        normalized_processing_mode = _normalize_processing_mode(processing_mode)
        normalized_overrides = _normalize_advanced_overrides(advanced_overrides)
        normalized_content_hash = _normalize_content_hash_sha256(content_hash_sha256)
        normalized_allow_duplicate = _normalize_allow_duplicate(allow_duplicate)
        overrides_json = _to_json_text(normalized_overrides)
        queue_priority = (
            str((normalized_overrides or {}).get("queue_priority") or "normal")
            .strip()
            .lower()
        )
        if queue_priority not in {"low", "normal", "high"}:
            queue_priority = "normal"
        now = utc_now_iso()
        document_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())

        with get_connection(settings) as connection:
            case = _fetch_case_or_404(connection, case_id)

        case_root = settings.cases_root / case["case_slug"]
        raw_dir = case_root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        original_filename = file.filename or "document"
        safe_name = _safe_filename(original_filename)
        stored_name = f"{document_id}_{safe_name}"
        stored_relative = Path("raw") / stored_name
        stored_path = case_root / stored_relative

        size_bytes, computed_content_hash = await _write_upload_and_hash(
            file, stored_path
        )
        if normalized_content_hash and normalized_content_hash != computed_content_hash:
            try:
                stored_path.unlink(missing_ok=True)
            except OSError:
                pass
            return _error_envelope(
                "content_hash_sha256 did not match uploaded file contents",
                data={
                    "provided_content_hash_sha256": normalized_content_hash,
                    "computed_content_hash_sha256": computed_content_hash,
                },
                status_code=400,
            )
        effective_content_hash = normalized_content_hash or computed_content_hash

        mime_type = file.content_type or "application/octet-stream"
        preflight = compute_ingestion_preflight(
            source_path=stored_path,
            mime_type=mime_type,
            ingest_profile=normalized_profile,
        )
        preflight_json = _to_json_text(preflight)

        try:
            with get_connection(settings) as connection:
                existing_matches = _list_matching_documents_by_hash(
                    connection,
                    case_id,
                    effective_content_hash,
                )
                if existing_matches and not normalized_allow_duplicate:
                    try:
                        stored_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    return _error_envelope(
                        "Duplicate evidence already exists in this case.",
                        data={
                            "content_hash_sha256": effective_content_hash,
                            "matches": existing_matches,
                        },
                        status_code=409,
                    )
                connection.execute(
                    "INSERT INTO document (id, case_id, original_filename, stored_file_path, content_hash_sha256, "
                    "mime_type, size_bytes, confidence_source_reliability, confidence_information_validity, "
                    "confidence_code, tags, notes, ingest_model_name, ingestion_status, ingestion_error, "
                    "created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        document_id,
                        case_id,
                        original_filename,
                        stored_relative.as_posix(),
                        effective_content_hash,
                        mime_type,
                        size_bytes,
                        source,
                        validity,
                        code,
                        tags,
                        notes,
                        None,
                        "queued",
                        None,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    "INSERT INTO ingestion_job (id, case_id, document_id, ingest_profile, processing_mode, advanced_overrides_json, preflight_json, "
                    "effective_config_json, complexity_class, eta_seconds, queue_priority, route_type, status, progress, started_at, "
                    "finished_at, parse_duration_s, insert_duration_s, finalize_duration_s, current_stage, error) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        job_id,
                        case_id,
                        document_id,
                        normalized_profile,
                        normalized_processing_mode,
                        overrides_json,
                        preflight_json,
                        None,
                        preflight["complexity_class"],
                        preflight["eta_seconds"],
                        queue_priority,
                        None,
                        "queued",
                        None,
                        now,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ),
                )
        except Exception:
            try:
                stored_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        append_job_log(
            settings,
            case_id=case_id,
            job_id=job_id,
            message=(
                f"Job queued for document '{original_filename}' with profile "
                f"'{normalized_profile}', mode '{normalized_processing_mode}', and confidence '{code}' "
                f"(complexity={preflight['complexity_class']}, eta={preflight['eta_seconds']}s, "
                f"priority={queue_priority})."
            ),
        )
        document_search_service.index_raw_document(
            case_id=case_id,
            document_id=document_id,
            original_filename=original_filename,
            stored_file_path=stored_relative.as_posix(),
            confidence_code=code,
            mime_type=mime_type,
            file_path=stored_path,
        )

        return _envelope(
            {
                "document_id": document_id,
                "job_id": job_id,
                "content_hash_sha256": effective_content_hash,
                "ingest_profile": normalized_profile,
                "processing_mode": normalized_processing_mode,
                "preflight": preflight,
                "advanced_overrides": normalized_overrides,
            },
            message="document uploaded",
        )

    @app.get("/api/cases/{case_id}/documents/search")
    def search_documents(
        case_id: str,
        q: str = Query(..., min_length=1),
        source: str = Query("all"),
        limit: int = Query(25, ge=1, le=100),
    ) -> JSONResponse:
        normalized_source = source.strip().lower()
        if normalized_source not in SEARCH_SOURCE_VALUES:
            raise HTTPException(
                status_code=400, detail="Unsupported document search source"
            )
        with get_connection(settings) as connection:
            case = _fetch_case_or_404(connection, case_id)
            rows = connection.execute(
                f"{_document_select_sql()} WHERE case_id = ?",
                (case_id,),
            ).fetchall()
            documents = [dict(row) for row in rows]
        document_search_service.ensure_case_index(
            case_id=case_id,
            case_root=settings.cases_root / case["case_slug"],
            documents=documents,
        )
        data = document_search_service.search(
            case_id=case_id,
            query=q,
            source=normalized_source,
            limit=limit,
        )
        return _envelope(data)

    @app.get("/api/cases/{case_id}/documents/{document_id}/search-preview")
    def get_document_search_preview(
        case_id: str,
        document_id: str,
        q: str = Query(..., min_length=1),
        source_kind: str = Query(...),
        segment_key: str = Query(..., min_length=1),
    ) -> JSONResponse:
        normalized_source = source_kind.strip().lower()
        if normalized_source not in {"raw", "processed"}:
            raise HTTPException(
                status_code=400, detail="Unsupported document search source"
            )
        with get_connection(settings) as connection:
            case = _fetch_case_or_404(connection, case_id)
            document = _fetch_document_or_404(connection, case_id, document_id)
        data = document_search_service.preview(
            case_id=case_id,
            case_root=settings.cases_root / case["case_slug"],
            document=document,
            source_kind=normalized_source,
            segment_key=segment_key,
            query=q,
        )
        if not data:
            raise HTTPException(status_code=404, detail="Search preview unavailable")
        return _envelope(data)

    @app.get("/api/cases/{case_id}/documents/{document_id}/reference-preview")
    def get_document_reference_preview(
        case_id: str,
        document_id: str,
        reference_id: str = Query(..., min_length=1),
        q: str = Query(""),
        snippet: str = Query(""),
    ) -> JSONResponse:
        with get_connection(settings) as connection:
            case = _fetch_case_or_404(connection, case_id)
            document = _fetch_document_or_404(connection, case_id, document_id)
        data = document_search_service.reference_preview(
            case_id=case_id,
            case_root=settings.cases_root / case["case_slug"],
            document=document,
            reference_id=reference_id,
            query=q,
            snippet=snippet,
        )
        if not data:
            raise HTTPException(status_code=404, detail="Reference preview unavailable")
        return _envelope(data)

    @app.get("/api/cases/{case_id}/documents/{document_id}")
    def get_document(case_id: str, document_id: str) -> JSONResponse:
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            doc = _fetch_document_or_404(connection, case_id, document_id)
        return _envelope(doc)

    @app.get("/api/cases/{case_id}/jobs")
    def list_jobs(case_id: str) -> JSONResponse:
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            rows = connection.execute(
                "SELECT id, case_id, document_id, ingest_profile, processing_mode, advanced_overrides_json, preflight_json, "
                "effective_config_json, complexity_class, eta_seconds, queue_priority, route_type, status, progress, "
                "started_at, finished_at, parse_duration_s, insert_duration_s, finalize_duration_s, current_stage, error "
                "FROM ingestion_job WHERE case_id = ? ORDER BY started_at DESC",
                (case_id,),
            ).fetchall()
        data = [_serialize_job_row(dict(row)) for row in rows]
        return _envelope(data)

    @app.get("/api/cases/{case_id}/jobs/{job_id}")
    def get_job(case_id: str, job_id: str) -> JSONResponse:
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            row = connection.execute(
                "SELECT id, case_id, document_id, ingest_profile, processing_mode, advanced_overrides_json, preflight_json, "
                "effective_config_json, complexity_class, eta_seconds, queue_priority, route_type, status, progress, "
                "started_at, finished_at, parse_duration_s, insert_duration_s, finalize_duration_s, current_stage, error "
                "FROM ingestion_job WHERE id = ? AND case_id = ?",
                (job_id, case_id),
            ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        return _envelope(_serialize_job_row(dict(row)))

    @app.get("/api/cases/{case_id}/jobs/{job_id}/logs")
    def get_job_logs(
        case_id: str,
        job_id: str,
        after_id: int = Query(0, ge=0),
        limit: int = Query(500, ge=1, le=2000),
    ) -> JSONResponse:
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            _fetch_job_or_404(connection, case_id, job_id)
        data = list_job_logs(
            settings=settings,
            case_id=case_id,
            job_id=job_id,
            after_id=after_id,
            limit=limit,
        )
        return _envelope(data)

    @app.get("/api/cases/{case_id}/graph")
    def get_graph(
        case_id: str,
        limit: int | None = Query(None, ge=1),
        entity_types: str | None = Query(None),
        keyword_filters: str | None = Query(None),
        focus_entity: str | None = Query(None),
        relation_types: str | None = Query(None),
        min_confidence: float | None = Query(None, ge=0.0, le=1.0),
        date_from: str | None = Query(None),
        date_to: str | None = Query(None),
        max_hops: int | None = Query(None, ge=0, le=3),
    ) -> JSONResponse:
        parsed_date_from: date | None = None
        parsed_date_to: date | None = None
        for value, target in ((date_from, "date_from"), (date_to, "date_to")):
            if not value:
                continue
            try:
                parsed = date.fromisoformat(value)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid {target}; expected ISO date (YYYY-MM-DD)",
                )
            if target == "date_from":
                parsed_date_from = parsed
            else:
                parsed_date_to = parsed

        store = _load_case_graph_store(case_id)
        data = store.graph_view(
            limit=limit,
            entity_types=parse_csv_query(entity_types),
            keyword_filters=parse_csv_query(keyword_filters),
            focus_entity=focus_entity.strip() if focus_entity else None,
            relation_types=parse_csv_query(relation_types),
            min_confidence=min_confidence,
            date_from=parsed_date_from,
            date_to=parsed_date_to,
            max_hops=max_hops,
        )
        return _envelope(data)

    @app.get("/api/cases/{case_id}/graph/export/entities.csv")
    def export_graph_entities_csv(case_id: str) -> Response:
        case, store = _load_case_graph_export_context(case_id)
        return _csv_download_response(
            store.export_entities_csv(),
            filename=f"{case['case_slug']}-entities.csv",
        )

    @app.get("/api/cases/{case_id}/graph/export/relations.csv")
    def export_graph_relations_csv(case_id: str) -> Response:
        case, store = _load_case_graph_export_context(case_id)
        return _csv_download_response(
            store.export_relations_csv(),
            filename=f"{case['case_slug']}-relations.csv",
        )

    # Actor-centric endpoints.

    @app.get("/api/cases/{case_id}/stats")
    def get_case_stats(case_id: str) -> JSONResponse:
        store = _load_case_graph_store(case_id)
        data = store.actor_stats()
        return _envelope(data)

    @app.get("/api/cases/{case_id}/tag-clusters")
    def get_tag_clusters(case_id: str) -> JSONResponse:
        # Primary buckets are relation types; also include document tags if present in docs table.
        store = _load_case_graph_store(case_id)
        relation_types = store.actor_stats().get("categories", [])
        clusters = [
            {
                "id": idx,
                "name": item["category"],
                "exemplars": [],
                "tagCount": item["count"],
            }
            for idx, item in enumerate(relation_types)
        ]
        return _envelope(clusters)

    @app.get("/api/cases/{case_id}/relationships")
    def list_relationships(
        case_id: str,
        limit: int = Query(500, ge=1, le=20000),
        categories: str | None = Query(None),
        clusters: str | None = Query(None),
        yearMin: int | None = Query(None),
        yearMax: int | None = Query(None),
        includeUndated: bool = Query(True),
        keywords: str | None = Query(None),
        maxHops: int | None = Query(None, ge=0, le=5),
    ) -> JSONResponse:
        store = _load_case_graph_store(case_id)
        data = store.relationships(
            limit=limit,
            categories=parse_csv_query(categories) or set(),
            clusters=parse_csv_query(clusters) or set(),
            year_min=yearMin,
            year_max=yearMax,
            include_undated=includeUndated,
            keywords=keywords,
            max_hops=maxHops,
            actor_name=None,
        )
        return _envelope(data)

    @app.get("/api/cases/{case_id}/actor/{name}/relationships")
    def list_actor_relationships(
        case_id: str,
        name: str,
        categories: str | None = Query(None),
        clusters: str | None = Query(None),
        yearMin: int | None = Query(None),
        yearMax: int | None = Query(None),
        includeUndated: bool = Query(True),
        keywords: str | None = Query(None),
        maxHops: int | None = Query(None, ge=0, le=5),
    ) -> JSONResponse:
        store = _load_case_graph_store(case_id)
        data = store.relationships(
            limit=5000,
            categories=parse_csv_query(categories) or set(),
            clusters=parse_csv_query(clusters) or set(),
            year_min=yearMin,
            year_max=yearMax,
            include_undated=includeUndated,
            keywords=keywords,
            max_hops=maxHops,
            actor_name=name,
        )
        return _envelope(data)

    @app.get("/api/cases/{case_id}/actor-counts")
    def get_actor_counts(
        case_id: str, limit: int = Query(300, ge=1, le=2000)
    ) -> JSONResponse:
        store = _load_case_graph_store(case_id)
        data = store.actor_counts(limit=limit)
        return _envelope(data)

    @app.get("/api/cases/{case_id}/actor/{name}/count")
    def get_actor_count(case_id: str, name: str) -> JSONResponse:
        store = _load_case_graph_store(case_id)
        counts = store.actor_counts(limit=5000)
        count = counts.get(name, 0)
        return _envelope({"count": count})

    @app.get("/api/cases/{case_id}/search")
    def search_actors(case_id: str, q: str = Query(..., min_length=1)) -> JSONResponse:
        store = _load_case_graph_store(case_id)
        data = store.search_actors(q)
        return _envelope(data)

    @app.get("/api/cases/{case_id}/graph/search")
    def search_graph(
        case_id: str,
        q: str = Query(..., min_length=1),
    ) -> JSONResponse:
        store = _load_case_graph_store(case_id)
        data = store.search(q)
        return _envelope(data)

    @app.get("/api/cases/{case_id}/graph/entity/{entity_id}")
    def get_graph_entity(case_id: str, entity_id: str) -> JSONResponse:
        store = _load_case_graph_store(case_id)
        data = store.entity_details(entity_id)
        if not data:
            raise HTTPException(status_code=404, detail="Entity not found")
        data = graph_insight_service.enrich_entity_detail(case_id, data)
        return _envelope(data)

    @app.get("/api/cases/{case_id}/graph/relationship")
    def get_graph_relationship(
        case_id: str,
        src_id: str = Query(..., min_length=1),
        tgt_id: str = Query(..., min_length=1),
    ) -> JSONResponse:
        store = _load_case_graph_store(case_id)
        data = store.relationship_details(src_id=src_id, tgt_id=tgt_id)
        if not data:
            raise HTTPException(status_code=404, detail="Relationship not found")
        data = graph_insight_service.enrich_relationship_detail(case_id, data)
        return _envelope(data)

    @app.post("/api/cases/{case_id}/graph/entity-resolution/merge")
    def merge_entity_resolution(
        case_id: str,
        payload: EntityResolutionMergeRequest,
    ) -> JSONResponse:
        try:
            data = entity_resolution_service.merge_entities(
                case_id=case_id,
                source_entities=payload.source_entities,
                target_entity=payload.target_entity,
                reason=payload.reason,
            )
            return _envelope(data, message="entity merge completed")
        except ValueError as exc:
            detail = str(exc)
            if "Case not found" in detail:
                raise HTTPException(status_code=404, detail=detail) from exc
            raise HTTPException(status_code=400, detail=detail) from exc

    @app.post("/api/cases/{case_id}/graph/entity-resolution/run")
    def run_entity_resolution(
        case_id: str,
    ) -> JSONResponse:
        try:
            data = entity_resolution_service.run_generic_resolution(
                case_id=case_id,
            )
            return _envelope(data, message="entity resolution run completed")
        except ValueError as exc:
            detail = str(exc)
            if "Case not found" in detail:
                raise HTTPException(status_code=404, detail=detail) from exc
            raise HTTPException(status_code=400, detail=detail) from exc

    @app.get("/api/cases/{case_id}/documents/{document_id}/download")
    def download_document(case_id: str, document_id: str) -> FileResponse:
        with get_connection(settings) as connection:
            case = _fetch_case_or_404(connection, case_id)
            doc = _fetch_document_or_404(connection, case_id, document_id)

        case_root = settings.cases_root / case["case_slug"]
        stored_path = _resolve_case_file(case_root, doc["stored_file_path"])
        if not stored_path.exists():
            raise HTTPException(status_code=404, detail="Document file not found")

        filename = doc["original_filename"] or stored_path.name
        media_type = doc["mime_type"] or "application/octet-stream"
        return FileResponse(stored_path, filename=filename, media_type=media_type)

    @app.post("/api/cases/{case_id}/documents/{document_id}/reingest")
    def reingest_document(
        case_id: str,
        document_id: str,
        ingest_profile: str | None = Query(None),
        processing_mode: str | None = Query(None),
        advanced_overrides: str | None = Query(None),
        notes: str | None = Query(None),
        confidence_source_reliability: str | None = Query(None),
        confidence_information_validity: str | None = Query(None),
    ) -> JSONResponse:
        normalized_profile = _normalize_ingest_profile(ingest_profile)
        normalized_processing_mode = _normalize_processing_mode(processing_mode)
        normalized_overrides = _normalize_advanced_overrides(advanced_overrides)
        overrides_json = _to_json_text(normalized_overrides)
        queue_priority = (
            str((normalized_overrides or {}).get("queue_priority") or "normal")
            .strip()
            .lower()
        )
        if queue_priority not in {"low", "normal", "high"}:
            queue_priority = "normal"
        now = utc_now_iso()
        job_id = str(uuid.uuid4())
        with get_connection(settings) as connection:
            case = _fetch_case_or_404(connection, case_id)
            document = _fetch_document_or_404(connection, case_id, document_id)
            if notes is not None:
                connection.execute(
                    "UPDATE document SET notes = ?, updated_at = ? WHERE id = ? AND case_id = ?",
                    (notes if notes else None, now, document_id, case_id),
                )
                document["notes"] = notes if notes else None
            if confidence_source_reliability is not None or confidence_information_validity is not None:
                src = (confidence_source_reliability or document.get("confidence_source_reliability", "A")).strip().upper()
                val = (confidence_information_validity or document.get("confidence_information_validity", "1")).strip()
                source, validity, code = _normalize_confidence(src, val)
                connection.execute(
                    "UPDATE document SET confidence_source_reliability = ?, confidence_information_validity = ?, "
                    "confidence_code = ?, updated_at = ? WHERE id = ? AND case_id = ?",
                    (source, validity, code, now, document_id, case_id),
                )
                document["confidence_source_reliability"] = source
                document["confidence_information_validity"] = validity
                document["confidence_code"] = code
        case_root = settings.cases_root / case["case_slug"]
        stored_path = _resolve_case_file(case_root, document["stored_file_path"])
        preflight = compute_ingestion_preflight(
            source_path=stored_path,
            mime_type=document.get("mime_type") or "application/octet-stream",
            ingest_profile=normalized_profile,
        )
        preflight_json = _to_json_text(preflight)
        try:
            cleanup_document_artifacts(
                case_root,
                document_id,
                settings.rag_embedding_dim_hint,
                case_id=case_id,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Unable to clean existing ingestion artifacts: {exc}",
            ) from exc
        document_search_service.clear_document_source(
            case_id=case_id,
            document_id=document_id,
            source_kind="processed",
        )
        with get_connection(settings) as connection:
            connection.execute(
                "INSERT INTO ingestion_job (id, case_id, document_id, ingest_profile, processing_mode, advanced_overrides_json, preflight_json, "
                "effective_config_json, complexity_class, eta_seconds, queue_priority, route_type, status, progress, started_at, "
                "finished_at, parse_duration_s, insert_duration_s, finalize_duration_s, current_stage, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    case_id,
                    document_id,
                    normalized_profile,
                    normalized_processing_mode,
                    overrides_json,
                    preflight_json,
                    None,
                    preflight["complexity_class"],
                    preflight["eta_seconds"],
                    queue_priority,
                    None,
                    "queued",
                    None,
                    now,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            )
            connection.execute(
                "UPDATE document SET ingestion_status = ?, ingestion_error = ?, updated_at = ? "
                "WHERE id = ? AND case_id = ?",
                ("queued", None, now, document_id, case_id),
            )
        append_job_log(
            settings,
            case_id=case_id,
            job_id=job_id,
            message=(
                f"Re-ingest queued for document '{document_id}' with profile "
                f"'{normalized_profile}' and mode '{normalized_processing_mode}' "
                f"(complexity={preflight['complexity_class']}, "
                f"eta={preflight['eta_seconds']}s, priority={queue_priority})."
            ),
        )
        return _envelope(
            {
                "job_id": job_id,
                "ingest_profile": normalized_profile,
                "processing_mode": normalized_processing_mode,
                "preflight": preflight,
                "advanced_overrides": normalized_overrides,
            },
            message="document re-ingest queued",
        )

    @app.post("/api/cases/{case_id}/resolve-entities")
    def trigger_entity_resolution(
        case_id: str,
        force: bool = Query(False),
    ) -> JSONResponse:
        """Manually trigger entity resolution for a case."""
        with get_connection(settings) as connection:
            case = _fetch_case_or_404(connection, case_id)
        try:
            data = entity_resolution_service.run_generic_resolution(
                case_id=case_id,
            )
            return _envelope(
                {"case_id": case_id, "force": force, **data},
                message="Entity resolution completed.",
            )
        except ValueError as exc:
            detail = str(exc)
            if "Case not found" in detail:
                raise HTTPException(status_code=404, detail=detail) from exc
            raise HTTPException(status_code=400, detail=detail) from exc

    @app.patch("/api/cases/{case_id}/documents/{document_id}")
    async def update_document_notes(case_id: str, document_id: str, request: Request) -> JSONResponse:
        body = await request.json()
        payload: dict[str, object] = body if isinstance(body, dict) else {}
        with get_connection(settings) as connection:
            _fetch_case_or_404(connection, case_id)
            doc = _fetch_document_or_404(connection, case_id, document_id)
            notes = payload.get("notes")
            if notes is not None and not isinstance(notes, str):
                raise HTTPException(status_code=400, detail="notes must be a string")
            src_rel = payload.get("confidence_source_reliability")
            if src_rel is not None and (not isinstance(src_rel, str) or src_rel.strip().upper() not in confidence_source_values):
                raise HTTPException(status_code=400, detail="Invalid confidence_source_reliability")
            info_val = payload.get("confidence_information_validity")
            if info_val is not None and (not isinstance(info_val, str) or info_val.strip() not in confidence_validity_values):
                raise HTTPException(status_code=400, detail="Invalid confidence_information_validity")
            now = utc_now_iso()
            updates: list[str] = []
            params: list[object] = []
            if "notes" in payload:
                updates.append("notes = ?")
                params.append(notes if notes else None)
            if src_rel is not None and info_val is not None:
                source = src_rel.strip().upper()
                validity = info_val.strip()
                code = f"{source}{validity}"
                updates.append("confidence_source_reliability = ?")
                params.append(source)
                updates.append("confidence_information_validity = ?")
                params.append(validity)
                updates.append("confidence_code = ?")
                params.append(code)
            if updates:
                updates.append("updated_at = ?")
                params.append(now)
                params.extend([document_id, case_id])
                connection.execute(
                    f"UPDATE document SET {', '.join(updates)} WHERE id = ? AND case_id = ?",
                    params,
                )
            doc = _fetch_document_or_404(connection, case_id, document_id)
        return _envelope(doc, message="document updated")

    @app.delete("/api/cases/{case_id}/documents/{document_id}")
    def delete_document(case_id: str, document_id: str) -> JSONResponse:
        with get_connection(settings) as connection:
            case = _fetch_case_or_404(connection, case_id)
            doc = _fetch_document_or_404(connection, case_id, document_id)

        case_root = settings.cases_root / case["case_slug"]
        try:
            cleanup_document_artifacts(
                case_root,
                document_id,
                settings.rag_embedding_dim_hint,
                case_id=case_id,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Unable to clean existing ingestion artifacts: {exc}",
            ) from exc
        document_search_service.clear_document(case_id=case_id, document_id=document_id)
        stored_path = _resolve_case_file(case_root, doc["stored_file_path"])
        if stored_path.exists():
            stored_path.unlink()

        remaining_docs = 0
        with get_connection(settings) as connection:
            connection.execute(
                "DELETE FROM ingestion_job WHERE case_id = ? AND document_id = ?",
                (case_id, document_id),
            )
            connection.execute(
                "DELETE FROM document WHERE id = ? AND case_id = ?",
                (document_id, case_id),
            )
            remaining_row = connection.execute(
                "SELECT COUNT(*) AS c FROM document WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            remaining_docs = int(remaining_row["c"]) if remaining_row is not None else 0

        if remaining_docs == 0:
            cleanup_case_ingestion_artifacts(case_root)
        try:
            case_summary_service.refresh_case_summary(case_id)
        except Exception:
            # Summary refresh is best-effort; evidence deletion must still succeed.
            pass

        return _envelope({"deleted": True}, message="document deleted")

    @app.exception_handler(HTTPException)
    def http_exception_handler(_, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "error", "message": exc.detail, "data": None},
        )

    return app


app = create_app()
