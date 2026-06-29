from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable
import httpx
from .cleanup import cleanup_document_graph_state, cleanup_orphan_lightrag_documents
from .db import get_connection
from .document_search import DocumentSearchService
from .entity_resolution import EntityResolutionService
from .fs import ensure_case_lightrag_dir, get_case_lightrag_root, resolve_case_lightrag_dir
from .graph_api import GraphStore
from .ingestion_preflight import detect_source_kind
from .job_logs import (
    append_job_log,
    clear_active_job_log_context,
    install_job_log_capture,
    set_active_job_log_context,
)
from .llm_tracer import LLMTracer
from .prompt_catalog import get_prompt_catalog
from .settings import Settings
from .utils import utc_now_iso

logger = logging.getLogger(__name__)

_LOW_VALUE_IMAGE_DESCRIPTION_PATTERNS = (
    "dark, circular object",
    "question mark",
    "stylized letter",
    "abstract logo",
    "abstract graphic",
    "indistinct and blurred",
    "faint, indistinct circular symbol",
    "low-res",
    "low-resolution",
)
_PLACEHOLDER_ENTITY_NAME_RE = re.compile(
    r"^(event|communication|document|asset)_\d+$", re.IGNORECASE
)
_NORMALIZATION_LOSSLESS_INSTRUCTIONS = """

Additional evidence-preservation rules:
- Preserve every atomic factual claim needed for later extraction. Do not omit facts merely because evidence is repetitive, tabular, log-like, or dense.
- For dense records, tables, logs, lists, or semi-structured text, keep each distinct record or relationship as its own item when possible.
- Preserve source relation wording, dates, roles, amounts, status values, identifiers, uncertainty, and source/provenance labels.
- Preserve complete named lists verbatim (e.g., all journalist names, all outlets, all policy distances, all dates). Do NOT truncate lists to representative examples.
- Preserve numeric values exactly as stated (e.g., "3 feet", "10 feet", "$50,000"). Do not round or approximate.
- Preserve exact source-specific phrases and direct quotes. Do not paraphrase or summarize quotes.
- Preserve cross-document references (page numbers, section titles, document identifiers) as provenance context.
- If the normalized JSON cannot fit every fact, prioritize preserving all relationship/event rows, unresolved references, named lists, and numeric values over concision.
""".strip()
_EXTRA_LIGHTRAG_PRIMARY_EXTRACTION_RULES = """

Additional generalized extraction rules:
- Preserve source-specific relationship wording in relation descriptions, including role phrases, validity dates, status values, amounts, identifiers, and provenance labels.
- When the allowed relation keyword is necessarily broad, use the closest allowed keyword but keep the original relation phrase verbatim in the relation description.
- For dense records, lists, logs, tables, or semi-structured text, extract every distinct factual relationship. Do not skip repeated rows if dates, roles, sources, status, or endpoints differ.
- If a source reference cannot be resolved to a human-readable name, preserve it in the description as unresolved provenance rather than dropping the relationship.
- Never add a descriptive prefix (such as 'Node', 'Row', 'Record', or 'ID') before a numeric or alphanumeric reference in entity names. Keep bare references as-is; keep identifiers in descriptions, not as prefixes.
""".strip()
_EXTRA_LIGHTRAG_CONTINUE_EXTRACTION_RULES = """

Additional generalized recovery rules:
- Focus this pass on missed relationships, dates, roles, status values, and provenance-bearing records, not only missed entities.
- Recover source-specific relationship wording in descriptions even when the relation keyword must stay within the allowed schema.
- For dense evidence, continue until every distinct factual relationship in the input has either been extracted or explicitly determined to be non-evidential.
- Do not add prefixes like 'Node', 'Row', 'Record', or 'ID' to numeric references in entity names. Keep references bare.
""".strip()
_INVESTIGATIVE_VISUAL_HINTS = (
    "aerial",
    "street",
    "vehicle",
    "vehicles",
    "officer",
    "agent",
    "shooter",
    "victim",
    "map",
    "scene",
    "location",
    "residential",
    "people",
    "individuals",
    "legend",
    "annotated",
)


@dataclass(frozen=True)
class IngestionRoute:
    route_type: str
    parse_method: str | None = None
    parser_kwargs: dict[str, Any] = field(default_factory=dict)
    direct_text: bool = False


class IngestionPipeline:
    """RAG-Anything + LightRAG ingestion workflow for a single document job."""

    _runtime_loop: asyncio.AbstractEventLoop | None = None
    _runtime_thread: threading.Thread | None = None
    _runtime_lock = threading.Lock()
    _runtime_ready = threading.Event()

    def __init__(
        self, settings: Settings, case_summary_service: Any | None = None
    ) -> None:
        self._settings = settings
        self._embedding_dim_cache: int | None = None
        self._entity_resolution_service = EntityResolutionService(settings)
        self._document_search_service = DocumentSearchService(settings)
        self._case_summary_service = case_summary_service
        self._prompt_catalog = get_prompt_catalog(
            path=self._settings.prompt_catalog_path,
            auto_reload=self._settings.prompt_catalog_auto_reload,
        )

    def process_job(self, job_id: str) -> None:
        loop = self._ensure_runtime_loop()
        future = asyncio.run_coroutine_threadsafe(self._process_job(job_id), loop)
        future.result()

    async def run_in_runtime_loop(self, coro: Any) -> Any:
        """Run a coroutine on the shared ingestion runtime loop and await its result."""
        runtime_loop = self._ensure_runtime_loop()
        current_loop = asyncio.get_running_loop()
        if current_loop is runtime_loop:
            return await coro
        future = asyncio.run_coroutine_threadsafe(coro, runtime_loop)
        return await asyncio.wrap_future(future)

    @classmethod
    def _ensure_runtime_loop(cls) -> asyncio.AbstractEventLoop:
        with cls._runtime_lock:
            if (
                cls._runtime_loop is not None
                and cls._runtime_thread is not None
                and cls._runtime_thread.is_alive()
                and not cls._runtime_loop.is_closed()
            ):
                return cls._runtime_loop

            cls._runtime_ready = threading.Event()

            def _run_loop() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                with cls._runtime_lock:
                    cls._runtime_loop = loop
                    cls._runtime_thread = threading.current_thread()
                    cls._runtime_ready.set()
                loop.run_forever()

            loop_thread = threading.Thread(
                target=_run_loop,
                daemon=True,
                name="rawabit-ingestion-loop",
            )
            cls._runtime_thread = loop_thread
            loop_thread.start()

        if not cls._runtime_ready.wait(timeout=5):
            raise RuntimeError("Failed to initialize ingestion runtime loop.")
        if cls._runtime_loop is None:
            raise RuntimeError("Ingestion runtime loop is unavailable.")
        return cls._runtime_loop

    async def _process_job(self, job_id: str) -> None:
        context = self._load_job_context(job_id)
        case_id = str(context["case_id"])
        set_active_job_log_context(case_id=case_id, job_id=job_id)
        self._active_tracer = LLMTracer(self._settings, case_id, job_id)
        case_root = self._settings.cases_root / context["case_slug"]
        active_lightrag_dir, lightrag_workspace = self._ensure_case_lightrag_context(
            case_root, case_id
        )
        source_path = self._resolve_case_file(case_root, context["stored_file_path"])
        ingest_profile = self._normalize_ingest_profile(context.get("ingest_profile"))
        processing_mode = self._normalize_processing_mode(
            context.get("processing_mode")
        )
        advanced_overrides = self._load_json_blob(
            context.get("advanced_overrides_json")
        )
        preflight = self._load_json_blob(context.get("preflight_json"))
        effective_config = self._build_effective_ingestion_config(
            ingest_profile=ingest_profile,
            processing_mode=processing_mode,
            advanced_overrides=advanced_overrides,
            preflight=preflight,
        )
        route = self._choose_ingestion_route(
            mime_type=str(context.get("mime_type") or ""),
            source_path=source_path,
            ingest_profile=ingest_profile,
            effective_config=effective_config,
        )
        started_at = utc_now_iso()
        status = "failed"
        error_message: str | None = None
        effective_parse_method = route.parse_method or self._settings.rag_parse_method
        stats: dict[str, Any] = {
            "ingest_profile": ingest_profile,
            "processing_mode": processing_mode,
            "route_type": route.route_type,
            "source_kind": effective_config.get("source_kind"),
            "complexity_class": effective_config.get("complexity_class"),
            "llm_model": effective_config.get("llm_model"),
            "vlm_model": effective_config.get("vlm_model"),
            "embedding_model": effective_config.get("embedding_model"),
            "primary_ingest_model": effective_config.get("primary_ingest_model"),
        }

        if not source_path.exists():
            raise FileNotFoundError(f"Missing evidence file: {source_path}")

        processed_dir = case_root / "processed" / context["document_id"]
        processed_dir.mkdir(parents=True, exist_ok=True)
        orphan_ids = self._cleanup_case_orphan_graph_state(
            case_id,
            case_root,
            active_lightrag_dir,
            lightrag_workspace,
        )
        if orphan_ids:
            self._log_job(
                case_id,
                job_id,
                (
                    "Removed orphan LightRAG document state before ingestion: "
                    + ", ".join(orphan_ids)
                ),
                level="warning",
            )
        self._set_job_telemetry(job_id, route_type=route.route_type)
        self._set_job_effective_config(job_id, effective_config)
        self._persist_json_artifact(
            processed_dir=processed_dir,
            filename="effective_ingestion_config.json",
            payload=effective_config,
        )
        if preflight:
            self._persist_json_artifact(
                processed_dir=processed_dir,
                filename="preflight_estimate.json",
                payload=preflight,
            )
        self._log_job(
            case_id,
            job_id,
            (
                f"Starting ingestion for '{context['original_filename']}' "
                f"(profile={ingest_profile}, mode={processing_mode}, route={route.route_type}, "
                f"parse_method={effective_parse_method}, ocr_mode={effective_config.get('ocr_mode')})."
            ),
        )

        rag = None
        try:
            rag = await self._initialize_rag(
                case_root,
                case_id=case_id,
                ingest_profile=ingest_profile,
                effective_config=effective_config,
            )
            rag._rawabit_case_id = case_id  # type: ignore[attr-defined]
            rag._rawabit_job_id = job_id  # type: ignore[attr-defined]
            self._set_status(job_id, "parsing", progress=15)
            self._set_job_telemetry(job_id, current_stage="parsing")
            self._log_job(case_id, job_id, "Parsing stage started.")
            parse_started = time.perf_counter()
            if route.direct_text:
                content_list = self._build_text_content_list(source_path)
                effective_parse_method = route.parse_method or "txt-direct"
            else:
                content_list, _ = await self._parse_document(
                    rag=rag,
                    file_path=source_path,
                    output_dir=processed_dir,
                    parse_method=route.parse_method,
                    parser_overrides=route.parser_kwargs,
                )
                effective_parse_method = (
                    route.parse_method or self._settings.rag_parse_method
                )
            parse_duration_s = round(time.perf_counter() - parse_started, 3)
            self._set_job_telemetry(job_id, parse_duration_s=parse_duration_s)
            stats["parse_duration_s"] = parse_duration_s
            stats["content_blocks"] = len(content_list)
            stats["content_type_counts"] = self._count_content_types(content_list)
            self._log_job(
                case_id,
                job_id,
                (
                    f"Parsing stage completed in {parse_duration_s}s with "
                    f"{len(content_list)} content blocks."
                ),
            )
            content_list = self._attach_content_provenance(
                content_list,
                document_id=context["document_id"],
                confidence_code=context["confidence_code"],
            )
            content_list = self._prepend_document_notes(
                content_list,
                notes=context.get("notes"),
            )
            self._set_job_telemetry(job_id, current_stage="enriching")
            content_list, enrichment_meta = await self._enrich_content_for_ingestion(
                content_list=content_list,
                rag=rag,
                ingest_profile=ingest_profile,
                effective_config=effective_config,
            )
            preinsert_text = self._compose_ingest_text_from_content_list(content_list)
            (
                normalized_evidence_text,
                normalization_meta,
            ) = await self._normalize_evidence_for_graph(
                source_text=preinsert_text,
                rag=rag,
                effective_config=effective_config,
            )
            insert_text = _coerce_text(normalized_evidence_text) or preinsert_text
            self._persist_preinsert_artifacts(
                processed_dir=processed_dir,
                content_list=content_list,
                preinsert_text=preinsert_text,
            )
            self._persist_text_artifact(
                processed_dir=processed_dir,
                filename="normalized_evidence.txt",
                text=insert_text,
            )
            self._persist_json_artifact(
                processed_dir=processed_dir,
                filename="enrichment_manifest.json",
                payload=enrichment_meta,
            )
            self._persist_json_artifact(
                processed_dir=processed_dir,
                filename="normalization_manifest.json",
                payload=normalization_meta,
            )
            stats["preinsert_enrichment"] = enrichment_meta
            stats["evidence_normalization"] = {
                key: value
                for key, value in normalization_meta.items()
                if key != "payload" and key != "raw_response"
            }
            stats["preinsert_text_chars"] = len(preinsert_text)
            stats["normalized_text_chars"] = len(insert_text)
            effective_config["used_vlm"] = bool(enrichment_meta.get("images_analyzed"))
            effective_config["primary_ingest_model"] = (
                self._resolve_primary_ingest_model(
                    effective_config=effective_config,
                    used_vlm=bool(enrichment_meta.get("images_analyzed")),
                )
            )
            stats["primary_ingest_model"] = effective_config.get("primary_ingest_model")
            self._set_job_effective_config(job_id, effective_config)
            self._persist_json_artifact(
                processed_dir=processed_dir,
                filename="effective_ingestion_config.json",
                payload=effective_config,
            )
            self._log_job(
                case_id,
                job_id,
                (
                    "Pre-insert enrichment completed "
                    f"(images_analyzed={enrichment_meta.get('images_analyzed', 0)}, "
                    f"visible_text_hits={enrichment_meta.get('visible_text_hits', 0)}, "
                    f"summary_added={enrichment_meta.get('summary_added', 0)})."
                ),
            )

            self._set_status(job_id, "inserting", progress=60)
            self._set_job_telemetry(job_id, current_stage="inserting")
            self._log_job(case_id, job_id, "Insertion stage started.")
            insert_started = time.perf_counter()
            try:
                await self._insert_content(
                    rag=rag,
                    content_list=content_list,
                    document_id=context["document_id"],
                    citation_file_path=context["stored_file_path"],
                    ingest_profile=ingest_profile,
                    processing_mode=processing_mode,
                    case_id=case_id,
                    job_id=job_id,
                    precomputed_fallback_text=insert_text,
                    effective_config=effective_config,
                )
            except RuntimeError as exc:
                if "different event loop" in str(exc):
                    self._log_job(
                        case_id, job_id,
                        "LightRAG async lock bound to a previous event loop (server restart). "
                        "Some relations may not be merged. This is a known LightRAG limitation. "
                        "Continuing with partial graph state.",
                        level="warning",
                    )
                else:
                    raise
            insert_duration_s = round(time.perf_counter() - insert_started, 3)
            self._set_job_telemetry(job_id, insert_duration_s=insert_duration_s)
            stats["insert_duration_s"] = insert_duration_s
            self._log_job(
                case_id,
                job_id,
                f"Insertion stage completed in {insert_duration_s}s.",
            )
            self._annotate_lightrag_provenance(
                working_dir=active_lightrag_dir,
                document_id=context["document_id"],
                confidence_code=context["confidence_code"],
                stored_file_path=context["stored_file_path"],
            )
            graph_counts = self._document_graph_counts(
                active_lightrag_dir, str(context["document_id"])
            )
            stats["graph_entity_count"] = graph_counts["entities"]
            stats["graph_relation_count"] = graph_counts["relations"]
            stats["graph_retry_used"] = False
            if graph_counts["entities"] == 0:
                retry_text = self._build_graph_retry_text(
                    preinsert_text,
                    content_list,
                    effective_config=effective_config,
                )
                if retry_text:
                    self._persist_json_artifact(
                        processed_dir=processed_dir,
                        filename="graph_retry_manifest.json",
                        payload={
                            "reason": "initial_graph_empty",
                            "original_text_chars": len(preinsert_text),
                            "retry_text_chars": len(retry_text),
                        },
                    )
                    self._persist_preinsert_artifacts(
                        processed_dir=processed_dir,
                        content_list=content_list,
                        preinsert_text=retry_text,
                    )
                    self._log_job(
                        case_id,
                        job_id,
                        "Initial graph extraction produced zero entities; retrying with graph-ready screenshot text.",
                        level="warning",
                    )
                    cleanup_document_graph_state(
                        get_case_lightrag_root(case_root),
                        active_lightrag_dir,
                        str(context["document_id"]),
                        self._settings.rag_embedding_dim_hint,
                        workspace=lightrag_workspace,
                    )
                    await self._insert_content(
                        rag=rag,
                        content_list=content_list,
                        document_id=context["document_id"],
                        citation_file_path=context["stored_file_path"],
                        ingest_profile=ingest_profile,
                        processing_mode=processing_mode,
                        case_id=case_id,
                        job_id=job_id,
                        precomputed_fallback_text=retry_text,
                        effective_config=effective_config,
                    )
                    self._annotate_lightrag_provenance(
                        working_dir=active_lightrag_dir,
                        document_id=context["document_id"],
                        confidence_code=context["confidence_code"],
                        stored_file_path=context["stored_file_path"],
                    )
                    graph_counts = self._document_graph_counts(
                        active_lightrag_dir, str(context["document_id"])
                    )
                    stats["graph_entity_count"] = graph_counts["entities"]
                    stats["graph_relation_count"] = graph_counts["relations"]
                    stats["graph_retry_used"] = True
                    if graph_counts["entities"] == 0:
                        self._log_job(
                            case_id,
                            job_id,
                            "Graph-ready retry still produced zero entities; review model quality or prompts.",
                            level="warning",
                        )
                else:
                    pass  # retry not available; extraction_failed handled below

            extraction_failed = graph_counts["entities"] == 0
            if extraction_failed:
                self._log_job(
                    case_id, job_id,
                    "Entity extraction produced zero graph entities for this document. "
                    "Check LLM availability and timeout settings.",
                    level="error",
                )

            # LightRAG indexing/final flush happens internally on insert.
            self._set_status(job_id, "indexing", progress=90)
            self._log_job(case_id, job_id, "Indexing stage started.")
            if self._case_has_pending_ingestion_jobs(case_id, exclude_job_id=job_id):
                self._log_job(
                    case_id,
                    job_id,
                    "Deferred case-level entity resolution until remaining queued ingestion jobs for this case finish.",
                )
            elif not self._settings.rag_resolution_auto_trigger:
                self._log_job(
                    case_id,
                    job_id,
                    "Entity resolution auto-trigger disabled; skipping resolution pass.",
                )
            else:
                maintenance_started = time.perf_counter()
                await self._run_entity_resolution_after_ingest(
                    case_id=case_id, job_id=job_id, rag=rag
                )
                stats["case_graph_maintenance_duration_s"] = round(
                    time.perf_counter() - maintenance_started, 3
                )
            if extraction_failed:
                status = "completed_with_warnings"
                self._log_job(
                    case_id, job_id,
                    "Entity extraction produced zero graph entities for this document. "
                    "Other documents' entities may still have been resolved successfully. "
                    "Re-ingest this document if important entities are missing.",
                    level="warning",
                )
            else:
                status = "complete"
        except Exception as exc:
            error_message = str(exc)
            self._log_job(
                case_id,
                job_id,
                f"Unhandled ingestion error: {error_message}",
                level="error",
            )
            raise
        finally:
            if rag is not None:
                self._set_job_telemetry(job_id, current_stage="finalizing")
                finalize_started = time.perf_counter()
                await self._finalize_rag(rag)
                finalize_duration_s = round(time.perf_counter() - finalize_started, 3)
                self._set_job_telemetry(job_id, finalize_duration_s=finalize_duration_s)
                stats["finalize_duration_s"] = finalize_duration_s
                self._log_job(
                    case_id,
                    job_id,
                    f"Finalize stage completed in {finalize_duration_s}s.",
                )
            if status in ("complete", "completed_with_warnings"):
                try:
                    self._set_job_telemetry(job_id, current_stage="indexing")
                    self._document_search_service.index_processed_document(
                        case_id=case_id,
                        document_id=str(context["document_id"]),
                        case_root=case_root,
                        original_filename=str(context["original_filename"]),
                        stored_file_path=str(context["stored_file_path"]),
                        confidence_code=str(context["confidence_code"]),
                    )
                    self._log_job(
                        case_id, job_id, "Processed document search index updated."
                    )
                except Exception as exc:
                    self._log_job(
                        case_id,
                        job_id,
                        f"Processed document search indexing skipped due to error: {exc}",
                        level="warning",
                    )
                self._mark_complete(
                    job_id,
                    status=status,
                    ingest_model_name=_coerce_text(
                        effective_config.get("primary_ingest_model")
                    )
                    or None,
                )
                if self._case_has_pending_ingestion_jobs(case_id, exclude_job_id=job_id):
                    self._log_job(
                        case_id,
                        job_id,
                        "Deferred case summary refresh until remaining queued ingestion jobs for this case finish.",
                    )
                else:
                    self._set_job_telemetry(job_id, current_stage="summarizing")
                    summary_started = time.perf_counter()
                    await self._refresh_case_summary_after_ingest(
                        case_id=case_id,
                        job_id=job_id,
                    )
                    stats["case_summary_refresh_duration_s"] = round(
                        time.perf_counter() - summary_started, 3
                    )
                self._log_job(case_id, job_id, "Job completed successfully.")
            self._write_manifest(
                processed_dir=processed_dir,
                context=context,
                started_at=started_at,
                finished_at=utc_now_iso(),
                status=status,
                parse_method=effective_parse_method,
                stats=stats,
                error=error_message,
            )
            self._persist_json_artifact(
                processed_dir=processed_dir,
                filename="stage_metrics.json",
                payload=stats,
            )
            self._log_job(
                case_id, job_id, f"Manifest written with terminal status '{status}'."
            )
            clear_active_job_log_context()

    async def _run_entity_resolution_after_ingest(
        self, *, case_id: str, job_id: str, rag: Any
    ) -> None:
        client = getattr(rag, "_rawabit_openai_client", None)
        try:
            result = await self._entity_resolution_service.arun_generic_resolution(
                case_id=case_id,
                client=client,
            )
            self._log_job(
                case_id,
                job_id,
                (
                    "Entity resolution pass completed: "
                    f"merged={result.get('merged_count', 0)}, "
                    f"proposed={result.get('proposed_count', 0)}."
                ),
            )
        except Exception as exc:
            self._log_job(
                case_id,
                job_id,
                f"Entity resolution pass skipped due to error: {exc}",
                level="warning",
            )

    async def _refresh_case_summary_after_ingest(
        self, *, case_id: str, job_id: str
    ) -> None:
        if self._case_summary_service is None:
            return
        try:
            await asyncio.to_thread(
                self._case_summary_service.refresh_case_summary,
                case_id,
                source_job_id=job_id,
            )
            self._log_job(case_id, job_id, "Case summary refreshed.")
        except Exception as exc:
            self._log_job(
                case_id,
                job_id,
                f"Case summary refresh skipped due to error: {exc}",
                level="warning",
            )

    def _case_has_pending_ingestion_jobs(
        self, case_id: str, *, exclude_job_id: str | None = None
    ) -> bool:
        with get_connection(self._settings) as connection:
            row = connection.execute(
                "SELECT 1 FROM ingestion_job WHERE case_id = ? "
                "AND (? IS NULL OR id <> ?) "
                "AND status IN ('queued', 'parsing', 'inserting', 'indexing') LIMIT 1",
                (case_id, exclude_job_id, exclude_job_id),
            ).fetchone()
        return row is not None

    def _normalize_ingest_profile(self, value: str | None) -> str:
        normalized = (value or self._settings.ingest_profile_default).strip().lower()
        if normalized == "balanced_fast":
            return "balanced_fast_intel"
        if normalized in {"balanced_fast_intel", "full_enrichment"}:
            return normalized
        return "balanced_fast_intel"

    @staticmethod
    def _normalize_processing_mode(value: Any) -> str:
        normalized = _coerce_text(value).lower()
        if normalized in {"multimodal", "text_first"}:
            return normalized
        return "multimodal"

    def _choose_ingestion_route(
        self,
        mime_type: str,
        source_path: Path,
        ingest_profile: str,
        effective_config: dict[str, Any] | None = None,
    ) -> IngestionRoute:
        config = effective_config or {}
        parse_override = _coerce_text(config.get("parse_method")).lower()
        ocr_mode = _coerce_text(config.get("ocr_mode")).lower()
        if not ocr_mode:
            ocr_mode = (
                "auto"
                if ingest_profile == "full_enrichment"
                else self._settings.rag_ocr_mode_default
            )
        source_kind = _coerce_text(
            config.get("source_kind")
        ).lower() or detect_source_kind(mime_type, source_path)

        if source_kind in {"text", "structured", "html"} and parse_override != "ocr":
            return IngestionRoute(
                route_type=f"{source_kind}_direct",
                parse_method="txt-direct",
                direct_text=True,
            )

        if source_kind == "image":
            image_parse_method = "auto"
            if parse_override in {"txt", "ocr", "auto"}:
                image_parse_method = (
                    "auto" if parse_override == "txt" else parse_override
                )
            elif ocr_mode == "force":
                image_parse_method = "ocr"
            elif ocr_mode == "auto":
                image_parse_method = "ocr"
            return IngestionRoute(
                route_type=f"image_{image_parse_method}",
                parse_method=image_parse_method,
                parser_kwargs={"table": False, "formula": False},
            )

        if source_kind == "pdf":
            parse_method = self._resolve_pdf_parse_method(
                source_path=source_path,
                ingest_profile=ingest_profile,
                parse_override=parse_override,
                ocr_mode=ocr_mode,
            )
            parser_kwargs = {
                "table": self._should_enable_tables_for_pdf(
                    source_path, ingest_profile
                ),
                "formula": False,
            }
            return IngestionRoute(
                route_type=f"pdf_{parse_method}",
                parse_method=parse_method,
                parser_kwargs=parser_kwargs,
            )

        if source_kind in {"office_document", "presentation", "spreadsheet", "email"}:
            parse_method = (
                parse_override if parse_override in {"txt", "ocr", "auto"} else "auto"
            )
            return IngestionRoute(
                route_type=f"{source_kind}_{parse_method}",
                parse_method=parse_method,
            )

        if source_kind in {"audio", "video"}:
            return IngestionRoute(
                route_type=f"media_{source_kind}",
                parse_method="auto",
            )

        if source_kind == "archive":
            return IngestionRoute(
                route_type="archive_auto",
                parse_method="auto",
            )

        parse_method = (
            parse_override
            if parse_override in {"txt", "ocr", "auto"}
            else self._settings.rag_parse_method
        )
        return IngestionRoute(
            route_type=f"{source_kind}_parse",
            parse_method=parse_method,
        )

    def _resolve_pdf_parse_method(
        self,
        source_path: Path,
        ingest_profile: str,
        parse_override: str = "",
        ocr_mode: str = "off",
    ) -> str:
        if parse_override in {"txt", "ocr", "auto"}:
            return parse_override
        if ocr_mode == "force":
            return "ocr"
        if ocr_mode == "off":
            return "txt"
        configured = (self._settings.rag_parse_method or "auto").strip().lower()
        if configured in {"txt", "ocr"}:
            return configured
        if ingest_profile == "full_enrichment":
            return "auto"
        chars_per_page = self._estimate_pdf_chars_per_page(source_path, max_pages=3)
        if chars_per_page is None:
            return configured
        threshold = max(1, int(self._settings.rag_pdf_probe_min_chars_per_page))
        return "txt" if chars_per_page >= threshold else "ocr"

    def _should_enable_tables_for_pdf(
        self, source_path: Path, ingest_profile: str
    ) -> bool:
        if not self._settings.rag_enable_table_processing:
            return False
        if ingest_profile == "full_enrichment":
            return True
        page_count = self._estimate_pdf_page_count(source_path)
        if page_count is None:
            return True
        return page_count <= max(1, int(self._settings.rag_balanced_table_max_pages))

    @staticmethod
    def _estimate_pdf_page_count(source_path: Path) -> int | None:
        try:
            from pypdf import PdfReader  # type: ignore[import-untyped]

            reader = PdfReader(str(source_path))
            return len(reader.pages)
        except Exception:
            return None

    @staticmethod
    def _estimate_pdf_chars_per_page(
        source_path: Path, max_pages: int = 3
    ) -> float | None:
        try:
            from pypdf import PdfReader  # type: ignore[import-untyped]

            reader = PdfReader(str(source_path))
            pages_to_scan = min(len(reader.pages), max(1, max_pages))
            if pages_to_scan == 0:
                return None
            chars = 0
            for idx in range(pages_to_scan):
                text = reader.pages[idx].extract_text() or ""
                chars += len(text.strip())
            return chars / float(pages_to_scan)
        except Exception:
            return None

    def _build_text_content_list(self, source_path: Path) -> list[dict[str, Any]]:
        raw = source_path.read_bytes()
        text = raw.decode("utf-8", errors="ignore").strip()
        if not text:
            text = raw.decode("latin-1", errors="ignore").strip()
        if not text:
            raise RuntimeError(
                "Unable to extract textual content from text evidence file."
            )
        return [{"type": "text", "text": text}]

    def _should_caption_image_with_vlm(
        self,
        ingest_profile: str,
        image_path: str,
        extracted_text: str,
        effective_config: dict[str, Any] | None = None,
    ) -> bool:
        config = effective_config or {}
        enable_vlm = config.get("enable_vlm")
        if enable_vlm is None:
            enable_vlm = self._settings.rag_enable_vlm
        if (
            not bool(enable_vlm)
            or not self._settings.rag_enable_vlm_image_analysis
            or not image_path
        ):
            return False
        if ingest_profile == "full_enrichment":
            return True
        if _coerce_text(config.get("parse_method")).lower() == "vlm-first":
            return True
        text_len = len(extracted_text.strip())
        return text_len < max(1, int(self._settings.rag_image_vlm_min_ocr_chars))

    def _image_caption_parallelism(
        self, ingest_profile: str, effective_config: dict[str, Any] | None = None
    ) -> int:
        config = effective_config or {}
        override = config.get("vlm_parallelism")
        if isinstance(override, int):
            return max(1, min(override, 16))
        if ingest_profile == "full_enrichment":
            return max(1, int(self._settings.rag_vlm_parallel_captions_full_enrichment))
        return max(1, int(self._settings.rag_vlm_parallel_captions_balanced))

    def _set_job_telemetry(self, job_id: str, **fields: Any) -> None:
        allowed = {
            "route_type",
            "parse_duration_s",
            "insert_duration_s",
            "finalize_duration_s",
            "current_stage",
            "complexity_class",
            "queue_priority",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        assignments = ", ".join(f"{name} = ?" for name in updates.keys())
        with get_connection(self._settings) as connection:
            connection.execute(
                f"UPDATE ingestion_job SET {assignments} WHERE id = ?",
                (*updates.values(), job_id),
            )

    @staticmethod
    def _set_llm_error_capture(rag: Any, enabled: bool) -> None:
        try:
            setattr(rag, "_rawabit_capture_llm_errors", bool(enabled))
        except Exception:
            return

    @staticmethod
    def _reset_llm_error_state(rag: Any) -> None:
        state = getattr(rag, "_rawabit_llm_error_state", None)
        if isinstance(state, dict):
            state["count"] = 0
            state["samples"] = []

    @staticmethod
    def _record_llm_error(rag: Any, exc: Exception) -> None:
        capture_enabled = bool(getattr(rag, "_rawabit_capture_llm_errors", False))
        if not capture_enabled:
            return
        state = getattr(rag, "_rawabit_llm_error_state", None)
        if not isinstance(state, dict):
            return
        count = int(state.get("count") or 0) + 1
        state["count"] = count
        samples = state.get("samples")
        if not isinstance(samples, list):
            samples = []
            state["samples"] = samples
        if len(samples) < 5:
            samples.append(str(exc))

    @staticmethod
    def _raise_if_llm_errors(rag: Any, *, stage: str) -> None:
        state = getattr(rag, "_rawabit_llm_error_state", None)
        if not isinstance(state, dict):
            return
        count = int(state.get("count") or 0)
        if count <= 0:
            return
        samples = state.get("samples")
        sample_rows = samples if isinstance(samples, list) else []
        sample_text = "; ".join(
            str(item).strip() for item in sample_rows if str(item).strip()
        )
        if sample_text:
            raise RuntimeError(
                f"LLM errors detected during {stage}: {count} failed call(s). Samples: {sample_text}"
            )
        raise RuntimeError(
            f"LLM errors detected during {stage}: {count} failed call(s)."
        )

    def _set_job_effective_config(self, job_id: str, payload: dict[str, Any]) -> None:
        serialized = json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        with get_connection(self._settings) as connection:
            connection.execute(
                "UPDATE ingestion_job SET effective_config_json = ? WHERE id = ?",
                (serialized, job_id),
            )

    def _resolve_primary_ingest_model(
        self,
        *,
        effective_config: dict[str, Any],
        used_vlm: bool | None = None,
    ) -> str:
        if used_vlm is None:
            used_vlm = bool(effective_config.get("used_vlm"))
        vlm_model = _coerce_text(effective_config.get("vlm_model"))
        llm_model = _coerce_text(effective_config.get("llm_model"))
        if used_vlm and vlm_model:
            return vlm_model
        return llm_model or vlm_model

    @staticmethod
    def _persist_json_artifact(
        processed_dir: Path, filename: str, payload: dict[str, Any]
    ) -> None:
        target = processed_dir / filename
        target.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _persist_text_artifact(processed_dir: Path, filename: str, text: str) -> None:
        (processed_dir / filename).write_text(text, encoding="utf-8")

    @staticmethod
    def _load_json_blob(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if not isinstance(value, str) or not value.strip():
            return {}
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, dict):
            return payload
        return {}

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        normalized = _coerce_text(text)
        if not normalized:
            return {}

        candidates = [normalized]
        candidates.extend(
            match.group(1)
            for match in re.finditer(
                r"```(?:json)?\s*(.*?)```", normalized, flags=re.IGNORECASE | re.DOTALL
            )
        )
        start = normalized.find("{")
        end = normalized.rfind("}")
        if start >= 0 and end > start:
            candidates.append(normalized[start : end + 1])
        decoder = json.JSONDecoder()
        for candidate in candidates:
            candidate = candidate.strip().lstrip("\ufeff")
            if candidate.lower().startswith("json\n"):
                candidate = candidate.split("\n", 1)[1].strip()
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                payload = None
                for match in re.finditer(r"\{", candidate):
                    try:
                        decoded, _ = decoder.raw_decode(candidate[match.start() :])
                    except json.JSONDecodeError:
                        continue
                    if isinstance(decoded, dict):
                        payload = decoded
                        break
            if isinstance(payload, dict):
                return payload
        return {}

    def _build_effective_ingestion_config(
        self,
        ingest_profile: str,
        processing_mode: str,
        advanced_overrides: dict[str, Any],
        preflight: dict[str, Any],
    ) -> dict[str, Any]:
        profile = ingest_profile
        ocr_mode = self._settings.rag_ocr_mode_default
        if profile == "full_enrichment":
            ocr_mode = "auto"

        source_kind = _coerce_text(preflight.get("source_kind")).lower() or "generic"
        complexity_class = (
            _coerce_text(preflight.get("complexity_class")).lower() or "medium"
        )
        eta_seconds = preflight.get("eta_seconds")
        eta_int = int(eta_seconds) if isinstance(eta_seconds, (int, float)) else None

        effective: dict[str, Any] = {
            "profile": profile,
            "processing_mode": processing_mode,
            "source_kind": source_kind,
            "complexity_class": complexity_class,
            "eta_seconds": eta_int,
            "ocr_mode": ocr_mode,
            "parse_method": "",
            "enable_vlm": bool(self._settings.rag_enable_vlm),
            "enable_vlm_visible_text": bool(
                self._settings.rag_enable_vlm_visible_text_extraction
            ),
            "enable_preinsert_summary": False,
            "summary_max_tokens": int(self._settings.rag_preinsert_summary_max_tokens),
            "vlm_parallelism": self._image_caption_parallelism(profile),
            "max_parallel_insert": self._max_parallel_insert_for_profile(profile),
            "queue_priority": "normal",
            "llm_model": self._settings.rag_llm_model,
            "vlm_model": self._settings.rag_vlm_model,
            "embedding_model": self._settings.rag_embedding_model,
            "used_vlm": False,
            "entity_extract_max_gleaning": 1 if advanced_overrides.get("enable_gleaning", True) else 0,
        }
        if processing_mode == "text_first":
            # Text-first mode intentionally skips multimodal-heavy extraction during insertion.
            effective["enable_vlm"] = False
            effective["enable_vlm_visible_text"] = False

        if isinstance(advanced_overrides.get("ocr_mode"), str):
            candidate = advanced_overrides["ocr_mode"].strip().lower()
            if candidate in {"off", "auto", "force"}:
                effective["ocr_mode"] = candidate

        if isinstance(advanced_overrides.get("parse_method"), str):
            candidate = advanced_overrides["parse_method"].strip().lower()
            if candidate in {
                "auto",
                "txt",
                "ocr",
                "native",
                "vlm-first",
                "transcript-first",
            }:
                effective["parse_method"] = candidate

        for bool_key in {
            "enable_vlm",
            "enable_vlm_visible_text",
            "enable_preinsert_summary",
        }:
            if isinstance(advanced_overrides.get(bool_key), bool):
                effective[bool_key] = advanced_overrides[bool_key]

        for int_key, lower, upper in (
            ("vlm_parallelism", 1, 16),
            ("max_parallel_insert", 1, 16),
            ("summary_max_tokens", 80, 1200),
        ):
            value = advanced_overrides.get(int_key)
            if isinstance(value, int):
                effective[int_key] = max(lower, min(value, upper))

        if isinstance(advanced_overrides.get("queue_priority"), str):
            candidate = advanced_overrides["queue_priority"].strip().lower()
            if candidate in {"low", "normal", "high"}:
                effective["queue_priority"] = candidate

        effective["primary_ingest_model"] = self._resolve_primary_ingest_model(
            effective_config=effective,
            used_vlm=False,
        )

        return effective

    @staticmethod
    def _image_bbox_area(item: dict[str, Any]) -> int:
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return 0
        try:
            x1, y1, x2, y2 = [int(value) for value in bbox]
        except Exception:
            return 0
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _sanitize_multiline_field(text: str) -> str:
        lines = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip().strip("-* ").strip()
            if stripped:
                lines.append(stripped)
        return "\n".join(lines).strip()

    @staticmethod
    def _visible_text_items(text: str) -> list[str]:
        items: list[str] = []
        normalized = _coerce_text(text)
        if not normalized:
            return items
        chunks = re.split(r"[\n,]+", normalized)
        for chunk in chunks:
            token = chunk.strip().strip('"').strip("'").strip()
            lowered = token.lower()
            if not token or lowered in {"none", "n/a", "null"}:
                continue
            if re.fullmatch(r"\d{2}:\d{2}:\d{2}:\d{2}", token):
                continue
            if lowered in {"play", "pause", "video"}:
                continue
            items.append(token)
        return _dedupe_preserve_order(items)

    @staticmethod
    def _is_low_value_image_description(description: str) -> bool:
        lowered = description.lower()
        return any(
            pattern in lowered for pattern in _LOW_VALUE_IMAGE_DESCRIPTION_PATTERNS
        )

    @staticmethod
    def _has_investigative_visual_signal(description: str) -> bool:
        lowered = description.lower()
        return any(token in lowered for token in _INVESTIGATIVE_VISUAL_HINTS)

    def _compose_image_evidence_block(self, item: dict[str, Any]) -> str:
        description = self._sanitize_multiline_field(
            _coerce_text(item.get("vlm_description"))
        )
        visible_text = self._sanitize_multiline_field(
            _coerce_text(item.get("vlm_visible_text"))
        )
        visible_items = self._visible_text_items(visible_text)

        raw_text = _coerce_text(item.get("text"))
        if raw_text and not visible_items and not description:
            visible_items = self._visible_text_items(raw_text)

        area = self._image_bbox_area(item)
        low_value_description = self._is_low_value_image_description(description)
        parts: list[str] = []
        if visible_items:
            parts.append("Image labels: " + "; ".join(visible_items))
        if (
            description
            and not low_value_description
            and self._has_investigative_visual_signal(description)
        ):
            parts.append(f"Visual depiction: {description}")
        if not parts and area < 4000:
            return ""
        if not parts and low_value_description:
            return ""
        return "\n".join(parts).strip()

    def _render_normalized_evidence_text(self, payload: dict[str, Any]) -> str:
        ordered_sections = (
            "primary_evidence_text",
            "entities_and_roles",
            "events_and_timeline",
            "movements_and_transfers",
            "source_and_reporting_context",
            "supporting_observations",
        )
        lines: list[str] = []
        for key in ordered_sections:
            value = payload.get(key)
            if isinstance(value, str):
                normalized = _coerce_text(value)
                if normalized:
                    lines.append(normalized)
                continue
            if not isinstance(value, list):
                continue
            for item in value:
                normalized = _coerce_text(item)
                if normalized:
                    lines.append(normalized)
        return "\n\n".join(_dedupe_preserve_order(lines)).strip()

    async def _normalize_evidence_for_graph(
        self,
        *,
        source_text: str,
        rag: Any,
        effective_config: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        normalized_source = _coerce_text(source_text)
        if not normalized_source:
            return "", {"used": False, "reason": "empty_source_text"}
        client = getattr(rag, "_rawabit_openai_client", None)
        if client is None:
            return normalized_source, {"used": False, "reason": "client_unavailable"}

        max_chars = max(4000, int(self._settings.rag_preinsert_summary_max_input_chars))
        prompt_source = normalized_source[:max_chars]
        prompt_text = self._prompt_catalog.render(
            "ingestion.evidence_normalization",
            {
                "source_kind": _coerce_text((effective_config or {}).get("source_kind"))
                or "generic",
                "source_text": prompt_source,
            },
        )
        prompt_text = f"{prompt_text}\n\n{_NORMALIZATION_LOSSLESS_INSTRUCTIONS}"
        _, _, messages = self._prompt_catalog.apply_external_overrides(
            messages=[{"role": "user", "content": prompt_text}]
        )
        try:
            response = await self._run_network_call_with_retry(
                operation="Evidence normalization LLM request",
                rag=rag,
                trace_stage="normalization",
                trace_model=self._settings.rag_llm_model,
                call_factory=lambda: client.chat.completions.create(
                    model=self._settings.rag_llm_model,
                    messages=messages or [{"role": "user", "content": prompt_text}],
                    max_tokens=min(
                        self._settings.rag_llm_max_tokens,
                        self._settings.rag_evidence_normalization_max_tokens,
                    ),
                    temperature=0,
                ),
            )
        except Exception:
            logger.debug("Evidence normalization failed", exc_info=True)
            return normalized_source, {"used": False, "reason": "llm_failure"}

        raw_response = _coerce_text(_extract_text_from_response(response))
        payload = self._extract_json_object(raw_response)
        normalized_text = self._render_normalized_evidence_text(payload)
        if not normalized_text:
            return normalized_source, {
                "used": False,
                "reason": "invalid_response",
                "raw_response": raw_response,
            }
        low_value_count = 0
        low_value_content = payload.get("low_value_content")
        if isinstance(low_value_content, list):
            low_value_count = sum(1 for item in low_value_content if _coerce_text(item))
        return normalized_text, {
            "used": True,
            "source_chars": len(prompt_source),
            "normalized_chars": len(normalized_text),
            "low_value_items": low_value_count,
            "payload": payload,
        }

    def _build_graph_retry_text(
        self,
        fallback_text: str,
        content_list: list[dict[str, Any]],
        effective_config: dict[str, Any] | None = None,
    ) -> str:
        config = effective_config or {}
        if _coerce_text(config.get("source_kind")).lower() != "image":
            return ""

        source_line = ""
        body_lines: list[str] = []
        for item in content_list:
            if (
                not isinstance(item, dict)
                or str(item.get("type", "")).lower() != "text"
            ):
                continue
            text_val = _coerce_text(item.get("text"))
            if not text_val or item.get("generated_by") == "preinsert_summary":
                continue
            compact = " ".join(text_val.split())
            if not source_line and "@" in compact and len(compact) <= 120:
                source_line = compact
                continue
            body_lines.append(compact)

        image_blocks = [
            self._compose_image_evidence_block(item)
            for item in content_list
            if isinstance(item, dict) and str(item.get("type", "")).lower() == "image"
        ]
        image_blocks = _dedupe_preserve_order(
            [block for block in image_blocks if block]
        )
        body_text = " ".join(_dedupe_preserve_order(body_lines)).strip()

        sections: list[str] = ["Evidence type: social media screenshot."]
        if source_line:
            match = re.match(r"^(?P<name>.+?)\s+(@\S+)$", source_line)
            if match:
                sections.append(
                    f"Source account: {match.group('name').strip()} ({match.group(2).strip()})."
                )
            else:
                sections.append(f"Source line: {source_line}")
        if body_text:
            sections.append(f"Post text: {body_text}")
        sections.extend(image_blocks)

        retry_text = "\n\n".join(
            section for section in sections if _coerce_text(section)
        ).strip()
        if retry_text == _coerce_text(fallback_text):
            return ""
        return retry_text

    def _document_graph_counts(
        self, working_dir: Path, document_id: str
    ) -> dict[str, int]:
        entity_count = 0
        relation_count = 0

        entity_payload = _load_json(working_dir / "kv_store_full_entities.json")
        if isinstance(entity_payload, dict):
            row = entity_payload.get(document_id)
            if isinstance(row, dict):
                raw_count = row.get("count")
                if isinstance(raw_count, int):
                    entity_count = raw_count
                elif isinstance(row.get("entity_names"), list):
                    entity_count = len(row["entity_names"])

        relation_payload = _load_json(working_dir / "kv_store_full_relations.json")
        if isinstance(relation_payload, dict):
            row = relation_payload.get(document_id)
            if isinstance(row, dict):
                raw_count = row.get("count")
                if isinstance(raw_count, int):
                    relation_count = raw_count
                elif isinstance(row.get("relation_pairs"), list):
                    relation_count = len(row["relation_pairs"])

        return {"entities": entity_count, "relations": relation_count}



    @staticmethod
    def _allocate_unique_entity_name(
        candidate: str,
        *,
        entity_type: str | None,
        reserved_names: set[str],
    ) -> str:
        if candidate not in reserved_names:
            return candidate
        base = candidate
        suffix_type = str(entity_type or "entity").strip().lower() or "entity"
        index = 2
        while True:
            variant = f"{base} ({suffix_type} {index})"
            if variant not in reserved_names:
                return variant
            index += 1

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod
    def _lightrag_workspace_for_dir(
        case_root: Path,
        case_id: str,
        working_dir: Path,
    ) -> str | None:
        return (
            case_id
            if working_dir.resolve() != get_case_lightrag_root(case_root).resolve()
            else None
        )

    def _ensure_case_lightrag_context(
        self,
        case_root: Path,
        case_id: str,
    ) -> tuple[Path, str | None]:
        working_dir = ensure_case_lightrag_dir(case_root, case_id)
        workspace = self._lightrag_workspace_for_dir(case_root, case_id, working_dir)
        return working_dir, workspace

    def _cleanup_case_orphan_graph_state(
        self,
        case_id: str,
        case_root: Path,
        working_dir: Path,
        workspace: str | None,
    ) -> list[str]:
        with get_connection(self._settings) as connection:
            rows = connection.execute(
                "SELECT id FROM document WHERE case_id = ?",
                (case_id,),
            ).fetchall()
        active_ids = {
            str(row["id"])
            for row in rows
            if isinstance(row["id"], str) and row["id"].strip()
        }
        if workspace is None:
            return []
        return cleanup_orphan_lightrag_documents(
            working_dir,
            active_document_ids=active_ids,
            embedding_dim_hint=self._settings.rag_embedding_dim_hint,
            workspace=workspace,
        )

    async def _enrich_content_for_ingestion(
        self,
        content_list: list[dict[str, Any]],
        rag: Any,
        ingest_profile: str,
        effective_config: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        config = effective_config or {}
        enriched: list[dict[str, Any]] = []
        for item in content_list:
            if isinstance(item, dict):
                enriched.append(dict(item))
            else:
                text_val = _coerce_text(item)
                if text_val:
                    enriched.append({"type": "text", "text": text_val})
        image_targets: list[tuple[int, str]] = []
        for idx, item in enumerate(enriched):
            item_type = str(item.get("type", "")).lower()
            if item_type != "image":
                continue
            image_path = _coerce_text(item.get("img_path"))
            if self._should_caption_image_with_vlm(
                ingest_profile=ingest_profile,
                image_path=image_path,
                extracted_text=_coerce_text(item.get("text")),
                effective_config=config,
            ):
                image_targets.append((idx, image_path))

        images_analyzed = 0
        visible_text_hits = 0
        if image_targets:
            analysis_rows = await self._analyze_images_batch(
                image_paths=[path for _, path in image_targets],
                rag=rag,
                ingest_profile=ingest_profile,
                effective_config=config,
            )
            for (item_idx, _), analysis in zip(image_targets, analysis_rows):
                item = enriched[item_idx]
                description = _coerce_text(analysis.get("description"))
                visible_text = _coerce_text(analysis.get("visible_text"))
                combined_text = _coerce_text(analysis.get("combined_text"))
                if description:
                    item["vlm_description"] = description
                    images_analyzed += 1
                if visible_text:
                    item["vlm_visible_text"] = visible_text
                    visible_text_hits += 1
                if combined_text:
                    item["normalized_image_text"] = combined_text

        summary_added = 0
        if bool(
            config.get(
                "enable_preinsert_summary", self._settings.rag_enable_preinsert_summary
            )
        ):
            source_text = self._build_summary_source_text(enriched)
            if source_text:
                summary = await self._summarize_ingestion_text(
                    source_text,
                    rag,
                    summary_max_tokens=int(
                        config.get(
                            "summary_max_tokens",
                            self._settings.rag_preinsert_summary_max_tokens,
                        )
                    ),
                )
                if summary:
                    enriched.insert(
                        0,
                        {
                            "type": "text",
                            "text": summary,
                            "generated_by": "preinsert_summary",
                        },
                    )
                    summary_added = 1

        return enriched, {
            "images_analyzed": images_analyzed,
            "visible_text_hits": visible_text_hits,
            "summary_added": summary_added,
        }

    @staticmethod
    def _persist_preinsert_artifacts(
        processed_dir: Path,
        content_list: list[dict[str, Any]],
        preinsert_text: str,
    ) -> None:
        content_list_path = processed_dir / "content_list_enriched.json"
        content_list_path.write_text(
            json.dumps(content_list, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        canonical_text_path = processed_dir / "canonical_text.txt"
        canonical_text_path.write_text(preinsert_text, encoding="utf-8")
        preinsert_text_path = processed_dir / "preinsert_text.txt"
        preinsert_text_path.write_text(preinsert_text, encoding="utf-8")

    def _compose_ingest_text_from_content_list(
        self, content_list: list[dict[str, Any]]
    ) -> str:
        blocks: list[str] = []
        for item in content_list:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).lower()
            text_val = _coerce_text(item.get("text"))
            if item_type == "text":
                if text_val:
                    blocks.append(text_val)
                continue
            if item_type == "table":
                table_text = _coerce_text(item.get("table_body")) or _coerce_text(
                    item.get("table_data")
                )
                if table_text:
                    blocks.append(table_text)
                continue
            if item_type == "equation":
                eq = _coerce_text(item.get("latex"))
                if eq and text_val:
                    blocks.append(f"Equation: {eq}\n{text_val}")
                elif eq:
                    blocks.append(f"Equation: {eq}")
                elif text_val:
                    blocks.append(text_val)
                continue
            if item_type == "image":
                composed = self._compose_image_evidence_block(item)
                if composed:
                    blocks.append(composed)
                continue
            generic = text_val or _coerce_text(item.get("content"))
            if generic:
                blocks.append(generic)

        return "\n\n".join(_dedupe_preserve_order(blocks))

    def _build_summary_source_text(self, content_list: list[dict[str, Any]]) -> str:
        max_chars = max(500, int(self._settings.rag_preinsert_summary_max_input_chars))
        blocks: list[str] = []
        total_chars = 0
        for item in content_list:
            block = self._content_item_text_for_summary(item)
            if not block:
                continue
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            clipped = block[:remaining]
            blocks.append(clipped)
            total_chars += len(clipped)
        return "\n\n".join(blocks).strip()

    def _content_item_text_for_summary(self, item: dict[str, Any]) -> str:
        item_type = str(item.get("type", "")).lower()
        text_val = _coerce_text(item.get("text"))
        if item_type == "text":
            return text_val
        if item_type == "table":
            return _coerce_text(item.get("table_body")) or _coerce_text(
                item.get("table_data")
            )
        if item_type == "equation":
            latex = _coerce_text(item.get("latex"))
            if latex and text_val:
                return f"Equation: {latex}\n{text_val}"
            if latex:
                return f"Equation: {latex}"
            return text_val
        if item_type == "image":
            return self._compose_image_evidence_block(item)
        return text_val or _coerce_text(item.get("content"))

    async def _summarize_ingestion_text(
        self, source_text: str, rag: Any, summary_max_tokens: int | None = None
    ) -> str:
        client = getattr(rag, "_rawabit_openai_client", None)
        if client is None:
            return ""
        prompt_text = self._prompt_catalog.render(
            "ingestion.preinsert_summary", {"source_text": source_text}
        )
        _, _, messages = self._prompt_catalog.apply_external_overrides(
            messages=[{"role": "user", "content": prompt_text}]
        )
        try:
            response = await self._run_network_call_with_retry(
                operation="Pre-insert summary LLM request",
                rag=rag,
                trace_stage="preinsert_summary",
                trace_model=self._settings.rag_llm_model,
                call_factory=lambda: client.chat.completions.create(
                    model=self._settings.rag_llm_model,
                    messages=messages or [{"role": "user", "content": prompt_text}],
                    max_tokens=min(
                        self._settings.rag_llm_max_tokens,
                        max(
                            100,
                            int(
                                summary_max_tokens
                                if summary_max_tokens is not None
                                else self._settings.rag_preinsert_summary_max_tokens
                            ),
                        ),
                    ),
                    temperature=0,
                ),
            )
            return _coerce_text(_extract_text_from_response(response))
        except Exception:
            logger.debug("Pre-insert summary generation failed", exc_info=True)
            return ""

    def _log_job(
        self, case_id: str, job_id: str, message: str, level: str = "info"
    ) -> None:
        append_job_log(
            self._settings,
            case_id=case_id,
            job_id=job_id,
            message=message,
            level=level,
        )

    @staticmethod
    def _network_retry_delay_seconds(attempt: int, remaining_seconds: float) -> float:
        # Exponential backoff (1s, 2s, 4s...) capped to keep retries responsive.
        backoff = min(30.0, float(2 ** max(0, attempt - 1)))
        return max(0.25, min(backoff, max(0.0, remaining_seconds)))

    def _network_retry_window_seconds(self) -> float:
        return max(0.0, float(self._settings.rag_network_retry_window_seconds))

    @staticmethod
    def _is_network_exception(exc: Exception) -> bool:
        try:
            from openai import APIConnectionError, APITimeoutError

            if isinstance(exc, (APIConnectionError, APITimeoutError)):
                return True
        except Exception:
            pass

        if isinstance(exc, asyncio.TimeoutError):
            return True

        try:
            import httpx

            if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
                return True
        except Exception:
            pass

        return False

    @staticmethod
    def _job_log_context_from_rag(rag: Any | None) -> tuple[str | None, str | None]:
        if rag is None:
            return None, None
        case_id = _coerce_text(getattr(rag, "_rawabit_case_id", None))
        job_id = _coerce_text(getattr(rag, "_rawabit_job_id", None))
        if case_id and job_id:
            return case_id, job_id
        return None, None

    async def _run_network_call_with_retry(
        self,
        *,
        operation: str,
        call_factory: Callable[[], Awaitable[Any]],
        rag: Any | None = None,
        trace_stage: str | None = None,
        trace_model: str | None = None,
    ) -> Any:
        retry_window_seconds = self._network_retry_window_seconds()
        deadline = time.monotonic() + retry_window_seconds
        attempt = 0
        started = time.monotonic()
        while True:
            try:
                result = await call_factory()
                tracer = getattr(self, "_active_tracer", None)
                if tracer and trace_stage and trace_model:
                    tracer.record(
                        stage=trace_stage,
                        model=trace_model,
                        started=started,
                        response=result,
                    )
                return result
            except Exception as exc:
                if not self._is_network_exception(exc):
                    tracer = getattr(self, "_active_tracer", None)
                    if tracer and trace_stage and trace_model:
                        tracer.record(
                            stage=trace_stage,
                            model=trace_model,
                            started=started,
                            error=str(exc)[:2000],
                        )
                    raise
                attempt += 1
                if retry_window_seconds <= 0:
                    raise RuntimeError(
                        f"{operation} failed due to network error: {exc}"
                    ) from exc
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise RuntimeError(
                        f"{operation} failed after retrying network errors for "
                        f"{int(retry_window_seconds)} seconds. Last error: {exc}"
                    ) from exc
                delay_seconds = self._network_retry_delay_seconds(
                    attempt, remaining_seconds
                )
                case_id, job_id = self._job_log_context_from_rag(rag)
                if case_id and job_id:
                    self._log_job(
                        case_id,
                        job_id,
                        (
                            f"{operation} network error (attempt {attempt}); retrying in "
                            f"{delay_seconds:.1f}s. Error: {exc}"
                        ),
                        level="warning",
                    )
                await asyncio.sleep(delay_seconds)

    async def _insert_content(
        self,
        rag: Any,
        content_list: list[dict[str, Any]],
        document_id: str,
        citation_file_path: str,
        ingest_profile: str = "balanced_fast",
        processing_mode: str = "multimodal",
        case_id: str | None = None,
        job_id: str | None = None,
        precomputed_fallback_text: str | None = None,
        effective_config: dict[str, Any] | None = None,
    ) -> None:
        config = effective_config or {}
        normalized_processing_mode = self._normalize_processing_mode(processing_mode)
        fallback_text = _coerce_text(precomputed_fallback_text)
        if not fallback_text:
            fallback_text, _ = await self._build_fallback_text(
                content_list,
                rag,
                ingest_profile=ingest_profile,
                effective_config=config,
            )
        if not fallback_text.strip():
            raise RuntimeError(
                "Canonical text insertion failed: no textual content available."
            )
        lightrag_instance = self._get_lightrag_instance(rag)
        if lightrag_instance is None or not hasattr(lightrag_instance, "ainsert"):
            raise RuntimeError(
                "Canonical text insertion failed: LightRAG instance unavailable."
            )
        if case_id and job_id:
            if normalized_processing_mode == "text_first":
                self._log_job(
                    case_id,
                    job_id,
                    "Processing mode 'text_first' is treated as a compatibility hint; using canonical text indexing.",
                )
            else:
                self._log_job(
                    case_id,
                    job_id,
                    "Using canonical text indexing for final LightRAG insertion.",
                )
        self._reset_llm_error_state(rag)
        self._set_llm_error_capture(rag, True)
        try:
            await lightrag_instance.ainsert(
                fallback_text,
                ids=document_id,
                file_paths=citation_file_path,
            )
        finally:
            self._set_llm_error_capture(rag, False)
        self._raise_if_llm_errors(rag, stage="canonical text insertion")
        await lightrag_instance._insert_done()
        if case_id and job_id:
            self._log_job(case_id, job_id, "Canonical text insertion completed.")

    async def _build_fallback_text(
        self,
        content_list: list[dict[str, Any]],
        rag: Any,
        ingest_profile: str = "balanced_fast",
        effective_config: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, int]]:
        config = effective_config or {}
        blocks: list[str] = []
        image_block_positions: list[int] = []
        image_paths_for_vlm: list[str] = []
        for item in content_list:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).lower()
            text_val = _coerce_text(item.get("text"))
            if item_type == "text":
                if text_val:
                    blocks.append(text_val)
                continue
            if item_type == "table":
                table_text = _coerce_text(item.get("table_body")) or _coerce_text(
                    item.get("table_data")
                )
                if table_text:
                    blocks.append(table_text)
                continue
            if item_type == "equation":
                eq = _coerce_text(item.get("latex"))
                if eq and text_val:
                    blocks.append(f"Equation: {eq}\n{text_val}")
                elif eq:
                    blocks.append(f"Equation: {eq}")
                elif text_val:
                    blocks.append(text_val)
                continue
            if item_type == "image":
                vlm_description = _coerce_text(item.get("vlm_description"))
                vlm_visible_text = _coerce_text(item.get("vlm_visible_text"))
                image_path = _coerce_text(item.get("img_path"))
                block_position = len(blocks)
                blocks.append(self._compose_image_evidence_block(item))
                if self._should_caption_image_with_vlm(
                    ingest_profile=ingest_profile,
                    image_path=image_path,
                    extracted_text=text_val,
                    effective_config=config,
                ) and not (vlm_description or vlm_visible_text):
                    image_block_positions.append(block_position)
                    image_paths_for_vlm.append(image_path)
                continue
            generic = text_val or _coerce_text(item.get("content"))
            if generic:
                blocks.append(generic)

        vlm_success = 0
        if image_paths_for_vlm:
            captions = await self._caption_images_batch(
                image_paths=image_paths_for_vlm,
                rag=rag,
                ingest_profile=ingest_profile,
                effective_config=config,
            )
            for position, caption in zip(image_block_positions, captions):
                normalized_caption = _coerce_text(caption)
                if not normalized_caption:
                    continue
                vlm_success += 1
                existing = _coerce_text(blocks[position])
                blocks[position] = (
                    f"{existing}\n{normalized_caption}".strip()
                    if existing
                    else normalized_caption
                )

        normalized_blocks = _dedupe_preserve_order(
            [block for block in blocks if _coerce_text(block)]
        )
        coalesced_blocks = _coalesce_short_text_blocks(
            normalized_blocks,
            target_tokens=max(250, min(600, int(self._settings.rag_chunk_token_size))),
            max_tokens=max(350, int(self._settings.rag_chunk_token_size)),
        )
        return "\n\n".join(coalesced_blocks), {
            "images_for_vlm": len(image_paths_for_vlm),
            "vlm_parallelism": self._image_caption_parallelism(ingest_profile, config),
            "vlm_success": vlm_success,
        }

    async def _analyze_images_batch(
        self,
        image_paths: list[str],
        rag: Any,
        ingest_profile: str,
        effective_config: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        config = effective_config or {}
        parallelism = self._image_caption_parallelism(ingest_profile, config)
        semaphore = asyncio.Semaphore(parallelism)

        async def _run_one(path: str) -> dict[str, str]:
            async with semaphore:
                return await self._analyze_image_with_vlm(path, rag, config)

        results = await asyncio.gather(
            *[_run_one(path) for path in image_paths], return_exceptions=True
        )
        analyses: list[dict[str, str]] = []
        for row in results:
            if isinstance(row, Exception):
                analyses.append(
                    {"description": "", "visible_text": "", "combined_text": ""}
                )
            else:
                analyses.append(
                    {
                        "description": _coerce_text(row.get("description")),
                        "visible_text": _coerce_text(row.get("visible_text")),
                        "combined_text": _coerce_text(row.get("combined_text")),
                    }
                )
        return analyses

    async def _caption_images_batch(
        self,
        image_paths: list[str],
        rag: Any,
        ingest_profile: str,
        effective_config: dict[str, Any] | None = None,
    ) -> list[str]:
        analyses = await self._analyze_images_batch(
            image_paths=image_paths,
            rag=rag,
            ingest_profile=ingest_profile,
            effective_config=effective_config,
        )
        return [row.get("combined_text", "") for row in analyses]

    async def _analyze_image_with_vlm(
        self, image_path: str, rag: Any, effective_config: dict[str, Any] | None = None
    ) -> dict[str, str]:
        config = effective_config or {}
        caption_text = await self._caption_image(image_path, rag, config)
        description, visible_text = self._parse_vlm_analysis_text(caption_text)
        combined_parts = []
        if description:
            combined_parts.append(f"Description: {description}")
        enable_visible_text = bool(
            config.get(
                "enable_vlm_visible_text",
                self._settings.rag_enable_vlm_visible_text_extraction,
            )
        )
        if enable_visible_text:
            combined_parts.append(f"Visible text: {visible_text or 'none'}")
        combined_text = "\n".join(part for part in combined_parts if part).strip()
        return {
            "description": description,
            "visible_text": visible_text,
            "combined_text": combined_text,
        }

    async def _caption_image(
        self, image_path: str, rag: Any, effective_config: dict[str, Any] | None = None
    ) -> str:
        config = effective_config or {}
        try:
            from base64 import b64encode
        except Exception:
            return ""

        client = getattr(rag, "_rawabit_openai_client", None)
        if client is None:
            return ""
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            return ""
        try:
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            image_data = b64encode(path.read_bytes()).decode("ascii")
            enable_visible_text = bool(
                config.get(
                    "enable_vlm_visible_text",
                    self._settings.rag_enable_vlm_visible_text_extraction,
                )
            )
            if enable_visible_text:
                task_prompt = self._prompt_catalog.render(
                    "ingestion.image_caption.with_visible_text"
                )
            else:
                task_prompt = self._prompt_catalog.render(
                    "ingestion.image_caption.description_only"
                )
            _, _, messages = self._prompt_catalog.apply_external_overrides(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": task_prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{image_data}"
                                },
                            },
                        ],
                    }
                ]
            )
            response = await self._run_network_call_with_retry(
                operation="Image caption LLM request",
                rag=rag,
                trace_stage="image_caption",
                trace_model=self._settings.rag_vlm_model,
                call_factory=lambda: client.chat.completions.create(
                    model=self._settings.rag_vlm_model,
                    messages=messages
                    or [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": task_prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime};base64,{image_data}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=min(self._settings.rag_llm_max_tokens, 4000),
                    temperature=0,
                ),
            )
            return _extract_text_from_response(response)
        except Exception:
            logger.debug(
                "Image caption fallback failed for %s", image_path, exc_info=True
            )
            return ""

    @staticmethod
    def _parse_vlm_analysis_text(text: str) -> tuple[str, str]:
        normalized = _coerce_text(text)
        if not normalized:
            return "", ""
        description_lines: list[str] = []
        visible_text_lines: list[str] = []
        current_section = ""
        for line in normalized.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if lowered.startswith("description:"):
                current_section = "description"
                remainder = stripped.split(":", 1)[1].strip()
                if remainder:
                    description_lines.append(remainder)
                continue
            if lowered.startswith("visible text:"):
                current_section = "visible_text"
                remainder = stripped.split(":", 1)[1].strip()
                if remainder:
                    visible_text_lines.append(remainder)
                continue
            if current_section == "description":
                description_lines.append(stripped)
                continue
            if current_section == "visible_text":
                visible_text_lines.append(stripped)
        description = IngestionPipeline._sanitize_multiline_field(
            "\n".join(description_lines)
        )
        visible_text = IngestionPipeline._sanitize_multiline_field(
            "\n".join(visible_text_lines)
        )
        if not description and not visible_text:
            description = normalized
        return description, visible_text

    async def _parse_document(
        self,
        rag: Any,
        file_path: Path,
        output_dir: Path,
        parse_method: str | None = None,
        parser_overrides: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        parser_kwargs = self._build_parser_kwargs()
        if parser_overrides:
            parser_kwargs.update(parser_overrides)
        parse_kwargs: dict[str, Any] = {
            "file_path": str(file_path),
            "output_dir": str(output_dir),
            "parse_method": parse_method or self._settings.rag_parse_method,
            "display_stats": self._settings.rag_display_stats,
            **parser_kwargs,
        }
        # Parser selection belongs to RAGAnythingConfig; passing parser= here leaks into
        # backend parser kwargs on some raganything versions and breaks MinerU.
        return await rag.parse_document(**parse_kwargs)

    def _apply_library_prompt_overrides(
        self, provider: str, target_prompts: dict[str, Any]
    ) -> None:
        overrides = self._prompt_catalog.get_library_prompt_overrides(provider)
        for key, value in overrides.items():
            if key not in target_prompts:
                raise RuntimeError(
                    f"Prompt override key not found in {provider}: {key}"
                )
            default_value = target_prompts.get(key)
            if isinstance(default_value, str):
                if not isinstance(value, str):
                    raise RuntimeError(
                        f"Prompt override type mismatch for {provider}.{key}: expected string."
                    )
                target_prompts[key] = value
                continue
            if isinstance(default_value, list):
                if not isinstance(value, list) or not all(
                    isinstance(item, str) for item in value
                ):
                    raise RuntimeError(
                        f"Prompt override type mismatch for {provider}.{key}: expected string array."
                    )
                target_prompts[key] = list(value)
                continue
            raise RuntimeError(
                f"Prompt override target {provider}.{key} has unsupported default type: {type(default_value).__name__}"
            )
        self._apply_extraction_prompt_enhancements(provider, target_prompts)
        if overrides:
            logger.info(
                "Applied %d prompt overrides for provider %s",
                len(overrides),
                provider,
            )

    @staticmethod
    def _append_once(source: str, addition: str) -> str:
        if addition in source:
            return source
        return f"{source.rstrip()}\n{addition}\n"

    @classmethod
    def _apply_extraction_prompt_enhancements(
        cls, provider: str, target_prompts: dict[str, Any]
    ) -> None:
        if provider != "lightrag":
            return
        primary_key = "entity_extraction_user_prompt"
        if isinstance(target_prompts.get(primary_key), str):
            target_prompts[primary_key] = cls._append_once(
                target_prompts[primary_key],
                _EXTRA_LIGHTRAG_PRIMARY_EXTRACTION_RULES,
            )
        continue_key = "entity_continue_extraction_user_prompt"
        if isinstance(target_prompts.get(continue_key), str):
            target_prompts[continue_key] = cls._append_once(
                target_prompts[continue_key],
                _EXTRA_LIGHTRAG_CONTINUE_EXTRACTION_RULES,
            )

    async def _build_runtime_components(
        self,
        *,
        max_parallel_insert: int,
        enable_vision: bool,
    ) -> dict[str, Any]:
        if not self._settings.llm_provider_api_key:
            raise RuntimeError(
                "LLM provider API key is not configured. Set RAWABIT_LLM_PROVIDER_API_KEY, RAWABIT_OPENROUTER_API_KEY, or OPENROUTER_API_KEY."
            )
        try:
            import numpy as np
            from lightrag.prompt import PROMPTS as LIGHTRAG_PROMPTS
            from lightrag.utils import EmbeddingFunc
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependencies for ingestion/query runtime. Install: lightrag-hku openai numpy"
            ) from exc

        self._apply_library_prompt_overrides("lightrag", LIGHTRAG_PROMPTS)
        if enable_vision:
            try:
                from raganything.prompt import PROMPTS as RAGANYTHING_PROMPTS
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependencies for multimodal ingestion. Install: raganything"
                ) from exc
            self._apply_library_prompt_overrides("raganything", RAGANYTHING_PROMPTS)

        headers: dict[str, str] = {}
        if self._settings.llm_provider_site_url:
            headers["HTTP-Referer"] = self._settings.llm_provider_site_url
        if self._settings.llm_provider_app_name:
            headers["X-Title"] = self._settings.llm_provider_app_name

        llm_client = AsyncOpenAI(
            base_url=self._settings.llm_provider_base_url,
            api_key=self._settings.llm_provider_api_key,
            timeout=float(self._settings.rag_llm_timeout_seconds),
            default_headers=headers or None,
        )
        embedding_client = AsyncOpenAI(
            base_url=self._settings.embedding_provider_base_url,
            api_key=self._settings.embedding_provider_api_key,
            timeout=float(self._settings.rag_embedding_timeout_seconds),
        )
        llm_error_state: dict[str, Any] = {"count": 0, "samples": []}
        runtime_holder: dict[str, Any] = {"runtime": None}
        embedding_dim = await self._resolve_embedding_dim(embedding_client)

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str:
            runtime = runtime_holder.get("runtime")
            raw_messages = kwargs.pop("messages", None)
            model_name = kwargs.pop("model", self._settings.rag_llm_model)
            chat_kwargs = self._sanitize_chat_kwargs(kwargs)

            if raw_messages:
                messages = [m for m in raw_messages if m]
            else:
                messages: list[dict[str, Any]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                if history_messages:
                    messages.extend(history_messages)
                messages.append({"role": "user", "content": prompt})

            prompt, system_prompt, messages = (
                self._prompt_catalog.apply_external_overrides(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    messages=messages,
                )
            )
            if not messages:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                if history_messages:
                    messages.extend(history_messages)
                if prompt:
                    messages.append({"role": "user", "content": prompt})

            if "max_tokens" not in chat_kwargs:
                chat_kwargs["max_tokens"] = self._settings.rag_llm_max_tokens
            if "temperature" not in chat_kwargs:
                chat_kwargs["temperature"] = self._settings.rag_llm_temperature

            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    response = await self._run_network_call_with_retry(
                        operation="LightRAG LLM completion request",
                        rag=runtime,
                        trace_stage="entity_extraction",
                        trace_model=model_name,
                        call_factory=lambda: llm_client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            **chat_kwargs,
                        ),
                    )
                    return _extract_text_from_response(response)
                except asyncio.TimeoutError:
                    last_error = None
                except Exception as exc:
                    last_error = exc
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
            if last_error is None:
                last_error = asyncio.TimeoutError(
                    "LLM call timed out after 3 attempts"
                )
            self._record_llm_error(runtime, last_error)
            raise last_error

        async def vision_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            image_data: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str:
            if messages:
                return await llm_model_func(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    model=self._settings.rag_vlm_model,
                    messages=messages,
                    **kwargs,
                )

            if image_data:
                image_messages: list[dict[str, Any]] = []
                if system_prompt:
                    image_messages.append({"role": "system", "content": system_prompt})
                image_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ],
                    }
                )
                return await llm_model_func(
                    prompt=prompt,
                    model=self._settings.rag_vlm_model,
                    messages=image_messages,
                    **kwargs,
                )

            return await llm_model_func(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )

        async def embedding_func(texts: list[str]):
            runtime = runtime_holder.get("runtime")
            response = await self._run_network_call_with_retry(
                operation="LightRAG embedding request",
                rag=runtime,
                trace_stage="embedding",
                trace_model=self._settings.rag_embedding_model,
                call_factory=lambda: embedding_client.embeddings.create(
                    model=self._settings.rag_embedding_model,
                    input=texts,
                ),
            )
            vectors = [row.embedding for row in response.data]
            return np.array(vectors)

        embedding_wrapper = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=embedding_func,
        )

        lightrag_kwargs = {
            "cosine_better_than_threshold": self._settings.rag_cosine_threshold,
            "default_llm_timeout": int(self._settings.rag_llm_timeout_seconds),
            "default_embedding_timeout": int(
                self._settings.rag_embedding_timeout_seconds
            ),
            "max_parallel_insert": max_parallel_insert,
            "llm_model_max_async": max(1, int(self._settings.rag_llm_max_async)),
            "embedding_func_max_async": max(
                1, int(self._settings.rag_embedding_max_async)
            ),
            "chunk_token_size": int(self._settings.rag_chunk_token_size),
            "chunk_overlap_token_size": int(self._settings.rag_chunk_overlap_token_size),
            "addon_params": {
                "language": "English",
                "entity_types": [
                    "person",
                    "organization",
                    "object",
                    "location",
                    "event",
                ],
            },
        }

        return {
            "client": llm_client,
            "embedding_client": embedding_client,
            "llm_error_state": llm_error_state,
            "runtime_holder": runtime_holder,
            "llm_model_func": llm_model_func,
            "vision_model_func": vision_model_func,
            "embedding_wrapper": embedding_wrapper,
            "lightrag_kwargs": lightrag_kwargs,
        }

    async def _initialize_rag(
        self,
        case_root: Path,
        *,
        case_id: str,
        ingest_profile: str,
        effective_config: dict[str, Any] | None = None,
    ) -> Any:
        import atexit

        effective = effective_config or {}
        _, lightrag_workspace = self._ensure_case_lightrag_context(case_root, case_id)
        if self._settings.rag_mineru_inter_op_threads is not None:
            os.environ["MINERU_INTER_OP_NUM_THREADS"] = str(
                self._settings.rag_mineru_inter_op_threads
            )
        if self._settings.rag_mineru_intra_op_threads is not None:
            os.environ["MINERU_INTRA_OP_NUM_THREADS"] = str(
                self._settings.rag_mineru_intra_op_threads
            )

        try:
            from raganything import RAGAnything, RAGAnythingConfig
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependencies for ingestion. "
                "Install: lightrag-hku raganything openai numpy"
            ) from exc
        # LightRAG configures non-propagating loggers at import time.
        # Re-install capture handlers after imports so those logs are routed to job logs.
        install_job_log_capture(self._settings)

        max_parallel_insert = self._max_parallel_insert_for_profile(
            ingest_profile, effective
        )
        runtime = await self._build_runtime_components(
            max_parallel_insert=max_parallel_insert,
            enable_vision=bool(
                effective.get("enable_vlm", self._settings.rag_enable_vlm)
            ),
        )

        config_kwargs: dict[str, Any] = {
            "working_dir": str(get_case_lightrag_root(case_root)),
            "enable_image_processing": self._settings.rag_enable_image_processing,
            "enable_table_processing": self._settings.rag_enable_table_processing,
            # Keep equation processing off across all ingestion profiles.
            "enable_equation_processing": False,
        }
        if self._settings.rag_parser:
            config_kwargs["parser"] = self._settings.rag_parser
        if self._settings.rag_parse_method:
            config_kwargs["parse_method"] = self._settings.rag_parse_method

        rag_config = RAGAnythingConfig(**config_kwargs)
        rag = RAGAnything(
            config=rag_config,
            llm_model_func=runtime["llm_model_func"],
            embedding_func=runtime["embedding_wrapper"],
            vision_model_func=runtime["vision_model_func"]
            if bool(effective.get("enable_vlm", self._settings.rag_enable_vlm))
            else None,
            lightrag_kwargs={
                **runtime["lightrag_kwargs"],
                "workspace": lightrag_workspace or "",
            },
        )
        try:
            atexit.unregister(rag.close)
        except Exception:
            pass
        runtime["runtime_holder"]["runtime"] = rag
        rag._rawabit_openai_client = runtime["client"]  # type: ignore[attr-defined]
        rag._rawabit_llm_error_state = runtime["llm_error_state"]  # type: ignore[attr-defined]
        rag._rawabit_capture_llm_errors = False  # type: ignore[attr-defined]
        await self._ensure_rag_initialized(rag)
        return rag

    async def _initialize_lightrag_runtime(
        self,
        *,
        case_root: Path,
        case_id: str,
        ingest_profile: str,
        effective_config: dict[str, Any] | None = None,
    ) -> Any:
        try:
            from lightrag import LightRAG
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependencies for query runtime. Install: lightrag-hku"
            ) from exc

        effective = effective_config or {}
        _, lightrag_workspace = self._ensure_case_lightrag_context(case_root, case_id)
        runtime = await self._build_runtime_components(
            max_parallel_insert=self._max_parallel_insert_for_profile(
                ingest_profile, effective
            ),
            enable_vision=False,
        )
        runtime["lightrag_kwargs"]["entity_extract_max_gleaning"] = effective.get(
            "entity_extract_max_gleaning", 1
        )

        settings_ref = self._settings
        rerank_model = getattr(settings_ref, "rag_rerank_model", None) or "cohere/rerank-v3.5"
        rerank_url = getattr(settings_ref, "rag_rerank_provider_base_url", None) or "https://openrouter.ai/api/v1/rerank"
        rerank_api_key = settings_ref.rag_rerank_provider_api_key


        async def rerank_func(query: str, documents: list[str], top_n: int) -> list[dict]:
            try:
                async with httpx.AsyncClient(timeout=30.0) as http:
                    resp = await http.post(
                        rerank_url,
                        headers={
                            "Authorization": f"Bearer {rerank_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": rerank_model,
                            "query": query,
                            "documents": documents,
                            "top_n": top_n,
                        },
                    )
                    resp.raise_for_status()
                    results = resp.json()["results"]
                    sorted_results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
                    return [
                        {"index": r["index"], "relevance_score": r["relevance_score"]}
                        for r in sorted_results
                    ]
            except Exception:
                logger.debug("Rerank failed, falling back to original order", exc_info=True)
                return [{"index": i, "relevance_score": 1.0 - i * 0.01} for i in range(len(documents))]
        
        lightrag = LightRAG(
            working_dir=str(get_case_lightrag_root(case_root)),
            workspace=lightrag_workspace or "",
            llm_model_func=runtime["llm_model_func"],
            embedding_func=runtime["embedding_wrapper"],
            rerank_model_func=rerank_func,
            **runtime["lightrag_kwargs"],
        )
        runtime["runtime_holder"]["runtime"] = lightrag
        lightrag._rawabit_openai_client = runtime["client"]  # type: ignore[attr-defined]
        lightrag._rawabit_llm_error_state = runtime["llm_error_state"]  # type: ignore[attr-defined]
        lightrag._rawabit_capture_llm_errors = False  # type: ignore[attr-defined]
        await lightrag.initialize_storages()
        try:
            from lightrag.kg.shared_storage import initialize_pipeline_status

            workspace = getattr(lightrag, "workspace", None)
            if workspace:
                await initialize_pipeline_status(workspace=workspace)
            else:
                await initialize_pipeline_status()
        except TypeError:
            from lightrag.kg.shared_storage import initialize_pipeline_status

            await initialize_pipeline_status()
        except ImportError:
            pass
        return lightrag

    async def initialize_rag_for_query(
        self,
        case_root: Path,
        case_id: str,
        ingest_profile: str = "balanced_fast_intel",
        effective_config: dict[str, Any] | None = None,
    ) -> Any:
        """Public wrapper for creating a case-scoped RAG runtime for querying."""
        return await self._initialize_lightrag_runtime(
            case_root=case_root,
            case_id=case_id,
            ingest_profile=ingest_profile,
            effective_config=effective_config,
        )

    @staticmethod
    def _get_lightrag_instance(rag: Any) -> Any | None:
        direct_query_api = getattr(rag, "aquery_llm", None)
        direct_insert_api = getattr(rag, "ainsert", None)
        if callable(direct_query_api) or callable(direct_insert_api):
            return rag
        return getattr(rag, "lightrag", None) or getattr(rag, "_lightrag", None)

    async def _ensure_rag_initialized(self, rag: Any) -> None:
        ensure_ready = getattr(rag, "_ensure_lightrag_initialized", None)
        if callable(ensure_ready):
            ensure_result = await ensure_ready()
            if isinstance(ensure_result, dict) and not ensure_result.get(
                "success", True
            ):
                error_message = _coerce_text(ensure_result.get("error"))
                raise RuntimeError(
                    error_message
                    or "RAGAnything failed to initialize LightRAG before ingestion."
                )

        init_rag_storages = getattr(rag, "initialize_storages", None)
        if callable(init_rag_storages):
            await init_rag_storages()

        lightrag_instance = self._get_lightrag_instance(rag)
        if lightrag_instance is None:
            return

        init_lightrag_storages = getattr(lightrag_instance, "initialize_storages", None)
        if callable(init_lightrag_storages):
            await init_lightrag_storages()

        try:
            from lightrag.kg.shared_storage import initialize_pipeline_status
        except ImportError:
            return

        workspace = getattr(lightrag_instance, "workspace", None)
        try:
            if workspace:
                await initialize_pipeline_status(workspace=workspace)
            else:
                await initialize_pipeline_status()
        except TypeError:
            await initialize_pipeline_status()

    def _max_parallel_insert_for_profile(
        self, ingest_profile: str, effective_config: dict[str, Any] | None = None
    ) -> int:
        effective = effective_config or {}
        override = effective.get("max_parallel_insert")
        if isinstance(override, int):
            return max(1, min(override, 16))
        if ingest_profile == "full_enrichment":
            configured = self._settings.rag_lightrag_max_parallel_insert_full_enrichment
        else:
            configured = self._settings.rag_lightrag_max_parallel_insert_balanced
        return max(
            1, int(configured or self._settings.rag_lightrag_max_parallel_insert)
        )

    async def _resolve_embedding_dim(self, embedding_client: Any) -> int:
        if self._embedding_dim_cache is not None:
            return self._embedding_dim_cache
        try:
            probe_text = self._prompt_catalog.render(
                "ingestion.embedding.dimension_probe_input"
            )
            sample = await self._run_network_call_with_retry(
                operation="Embedding dimension probe request",
                trace_stage="embedding_probe",
                trace_model=self._settings.rag_embedding_model,
                call_factory=lambda: embedding_client.embeddings.create(
                    model=self._settings.rag_embedding_model,
                    input=[probe_text],
                ),
            )
            dim = len(sample.data[0].embedding)
        except Exception:
            dim = self._settings.rag_embedding_dim_hint
        self._embedding_dim_cache = dim
        return dim

    async def _finalize_rag(self, rag: Any) -> None:
        lightrag_instance = self._get_lightrag_instance(rag)
        if lightrag_instance is not None:
            await self._shutdown_lightrag_priority_workers(lightrag_instance)

        if lightrag_instance is not None and hasattr(
            lightrag_instance, "finalize_storages"
        ):
            try:
                await lightrag_instance.finalize_storages()
            except Exception:
                logger.debug("LightRAG finalize failed", exc_info=True)

        client = getattr(rag, "_rawabit_openai_client", None)
        if client is not None:
            try:
                await client.close()
            except Exception:
                logger.debug("OpenRouter client close failed", exc_info=True)

    async def _shutdown_lightrag_priority_workers(
        self,
        lightrag_instance: Any,
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        async def _shutdown_wrapper(target: Any, label: str) -> None:
            shutdown = getattr(target, "shutdown", None)
            if not callable(shutdown):
                return
            try:
                result = shutdown()
                if asyncio.iscoroutine(result):
                    await asyncio.wait_for(result, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.warning(
                    "%s shutdown timed out after %.1fs",
                    label,
                    timeout_seconds,
                )
            except Exception:
                logger.debug("%s shutdown failed", label, exc_info=True)

        llm_wrapper = getattr(lightrag_instance, "llm_model_func", None)
        embedding_func = getattr(lightrag_instance, "embedding_func", None)
        embedding_wrapper = getattr(embedding_func, "func", None)

        await _shutdown_wrapper(llm_wrapper, "LightRAG LLM worker queue")
        await _shutdown_wrapper(embedding_wrapper, "LightRAG embedding worker queue")

    async def finalize_rag_runtime(self, rag: Any) -> None:
        """Public wrapper for finalizing a case-scoped RAG runtime."""
        await self._finalize_rag(rag)

    def _build_parser_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self._settings.rag_parser_lang:
            kwargs["lang"] = self._settings.rag_parser_lang
        if self._settings.rag_parser_device:
            kwargs["device"] = self._settings.rag_parser_device
        if self._settings.rag_parser_backend:
            kwargs["backend"] = self._settings.rag_parser_backend
        if self._settings.rag_parser_start_page is not None:
            kwargs["start_page"] = self._settings.rag_parser_start_page
        if self._settings.rag_parser_end_page is not None:
            kwargs["end_page"] = self._settings.rag_parser_end_page
        if self._settings.rag_parser.lower() == "mineru":
            kwargs["table"] = self._settings.rag_enable_table_processing
            kwargs["formula"] = False
        return kwargs

    @staticmethod
    def _count_content_types(content_list: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in content_list:
            if not isinstance(item, dict):
                continue
            key = str(item.get("type", "unknown")).lower() or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _attach_content_provenance(
        content_list: list[dict[str, Any]], document_id: str, confidence_code: str
    ) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for item in content_list:
            clone = dict(item)
            clone["document_id"] = document_id
            clone["confidence_code"] = confidence_code
            enriched.append(clone)
        return enriched

    def _write_manifest(
        self,
        processed_dir: Path,
        context: dict[str, Any],
        started_at: str,
        finished_at: str,
        status: str,
        parse_method: str,
        stats: dict[str, Any],
        error: str | None = None,
    ) -> None:
        manifest = {
            "document_id": context["document_id"],
            "case_id": context["case_id"],
            "case_slug": context["case_slug"],
            "original_filename": context["original_filename"],
            "parser": self._settings.rag_parser,
            "parse_method": parse_method,
            "started_at": started_at,
            "finished_at": finished_at,
            "status": status,
            "stats": stats,
        }
        if error:
            manifest["error"] = error
        manifest_path = processed_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def _annotate_lightrag_provenance(
        self,
        working_dir: Path,
        document_id: str,
        confidence_code: str,
        stored_file_path: str,
    ) -> None:
        text_chunks_path = working_dir / "kv_store_text_chunks.json"
        text_chunks = _load_json(text_chunks_path)
        chunk_ids: set[str] = set()
        if isinstance(text_chunks, dict):
            for chunk_id, chunk in text_chunks.items():
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("full_doc_id") == document_id:
                    chunk["document_id"] = document_id
                    chunk["confidence_code"] = confidence_code
                    chunk_ids.add(str(chunk_id))
                    if isinstance(chunk.get("_id"), str):
                        chunk_ids.add(chunk["_id"])
            _save_json(text_chunks_path, text_chunks)

        by_doc_paths = [
            working_dir / "kv_store_doc_status.json",
            working_dir / "kv_store_full_docs.json",
            working_dir / "kv_store_full_entities.json",
            working_dir / "kv_store_full_relations.json",
        ]
        for path in by_doc_paths:
            payload = _load_json(path)
            if isinstance(payload, dict) and isinstance(payload.get(document_id), dict):
                row = payload[document_id]
                row["document_id"] = document_id
                row["confidence_code"] = confidence_code
                row.setdefault("file_path", stored_file_path)
                _save_json(path, payload)

        chunk_index_paths = [
            working_dir / "kv_store_entity_chunks.json",
            working_dir / "kv_store_relation_chunks.json",
        ]
        for path in chunk_index_paths:
            payload = _load_json(path)
            if isinstance(payload, dict):
                changed = False
                for _, row in payload.items():
                    if not isinstance(row, dict):
                        continue
                    row_chunk_ids = row.get("chunk_ids")
                    if not isinstance(row_chunk_ids, list):
                        continue
                    if any(str(c) in chunk_ids for c in row_chunk_ids):
                        row["document_id"] = document_id
                        row["confidence_code"] = confidence_code
                        changed = True
                if changed:
                    _save_json(path, payload)

        vdb_paths = [
            working_dir / "vdb_chunks.json",
            working_dir / "vdb_entities.json",
            working_dir / "vdb_relationships.json",
        ]
        for path in vdb_paths:
            payload = _load_json(path)
            if not isinstance(payload, dict):
                continue
            records = payload.get("data")
            if not isinstance(records, list):
                continue
            changed = False
            for record in records:
                if not isinstance(record, dict):
                    continue
                is_doc_chunk = record.get("full_doc_id") == document_id
                by_source = False
                source_id = record.get("source_id")
                if isinstance(source_id, str) and source_id:
                    by_source = any(
                        token in chunk_ids for token in _split_source_tokens(source_id)
                    )
                if is_doc_chunk or by_source:
                    record["document_id"] = document_id
                    record["confidence_code"] = confidence_code
                    record.setdefault("file_path", stored_file_path)
                    changed = True
            if changed:
                _save_json(path, payload)

    def _load_job_context(self, job_id: str) -> dict[str, Any]:
        with get_connection(self._settings) as connection:
            row = connection.execute(
                "SELECT j.id, j.case_id, j.document_id, j.ingest_profile, j.processing_mode, j.advanced_overrides_json, "
                "j.preflight_json, j.complexity_class, j.eta_seconds, c.case_slug, "
                "d.stored_file_path, d.original_filename, d.confidence_code, d.mime_type, d.notes "
                "FROM ingestion_job j "
                'JOIN "case" c ON c.id = j.case_id '
                "JOIN document d ON d.id = j.document_id AND d.case_id = j.case_id "
                "WHERE j.id = ?",
                (job_id,),
            ).fetchone()
        if not row:
            raise RuntimeError(f"Ingestion job not found: {job_id}")
        return dict(row)

    @staticmethod
    def _prepend_document_notes(
        content_list: list[dict[str, Any]], notes: str | None
    ) -> list[dict[str, Any]]:
        normalized_notes = _coerce_text(notes)
        if not normalized_notes:
            return content_list
        return [
            {
                "type": "text",
                "text": f"Analyst notes for this evidence:\n{normalized_notes}",
                "generated_by": "document_notes",
            },
            *content_list,
        ]

    def _resolve_case_file(self, case_root: Path, stored_path: str) -> Path:
        candidate = Path(stored_path)
        resolved = candidate if candidate.is_absolute() else case_root / candidate
        resolved = resolved.resolve()
        case_root_resolved = case_root.resolve()
        if (
            resolved != case_root_resolved
            and case_root_resolved not in resolved.parents
        ):
            raise RuntimeError(f"Document path escapes case workspace: {stored_path}")
        return resolved

    def _set_status(
        self, job_id: str, status: str, progress: int | None = None
    ) -> None:
        now = utc_now_iso()
        with get_connection(self._settings) as connection:
            connection.execute(
                "UPDATE ingestion_job SET status = ?, progress = ?, error = NULL, finished_at = NULL "
                "WHERE id = ?",
                (status, progress, job_id),
            )
            connection.execute(
                "UPDATE document SET ingestion_status = ?, ingestion_error = NULL, updated_at = ? "
                "WHERE id = (SELECT document_id FROM ingestion_job WHERE id = ?)",
                (status, now, job_id),
            )

    def _mark_complete(
        self, job_id: str, *, status: str = "complete", ingest_model_name: str | None = None
    ) -> None:
        now = utc_now_iso()
        with get_connection(self._settings) as connection:
            connection.execute(
                "UPDATE ingestion_job SET status = ?, progress = ?, finished_at = ?, error = NULL "
                "WHERE id = ?",
                (status, 100, now, job_id),
            )
            connection.execute(
                "UPDATE document SET ingestion_status = ?, ingestion_error = NULL, ingest_model_name = COALESCE(?, ingest_model_name), updated_at = ? "
                "WHERE id = (SELECT document_id FROM ingestion_job WHERE id = ?)",
                (status, ingest_model_name, now, job_id),
            )

    @staticmethod
    def _sanitize_chat_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "temperature",
            "top_p",
            "max_tokens",
            "n",
            "stop",
            "stream",
            "presence_penalty",
            "frequency_penalty",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
        }
        return {key: value for key, value in kwargs.items() if key in allowed}


def _extract_text_from_response(response: Any) -> str:
    if not response or not getattr(response, "choices", None):
        return ""
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text_value = getattr(item, "text", None)
            if text_value:
                parts.append(text_value)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content or "")


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [str(v).strip() for v in value if str(v).strip()]
        return "\n".join(parts)
    text = str(value).strip()
    return text

def _rough_token_count(value: str) -> int:
    """Cheap token estimate used only to group parsed PDF text blocks before LightRAG insertion."""
    text = _coerce_text(value)
    if not text:
        return 0
    return max(1, len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)))


def _looks_like_structural_boundary(value: str) -> bool:
    """Keep obvious headings/images/tables as boundaries while merging tiny prose blocks."""
    text = _coerce_text(value)
    if not text:
        return False
    if text.startswith(("Image:", "Table:", "Equation:", "Page ")):
        return True
    if len(text) <= 90 and not text.endswith((".", "?", "!", ":", ";")):
        # Usually a title or section heading.
        return True
    return False


def _coalesce_short_text_blocks(
    blocks: list[str],
    *,
    target_tokens: int = 500,
    max_tokens: int = 600,
) -> list[str]:
    """
    MinerU often emits one sentence per block. If those blocks are inserted as-is,
    LightRAG receives microchunks and retrieval misses neighbouring evidence.

    This groups consecutive short parsed text blocks into evidence windows before
    LightRAG performs its own token chunking. It preserves order and keeps image,
    table, equation, and heading-like boundaries separate.
    """
    output: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            output.append("\n".join(current).strip())
        current = []
        current_tokens = 0

    for raw_block in blocks:
        block = _coerce_text(raw_block)
        if not block:
            continue
        block_tokens = _rough_token_count(block)
        boundary = _looks_like_structural_boundary(block)

        if boundary:
            flush()
            output.append(block)
            continue

        if current and current_tokens + block_tokens > max_tokens:
            flush()

        current.append(block)
        current_tokens += block_tokens

        if current_tokens >= target_tokens:
            flush()

    flush()
    return output

def _split_source_tokens(value: str) -> set[str]:
    tokens = {value}
    for sep in ("<SEP>", ",", ";", "|"):
        if sep in value:
            for part in value.split(sep):
                stripped = part.strip()
                if stripped:
                    tokens.add(stripped)
    return tokens


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _coerce_text(value)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output




def _derive_descriptive_entity_name(
    entity_type: str | None,
    description: str | None,
) -> str | None:
    normalized_type = str(entity_type or "").strip().lower()
    if normalized_type not in {"event", "communication", "document", "asset"}:
        return None
    if not isinstance(description, str) or not description.strip():
        return None

    raw_text = description.replace("<SEP>", "\n")
    for segment in raw_text.splitlines():
        candidate = re.sub(r"\s+", " ", segment).strip()
        if not candidate:
            continue
        candidate = candidate.rstrip(" .;,:-\t")
        if normalized_type != "document":
            candidate = re.sub(r"^(?:the|a|an)\s+", "", candidate, flags=re.IGNORECASE)
        candidate = candidate.strip()
        if not candidate or re.match(r"^(event|communication|document|asset)_\d+$", candidate, re.IGNORECASE):
            continue
        if candidate.lower() in {"event", "communication", "document", "asset"}:
            continue
        if candidate and candidate[0].islower():
            candidate = candidate[0].upper() + candidate[1:]
        return candidate
    return None


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
