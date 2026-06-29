from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .graph_api import GraphStore
from .ingestion_pipeline import IngestionPipeline
from .settings import Settings


_LIGHTRAG_FILEPATH_SEP = "<SEP>"
_DEFAULT_ASSISTANT_FAILURE = (
    "I could not complete this query right now. Please retry in a moment."
)
_OOM_ASSISTANT_FAILURE = "The initial query exceeded the current model memory budget. Please retry with a narrower question."
_OOM_ERROR_PATTERNS = (
    "out of memory",
    "error code: out of memory",
    "cuda out of memory",
    "oom",
)


@dataclass(frozen=True)
class _CaseDocument:
    id: str
    original_filename: str
    stored_file_path: str
    confidence_code: str | None


@dataclass(frozen=True)
class ChatQueryResult:
    assistant_content: str
    highlight: dict[str, Any]
    retrieval_eval: dict[str, Any]
    references: list[dict[str, str]]
    chunks: list[dict[str, str]]
    metadata: dict[str, Any]


class ChatService:
    """Case-scoped chat query adapter over RAGAnything/LightRAG query APIs."""

    _QUERY_ALLOWED_OPTIONS = {
        "only_need_context",
        "only_need_prompt",
        "response_type",
        "top_k",
        "chunk_top_k",
        "max_entity_tokens",
        "max_relation_tokens",
        "max_total_tokens",
        "hl_keywords",
        "ll_keywords",
        "history_turns",
        "user_prompt",
        "enable_rerank",
        "model_func",
    }
    _DEFAULT_QUERY_OPTION_LIMITS: dict[str, int] = {
        "history_turns": 6,
        "max_entity_tokens": 20000,
        "max_relation_tokens": 20000,
        "max_total_tokens": 40000,
    }
    _QUERY_OPTION_MAX_LIMITS: dict[str, int] = {
        "top_k": 100,
        "chunk_top_k": 100,
        "history_turns": 20,
        "max_entity_tokens": 20000,
        "max_relation_tokens": 20000,
        "max_total_tokens": 40000,
    }
    _OOM_FALLBACK_OPTION_LIMITS: dict[str, int] = {
        "top_k": 3,
        "chunk_top_k": 3,
        "history_turns": 2,
        "max_entity_tokens": 4000,
        "max_relation_tokens": 4000,
        "max_total_tokens": 10000,
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ingestion_pipeline = IngestionPipeline(settings)

    async def query_case_message(
        self,
        *,
        case_root: Path,
        case_id: str | None = None,
        user_content: str,
        mode: str,
        conversation_history: list[dict[str, str]],
        case_documents: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        allow_mode_fallback: bool = True,
    ) -> ChatQueryResult:
        query_options = dict(options or {})
        system_prompt = (
            str(query_options.pop("system_prompt")).strip()
            if "system_prompt" in query_options
            and query_options["system_prompt"] is not None
            else None
        )

        selected_entities = query_options.pop("selected_entities", None)
        selected_relationships = query_options.pop("selected_relationships", None)
        if selected_entities or selected_relationships:
            context_parts: list[str] = []
            if isinstance(selected_entities, list) and selected_entities:
                entity_descriptions: list[str] = []
                for ent in selected_entities:
                    if isinstance(ent, dict):
                        ent_name = str(ent.get("name", "") or "").strip()
                        ent_type = str(ent.get("type", "") or "").strip()
                        if ent_name and ent_type:
                            entity_descriptions.append(f"[{ent_type}] {ent_name}")
                        elif ent_name:
                            entity_descriptions.append(ent_name)
                if entity_descriptions:
                    context_parts.append(
                        f"The user has selected the following entities for additional context: {', '.join(entity_descriptions)}."
                    )
            if isinstance(selected_relationships, list) and selected_relationships:
                rel_descriptions: list[str] = []
                for rel in selected_relationships:
                    if isinstance(rel, dict):
                        src_name = str(rel.get("src_name", "") or "").strip()
                        tgt_name = str(rel.get("tgt_name", "") or "").strip()
                        rel_type = str(rel.get("relation_type", "") or "").strip()
                        if src_name and tgt_name and rel_type:
                            rel_descriptions.append(f"{rel_type} between {src_name} and {tgt_name}")
                if rel_descriptions:
                    context_parts.append(
                        f"The user has selected the following relationships for additional context: {', '.join(rel_descriptions)}."
                    )
            if context_parts:
                context_str = " ".join(context_parts)
                user_content = f"{context_str}\n\nUser question: {user_content}"

        filtered_options = {
            key: value
            for key, value in query_options.items()
            if key in self._QUERY_ALLOWED_OPTIONS
        }
        primary_options = self._build_query_options(filtered_options)
        retrieval_top_k = self._normalize_retrieval_top_k(primary_options.get("top_k"))
        fallback_options = self._build_query_options(
            filtered_options,
            base_limits=self._OOM_FALLBACK_OPTION_LIMITS,
        )
        resolved_case_id = str(case_id or "").strip()
        if not resolved_case_id:
            for document in case_documents:
                candidate = str(document.get("case_id") or "").strip()
                if candidate:
                    resolved_case_id = candidate
                    break
        if not resolved_case_id:
            resolved_case_id = case_root.name

        async def _run_query_on_runtime() -> dict[str, Any] | None:
            rag = None
            rag = await self._ingestion_pipeline.initialize_rag_for_query(
                case_root=case_root,
                case_id=resolved_case_id,
                ingest_profile="balanced_fast_intel",
                effective_config=None,
            )
            try:
                direct_query_api = getattr(rag, "aquery_llm", None)
                if callable(direct_query_api):
                    lightrag = rag
                else:
                    lightrag = getattr(rag, "lightrag", None) or getattr(
                        rag, "_lightrag", None
                    )
                if lightrag is None:
                    raise RuntimeError(
                        "LightRAG runtime is unavailable for chat queries."
                    )

                from lightrag import QueryParam

                naive_threshold_override: float | None = None
                if resolved_case_id:
                    try:
                        naive_threshold_override = self._settings.rag_naive_cosine_threshold
                    except AttributeError:
                        naive_threshold_override = None

                async def _execute_query(
                    query_mode: str, query_options: dict[str, Any]
                ) -> dict[str, Any] | None:
                    lower_threshold = (
                        naive_threshold_override is not None
                        and query_mode == "naive"
                    )
                    original_chunk_threshold = None
                    if lower_threshold:
                        chunks_vdb = getattr(lightrag, "chunks_vdb", None)
                        if chunks_vdb is not None and hasattr(
                            chunks_vdb, "cosine_better_than_threshold"
                        ):
                            original_chunk_threshold = chunks_vdb.cosine_better_than_threshold
                            chunks_vdb.cosine_better_than_threshold = naive_threshold_override
                    try:
                        query_param = QueryParam(
                            mode=query_mode,
                            conversation_history=conversation_history,
                            include_references=True,
                            stream=False,
                            enable_rerank=self._settings.rag_rerank_enabled,
                            **query_options,
                        )
                        return await lightrag.aquery_llm(
                            user_content,
                            param=query_param,
                            system_prompt=system_prompt or None,
                        )
                    finally:
                        if lower_threshold and original_chunk_threshold is not None:
                            chunks_vdb = getattr(lightrag, "chunks_vdb", None)
                            if chunks_vdb is not None and hasattr(
                                chunks_vdb, "cosine_better_than_threshold"
                            ):
                                chunks_vdb.cosine_better_than_threshold = original_chunk_threshold

                raw_result = await _execute_query(mode, primary_options)
                resolved_mode = mode
                fallback_attempts: list[dict[str, Any]] = []

                if allow_mode_fallback and self._should_retry_after_failure(raw_result):
                    for fallback_mode in self._fallback_modes_for(mode):
                        fallback_attempts.append(
                            {
                                "from_mode": resolved_mode,
                                "to_mode": fallback_mode,
                                "reason": "oom",
                            }
                        )
                        raw_result = await _execute_query(
                            fallback_mode, fallback_options
                        )
                        resolved_mode = fallback_mode
                        if not self._should_retry_after_failure(raw_result):
                            break

                if isinstance(raw_result, dict):
                    normalized_result = dict(raw_result)
                    metadata = (
                        normalized_result.get("metadata")
                        if isinstance(normalized_result.get("metadata"), dict)
                        else {}
                    )
                    metadata = dict(metadata)
                    metadata["mode"] = resolved_mode
                    metadata["requested_mode"] = mode
                    if fallback_attempts:
                        metadata["fallback_attempts"] = fallback_attempts
                        metadata["fallback_applied"] = resolved_mode != mode
                    normalized_result["metadata"] = metadata
                    return normalized_result
                return raw_result
            finally:
                if rag is not None:
                    await self._ingestion_pipeline.finalize_rag_runtime(rag)

        raw_result = await self._ingestion_pipeline.run_in_runtime_loop(
            _run_query_on_runtime()
        )
        return self._normalize_query_result(
            case_root=case_root,
            case_id=resolved_case_id,
            mode=mode,
            raw_result=raw_result,
            case_documents=case_documents,
            retrieval_top_k=retrieval_top_k,
        )

    def build_failure_result(self, *, mode: str, error_message: str) -> ChatQueryResult:
        highlight = {
            "highlight_entities": [],
            "highlight_relationships": [],
            "references": [],
        }
        retrieval_eval = self._empty_retrieval_eval(mode=mode, top_k=5)
        return ChatQueryResult(
            assistant_content=_DEFAULT_ASSISTANT_FAILURE,
            highlight=highlight,
            retrieval_eval=retrieval_eval,
            references=[],
            chunks=[],
            metadata={
                "status": "failed",
                "mode": mode,
                "model_name": self._settings.rag_llm_model,
                "error": error_message,
                "highlight": highlight,
                "retrieval_eval": retrieval_eval,
                "references": [],
                "chunks": [],
            },
        )

    def _normalize_query_result(
        self,
        *,
        case_root: Path,
        case_id: str,
        mode: str,
        raw_result: dict[str, Any] | None,
        case_documents: list[dict[str, Any]],
        retrieval_top_k: int,
    ) -> ChatQueryResult:
        payload = raw_result if isinstance(raw_result, dict) else {}
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        llm_response = (
            payload.get("llm_response")
            if isinstance(payload.get("llm_response"), dict)
            else {}
        )
        payload_metadata = (
            payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        )
        resolved_mode = str(payload_metadata.get("mode") or "").strip().lower() or mode
        references_raw = (
            data.get("references") if isinstance(data.get("references"), list) else []
        )
        chunks_raw = data.get("chunks") if isinstance(data.get("chunks"), list) else []
        entities_raw = self._normalize_raw_rows(
            data.get("entities"),
            fallback_values=[data.get("nodes"), data.get("retrieved_entities")],
        )
        relationships_raw = self._normalize_raw_rows(
            data.get("relationships"),
            fallback_values=[
                data.get("relations"),
                data.get("edges"),
                data.get("retrieved_relations"),
            ],
        )
        resolver = _CaseDocumentResolver(case_documents)
        references = self._normalize_references(references_raw, resolver)
        chunks = self._normalize_chunks(chunks_raw, resolver, references)
        if not references and chunks:
            references = self._references_from_chunks(chunks)

        entity_records = self._extract_entity_records(entities_raw)
        relation_records = self._extract_relation_records(relationships_raw)
        retrieval_entity_records = list(entity_records)
        retrieval_relation_records = list(relation_records)
        if not retrieval_entity_records and not retrieval_relation_records and chunks:
            (
                retrieval_entity_records,
                retrieval_relation_records,
            ) = self._infer_retrieval_records_from_chunks(chunks)

        matched_entities, matched_relationships = self._resolve_graph_highlight(
            case_root=case_root,
            case_id=case_id,
            case_documents=case_documents,
            entity_records=entity_records,
            relation_records=relation_records,
        )
        highlight = {
            "highlight_entities": matched_entities,
            "highlight_relationships": matched_relationships,
            "supporting_chunks": chunks,
            "references": references,
        }
        retrieval_eval = self._build_retrieval_eval(
            mode=resolved_mode,
            top_k=retrieval_top_k,
            entity_records=retrieval_entity_records,
            relation_records=retrieval_relation_records,
        )
        model_name = self._extract_model_name(
            payload=payload,
            llm_response=llm_response,
        )

        assistant_content = str(llm_response.get("content") or "").strip()
        payload_message = str(payload.get("message") or "").strip()
        payload_status = str(payload.get("status") or "success").strip().lower()
        if not assistant_content:
            if payload_status != "success" and self._is_oom_error_text(payload_message):
                assistant_content = _OOM_ASSISTANT_FAILURE
            else:
                assistant_content = payload_message or _DEFAULT_ASSISTANT_FAILURE

        assistant_content = re.sub(
            r"\n+\s*#{1,3}\s*References\s*\n+.*",
            "",
            assistant_content,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        if (
            not retrieval_entity_records
            and not retrieval_relation_records
            and not chunks_raw
        ):
            return ChatQueryResult(
                assistant_content=(
                    "I don't have enough information to answer this question "
                    "based on the available evidence."
                ),
                highlight={
                    "highlight_entities": [],
                    "highlight_relationships": [],
                    "supporting_chunks": [],
                    "references": [],
                },
                retrieval_eval=self._empty_retrieval_eval(
                    mode=resolved_mode, top_k=retrieval_top_k
                ),
                references=[],
                chunks=[],
                metadata={
                    "status": "success",
                    "mode": resolved_mode,
                    "requested_mode": str(
                        payload_metadata.get("requested_mode") or mode
                    )
                    .strip()
                    .lower(),
                    "model_name": model_name,
                    "highlight": {
                        "highlight_entities": [],
                        "highlight_relationships": [],
                        "references": [],
                    },
                    "retrieval_eval": {},
                    "references": [],
                    "chunks": [],
                },
            )

        metadata = {
            "status": str(payload.get("status") or "success"),
            "mode": resolved_mode,
            "query_metadata": payload_metadata,
            "model_name": model_name,
            "highlight": highlight,
            "retrieval_eval": retrieval_eval,
            "references": references,
            "chunks": chunks,
        }
        requested_mode = (
            str(payload_metadata.get("requested_mode") or "").strip().lower()
        )
        if requested_mode:
            metadata["requested_mode"] = requested_mode
        if "fallback_applied" in payload_metadata:
            metadata["fallback_applied"] = bool(
                payload_metadata.get("fallback_applied")
            )
        if isinstance(payload_metadata.get("fallback_attempts"), list):
            metadata["fallback_attempts"] = list(
                payload_metadata.get("fallback_attempts") or []
            )
        if payload_status != "success" and payload_message:
            metadata["error"] = payload_message
        return ChatQueryResult(
            assistant_content=assistant_content,
            highlight=highlight,
            retrieval_eval=retrieval_eval,
            references=references,
            chunks=chunks,
            metadata=metadata,
        )

    @staticmethod
    def _references_from_chunks(chunks: list[dict[str, str]]) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for chunk in chunks:
            reference_id = str(chunk.get("reference_id") or "").strip()
            file_path = str(chunk.get("file_path") or "").strip()
            if not reference_id or not file_path:
                continue
            key = (reference_id, file_path)
            if key in seen:
                continue
            seen.add(key)
            output.append({"reference_id": reference_id, "file_path": file_path})
        return output

    def _resolve_graph_highlight(
        self,
        *,
        case_root: Path,
        case_id: str,
        case_documents: list[dict[str, Any]],
        entity_records: list[dict[str, str]],
        relation_records: list[dict[str, str]],
    ) -> tuple[list[str], list[dict[str, str]]]:
        store = GraphStore(
            case_root=case_root,
            case_id=case_id,
            documents=case_documents,
        )
        payload = store.graph_view(limit=1_000_000)
        return self._match_graph_highlight(
            payload=payload,
            entity_records=entity_records,
            relation_records=relation_records,
        )

    @staticmethod
    def _match_graph_highlight(
        *,
        payload: dict[str, Any],
        entity_records: list[dict[str, str]],
        relation_records: list[dict[str, str]],
    ) -> tuple[list[str], list[dict[str, str]]]:
        node_ids = {
            str(node.get("id") or "").strip()
            for node in payload.get("nodes", [])
            if isinstance(node, dict) and str(node.get("id") or "").strip()
        }

        matched_entities: list[str] = []
        seen_entities: set[str] = set()
        for row in entity_records:
            entity_id = str(row.get("id") or "").strip()
            if not entity_id or entity_id not in node_ids or entity_id in seen_entities:
                continue
            seen_entities.add(entity_id)
            matched_entities.append(entity_id)

        edge_rows = [row for row in payload.get("edges", []) if isinstance(row, dict)]
        edge_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for edge in edge_rows:
            src_id = str(edge.get("src_id") or "").strip()
            tgt_id = str(edge.get("tgt_id") or "").strip()
            if not src_id or not tgt_id:
                continue
            edge_by_pair[(src_id, tgt_id)] = edge
            edge_by_pair[(tgt_id, src_id)] = edge

        matched_relationships: list[dict[str, str]] = []
        seen_relationships: set[tuple[str, str, str]] = set()
        for row in relation_records:
            src_id = str(row.get("src_id") or "").strip()
            tgt_id = str(row.get("tgt_id") or "").strip()
            if not src_id or not tgt_id:
                continue
            edge = edge_by_pair.get((src_id, tgt_id))
            if not edge:
                continue
            actual_src = str(edge.get("src_id") or src_id).strip() or src_id
            actual_tgt = str(edge.get("tgt_id") or tgt_id).strip() or tgt_id
            edge_id = str(edge.get("id") or "").strip()
            key = (actual_src, actual_tgt, edge_id)
            if key in seen_relationships:
                continue
            seen_relationships.add(key)
            matched_row: dict[str, str] = {
                "src_id": actual_src,
                "tgt_id": actual_tgt,
            }
            if edge_id:
                matched_row["edge_id"] = edge_id
            relation_type = str(row.get("type") or "").strip()
            if relation_type:
                matched_row["relation_type"] = relation_type
            matched_relationships.append(matched_row)

        return matched_entities, matched_relationships

    def _extract_model_name(
        self,
        *,
        payload: dict[str, Any],
        llm_response: dict[str, Any],
    ) -> str:
        metadata = (
            payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        )
        candidates = (
            llm_response.get("model"),
            llm_response.get("model_name"),
            payload.get("model"),
            payload.get("model_name"),
            metadata.get("model"),
            metadata.get("model_name"),
            metadata.get("llm_model"),
            metadata.get("primary_model"),
            self._settings.rag_llm_model,
        )
        for candidate in candidates:
            normalized = str(candidate or "").strip()
            if normalized:
                return normalized
        return self._settings.rag_llm_model

    def _build_query_options(
        self,
        options: dict[str, Any],
        *,
        base_limits: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        normalized = dict(options)
        if base_limits is not None:
            defaults = dict(base_limits)
            limits = dict(base_limits)
        else:
            defaults = dict(self._DEFAULT_QUERY_OPTION_LIMITS)
            defaults.setdefault("top_k", self._settings.rag_default_top_k)
            defaults.setdefault("chunk_top_k", self._settings.rag_default_chunk_top_k)
            limits = dict(self._QUERY_OPTION_MAX_LIMITS)
        for key, value in defaults.items():
            if key not in normalized or normalized[key] in (None, "", 0, False):
                normalized[key] = value
            elif isinstance(normalized[key], bool):
                normalized[key] = value
            else:
                try:
                    normalized[key] = max(
                        1,
                        min(int(float(normalized[key])), limits.get(key, value)),
                    )
                except (TypeError, ValueError):
                    normalized[key] = value
        return normalized

    @staticmethod
    def _is_oom_error_text(text: str) -> bool:
        normalized = text.strip().lower()
        if not normalized:
            return False
        return any(pattern in normalized for pattern in _OOM_ERROR_PATTERNS)

    @classmethod
    def _should_retry_after_failure(cls, raw_result: dict[str, Any] | None) -> bool:
        if not isinstance(raw_result, dict):
            return False
        status = str(raw_result.get("status") or "success").strip().lower()
        if status == "success":
            return False
        message = str(raw_result.get("message") or "").strip()
        llm_response = (
            raw_result.get("llm_response")
            if isinstance(raw_result.get("llm_response"), dict)
            else {}
        )
        content = str(llm_response.get("content") or "").strip()
        return cls._is_oom_error_text(message) or cls._is_oom_error_text(content)

    @staticmethod
    def _fallback_modes_for(mode: str) -> tuple[str, ...]:
        normalized = mode.strip().lower()
        if normalized in {"hybrid", "mix", "global"}:
            return ("local", "naive")
        if normalized == "local":
            return ("naive",)
        return ()

    @staticmethod
    def _normalize_retrieval_top_k(raw_top_k: Any) -> int:
        if isinstance(raw_top_k, bool):
            return 5
        if isinstance(raw_top_k, (int, float)):
            value = int(raw_top_k)
            return 5 if value <= 0 else min(value, 100)
        if isinstance(raw_top_k, str):
            stripped = raw_top_k.strip()
            if stripped:
                try:
                    value = int(float(stripped))
                    return 5 if value <= 0 else min(value, 100)
                except ValueError:
                    return 5
        return 5

    @staticmethod
    def _normalize_raw_rows(primary: Any, *, fallback_values: list[Any]) -> list[Any]:
        if isinstance(primary, list):
            return primary
        for value in fallback_values:
            if isinstance(value, list):
                return value
        return []

    @staticmethod
    def _extract_entity_records(entities_raw: list[Any]) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in entities_raw:
            entity_id = ""
            entity_type = ""
            if isinstance(item, dict):
                entity_id = str(
                    item.get("entity_id")
                    or item.get("id")
                    or item.get("node_id")
                    or item.get("entity_name")
                    or item.get("name")
                    or item.get("label")
                    or ""
                ).strip()
                entity_type = (
                    str(
                        item.get("entity_type")
                        or item.get("type")
                        or item.get("node_type")
                        or item.get("category")
                        or ""
                    )
                    .strip()
                    .lower()
                )
            elif isinstance(item, str):
                entity_id = item.strip()
            if not entity_id or entity_id in seen:
                continue
            seen.add(entity_id)
            output.append({"id": entity_id, "type": entity_type})
        return output

    @staticmethod
    def _extract_relation_records(relationships_raw: list[Any]) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in relationships_raw:
            relation_id = ""
            relation_type = ""
            src_id = ""
            tgt_id = ""
            if isinstance(item, dict):
                src_id = str(
                    item.get("src_id")
                    or item.get("source")
                    or item.get("source_id")
                    or item.get("from")
                    or ""
                ).strip()
                tgt_id = str(
                    item.get("tgt_id")
                    or item.get("target")
                    or item.get("target_id")
                    or item.get("to")
                    or ""
                ).strip()
                relation_id = str(
                    item.get("id")
                    or item.get("edge_id")
                    or item.get("relationship_id")
                    or ""
                ).strip()
                keywords = str(item.get("keywords") or "").strip()
                if keywords:
                    relation_type = keywords.split(",")[0].strip().upper()
                else:
                    relation_type = (
                        str(
                            item.get("relation_type")
                            or item.get("rel_type")
                            or item.get("type")
                            or item.get("predicate")
                            or item.get("label")
                            or ""
                        )
                        .strip()
                        .upper()
                    )
                if not relation_id and src_id and tgt_id:
                    seed = relation_type or "REL"
                    relation_id = f"{src_id}__{seed}__{tgt_id}"
                if not relation_type and relation_id and "__" in relation_id:
                    parts = relation_id.split("__")
                    if len(parts) >= 3:
                        relation_type = parts[-2].strip().upper()
            elif isinstance(item, str):
                relation_id = item.strip()
            if not relation_id:
                continue
            if relation_id in seen:
                continue
            seen.add(relation_id)
            output.append(
                {
                    "id": relation_id,
                    "type": relation_type or "REL",
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                }
            )
        return output

    @classmethod
    def _infer_retrieval_records_from_chunks(
        cls, chunks: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        entities: list[dict[str, str]] = []
        relations: list[dict[str, str]] = []
        seen_entities: set[str] = set()
        seen_relations: set[str] = set()

        for chunk in chunks:
            text = str(chunk.get("full_text") or chunk.get("snippet") or "").strip()
            if not text or "," not in text:
                continue
            tokens = [cls._clean_inferred_token(part) for part in text.split(",")]
            tokens = [token for token in tokens if token and not token.isdigit()]
            if len(tokens) < 3:
                continue
            src_id = tokens[0]
            tgt_id = tokens[-1]
            relation_phrase = " ".join(tokens[1:-1])
            relation_type = cls._normalize_inferred_relation_type(relation_phrase)
            for entity_id in (src_id, tgt_id):
                if entity_id and entity_id not in seen_entities:
                    seen_entities.add(entity_id)
                    entities.append({"id": entity_id, "type": ""})
            if src_id and tgt_id and relation_type:
                relation_id = f"{src_id}__{relation_type}__{tgt_id}"
                if relation_id not in seen_relations:
                    seen_relations.add(relation_id)
                    relations.append(
                        {
                            "id": relation_id,
                            "type": relation_type,
                            "src_id": src_id,
                            "tgt_id": tgt_id,
                        }
                    )

        return entities, relations

    @staticmethod
    def _clean_inferred_token(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip(" \t\r\n\"'`.;:")

    @staticmethod
    def _normalize_inferred_relation_type(value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").upper()
        if "SHAREHOLDER" in normalized or "MEMBER" in normalized:
            return "MEMBER_OF"
        if "OWNER" in normalized or "OWNS" in normalized:
            return "OWNS"
        return normalized or "RELATED_TO"

    @staticmethod
    def _empty_retrieval_eval(*, mode: str, top_k: int) -> dict[str, Any]:
        return {
            "mode": mode,
            "top_k": top_k,
            "retrieved_entity_ids_topk": [],
            "retrieved_entity_types_topk": [],
            "retrieved_relation_ids_topk": [],
            "retrieved_relation_types_topk": [],
            "quality_flags": {
                "entities_present": False,
                "relations_present": False,
                "non_empty_payload": False,
            },
        }

    @classmethod
    def _build_retrieval_eval(
        cls,
        *,
        mode: str,
        top_k: int,
        entity_records: list[dict[str, str]],
        relation_records: list[dict[str, str]],
    ) -> dict[str, Any]:
        if top_k <= 0:
            top_k = 5
        top_entities = entity_records[:top_k]
        top_relations = relation_records[:top_k]
        entity_ids = [row["id"] for row in top_entities if row.get("id")]
        entity_types = [row["type"].lower() for row in top_entities if row.get("type")]
        relation_ids = [row["id"] for row in top_relations if row.get("id")]
        relation_types = [
            row["type"].upper() for row in top_relations if row.get("type")
        ]
        return {
            "mode": mode,
            "top_k": top_k,
            "retrieved_entity_ids_topk": entity_ids,
            "retrieved_entity_types_topk": entity_types,
            "retrieved_relation_ids_topk": relation_ids,
            "retrieved_relation_types_topk": relation_types,
            "quality_flags": {
                "entities_present": bool(entity_ids),
                "relations_present": bool(relation_ids),
                "non_empty_payload": bool(entity_ids or relation_ids),
            },
        }

    @staticmethod
    def _normalize_references(
        references_raw: list[dict[str, Any]],
        resolver: "_CaseDocumentResolver",
    ) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in references_raw:
            if not isinstance(item, dict):
                continue
            reference_id = str(item.get("reference_id") or "").strip()
            source_file_path = str(item.get("file_path") or "").strip()
            if not reference_id or not source_file_path:
                continue
            resolved_file_path = resolver.resolve_file_path(source_file_path)
            if not resolved_file_path:
                continue
            key = (reference_id, resolved_file_path)
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "reference_id": reference_id,
                    "file_path": resolved_file_path,
                }
            )
        return output

    @staticmethod
    def _normalize_chunks(
        chunks_raw: list[dict[str, Any]],
        resolver: "_CaseDocumentResolver",
        references: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        ref_ids_by_path = {
            row["file_path"]: row["reference_id"]
            for row in references
            if row.get("file_path")
        }
        seen: set[tuple[str, str, str]] = set()
        for item in chunks_raw:
            if not isinstance(item, dict):
                continue
            source_file_path = str(item.get("file_path") or "").strip()
            if not source_file_path:
                continue
            resolved_file_path = resolver.resolve_file_path(source_file_path)
            if not resolved_file_path:
                continue
            reference_id = str(item.get("reference_id") or "").strip()
            if not reference_id:
                reference_id = ref_ids_by_path.get(resolved_file_path, "")
            if not reference_id:
                continue
            full_text = str(item.get("content") or "").strip()
            snippet = full_text
            if len(snippet) > 480:
                snippet = f"{snippet[:477].rstrip()}..."
            key = (reference_id, resolved_file_path, snippet)
            if key in seen:
                continue
            seen.add(key)
            row = {
                "reference_id": reference_id,
                "file_path": resolved_file_path,
            }
            if snippet:
                row["snippet"] = snippet
            if full_text:
                row["full_text"] = full_text
            output.append(row)
        return output


class _CaseDocumentResolver:
    def __init__(self, case_documents: list[dict[str, Any]]) -> None:
        self._by_key: dict[str, _CaseDocument] = {}
        self._documents: list[_CaseDocument] = []
        for row in case_documents:
            document = _CaseDocument(
                id=str(row.get("id") or ""),
                original_filename=str(row.get("original_filename") or ""),
                stored_file_path=str(row.get("stored_file_path") or ""),
                confidence_code=(
                    str(row["confidence_code"])
                    if row.get("confidence_code") is not None
                    else None
                ),
            )
            if not document.id or not document.stored_file_path:
                continue
            self._documents.append(document)
            for key in self._candidate_keys(document.stored_file_path):
                self._by_key.setdefault(key, document)
            for key in self._candidate_keys(document.original_filename):
                self._by_key.setdefault(key, document)

    def resolve_file_path(self, source_file_path: str) -> str | None:
        candidates = self._split_file_path_candidates(source_file_path)
        for candidate in candidates:
            for key in self._candidate_keys(candidate):
                document = self._by_key.get(key)
                if document is not None:
                    return document.stored_file_path
        return None

    @staticmethod
    def _split_file_path_candidates(value: str) -> list[str]:
        normalized = str(value or "").strip()
        if not normalized:
            return []
        parts = [
            item.strip()
            for item in normalized.split(_LIGHTRAG_FILEPATH_SEP)
            if item and item.strip()
        ]
        if not parts:
            return [normalized]
        return parts

    @staticmethod
    def _candidate_keys(value: str) -> list[str]:
        normalized = str(value or "").replace("\\", "/").strip()
        if not normalized:
            return []
        lowered = normalized.lower()
        parts = [segment for segment in lowered.split("/") if segment]
        basename = parts[-1] if parts else lowered
        keys = {lowered, basename}
        return [item for item in keys if item]
