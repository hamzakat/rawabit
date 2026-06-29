from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from .chat_service import ChatQueryResult, ChatService
from .graph_api import GraphStore
from .prompt_catalog import get_prompt_catalog
from .settings import Settings

ANALYSIS_TYPES = {"link", "event", "flow"}
MAX_ANALYSIS_REPAIR_ATTEMPTS = 5
MAX_ANALYSIS_CHARTS = 3
ANALYSIS_CHART_KINDS = {
    "link": {
        "relationship_map",
        "operational_hierarchy",
        "affiliation_structure",
    },
    "event": {
        "chronological_timeline",
        "event_dependencies",
        "actor_event_matrix",
    },
    "flow": {
        "commodity_flow",
        "activity_flow",
        "event_flow",
    },
}
_FORBIDDEN_MERMAID = (
    "%%{",
    "click ",
    "href",
    "javascript:",
    "<script",
    "<iframe",
    "<img",
    "classdef ",
    "style ",
    "linkstyle ",
)
_SOURCE_MARKER = re.compile(r"^\s*%%\s*source-id:\s*(.+?)\s*$", re.IGNORECASE)
_FLOWCHART_DECLARATION = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*[\[\(\{]")
logger = logging.getLogger(__name__)


class AnalysisProviderUnavailable(RuntimeError):
    pass


class AnalysisService:
    """Builds UNODC-style Mermaid analyses from case RAG results."""

    def __init__(self, settings: Settings, *, chat_service: ChatService | None = None) -> None:
        self._settings = settings
        self._chat_service = chat_service or ChatService(settings)
        self._prompt_catalog = get_prompt_catalog(
            path=settings.prompt_catalog_path,
            auto_reload=settings.prompt_catalog_auto_reload,
        )

    async def generate_analysis(
        self,
        *,
        case_id: str,
        case_root: Path,
        prompt: str,
        analysis_type: str,
        case_documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        normalized_type = self._normalize_analysis_type(analysis_type)
        rag_result = await self._chat_service.query_case_message(
            case_root=case_root,
            case_id=case_id,
            user_content=prompt,
            mode="hybrid",
            conversation_history=[],
            case_documents=case_documents,
            options={
                "top_k": self._settings.rag_default_top_k,
                "chunk_top_k": self._settings.rag_default_chunk_top_k,
            },
            allow_mode_fallback=False,
        )
        highlight = dict(rag_result.highlight or {})
        subgraph = self._build_subgraph(
            case_root=case_root,
            case_id=case_id,
            case_documents=case_documents,
            highlight=highlight,
        )
        context = self._build_generation_context(
            prompt=prompt,
            analysis_type=normalized_type,
            rag_result=rag_result,
            highlight=highlight,
            subgraph=subgraph,
        )
        deadline = time.monotonic() + max(
            0.0, float(self._settings.rag_network_retry_window_seconds)
        )
        bundle = await self._generate_valid_bundle(
            analysis_type=normalized_type,
            context=context,
            initial_prompt=(
                self._prompt_catalog.render(
                    f"analysis.{normalized_type}_projection",
                    context,
                )
                + self._combined_bundle_instruction()
            ),
            deadline=deadline,
        )
        return {
            "analysis_type": normalized_type,
            "prompt": prompt,
            "title": self.build_title(prompt),
            "status": "complete",
            "rag_answer": rag_result.assistant_content,
            "summary_text": bundle["summary_text"],
            "charts": bundle["charts"],
            "highlight": highlight,
            "subgraph": subgraph,
            "references": rag_result.references,
            "chunks": rag_result.chunks,
            "model_name": self._extract_model_name(rag_result),
        }

    async def _generate_valid_bundle(
        self,
        *,
        analysis_type: str,
        context: dict[str, str],
        initial_prompt: str,
        deadline: float,
    ) -> dict[str, Any]:
        raw_bundle = await self._call_llm(
            initial_prompt,
            deadline=deadline,
            operation="Analysis generation",
        )
        for attempt in range(MAX_ANALYSIS_REPAIR_ATTEMPTS + 1):
            payload, parse_error = self._parse_bundle(raw_bundle)
            errors = [parse_error] if parse_error else self._validate_bundle(payload, analysis_type)
            if not errors:
                return {
                    "charts": [
                        {
                            **chart,
                            "repair_attempts": 0,
                        }
                        for chart in payload["charts"]
                    ],
                    "summary_text": str(payload["summary_text"]).strip(),
                }
            if attempt >= MAX_ANALYSIS_REPAIR_ATTEMPTS:
                raise ValueError(
                    "Unable to generate a valid analysis chart bundle after "
                    f"{MAX_ANALYSIS_REPAIR_ATTEMPTS} repair attempts: "
                    + "; ".join(errors)
                )
            logger.warning(
                "Analysis bundle validation failed; repair attempt %s of %s: %s",
                attempt + 1,
                MAX_ANALYSIS_REPAIR_ATTEMPTS,
                "; ".join(errors),
            )
            raw_bundle = await self._call_llm(
                (
                    self._prompt_catalog.render(
                        "analysis.bundle_repair",
                        {
                            **context,
                            "validation_errors": json.dumps(errors, ensure_ascii=True),
                            "rejected_bundle": self._clean_json_output(raw_bundle),
                        },
                    )
                    + self._combined_bundle_instruction()
                ),
                deadline=deadline,
                operation="Analysis contract repair",
            )
        raise ValueError("Unable to generate a valid analysis chart bundle.")

    async def repair_chart(
        self,
        *,
        analysis_type: str,
        chart: dict[str, Any],
        user_prompt: str,
        rag_answer: str,
        graph: dict[str, Any],
        error: str,
        mermaid_code: str,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        repaired = self._clean_mermaid_output(
            await self._call_llm(
                self._prompt_catalog.render(
                    "analysis.mermaid_repair",
                    {
                        "analysis_type": self._normalize_analysis_type(analysis_type),
                        "chart_kind": str(chart.get("kind") or ""),
                        "chart_title": str(chart.get("title") or ""),
                        "item_ids_json": json.dumps(chart.get("item_ids") or [], ensure_ascii=True),
                        "user_prompt": user_prompt,
                        "rag_answer": rag_answer,
                        "graph_json": json.dumps(graph, ensure_ascii=True, default=str),
                        "render_error": error,
                        "mermaid_code": mermaid_code,
                    },
                ),
                deadline=deadline,
                operation="Mermaid chart repair",
            )
        )
        updated = {**chart, "mermaid_code": repaired}
        errors = self._validate_chart(updated, self._normalize_analysis_type(analysis_type))
        if errors:
            raise ValueError("Repaired Mermaid is invalid: " + "; ".join(errors))
        return updated

    async def _call_llm(
        self,
        prompt_text: str,
        *,
        deadline: float | None = None,
        operation: str = "Analyzer LLM request",
    ) -> str:
        if not self._settings.llm_provider_api_key:
            raise RuntimeError("LLM API key is not configured.")
        retry_window = max(
            0.0, float(self._settings.rag_network_retry_window_seconds)
        )
        deadline = deadline if deadline is not None else time.monotonic() + retry_window
        attempt = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if attempt > 0 or retry_window > 0:
                    raise AnalysisProviderUnavailable(
                        "The analysis provider is temporarily unavailable. Retry the analysis later."
                    )
                remaining = max(0.25, float(self._settings.rag_llm_timeout_seconds))
            try:
                return await self._request_llm_once(prompt_text, remaining)
            except Exception as exc:
                if not self._is_network_exception(exc):
                    raise
                attempt += 1
                remaining = deadline - time.monotonic()
                if retry_window <= 0 or remaining <= 0:
                    logger.error(
                        "%s exhausted its network retry budget after %s attempts (%s).",
                        operation,
                        attempt,
                        type(exc).__name__,
                    )
                    raise AnalysisProviderUnavailable(
                        "The analysis provider is temporarily unavailable. Retry the analysis later."
                    ) from exc
                delay = min(30.0, float(2 ** max(0, attempt - 1)), remaining)
                logger.warning(
                    "%s network failure on attempt %s (%s); retrying in %.1fs.",
                    operation,
                    attempt,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)

    async def _request_llm_once(self, prompt_text: str, remaining_seconds: float) -> str:
        try:
            import httpx
            import openai
        except ImportError as exc:
            raise RuntimeError("OpenAI client is unavailable.") from exc

        request_timeout = min(
            max(0.25, remaining_seconds),
            max(0.25, float(self._settings.rag_llm_timeout_seconds)),
        )
        headers: dict[str, str] = {}
        if self._settings.llm_provider_site_url:
            headers["HTTP-Referer"] = self._settings.llm_provider_site_url
        if self._settings.llm_provider_app_name:
            headers["X-Title"] = self._settings.llm_provider_app_name
        client = openai.AsyncOpenAI(
            api_key=self._settings.llm_provider_api_key,
            base_url=self._settings.llm_provider_base_url,
            timeout=httpx.Timeout(
                request_timeout,
                connect=min(30.0, request_timeout),
            ),
            max_retries=0,
            default_headers=headers or None,
        )
        try:
            response = await client.chat.completions.create(
                model=self._settings.rag_llm_model,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=self._settings.rag_llm_max_tokens,
                temperature=0,
            )
            return str(response.choices[0].message.content or "")
        finally:
            await client.close()

    @staticmethod
    def _is_network_exception(exc: Exception) -> bool:
        if isinstance(exc, asyncio.TimeoutError):
            return True
        try:
            from openai import APIConnectionError, APITimeoutError

            if isinstance(exc, (APIConnectionError, APITimeoutError)):
                return True
        except ImportError:
            pass
        try:
            import httpx

            return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))
        except ImportError:
            return False

    @classmethod
    def _validate_bundle(cls, payload: Any, analysis_type: str) -> list[str]:
        if not isinstance(payload, dict):
            return ["Bundle must be a JSON object."]
        summary_text = payload.get("summary_text")
        if not isinstance(summary_text, str) or not summary_text.strip():
            return ["Bundle requires a non-empty summary_text string."]
        charts = payload.get("charts")
        if not isinstance(charts, list) or not charts:
            return ["Bundle requires a non-empty charts array."]
        if len(charts) > MAX_ANALYSIS_CHARTS:
            return [f"Bundle may contain at most {MAX_ANALYSIS_CHARTS} charts."]
        errors: list[str] = []
        chart_ids: set[str] = set()
        for index, chart in enumerate(charts):
            if not isinstance(chart, dict):
                errors.append(f"Chart {index + 1} must be an object.")
                continue
            chart_id = str(chart.get("id") or "").strip()
            if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", chart_id):
                errors.append(f"Chart {index + 1} has an invalid id.")
            elif chart_id in chart_ids:
                errors.append(f"Duplicate chart id: {chart_id}")
            chart_ids.add(chart_id)
            errors.extend(cls._validate_chart(chart, analysis_type))
        return list(dict.fromkeys(errors))

    @classmethod
    def _validate_chart(cls, chart: dict[str, Any], analysis_type: str) -> list[str]:
        errors: list[str] = []
        kind = str(chart.get("kind") or "").strip()
        title = str(chart.get("title") or "").strip()
        code = str(chart.get("mermaid_code") or "").strip()
        item_ids = chart.get("item_ids")
        if kind not in ANALYSIS_CHART_KINDS[analysis_type]:
            errors.append(f"Unsupported {analysis_type} chart kind: {kind or 'empty'}")
        if not title or len(title) > 100:
            errors.append(f"Chart {kind or 'unknown'} requires a concise title.")
        if not isinstance(item_ids, list) or not item_ids or not all(
            isinstance(item, str) and item.strip() for item in item_ids
        ):
            errors.append(f"Chart {kind or 'unknown'} requires non-empty item_ids.")
            item_ids = []
        normalized_ids = [str(item).strip() for item in item_ids]
        if len(set(normalized_ids)) != len(normalized_ids):
            errors.append(f"Chart {kind or 'unknown'} has duplicate item_ids.")
        if not code:
            errors.append(f"Chart {kind or 'unknown'} has no Mermaid code.")
            return errors
        lower_code = code.lower()
        for forbidden in _FORBIDDEN_MERMAID:
            if forbidden in lower_code:
                errors.append(f"Chart {kind or 'unknown'} contains forbidden Mermaid: {forbidden.strip()}")
        first_line = next((line.strip().lower() for line in code.splitlines() if line.strip()), "")
        if kind == "chronological_timeline":
            if first_line != "timeline":
                errors.append("chronological_timeline must use Mermaid timeline syntax.")
        elif not first_line.startswith(("flowchart ", "graph ")):
            errors.append(f"{kind or 'Chart'} must use Mermaid flowchart syntax.")

        markers, labels, declaration_ids = cls._extract_marked_items(code, kind)
        if len(markers) != len(set(markers)):
            errors.append(f"Chart {kind or 'unknown'} repeats source-id markers.")
        if set(markers) != set(normalized_ids):
            errors.append(f"Chart {kind or 'unknown'} source-id markers must match item_ids.")
        normalized_labels = [cls._normalize_label(label) for label in labels]
        normalized_labels = [label for label in normalized_labels if label]
        if len(normalized_labels) != len(set(normalized_labels)):
            errors.append(f"Chart {kind or 'unknown'} contains duplicate normalized labels.")
        if len(declaration_ids) != len(set(declaration_ids)):
            errors.append(f"Chart {kind or 'unknown'} declares a Mermaid node more than once.")
        return errors

    @staticmethod
    def _extract_marked_items(
        code: str, kind: str
    ) -> tuple[list[str], list[str], list[str]]:
        lines = code.splitlines()
        markers: list[str] = []
        labels: list[str] = []
        declaration_ids: list[str] = []
        for index, line in enumerate(lines):
            marker = _SOURCE_MARKER.match(line)
            if not marker:
                continue
            markers.append(marker.group(1).strip())
            declaration = ""
            for candidate in lines[index + 1 :]:
                stripped = candidate.strip()
                if stripped and not stripped.startswith("%%"):
                    declaration = stripped
                    break
            labels.append(declaration)
            if kind != "chronological_timeline":
                match = _FLOWCHART_DECLARATION.match(declaration)
                if match:
                    declaration_ids.append(match.group(1))
        return markers, labels, declaration_ids

    @staticmethod
    def _normalize_label(declaration: str) -> str:
        label = re.sub(r"^[A-Za-z][A-Za-z0-9_]*\s*", "", declaration)
        label = re.sub(r"<br\s*/?>|\\n", " ", label, flags=re.IGNORECASE)
        label = re.sub(r"[^a-z0-9]+", " ", label.lower())
        return re.sub(r"\s+", " ", label).strip()

    @classmethod
    def _parse_bundle(cls, raw_text: str) -> tuple[Any, str | None]:
        text = cls._clean_json_output(raw_text)
        try:
            return json.loads(text), None
        except json.JSONDecodeError as exc:
            return None, f"Invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}."

    @staticmethod
    def _clean_json_output(raw_text: str) -> str:
        text = str(raw_text or "").strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        return text.strip()

    @staticmethod
    def _clean_mermaid_output(raw_text: str) -> str:
        text = str(raw_text or "").strip()
        fence = re.search(r"```(?:mermaid)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        return "\n".join(line.rstrip() for line in text.splitlines()).strip()

    @staticmethod
    def _combined_bundle_instruction() -> str:
        return (
            "\n\n---Combined response requirement---\n"
            "The strict JSON object must contain both `charts` and a non-empty "
            "`summary_text` string. The summary must start with the main inference, "
            "then separate confirmed facts, plausible interpretations, evidence gaps, "
            "and collection questions. Return exactly one JSON object and make no "
            "second narrative call."
        )

    def _build_generation_context(
        self,
        *,
        prompt: str,
        analysis_type: str,
        rag_result: ChatQueryResult,
        highlight: dict[str, Any],
        subgraph: dict[str, Any],
    ) -> dict[str, str]:
        evidence_payload = {
            "highlight": highlight,
            "references": rag_result.references,
            "chunks": self._compact_chunks(rag_result.chunks),
        }
        return {
            "analysis_type": analysis_type,
            "user_prompt": prompt,
            "rag_answer": rag_result.assistant_content,
            "graph_json": json.dumps(subgraph, ensure_ascii=True, default=str),
            "evidence_json": json.dumps(evidence_payload, ensure_ascii=True, default=str),
        }

    @staticmethod
    def _build_subgraph(
        *,
        case_root: Path,
        case_id: str,
        case_documents: list[dict[str, Any]],
        highlight: dict[str, Any],
    ) -> dict[str, Any]:
        store = GraphStore(case_root=case_root, case_id=case_id, documents=case_documents)
        graph = store.graph_view(limit=1_000_000)
        nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
        edges = [edge for edge in graph.get("edges", []) if isinstance(edge, dict)]
        highlighted_entities = {
            str(entity).strip()
            for entity in highlight.get("highlight_entities", [])
            if str(entity).strip()
        }
        highlighted_edge_keys: set[tuple[str, str]] = set()
        highlighted_edge_ids: set[str] = set()
        for item in highlight.get("highlight_relationships", []):
            if not isinstance(item, dict):
                continue
            src_id = str(item.get("src_id") or "").strip()
            tgt_id = str(item.get("tgt_id") or "").strip()
            edge_id = str(item.get("edge_id") or "").strip()
            if src_id and tgt_id:
                highlighted_edge_keys.add((src_id, tgt_id))
                highlighted_entities.update({src_id, tgt_id})
            if edge_id:
                highlighted_edge_ids.add(edge_id)

        selected_edges = [
            edge
            for edge in edges
            if (str(edge.get("src_id") or ""), str(edge.get("tgt_id") or ""))
            in highlighted_edge_keys
            or str(edge.get("id") or "") in highlighted_edge_ids
        ]
        for edge in selected_edges:
            highlighted_entities.update(
                {
                    str(edge.get("src_id") or "").strip(),
                    str(edge.get("tgt_id") or "").strip(),
                }
            )
        selected_nodes = [
            node
            for node in nodes
            if str(node.get("id") or "").strip() in highlighted_entities
            or str(node.get("label") or "").strip() in highlighted_entities
        ]
        if not selected_nodes and not selected_edges:
            selected_nodes = nodes[:25]
            node_ids = {str(node.get("id") or "") for node in selected_nodes}
            selected_edges = [
                edge
                for edge in edges
                if str(edge.get("src_id") or "") in node_ids
                and str(edge.get("tgt_id") or "") in node_ids
            ][:40]
        return {
            "nodes": selected_nodes[:60],
            "edges": selected_edges[:80],
            "truncated": bool(graph.get("truncated")),
        }

    @staticmethod
    def _compact_chunks(chunks: list[dict[str, str]]) -> list[dict[str, str]]:
        output = []
        for chunk in chunks[:12]:
            item = dict(chunk)
            if len(str(item.get("snippet") or "")) > 800:
                item["snippet"] = str(item["snippet"])[:797] + "..."
            if len(str(item.get("full_text") or "")) > 1200:
                item["full_text"] = str(item["full_text"])[:1197] + "..."
            output.append(item)
        return output

    @staticmethod
    def build_title(prompt: str) -> str:
        title = re.sub(r"\s+", " ", prompt).strip()
        if len(title) <= 80:
            return title or "New analysis"
        return title[:77].rstrip() + "..."

    @staticmethod
    def _extract_model_name(rag_result: ChatQueryResult) -> str | None:
        metadata = rag_result.metadata if isinstance(rag_result.metadata, dict) else {}
        model_name = str(metadata.get("model_name") or "").strip()
        return model_name or None

    @staticmethod
    def _normalize_analysis_type(value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in ANALYSIS_TYPES:
            raise ValueError("Analysis type must be one of: link, event, flow.")
        return normalized
