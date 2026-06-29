from __future__ import annotations

import json
import logging
import re
from typing import Any

from .db import get_connection
from .graph_api import GraphStore
from .prompt_catalog import get_prompt_catalog
from .settings import Settings
from .utils import utc_now_iso

logger = logging.getLogger(__name__)

FIVE_W_ONE_H_KEYS = ("who", "what", "when", "where", "why", "how")


class CaseSummaryService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._prompt_catalog = get_prompt_catalog(
            path=self._settings.prompt_catalog_path,
            auto_reload=self._settings.prompt_catalog_auto_reload,
        )

    def get_case_summary(self, case_id: str) -> dict[str, Any]:
        with get_connection(self._settings) as connection:
            case = self._fetch_case_or_none(connection, case_id)
            if case is None:
                raise ValueError("Case not found")
            cached = self._load_cached_summary(connection, case_id)
        if cached:
            return cached
        return self.refresh_case_summary(case_id)

    def refresh_case_summary(
        self,
        case_id: str,
        *,
        source_job_id: str | None = None,
    ) -> dict[str, Any]:
        with get_connection(self._settings) as connection:
            case = self._fetch_case_or_none(connection, case_id)
            if case is None:
                raise ValueError("Case not found")
            documents = self._list_case_documents(connection, case_id)
            existing = self._load_cached_summary(connection, case_id)
            latest_complete_job_id = self._latest_complete_job_id(connection, case_id)

        resolved_job_id = source_job_id or latest_complete_job_id
        summary = self._build_base_summary(case, documents, resolved_job_id)
        llm_payload = self._generate_summary_with_llm(case, summary)

        if llm_payload is None:
            merged = self._merge_summary(summary, existing or {})
            self._persist_summary(
                case=case,
                summary=merged,
                source_document_count=len(documents),
                source_job_id=resolved_job_id,
            )
            return merged

        merged = self._merge_summary(summary, llm_payload)
        self._persist_summary(
            case=case,
            summary=merged,
            source_document_count=len(documents),
            source_job_id=resolved_job_id,
        )
        return merged

    @staticmethod
    def _fetch_case_or_none(connection, case_id: str) -> dict[str, Any] | None:
        row = connection.execute(
            'SELECT id, name, case_slug, updated_at FROM "case" WHERE id = ?',
            (case_id,),
        ).fetchone()
        if not row:
            return None
        return dict(row)

    @staticmethod
    def _list_case_documents(connection, case_id: str) -> list[dict[str, Any]]:
        rows = connection.execute(
            "SELECT id, original_filename, stored_file_path, confidence_code, tags, ingestion_status, updated_at "
            "FROM document WHERE case_id = ? ORDER BY updated_at DESC",
            (case_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _latest_complete_job_id(connection, case_id: str) -> str | None:
        row = connection.execute(
            "SELECT id FROM ingestion_job WHERE case_id = ? AND status = 'complete' "
            "ORDER BY finished_at DESC LIMIT 1",
            (case_id,),
        ).fetchone()
        return str(row["id"]) if row and row["id"] else None

    @staticmethod
    def _load_cached_summary(connection, case_id: str) -> dict[str, Any] | None:
        row = connection.execute(
            "SELECT summary_json FROM case_summary_cache WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        if not row:
            return None
        payload = row["summary_json"]
        if not isinstance(payload, str) or not payload.strip():
            return None
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _build_base_summary(
        self,
        case: dict[str, Any],
        documents: list[dict[str, Any]],
        source_job_id: str | None,
    ) -> dict[str, Any]:
        case_root = self._settings.cases_root / str(case["case_slug"])
        store = GraphStore(
            case_root=case_root,
            case_id=str(case["id"]),
            documents=documents,
        )
        metrics = store.case_summary_metrics()
        context = store.summary_context()
        completed_document_count = sum(
            1
            for item in documents
            if str(item.get("ingestion_status") or "").lower() in ("complete", "completed_with_warnings")
        )
        latest_activity_at = str(case.get("updated_at") or "").strip() or None
        for item in documents:
            updated_at = str(item.get("updated_at") or "").strip()
            if updated_at and (
                latest_activity_at is None or updated_at > latest_activity_at
            ):
                latest_activity_at = updated_at

        return {
            "case_id": case["id"],
            "case_name": case["name"],
            "case_slug": case["case_slug"],
            "document_count": len(documents),
            "completed_document_count": completed_document_count,
            "evidence_count": len(documents),
            "entity_count": metrics["entity_count"],
            "relationship_count": metrics["relationship_count"],
            "top_entity_types": metrics["top_entity_types"],
            "graph_context": context,
            "latest_activity_at": latest_activity_at,
            "five_w_one_h": {key: None for key in FIVE_W_ONE_H_KEYS},
            "unknowns": [],
            "intelligence_summary": None,
            "investigation_summary": None,
            "summary_text": None,
            "last_refreshed_at": utc_now_iso(),
            "source_job_id": source_job_id,
        }

    def _generate_summary_with_llm(
        self,
        case: dict[str, Any],
        base_summary: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self._settings.llm_provider_api_key:
            return None

        try:
            from openai import OpenAI
        except Exception:
            logger.warning(
                "OpenAI client is unavailable; skipping case summary generation."
            )
            return None

        prompt_text = self._prompt_catalog.render(
            "summary.case_overview",
            {
                "case_name": str(case.get("name") or ""),
                "document_count": base_summary["document_count"],
                "completed_document_count": base_summary["completed_document_count"],
                "evidence_count": base_summary["evidence_count"],
                "entity_count": base_summary["entity_count"],
                "relationship_count": base_summary["relationship_count"],
                "top_entity_types_json": json.dumps(
                    base_summary.get("top_entity_types", []),
                    ensure_ascii=True,
                ),
                "graph_context_json": json.dumps(
                    base_summary.get("graph_context", {}),
                    ensure_ascii=True,
                    default=str,
                ),
            },
        )
        _, _, messages = self._prompt_catalog.apply_external_overrides(
            messages=[{"role": "user", "content": prompt_text}]
        )
        request_messages = messages or [{"role": "user", "content": prompt_text}]

        headers: dict[str, str] = {}
        if self._settings.llm_provider_site_url:
            headers["HTTP-Referer"] = self._settings.llm_provider_site_url
        if self._settings.llm_provider_app_name:
            headers["X-Title"] = self._settings.llm_provider_app_name

        client = OpenAI(
            base_url=self._settings.llm_provider_base_url,
            api_key=self._settings.llm_provider_api_key,
            timeout=float(self._settings.rag_llm_timeout_seconds),
            default_headers=headers or None,
        )

        try:
            response = client.chat.completions.create(
                model=self._settings.rag_llm_model,
                messages=request_messages,
                max_tokens=min(self._settings.rag_llm_max_tokens, 4000),
                temperature=0,
            )
        except Exception:
            logger.warning("Case summary generation failed", exc_info=True)
            return None

        parsed = self._parse_llm_summary_response(
            self._extract_text_from_response(response)
        )
        return parsed

    @staticmethod
    def _extract_text_from_response(response: Any) -> str:
        if not response or not getattr(response, "choices", None):
            return ""
        message = response.choices[0].message
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text_value = getattr(item, "text", None)
                if text_value:
                    parts.append(str(text_value))
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts).strip()
        return str(content or "").strip()

    def _parse_llm_summary_response(self, raw: str) -> dict[str, Any] | None:
        if not raw.strip():
            return None

        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        payload: dict[str, Any] | None = None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, dict):
                        payload = parsed
                except json.JSONDecodeError:
                    payload = None

        if payload is None:
            return None

        intelligence_summary = self._normalize_text(payload.get("intelligence_summary"))
        investigation_summary = self._normalize_text(
            payload.get("investigation_summary")
        )
        five_w_one_h_payload = payload.get("five_w_one_h")
        five_w_one_h: dict[str, str | None] = {key: None for key in FIVE_W_ONE_H_KEYS}
        if isinstance(five_w_one_h_payload, dict):
            for key in FIVE_W_ONE_H_KEYS:
                five_w_one_h[key] = self._normalize_fact_value(
                    five_w_one_h_payload.get(key)
                )
        unknowns = self._normalize_unknowns(payload.get("unknowns"))

        summary_text_parts = [
            value for value in (intelligence_summary, investigation_summary) if value
        ]
        summary_text = (
            "\n\n".join(summary_text_parts).strip() if summary_text_parts else None
        )
        return {
            "intelligence_summary": intelligence_summary,
            "investigation_summary": investigation_summary,
            "five_w_one_h": five_w_one_h,
            "unknowns": unknowns,
            "summary_text": summary_text,
        }

    @staticmethod
    def _normalize_text(value: Any, max_chars: int = 2500) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        return text

    @staticmethod
    def _normalize_fact_value(value: Any) -> str | None:
        text = CaseSummaryService._normalize_text(value, max_chars=500)
        if not text:
            return None
        normalized = text.strip().lower()
        if normalized in {
            "unknown",
            "n/a",
            "none",
            "not available",
            "insufficient evidence",
        }:
            return None
        return text

    @staticmethod
    def _normalize_unknowns(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        output: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(text)
            if len(output) >= 20:
                break
        return output

    def _merge_summary(
        self,
        base: dict[str, Any],
        llm_payload: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(base)
        merged["intelligence_summary"] = llm_payload.get("intelligence_summary")
        merged["investigation_summary"] = llm_payload.get("investigation_summary")
        merged["summary_text"] = llm_payload.get("summary_text")
        merged["five_w_one_h"] = llm_payload.get("five_w_one_h", merged["five_w_one_h"])
        merged["unknowns"] = llm_payload.get("unknowns", [])
        merged["last_refreshed_at"] = utc_now_iso()
        return merged

    def _persist_summary(
        self,
        *,
        case: dict[str, Any],
        summary: dict[str, Any],
        source_document_count: int,
        source_job_id: str | None,
    ) -> None:
        serialized = json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
        refreshed_at = summary.get("last_refreshed_at") or utc_now_iso()
        try:
            with get_connection(self._settings) as connection:
                connection.execute(
                    "INSERT INTO case_summary_cache (case_id, summary_json, source_document_count, source_completed_job_id, last_refreshed_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(case_id) DO UPDATE SET "
                    "summary_json = excluded.summary_json, "
                    "source_document_count = excluded.source_document_count, "
                    "source_completed_job_id = excluded.source_completed_job_id, "
                    "last_refreshed_at = excluded.last_refreshed_at",
                    (
                        case["id"],
                        serialized,
                        int(source_document_count),
                        source_job_id,
                        refreshed_at,
                    ),
                )
        except Exception:
            logger.warning("Failed to persist case summary cache", exc_info=True)

        try:
            summary_path = (
                self._settings.cases_root
                / str(case["case_slug"])
                / "summaries"
                / "case-summary.json"
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.warning("Failed to persist case summary artifact", exc_info=True)

    @staticmethod
    def list_summary_snippets(connection) -> dict[str, str]:
        rows = connection.execute(
            "SELECT case_id, summary_json FROM case_summary_cache"
        ).fetchall()
        snippets: dict[str, str] = {}
        for row in rows:
            case_id = str(row["case_id"])
            payload = row["summary_json"]
            if not isinstance(payload, str) or not payload.strip():
                continue
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            text = parsed.get("summary_text")
            if not isinstance(text, str) or not text.strip():
                continue
            snippet = " ".join(text.strip().split())
            if len(snippet) > 220:
                snippet = snippet[:217].rstrip() + "..."
            snippets[case_id] = snippet
        return snippets
