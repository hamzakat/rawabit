from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .db import get_connection
from .fs import resolve_case_lightrag_dir
from .settings import Settings


SEARCH_SOURCE_VALUES = {"all", "raw", "processed"}
TEXT_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".rtf",
    ".text",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
MAX_RAW_TEXT_BYTES = 5 * 1024 * 1024
MAX_INDEXED_SEGMENT_CHARS = 120_000
MAX_PREVIEW_CHARS = 6000
PREVIEW_CONTEXT_CHARS = 2400
SNIPPET_START = "\u001e"
SNIPPET_END = "\u001f"


def _coerce_preview_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


class DocumentSearchService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def index_raw_document(
        self,
        *,
        case_id: str,
        document_id: str,
        original_filename: str,
        stored_file_path: str,
        confidence_code: str,
        mime_type: str,
        file_path: Path,
    ) -> None:
        text = self._read_raw_text(file_path, mime_type=mime_type)
        self.clear_document_source(
            case_id=case_id,
            document_id=document_id,
            source_kind="raw",
        )
        if not text:
            return
        self._insert_segment(
            case_id=case_id,
            document_id=document_id,
            source_kind="raw",
            segment_key="raw",
            original_filename=original_filename,
            stored_file_path=stored_file_path,
            confidence_code=confidence_code,
            content=text,
        )

    def index_processed_document(
        self,
        *,
        case_id: str,
        document_id: str,
        case_root: Path,
        original_filename: str,
        stored_file_path: str,
        confidence_code: str,
    ) -> None:
        self.clear_document_source(
            case_id=case_id,
            document_id=document_id,
            source_kind="processed",
        )
        segments = self._processed_segments(
            case_root=case_root,
            case_id=case_id,
            document_id=document_id,
        )
        for segment_key, content in segments:
            self._insert_segment(
                case_id=case_id,
                document_id=document_id,
                source_kind="processed",
                segment_key=segment_key,
                original_filename=original_filename,
                stored_file_path=stored_file_path,
                confidence_code=confidence_code,
                content=content,
            )

    def clear_document(
        self,
        *,
        case_id: str,
        document_id: str,
    ) -> None:
        with get_connection(self._settings) as connection:
            connection.execute(
                "DELETE FROM document_search_fts WHERE case_id = ? AND document_id = ?",
                (case_id, document_id),
            )

    def clear_document_source(
        self,
        *,
        case_id: str,
        document_id: str,
        source_kind: str,
    ) -> None:
        with get_connection(self._settings) as connection:
            connection.execute(
                "DELETE FROM document_search_fts "
                "WHERE case_id = ? AND document_id = ? AND source_kind = ?",
                (case_id, document_id, source_kind),
            )

    def ensure_case_index(
        self,
        *,
        case_id: str,
        case_root: Path,
        documents: list[dict[str, Any]],
    ) -> None:
        for document in documents:
            document_id = str(document.get("id") or "").strip()
            stored_file_path = str(document.get("stored_file_path") or "").strip()
            if not document_id or not stored_file_path:
                continue
            original_filename = str(document.get("original_filename") or "").strip()
            confidence_code = str(document.get("confidence_code") or "").strip()
            mime_type = str(document.get("mime_type") or "").strip()

            if not self._has_source_row(
                case_id=case_id,
                document_id=document_id,
                source_kind="raw",
            ):
                try:
                    raw_path = self._resolve_case_file(case_root, stored_file_path)
                except ValueError:
                    raw_path = None
                if raw_path is not None:
                    self.index_raw_document(
                        case_id=case_id,
                        document_id=document_id,
                        original_filename=original_filename,
                        stored_file_path=stored_file_path,
                        confidence_code=confidence_code,
                        mime_type=mime_type,
                        file_path=raw_path,
                    )

            if str(document.get("ingestion_status") or "") not in ("complete", "completed_with_warnings"):
                continue
            if self._has_source_row(
                case_id=case_id,
                document_id=document_id,
                source_kind="processed",
            ):
                continue
            self.index_processed_document(
                case_id=case_id,
                document_id=document_id,
                case_root=case_root,
                original_filename=original_filename,
                stored_file_path=stored_file_path,
                confidence_code=confidence_code,
            )

    def search(
        self,
        *,
        case_id: str,
        query: str,
        source: str = "all",
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        normalized_source = source.strip().lower()
        if normalized_source not in SEARCH_SOURCE_VALUES:
            raise ValueError("Unsupported document search source")
        match_query = self._to_fts_query(query)
        if not match_query:
            return []
        clamped_limit = max(1, min(int(limit), 100))
        with get_connection(self._settings) as connection:
            rows = connection.execute(
                """
                SELECT
                    document_id,
                    source_kind,
                    segment_key,
                    original_filename,
                    stored_file_path,
                    confidence_code,
                    snippet(document_search_fts, 7, ?, ?, '...', 28) AS snippet,
                    bm25(document_search_fts) AS score
                FROM document_search_fts
                WHERE document_search_fts MATCH ?
                    AND case_id = ?
                    AND (? = 'all' OR source_kind = ?)
                ORDER BY score ASC
                LIMIT ?
                """,
                (
                    SNIPPET_START,
                    SNIPPET_END,
                    match_query,
                    case_id,
                    normalized_source,
                    normalized_source,
                    clamped_limit,
                ),
            ).fetchall()
        return [self._format_search_row(row) for row in rows]

    def preview(
        self,
        *,
        case_id: str,
        case_root: Path,
        document: dict[str, Any],
        source_kind: str,
        segment_key: str,
        query: str,
    ) -> dict[str, Any] | None:
        document_id = str(document.get("id") or "").strip()
        if not document_id or str(document.get("case_id") or "").strip() != case_id:
            return None
        normalized_source = source_kind.strip().lower()
        if normalized_source not in {"raw", "processed"}:
            return None

        content = ""
        if normalized_source == "raw":
            stored_file_path = str(document.get("stored_file_path") or "").strip()
            mime_type = str(document.get("mime_type") or "").strip()
            try:
                raw_path = self._resolve_case_file(case_root, stored_file_path)
            except ValueError:
                return None
            content = self._read_raw_text(raw_path, mime_type=mime_type)
        else:
            segments = self._processed_segments(
                case_root=case_root,
                case_id=case_id,
                document_id=document_id,
            )
            for key, value in segments:
                if key == segment_key:
                    content = value
                    break
            if not content and segment_key in {"canonical_text", "preinsert_text"}:
                canonical = case_root / "processed" / document_id / "canonical_text.txt"
                preinsert = case_root / "processed" / document_id / "preinsert_text.txt"
                fallback_path = canonical if canonical.exists() else preinsert
                if fallback_path.exists():
                    try:
                        content = fallback_path.read_text(
                            encoding="utf-8", errors="replace"
                        )
                    except OSError:
                        content = ""

        if not content.strip():
            return None
        window, offset = self._preview_window(content, query)
        ranges = self._match_ranges(window, self._query_tokens(query))
        return {
            "document_id": document_id,
            "original_filename": str(document.get("original_filename") or "").strip(),
            "confidence_code": str(document.get("confidence_code") or "").strip(),
            "source_kind": normalized_source,
            "segment_key": segment_key,
            "content": window,
            "match_ranges": ranges,
            "window_start": offset,
            "truncated": len(window) < len(content),
        }

    def reference_preview(
        self,
        *,
        case_id: str,
        case_root: Path,
        document: dict[str, Any],
        reference_id: str,
        query: str = "",
        snippet: str = "",
    ) -> dict[str, Any] | None:
        normalized_reference = reference_id.strip()
        if not normalized_reference:
            return None

        data = self.preview(
            case_id=case_id,
            case_root=case_root,
            document=document,
            source_kind="processed",
            segment_key=normalized_reference,
            query=query or snippet or normalized_reference,
        )
        if data:
            return data

        data = self.preview(
            case_id=case_id,
            case_root=case_root,
            document=document,
            source_kind="raw",
            segment_key="raw",
            query=query or snippet or normalized_reference,
        )
        if data:
            data["segment_key"] = normalized_reference
            return data

        normalized_snippet = _coerce_preview_text(snippet)
        if not normalized_snippet:
            return None
        window, offset = self._preview_window(
            normalized_snippet,
            query or normalized_snippet,
        )
        ranges = self._match_ranges(
            window, self._query_tokens(query or normalized_snippet)
        )
        return {
            "document_id": str(document.get("id") or "").strip(),
            "original_filename": str(document.get("original_filename") or "").strip(),
            "confidence_code": str(document.get("confidence_code") or "").strip(),
            "source_kind": "processed",
            "segment_key": normalized_reference,
            "content": window,
            "match_ranges": ranges,
            "window_start": offset,
            "truncated": len(window) < len(normalized_snippet),
        }

    def _format_search_row(self, row: Any) -> dict[str, Any]:
        snippet = self._clean_snippet(row["snippet"])
        return {
            "document_id": row["document_id"],
            "original_filename": row["original_filename"],
            "confidence_code": row["confidence_code"],
            "source_kind": row["source_kind"],
            "segment_key": row["segment_key"],
            "stored_file_path": row["stored_file_path"],
            "snippet": snippet,
            "snippet_parts": self._snippet_parts(row["snippet"]),
            "score": float(row["score"]),
        }

    @staticmethod
    def _snippet_parts(value: Any) -> list[dict[str, Any]]:
        text = str(value or "")
        parts: list[dict[str, Any]] = []
        buffer: list[str] = []
        in_match = False

        def flush() -> None:
            nonlocal buffer
            if not buffer:
                return
            text_part = "".join(buffer)
            if text_part:
                parts.append({"text": text_part, "match": in_match})
            buffer = []

        idx = 0
        while idx < len(text):
            char = text[idx]
            if char == SNIPPET_START:
                flush()
                in_match = True
            elif char == SNIPPET_END:
                flush()
                in_match = False
            else:
                buffer.append(char)
            idx += 1
        flush()
        return parts or [
            {"text": DocumentSearchService._clean_snippet(value), "match": False}
        ]

    @staticmethod
    def _query_tokens(query: str) -> list[str]:
        tokens = re.findall(r"[\w]+", query.lower(), flags=re.UNICODE)
        return [token for token in tokens if token][:12]

    @staticmethod
    def _match_ranges(content: str, tokens: list[str]) -> list[dict[str, int]]:
        if not content or not tokens:
            return []
        ranges: list[tuple[int, int]] = []
        lowered = content.lower()
        for token in tokens:
            pattern = re.compile(rf"\b{re.escape(token)}\w*", flags=re.IGNORECASE)
            for match in pattern.finditer(lowered):
                ranges.append((match.start(), match.end()))
        if not ranges:
            return []
        ranges.sort()
        merged: list[tuple[int, int]] = []
        for start, end in ranges:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return [{"start": start, "end": end} for start, end in merged]

    @classmethod
    def _preview_window(cls, content: str, query: str) -> tuple[str, int]:
        if len(content) <= MAX_PREVIEW_CHARS:
            return content, 0
        tokens = cls._query_tokens(query)
        lowered = content.lower()
        first_match = -1
        for token in tokens:
            idx = lowered.find(token)
            if idx >= 0 and (first_match < 0 or idx < first_match):
                first_match = idx
        center = first_match if first_match >= 0 else 0
        start = max(0, center - PREVIEW_CONTEXT_CHARS)
        end = min(len(content), start + MAX_PREVIEW_CHARS)
        start = max(0, end - MAX_PREVIEW_CHARS)
        prefix = "... " if start > 0 else ""
        suffix = " ..." if end < len(content) else ""
        return f"{prefix}{content[start:end]}{suffix}", start

    def _insert_segment(
        self,
        *,
        case_id: str,
        document_id: str,
        source_kind: str,
        segment_key: str,
        original_filename: str,
        stored_file_path: str,
        confidence_code: str,
        content: str,
    ) -> None:
        normalized = " ".join(content.split())
        if not normalized:
            return
        with get_connection(self._settings) as connection:
            connection.execute(
                "INSERT INTO document_search_fts "
                "(case_id, document_id, source_kind, segment_key, original_filename, "
                "stored_file_path, confidence_code, content) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    case_id,
                    document_id,
                    source_kind,
                    segment_key,
                    original_filename,
                    stored_file_path,
                    confidence_code,
                    normalized[:MAX_INDEXED_SEGMENT_CHARS],
                ),
            )

    def _has_source_row(
        self,
        *,
        case_id: str,
        document_id: str,
        source_kind: str,
    ) -> bool:
        with get_connection(self._settings) as connection:
            row = connection.execute(
                "SELECT rowid FROM document_search_fts "
                "WHERE case_id = ? AND document_id = ? AND source_kind = ? LIMIT 1",
                (case_id, document_id, source_kind),
            ).fetchone()
        return row is not None

    @staticmethod
    def _to_fts_query(query: str) -> str:
        tokens = re.findall(r"[\w]+", query.lower(), flags=re.UNICODE)
        tokens = [token for token in tokens if token]
        if not tokens:
            return ""
        return " AND ".join(f"{token}*" for token in tokens[:12])

    @staticmethod
    def _clean_snippet(value: Any) -> str:
        text = (
            str(value or "").replace(SNIPPET_START, "").replace(SNIPPET_END, "").strip()
        )
        return " ".join(text.split())

    @staticmethod
    def _read_raw_text(path: Path, *, mime_type: str) -> str:
        suffix = path.suffix.lower()
        normalized_mime = mime_type.lower()
        is_text = normalized_mime.startswith("text/") or suffix in TEXT_EXTENSIONS
        if not is_text or not path.exists() or path.stat().st_size > MAX_RAW_TEXT_BYTES:
            return ""
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _resolve_case_file(case_root: Path, stored_path: str) -> Path:
        candidate = Path(stored_path)
        resolved = candidate if candidate.is_absolute() else case_root / candidate
        resolved = resolved.resolve()
        case_root_resolved = case_root.resolve()
        if (
            resolved != case_root_resolved
            and case_root_resolved not in resolved.parents
        ):
            raise ValueError("Document path escapes case workspace")
        return resolved

    @staticmethod
    def _processed_segments(
        case_root: Path,
        case_id: str,
        document_id: str,
    ) -> list[tuple[str, str]]:
        output: list[tuple[str, str]] = []
        chunks_path = (
            resolve_case_lightrag_dir(case_root, case_id) / "kv_store_text_chunks.json"
        )
        payload = DocumentSearchService._read_json(chunks_path)
        if isinstance(payload, dict):
            for chunk_id, row in payload.items():
                if not isinstance(row, dict):
                    continue
                if (
                    str(row.get("full_doc_id") or row.get("document_id") or "")
                    != document_id
                ):
                    continue
                content = DocumentSearchService._pick_text(
                    row, ("content", "text", "summary")
                )
                if content:
                    output.append((str(chunk_id), content))
        if output:
            return output

        canonical_path = case_root / "processed" / document_id / "canonical_text.txt"
        preinsert_path = case_root / "processed" / document_id / "preinsert_text.txt"
        fallback_path = canonical_path if canonical_path.exists() else preinsert_path
        if fallback_path.exists():
            try:
                text = fallback_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            if text.strip():
                key = (
                    "canonical_text"
                    if fallback_path == canonical_path
                    else "preinsert_text"
                )
                output.append((key, text))
        return output

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _pick_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
