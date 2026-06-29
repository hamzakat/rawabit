from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {".txt", ".md", ".log", ".rtf"}
STRUCTURED_SUFFIXES = {".csv", ".json", ".jsonl", ".xml"}
HTML_SUFFIXES = {".html", ".htm"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
OFFICE_DOC_SUFFIXES = {".doc", ".docx", ".odt"}
PRESENTATION_SUFFIXES = {".ppt", ".pptx", ".odp"}
SPREADSHEET_SUFFIXES = {".xls", ".xlsx", ".ods"}
PDF_SUFFIXES = {".pdf"}
EMAIL_SUFFIXES = {".eml", ".msg"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ARCHIVE_SUFFIXES = {".zip"}


def detect_source_kind(mime_type: str, source_path: Path) -> str:
    mime = (mime_type or "").strip().lower()
    suffix = source_path.suffix.lower()
    if suffix in PDF_SUFFIXES or mime == "application/pdf":
        return "pdf"
    if suffix in IMAGE_SUFFIXES or mime.startswith("image/"):
        return "image"
    if suffix in TEXT_SUFFIXES or mime.startswith("text/"):
        return "text"
    if suffix in STRUCTURED_SUFFIXES:
        return "structured"
    if suffix in HTML_SUFFIXES or "html" in mime:
        return "html"
    if suffix in OFFICE_DOC_SUFFIXES:
        return "office_document"
    if suffix in PRESENTATION_SUFFIXES:
        return "presentation"
    if suffix in SPREADSHEET_SUFFIXES:
        return "spreadsheet"
    if suffix in EMAIL_SUFFIXES or "message/rfc822" in mime:
        return "email"
    if suffix in AUDIO_SUFFIXES or mime.startswith("audio/"):
        return "audio"
    if suffix in VIDEO_SUFFIXES or mime.startswith("video/"):
        return "video"
    if suffix in ARCHIVE_SUFFIXES or mime in {
        "application/zip",
        "application/x-zip-compressed",
    }:
        return "archive"
    return "generic"


def compute_ingestion_preflight(
    source_path: Path, mime_type: str, ingest_profile: str
) -> dict[str, Any]:
    size_bytes = source_path.stat().st_size if source_path.exists() else 0
    size_mb = size_bytes / (1024.0 * 1024.0)
    source_kind = detect_source_kind(mime_type, source_path)
    metrics: dict[str, Any] = {
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 3),
    }

    if source_kind == "pdf":
        page_count = _estimate_pdf_page_count(source_path)
        if page_count is not None:
            metrics["pdf_pages"] = page_count
    elif source_kind in {"structured", "text", "html"}:
        line_count = _estimate_line_count(source_path, max_bytes=2_000_000)
        if line_count is not None:
            metrics["line_count"] = line_count
        if source_kind == "structured":
            if source_path.suffix.lower() in {".json", ".jsonl"}:
                nested_depth = _estimate_json_depth(source_path, max_bytes=500_000)
                if nested_depth is not None:
                    metrics["json_depth"] = nested_depth
    elif source_kind == "archive":
        archive_metrics = _estimate_zip_metrics(source_path)
        metrics.update(archive_metrics)

    complexity_class = _classify_complexity(source_kind=source_kind, metrics=metrics)
    eta_seconds = _estimate_eta_seconds(
        source_kind=source_kind,
        complexity_class=complexity_class,
        ingest_profile=ingest_profile,
    )

    warnings: list[str] = []
    if complexity_class in {"large", "very_large"}:
        warnings.append(
            "This file is likely to take longer than usual. Ingestion will run in background."
        )
    if source_kind == "archive":
        warnings.append("Archive ingestion may recursively process multiple embedded files.")

    return {
        "source_kind": source_kind,
        "mime_type": mime_type or "application/octet-stream",
        "extension": source_path.suffix.lower(),
        "complexity_class": complexity_class,
        "eta_seconds": eta_seconds,
        "metrics": metrics,
        "warnings": warnings,
    }


def _estimate_pdf_page_count(source_path: Path) -> int | None:
    try:
        from pypdf import PdfReader  # type: ignore[import-untyped]

        reader = PdfReader(str(source_path))
        return len(reader.pages)
    except Exception:
        return None


def _estimate_line_count(source_path: Path, max_bytes: int) -> int | None:
    try:
        raw = source_path.read_bytes()[:max_bytes]
        if not raw:
            return 0
        text = raw.decode("utf-8", errors="ignore")
        return max(1, text.count("\n") + 1)
    except Exception:
        return None


def _estimate_json_depth(source_path: Path, max_bytes: int) -> int | None:
    try:
        raw = source_path.read_bytes()[:max_bytes]
        payload = json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception:
        return None

    def _depth(value: Any, current: int = 1) -> int:
        if isinstance(value, dict):
            if not value:
                return current
            return max(_depth(v, current + 1) for v in value.values())
        if isinstance(value, list):
            if not value:
                return current
            return max(_depth(v, current + 1) for v in value)
        return current

    return _depth(payload)


def _estimate_zip_metrics(source_path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(source_path, "r") as archive:
            infos = archive.infolist()
            return {
                "archive_entries": len(infos),
                "archive_uncompressed_bytes": sum(max(0, row.file_size) for row in infos),
            }
    except Exception:
        return {}


def _classify_complexity(source_kind: str, metrics: dict[str, Any]) -> str:
    size_mb = float(metrics.get("size_mb") or 0.0)
    if source_kind == "pdf":
        pages = metrics.get("pdf_pages")
        if isinstance(pages, int):
            if pages <= 20:
                return "small"
            if pages <= 80:
                return "medium"
            if pages <= 220:
                return "large"
            return "very_large"
    if source_kind in {"text", "html", "structured"}:
        if size_mb <= 0.8:
            return "small"
        if size_mb <= 4:
            return "medium"
        if size_mb <= 20:
            return "large"
        return "very_large"
    if source_kind == "image":
        if size_mb <= 3:
            return "small"
        if size_mb <= 10:
            return "medium"
        if size_mb <= 30:
            return "large"
        return "very_large"
    if source_kind in {"audio", "video"}:
        if size_mb <= 25:
            return "small"
        if size_mb <= 120:
            return "medium"
        if size_mb <= 600:
            return "large"
        return "very_large"
    if source_kind == "archive":
        entries = metrics.get("archive_entries")
        if isinstance(entries, int):
            if entries <= 20:
                return "small"
            if entries <= 120:
                return "medium"
            if entries <= 500:
                return "large"
            return "very_large"
    if size_mb <= 2:
        return "small"
    if size_mb <= 15:
        return "medium"
    if size_mb <= 45:
        return "large"
    return "very_large"


def _estimate_eta_seconds(
    source_kind: str, complexity_class: str, ingest_profile: str
) -> int:
    base_map = {
        "small": 90,
        "medium": 210,
        "large": 480,
        "very_large": 960,
    }
    eta = base_map.get(complexity_class, 210)
    if source_kind in {"text", "html"}:
        eta = int(eta * 0.7)
    elif source_kind in {"structured", "image"}:
        eta = int(eta * 0.85)
    elif source_kind in {"audio", "video", "archive"}:
        eta = int(eta * 1.2)
    elif source_kind == "pdf":
        # PDF workflows combine parsing + enrichment + insertion and are usually slower
        # than generic "small file" expectations.
        eta = int(eta * 1.45)

    profile = (ingest_profile or "").strip().lower()
    if profile == "full_enrichment":
        eta = int(eta * 1.6)
    elif profile == "balanced_fast_intel":
        eta = int(eta * 1.2)

    return max(45, eta)
