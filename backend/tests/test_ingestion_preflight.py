from __future__ import annotations

from pathlib import Path
import shutil
import uuid
import zipfile

from backend.app.ingestion_preflight import compute_ingestion_preflight, detect_source_kind


def test_detect_source_kind_supports_mixed_evidence_types() -> None:
    assert detect_source_kind("application/pdf", Path("evidence.pdf")) == "pdf"
    assert detect_source_kind("image/png", Path("photo.png")) == "image"
    assert (
        detect_source_kind(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            Path("report.docx"),
        )
        == "office_document"
    )
    assert detect_source_kind("text/csv", Path("table.csv")) == "text"
    assert detect_source_kind("video/mp4", Path("cam.mp4")) == "video"
    assert detect_source_kind("application/zip", Path("bundle.zip")) == "archive"


def test_compute_ingestion_preflight_for_text_file() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"preflight-text-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        source = temp_dir / "notes.txt"
        source.write_text("alpha\nbravo\ncharlie", encoding="utf-8")
        preflight = compute_ingestion_preflight(
            source_path=source,
            mime_type="text/plain",
            ingest_profile="balanced_fast_intel",
        )
        assert preflight["source_kind"] == "text"
        assert preflight["complexity_class"] in {"small", "medium"}
        assert preflight["eta_seconds"] > 0
        assert preflight["metrics"]["size_bytes"] == source.stat().st_size
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_compute_ingestion_preflight_for_archive_file_counts_entries() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"preflight-archive-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        source = temp_dir / "bundle.zip"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("a.txt", "aaa")
            archive.writestr("b.txt", "bbb")
            archive.writestr("nested/c.txt", "ccc")
        preflight = compute_ingestion_preflight(
            source_path=source,
            mime_type="application/zip",
            ingest_profile="balanced_fast_intel",
        )
        assert preflight["source_kind"] == "archive"
        assert preflight["metrics"]["archive_entries"] == 3
        assert preflight["eta_seconds"] >= 45
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
