"""
Create a Rawabit case, upload a casepack, and wait for ingestion.

The script is self-contained and accepts any top-level regular files. MIME types
are inferred from filenames; unknown formats use ``application/octet-stream``.
Rawabit remains responsible for deciding whether and how each format is parsed.

Dependency:
    httpx>=0.25

Examples:
    python setup_casepack.py --casepack-dir ../casepack
    python setup_casepack.py --casepack-dir ../casepack --name "Casepack [Eval]"
    python setup_casepack.py --casepack-dir ../casepack \
        --include "*.pdf" --include "*.csv" --exclude "*golden*"

Environment variables:
    RAWABIT_BASE_URL       Default: http://localhost:8000
    RAWABIT_REQUEST_TIMEOUT Default: 120 seconds
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import logging
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import httpx


DEFAULT_BASE_URL = os.environ.get("RAWABIT_BASE_URL", "http://localhost:8000")
DEFAULT_REQUEST_TIMEOUT = float(os.environ.get("RAWABIT_REQUEST_TIMEOUT", "120"))
DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_MAX_POLL_TIME = 6000.0
MAX_CONSECUTIVE_POLL_ERRORS = 5
HASH_CHUNK_SIZE = 1024 * 1024

logger = logging.getLogger(__name__)


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def find_casepack_files(
    casepack_dir: Path,
    include_patterns: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
) -> list[Path]:
    """Return matching top-level, non-hidden regular files in stable order."""
    includes = tuple(include_patterns or ("*",))
    excludes = tuple(exclude_patterns or ())

    files = [
        path
        for path in casepack_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and any(fnmatch.fnmatchcase(path.name, pattern) for pattern in includes)
        and not any(fnmatch.fnmatchcase(path.name, pattern) for pattern in excludes)
    ]
    files.sort(key=lambda path: (path.name.casefold(), path.name))
    return files


def detect_mime_type(file_path: Path) -> str:
    """Infer a MIME type from the filename, with a binary fallback."""
    mime_type, _encoding = mimetypes.guess_type(file_path.name)
    return mime_type or "application/octet-stream"


def sha256_file(file_path: Path) -> str:
    """Calculate a file hash without loading the entire file into memory."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def response_data(response: httpx.Response, operation: str) -> Any:
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"{operation} failed: {payload}")
    return payload["data"]


def create_case(
    name: str,
    description: str,
    client: httpx.Client,
    base_url: str,
) -> dict[str, Any]:
    response = client.post(
        f"{base_url}/api/cases",
        json={"name": name, "description": description},
    )
    return response_data(response, "Case creation")


def upload_document(
    *,
    case_id: str,
    file_path: Path,
    client: httpx.Client,
    base_url: str,
    source_reliability: str,
    information_validity: str,
    ingest_profile: str,
    processing_mode: str,
    tags: str,
    notes_prefix: str,
    allow_duplicate: bool,
    request_timeout: float,
) -> dict[str, Any]:
    mime_type = detect_mime_type(file_path)
    content_hash = sha256_file(file_path)
    notes = f"{notes_prefix} — {file_path.name}" if notes_prefix else file_path.name

    with file_path.open("rb") as handle:
        response = client.post(
            f"{base_url}/api/cases/{case_id}/documents",
            files={"file": (file_path.name, handle, mime_type)},
            data={
                "confidence_source_reliability": source_reliability,
                "confidence_information_validity": information_validity,
                "ingest_profile": ingest_profile,
                "processing_mode": processing_mode,
                "content_hash_sha256": content_hash,
                "allow_duplicate": str(allow_duplicate).lower(),
                "tags": tags,
                "notes": notes,
            },
            timeout=request_timeout,
        )
    return response_data(response, f"Upload of {file_path.name}")


def get_case_jobs(
    case_id: str,
    client: httpx.Client,
    base_url: str,
) -> list[dict[str, Any]]:
    response = client.get(f"{base_url}/api/cases/{case_id}/jobs")
    return response_data(response, "Job listing")


def wait_for_ingestion(
    *,
    case_id: str,
    job_ids: Sequence[str],
    client: httpx.Client,
    base_url: str,
    poll_interval: float,
    max_poll_time: float,
) -> None:
    """Wait until every uploaded job completes, or fail on errors/timeouts."""
    expected_ids = set(job_ids)
    if not expected_ids:
        return

    logger.info(
        "Waiting for %d ingestion job(s) (timeout: %.0fs)",
        len(expected_ids),
        max_poll_time,
    )
    started_at = time.monotonic()
    consecutive_errors = 0
    last_statuses: dict[str, str] = {}

    while time.monotonic() - started_at < max_poll_time:
        try:
            jobs = get_case_jobs(case_id, client, base_url)
            consecutive_errors = 0
        except httpx.TransportError as exc:
            consecutive_errors += 1
            logger.warning(
                "Connection error polling jobs (%d/%d): %s",
                consecutive_errors,
                MAX_CONSECUTIVE_POLL_ERRORS,
                exc,
            )
            if consecutive_errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                raise RuntimeError(
                    "Too many consecutive connection errors while polling "
                    f"case {case_id}. Re-run the script only if a new case is intended."
                ) from exc
            time.sleep(poll_interval * 2)
            continue

        tracked_jobs = {
            str(job.get("id")): job
            for job in jobs
            if str(job.get("id")) in expected_ids
        }
        last_statuses = {
            job_id: str(tracked_jobs.get(job_id, {}).get("status", "not-listed"))
            for job_id in expected_ids
        }

        failed_jobs = [
            tracked_jobs[job_id]
            for job_id, status in last_statuses.items()
            if status == "failed"
        ]
        if failed_jobs:
            details = [
                {
                    "id": job.get("id"),
                    "document_id": job.get("document_id"),
                    "error": job.get("error"),
                }
                for job in failed_jobs
            ]
            raise RuntimeError(f"Ingestion job failed: {details}")

        completed = sum(status == "complete" for status in last_statuses.values())
        logger.info(
            "Jobs: %d/%d complete; statuses=%s",
            completed,
            len(expected_ids),
            last_statuses,
        )
        if completed == len(expected_ids):
            logger.info("All uploaded documents ingested successfully.")
            return

        time.sleep(poll_interval)

    raise TimeoutError(
        f"Ingestion did not complete within {max_poll_time:.0f}s. "
        f"Tracked statuses: {last_statuses}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Rawabit case and upload matching top-level casepack files."
        )
    )
    parser.add_argument(
        "--casepack-dir",
        required=True,
        help="Directory containing the evidence files to upload.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        metavar="GLOB",
        help='Include filename glob; repeatable (default: "*").',
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="GLOB",
        help="Exclude filename glob; repeatable and applied after includes.",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Base case name (default: casepack directory name).",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Case description (default: derived from the directory name).",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model or run label appended to the timestamped case name.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Rawabit API base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--ingest-profile",
        choices=("balanced_fast", "balanced_fast_intel", "full_enrichment"),
        default="balanced_fast_intel",
    )
    parser.add_argument(
        "--processing-mode",
        choices=("multimodal", "text_first"),
        default="multimodal",
    )
    parser.add_argument(
        "--source-reliability",
        choices=("A", "B", "C", "X"),
        default="A",
    )
    parser.add_argument(
        "--information-validity",
        choices=("1", "2", "3", "4"),
        default="1",
    )
    parser.add_argument(
        "--tags",
        default="evaluation",
        help="Comma-separated tags attached to every uploaded file.",
    )
    parser.add_argument(
        "--notes-prefix",
        default="Evaluation casepack",
        help="Prefix for each document note; use an empty value for filenames only.",
    )
    duplicate_group = parser.add_mutually_exclusive_group()
    duplicate_group.add_argument(
        "--allow-duplicate",
        dest="allow_duplicate",
        action="store_true",
        default=True,
        help="Allow duplicate file content within the created case (default).",
    )
    duplicate_group.add_argument(
        "--reject-duplicates",
        dest="allow_duplicate",
        action="store_false",
        help="Reject duplicate file content within the created case.",
    )
    parser.add_argument(
        "--request-timeout",
        type=positive_float,
        default=DEFAULT_REQUEST_TIMEOUT,
        metavar="SECONDS",
    )
    parser.add_argument(
        "--poll-interval",
        type=positive_float,
        default=DEFAULT_POLL_INTERVAL,
        metavar="SECONDS",
    )
    parser.add_argument(
        "--max-ingestion-time",
        type=positive_float,
        default=DEFAULT_MAX_POLL_TIME,
        metavar="SECONDS",
    )
    return parser


def run(args: argparse.Namespace) -> str:
    casepack_dir = Path(args.casepack_dir).expanduser().resolve()
    if not casepack_dir.is_dir():
        raise ValueError(f"Casepack directory does not exist: {casepack_dir}")

    files = find_casepack_files(casepack_dir, args.include, args.exclude)
    if not files:
        raise ValueError(
            f"No matching top-level files found in {casepack_dir}. "
            "Check --include and --exclude patterns."
        )

    base_url = args.base_url.rstrip("/")
    base_name = args.name.strip() or casepack_dir.name
    description = args.description.strip() or (
        f"Evaluation case created from the {casepack_dir.name} casepack."
    )
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.model.strip() or "default"
    case_name = f"{base_name} — {model_slug} — {timestamp}"

    logger.info("Connecting to Rawabit at %s", base_url)
    logger.info("Selected %d file(s): %s", len(files), [path.name for path in files])

    with httpx.Client(timeout=args.request_timeout) as client:
        case = create_case(case_name, description, client, base_url)
        case_id = str(case["id"])
        logger.info(
            "Case created: %s (slug: %s)",
            case_id,
            case.get("slug", "?"),
        )

        job_ids: list[str] = []
        for file_path in files:
            mime_type = detect_mime_type(file_path)
            logger.info("Uploading %s (%s)", file_path.name, mime_type)
            uploaded = upload_document(
                case_id=case_id,
                file_path=file_path,
                client=client,
                base_url=base_url,
                source_reliability=args.source_reliability,
                information_validity=args.information_validity,
                ingest_profile=args.ingest_profile,
                processing_mode=args.processing_mode,
                tags=args.tags,
                notes_prefix=args.notes_prefix,
                allow_duplicate=args.allow_duplicate,
                request_timeout=args.request_timeout,
            )
            job_id = str(uploaded["job_id"])
            job_ids.append(job_id)
            logger.info(
                "Uploaded document %s; job %s",
                uploaded.get("document_id", "?"),
                job_id,
            )

        wait_for_ingestion(
            case_id=case_id,
            job_ids=job_ids,
            client=client,
            base_url=base_url,
            poll_interval=args.poll_interval,
            max_poll_time=args.max_ingestion_time,
        )

    return case_id


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    try:
        case_id = run(build_parser().parse_args())
    except (ValueError, RuntimeError, TimeoutError, httpx.HTTPError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    print(case_id)


if __name__ == "__main__":
    main()
