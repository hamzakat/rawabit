from __future__ import annotations

import json
import os
import shutil
import stat
import time
from pathlib import Path


_LIGHTRAG_WORKSPACE_MARKER = ".rawabit_lightrag_workspace.json"


def get_case_lightrag_root(case_root: Path) -> Path:
    return case_root / "lightrag"


def get_case_lightrag_workspace(case_id: str) -> str:
    return case_id.strip()


def get_case_lightrag_workspace_dir(case_root: Path, case_id: str) -> Path:
    return get_case_lightrag_root(case_root) / get_case_lightrag_workspace(case_id)


def get_case_lightrag_workspace_marker(case_root: Path) -> Path:
    return get_case_lightrag_root(case_root) / _LIGHTRAG_WORKSPACE_MARKER


def _read_case_lightrag_workspace(case_root: Path) -> str | None:
    marker_path = get_case_lightrag_workspace_marker(case_root)
    if not marker_path.exists():
        return None
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    workspace = str(payload.get("workspace") or "").strip()
    return workspace or None


def _write_case_lightrag_workspace(case_root: Path, workspace: str) -> None:
    marker_path = get_case_lightrag_workspace_marker(case_root)
    marker_path.write_text(
        json.dumps({"workspace": workspace}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def case_has_legacy_lightrag_data(case_root: Path) -> bool:
    lightrag_root = get_case_lightrag_root(case_root)
    if not lightrag_root.exists():
        return False
    for name in (
        "graph_chunk_entity_relation.graphml",
        "kv_store_doc_status.json",
        "kv_store_full_docs.json",
        "kv_store_full_entities.json",
        "kv_store_full_relations.json",
        "kv_store_text_chunks.json",
        "kv_store_parse_cache.json",
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
    ):
        if (lightrag_root / name).exists():
            return True
    return False


def ensure_case_lightrag_dir(case_root: Path, case_id: str) -> Path:
    lightrag_root = get_case_lightrag_root(case_root)
    lightrag_root.mkdir(parents=True, exist_ok=True)

    workspace = get_case_lightrag_workspace(case_id)
    stored_workspace = _read_case_lightrag_workspace(case_root)
    if stored_workspace:
        if stored_workspace != workspace:
            raise RuntimeError(
                f"Case LightRAG workspace mismatch: expected '{workspace}', found '{stored_workspace}'."
            )
        workspace_dir = lightrag_root / stored_workspace
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return workspace_dir

    if case_has_legacy_lightrag_data(case_root):
        return lightrag_root

    _write_case_lightrag_workspace(case_root, workspace)
    workspace_dir = lightrag_root / workspace
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return workspace_dir


def resolve_case_lightrag_dir(case_root: Path, case_id: str) -> Path:
    lightrag_root = get_case_lightrag_root(case_root)
    stored_workspace = _read_case_lightrag_workspace(case_root)
    if stored_workspace:
        return lightrag_root / stored_workspace
    return lightrag_root


def create_case_workspace(cases_root: Path, case_slug: str) -> Path:
    workspace_path = cases_root / case_slug
    workspace_path.mkdir(parents=True, exist_ok=False)
    for folder in (
        "raw",
        "processed",
        "lightrag",
        "summaries",
        "analysis",
        "analysis/raw",
        "analysis/link",
        "analysis/event",
        "analysis/flow",
        "exports",
        "tmp",
    ):
        (workspace_path / folder).mkdir(parents=True, exist_ok=True)
    return workspace_path


def _handle_remove_readonly(func, path, exc_info) -> None:
    exc_type, exc, _ = exc_info
    if isinstance(exc, PermissionError):
        os.chmod(path, stat.S_IWRITE)
        func(path)
        return
    raise exc


def delete_case_workspace(cases_root: Path, case_slug: str) -> None:
    workspace_path = cases_root / case_slug
    if not workspace_path.exists():
        return

    attempts = 3
    for attempt in range(attempts):
        try:
            shutil.rmtree(workspace_path, onerror=_handle_remove_readonly)
            return
        except (PermissionError, OSError):
            if attempt == attempts - 1:
                raise
            time.sleep(0.25 * (attempt + 1))
