from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

from .fs import (
    ensure_case_lightrag_dir,
    get_case_lightrag_root,
    resolve_case_lightrag_dir,
)


def _handle_remove_readonly(func, path, exc_info) -> None:
    exc_type, exc, _ = exc_info
    if isinstance(exc, PermissionError):
        os.chmod(path, stat.S_IWRITE)
        func(path)
        return
    raise exc


def _delete_processed_dir(case_root: Path, document_id: str) -> None:
    processed_dir = case_root / "processed" / document_id
    if processed_dir.exists():
        shutil.rmtree(processed_dir, onerror=_handle_remove_readonly)


def _delete_workspace_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=_handle_remove_readonly)


async def _delete_lightrag_doc(
    working_dir: Path,
    storage_dir: Path,
    document_id: str,
    workspace: str | None = None,
) -> tuple[str, str]:
    import numpy as np
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    embedding_dim = _detect_embedding_dim(storage_dir)

    async def _noop_llm(*args, **kwargs) -> str:
        return ""

    async def _noop_embed(texts: list[str]):
        return np.zeros((len(texts), embedding_dim), dtype=float)

    rag = LightRAG(
        working_dir=str(working_dir),
        workspace=(workspace or ""),
        llm_model_func=_noop_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=_noop_embed,
        ),
    )
    await rag.initialize_storages()
    try:
        result = await rag.adelete_by_doc_id(document_id)
        return result.status, result.message
    finally:
        await rag.finalize_storages()


def _detect_embedding_dim(working_dir: Path) -> int:
    vdb = working_dir / "vdb_entities.json"
    if not vdb.exists():
        return 1024
    try:
        payload = json.loads(vdb.read_text(encoding="utf-8"))
        dim = payload.get("embedding_dim")
        if isinstance(dim, int) and dim > 0:
            return dim
    except Exception:
        pass
    return 1024


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _save_json_dict(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _split_source_tokens(value: str) -> set[str]:
    tokens = {value}
    for sep in ("<SEP>", ",", ";", "|"):
        if sep in value:
            for part in value.split(sep):
                stripped = part.strip()
                if stripped:
                    tokens.add(stripped)
    return tokens


def _manual_delete_lightrag_data(working_dir: Path, document_id: str) -> None:
    chunk_ids: set[str] = set()
    file_paths: set[str] = set()

    def _track_row(row: Any) -> None:
        if not isinstance(row, dict):
            return

        file_path = row.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            file_paths.add(file_path)

        chunks = row.get("chunks_list")
        if isinstance(chunks, list):
            for chunk_id in chunks:
                if isinstance(chunk_id, str) and chunk_id:
                    chunk_ids.add(chunk_id)

        row_id = row.get("_id")
        if isinstance(row_id, str) and row_id:
            chunk_ids.add(row_id)

        source_id = row.get("source_id")
        if isinstance(source_id, str) and source_id:
            chunk_ids.update(_split_source_tokens(source_id))

    doc_status_path = working_dir / "kv_store_doc_status.json"
    doc_status = _load_json_dict(doc_status_path)
    if doc_status and document_id in doc_status:
        _track_row(doc_status.pop(document_id))
        _save_json_dict(doc_status_path, doc_status)

    for name in (
        "kv_store_full_docs.json",
        "kv_store_full_entities.json",
        "kv_store_full_relations.json",
    ):
        path = working_dir / name
        payload = _load_json_dict(path)
        if not payload:
            continue

        keys_to_delete: set[str] = set()
        if document_id in payload:
            keys_to_delete.add(document_id)
        for key, row in payload.items():
            if not isinstance(row, dict):
                continue
            if (
                row.get("document_id") == document_id
                or row.get("full_doc_id") == document_id
                or row.get("_id") == document_id
            ):
                keys_to_delete.add(str(key))

        if keys_to_delete:
            for key in keys_to_delete:
                _track_row(payload.pop(key, None))
            _save_json_dict(path, payload)

    text_chunks_path = working_dir / "kv_store_text_chunks.json"
    text_chunks = _load_json_dict(text_chunks_path)
    if text_chunks:
        keys_to_delete: list[str] = []
        for key, row in text_chunks.items():
            if not isinstance(row, dict):
                continue
            if (
                row.get("full_doc_id") == document_id
                or row.get("document_id") == document_id
            ):
                keys_to_delete.append(str(key))
        if keys_to_delete:
            for key in keys_to_delete:
                removed = text_chunks.pop(key, None)
                chunk_ids.add(key)
                _track_row(removed)
            _save_json_dict(text_chunks_path, text_chunks)

    for name in ("kv_store_entity_chunks.json", "kv_store_relation_chunks.json"):
        path = working_dir / name
        payload = _load_json_dict(path)
        if not payload:
            continue

        changed = False
        keys_to_delete: list[str] = []
        for key, row in payload.items():
            if not isinstance(row, dict):
                continue

            remove_row = bool(
                row.get("document_id") == document_id
                or row.get("full_doc_id") == document_id
            )
            row_chunk_ids = row.get("chunk_ids")
            if isinstance(row_chunk_ids, list):
                normalized = [
                    str(item) for item in row_chunk_ids if isinstance(item, str)
                ]
                filtered = [item for item in normalized if item not in chunk_ids]
                if len(filtered) != len(normalized):
                    changed = True
                    if filtered:
                        row["chunk_ids"] = filtered
                        row["count"] = len(filtered)
                    else:
                        remove_row = True

            if remove_row:
                keys_to_delete.append(str(key))
                changed = True

        for key in keys_to_delete:
            payload.pop(key, None)

        if changed:
            _save_json_dict(path, payload)

    def _matches_vdb_record(record: dict[str, Any]) -> bool:
        if (
            record.get("document_id") == document_id
            or record.get("full_doc_id") == document_id
        ):
            return True

        if chunk_ids:
            for key in ("source_id", "chunk_id", "_id", "id", "__id__"):
                value = record.get(key)
                if not isinstance(value, str) or not value:
                    continue
                values = _split_source_tokens(value) if key == "source_id" else {value}
                if any(item in chunk_ids for item in values):
                    return True

            list_ids = record.get("chunk_ids")
            if isinstance(list_ids, list):
                for item in list_ids:
                    if isinstance(item, str) and item in chunk_ids:
                        return True

        if not chunk_ids and file_paths:
            file_path = record.get("file_path")
            if isinstance(file_path, str) and file_path in file_paths:
                return True

        return False

    from nano_vectordb import NanoVectorDB

    for name in ("vdb_chunks.json", "vdb_entities.json", "vdb_relationships.json"):
        path = working_dir / name
        payload = _load_json_dict(path)
        if not payload:
            continue

        records = payload.get("data")
        if not isinstance(records, list):
            continue

        ids_to_delete: list[str] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            if not _matches_vdb_record(record):
                continue
            row_id = record.get("__id__")
            if isinstance(row_id, str) and row_id:
                ids_to_delete.append(row_id)

        if not ids_to_delete:
            continue

        embedding_dim = payload.get("embedding_dim")
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            embedding_dim = _detect_embedding_dim(working_dir)

        vdb = NanoVectorDB(embedding_dim=embedding_dim, storage_file=str(path))
        vdb.delete(ids_to_delete)
        vdb.save()


def _delete_lightrag_data(
    lightrag_root: Path,
    storage_dir: Path,
    document_id: str,
    embedding_dim_hint: int | None = None,
    workspace: str | None = None,
) -> None:
    del embedding_dim_hint
    if not storage_dir.exists():
        return
    if not any(storage_dir.iterdir()):
        return

    try:
        status, message = asyncio.run(
            _delete_lightrag_doc(
                lightrag_root,
                storage_dir,
                document_id,
                workspace=workspace,
            )
        )
    except Exception:
        _manual_delete_lightrag_data(storage_dir, document_id)
        return

    if status in {"success", "not_found"}:
        return

    try:
        _manual_delete_lightrag_data(storage_dir, document_id)
    except Exception as exc:
        raise RuntimeError(
            message or f"LightRAG cleanup failed for doc {document_id}"
        ) from exc


def cleanup_document_artifacts(
    case_root: Path,
    document_id: str,
    embedding_dim_hint: int | None = None,
    case_id: str | None = None,
) -> None:
    _delete_processed_dir(case_root, document_id)
    lightrag_root = get_case_lightrag_root(case_root)
    if case_id:
        storage_dir = ensure_case_lightrag_dir(case_root, case_id)
        workspace = (
            case_id if storage_dir.resolve() != lightrag_root.resolve() else None
        )
    else:
        storage_dir = resolve_case_lightrag_dir(case_root, "")
        workspace = None
    _delete_lightrag_data(
        lightrag_root,
        storage_dir,
        document_id,
        embedding_dim_hint,
        workspace=workspace,
    )


def cleanup_document_graph_state(
    lightrag_root: Path,
    storage_dir: Path,
    document_id: str,
    embedding_dim_hint: int | None = None,
    workspace: str | None = None,
) -> None:
    _delete_lightrag_data(
        lightrag_root,
        storage_dir,
        document_id,
        embedding_dim_hint,
        workspace=workspace,
    )


def cleanup_orphan_lightrag_documents(
    working_dir: Path,
    active_document_ids: set[str],
    embedding_dim_hint: int | None = None,
    workspace: str | None = None,
) -> list[str]:
    if not working_dir.exists() or not any(working_dir.iterdir()):
        return []

    discovered_ids: set[str] = set()

    def _collect_document_ids(
        path: Path,
        include_keys: bool = True,
        tracked_fields: tuple[str, ...] = ("document_id", "full_doc_id", "_id"),
    ) -> None:
        payload = _load_json_dict(path)
        if not payload:
            return
        for key, row in payload.items():
            if include_keys and isinstance(key, str) and key:
                discovered_ids.add(key)
            if not isinstance(row, dict):
                continue
            for field in tracked_fields:
                value = row.get(field)
                if isinstance(value, str) and value:
                    discovered_ids.add(value)

    for name in (
        "kv_store_doc_status.json",
        "kv_store_full_docs.json",
        "kv_store_full_entities.json",
        "kv_store_full_relations.json",
    ):
        _collect_document_ids(working_dir / name)
    _collect_document_ids(
        working_dir / "kv_store_text_chunks.json",
        include_keys=False,
        tracked_fields=("document_id", "full_doc_id"),
    )

    orphan_ids = sorted(
        document_id
        for document_id in discovered_ids
        if document_id and document_id not in active_document_ids
    )
    for document_id in orphan_ids:
        _delete_lightrag_data(
            working_dir.parent if workspace else working_dir,
            working_dir,
            document_id,
            embedding_dim_hint,
            workspace=workspace,
        )
    return orphan_ids


def cleanup_case_ingestion_artifacts(case_root: Path) -> None:
    _delete_workspace_dir(case_root / "lightrag")
    _delete_workspace_dir(case_root / "processed")
