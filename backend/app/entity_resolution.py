from __future__ import annotations

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .db import get_connection
from .fs import get_case_lightrag_root, resolve_case_lightrag_dir
from .graph_api import GraphStore
from .settings import Settings

_POLE_TYPE_PREFIX_RE = re.compile(
    r"^\[(PERSON|ORGANIZATION|OBJECT|LOCATION|EVENT)\]\s+", re.IGNORECASE
)


class EntityResolutionService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # Kept for a future analyst-facing merge flow.

    async def amerge_entities(
        self,
        *,
        case_id: str,
        source_entities: list[str],
        target_entity: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        normalized_target = target_entity.strip()
        if not normalized_target:
            raise ValueError("target_entity must not be empty.")

        normalized_sources: list[str] = []
        seen: set[str] = set()
        for value in source_entities:
            item = str(value).strip()
            if not item or item == normalized_target or item in seen:
                continue
            seen.add(item)
            normalized_sources.append(item)
        if not normalized_sources:
            raise ValueError(
                "source_entities must contain at least one entity distinct from target_entity."
            )

        store = self._load_case_graph_store(case_id)
        graph = store.graph_view(limit=1000000)
        existing_ids = {
            str(node.get("id")) for node in graph.get("nodes", []) if node.get("id")
        }
        missing = [
            entity
            for entity in [normalized_target] + normalized_sources
            if entity not in existing_ids
        ]
        if missing:
            raise ValueError(
                f"Unknown entities for this case: {', '.join(sorted(set(missing)))}"
            )

        merge_result = await self._amerge_lightrag_entities(
            case_id=case_id,
            source_entities=normalized_sources,
            target_entity=normalized_target,
        )
        return {
            "source_entities": normalized_sources,
            "target_entity": normalized_target,
            "reason": reason.strip()
            if isinstance(reason, str) and reason.strip()
            else None,
            "merge_result": merge_result,
        }

    def merge_entities(
        self,
        *,
        case_id: str,
        source_entities: list[str],
        target_entity: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return asyncio.run(
            self.amerge_entities(
                case_id=case_id,
                source_entities=source_entities,
                target_entity=target_entity,
                reason=reason,
            )
        )

    async def arun_generic_resolution(
        self,
        *,
        case_id: str,
        client: Any | None = None,
    ) -> dict[str, Any]:
        """
        Generic LLM-powered entity resolution.

        Groups entities by type, sends each group to an LLM to identify
        duplicates/aliases based on names, descriptions, and graph neighbours,
        then auto-merges proposals above the configured confidence threshold.

        Idempotent: already-merged pairs are tracked in a state file and
        skipped on re-runs.
        """
        storage_dir = self._resolve_storage_dir(case_id)
        state_path = storage_dir / ".entity_resolution_state.json"
        resolved_map: dict[str, str] = {}
        if state_path.exists():
            try:
                resolved_map = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        store = self._load_case_graph_store(case_id)
        graph = store.graph_view(limit=1_000_000)
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        if len(nodes) < 2:
            return {
                "merged_count": 0,
                "proposed_count": 0,
                "merges": [],
                "skipped_already_resolved": len(resolved_map),
            }

        # Neighbour links help the model distinguish names that look alike.
        adjacency: dict[str, list[dict[str, str]]] = {}
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            src = str(edge.get("src_id", "")).strip()
            tgt = str(edge.get("tgt_id", "")).strip()
            rel = str(edge.get("relation_type", "")).strip()
            if src and tgt and rel:
                adjacency.setdefault(src, []).append({"id": tgt, "relation": rel})
                adjacency.setdefault(tgt, []).append({"id": src, "relation": rel})

        by_type: dict[str, list[dict[str, Any]]] = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            nid = str(node.get("id", "")).strip()
            if not nid or nid in resolved_map:
                continue
            etype = str(node.get("entity_type", "other")).strip().lower() or "other"
            by_type.setdefault(etype, []).append(node)

        llm_client = client or self._create_llm_client()
        if llm_client is None:
            return {
                "merged_count": 0,
                "proposed_count": 0,
                "merges": [],
                "error": "LLM client unavailable",
            }

        threshold = self._settings.rag_resolution_confidence_threshold
        merged_total = 0
        all_merges: list[dict[str, Any]] = []

        for etype, group in by_type.items():
            if len(group) < 2:
                continue
            inventory = self._build_inventory(group, adjacency)
            if not inventory:
                continue

            prompt_text = self._render_resolution_prompt(inventory)
            try:
                response = await self._call_llm(llm_client, prompt_text)
            except Exception:
                continue

            payload = self._extract_json(response)
            proposals = payload.get("merges", [])
            if not isinstance(proposals, list):
                continue

            for prop in proposals:
                if not isinstance(prop, dict):
                    continue
                keep = str(prop.get("keep", "")).strip()
                merge_into = str(prop.get("merge_into", "")).strip()
                confidence = float(prop.get("confidence", 0))
                if not keep or not merge_into or keep == merge_into:
                    continue
                if confidence < threshold:
                    continue
                if keep in resolved_map or merge_into in resolved_map:
                    continue

                # The graph may have changed while the LLM was running.
                keep_exists = any(
                    str(n.get("id", "")).strip() == keep for n in nodes
                )
                target_exists = any(
                    str(n.get("id", "")).strip() == merge_into for n in nodes
                )
                if not keep_exists or not target_exists:
                    continue

                try:
                    await self._amerge_lightrag_entities(
                        case_id=case_id,
                        source_entities=[keep],
                        target_entity=merge_into,
                    )
                    resolved_map[keep] = merge_into
                    merged_total += 1
                    all_merges.append({
                        "source": keep,
                        "target": merge_into,
                        "confidence": round(confidence, 3),
                        "reasoning": str(prop.get("reasoning", "")),
                    })
                except Exception:
                    continue

        try:
            state_path.write_text(
                json.dumps(resolved_map, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

        return {
            "merged_count": merged_total,
            "proposed_count": len(all_merges),
            "merges": all_merges,
            "skipped_already_resolved": len(resolved_map) - merged_total,
        }

    def run_generic_resolution(
        self,
        *,
        case_id: str,
        client: Any | None = None,
    ) -> dict[str, Any]:
        return asyncio.run(
            self.arun_generic_resolution(case_id=case_id, client=client)
        )

    def _resolve_storage_dir(self, case_id: str) -> Path:
        with get_connection(self._settings) as connection:
            case_row = connection.execute(
                'SELECT case_slug FROM "case" WHERE id = ?', (case_id,)
            ).fetchone()
        if not case_row:
            raise ValueError("Case not found.")
        case_root = self._settings.cases_root / str(case_row["case_slug"])
        return resolve_case_lightrag_dir(case_root, case_id)

    def _build_inventory(
        self,
        group: list[dict[str, Any]],
        adjacency: dict[str, list[dict[str, str]]],
    ) -> str:
        lines: list[str] = []
        for node in group:
            nid = str(node.get("id", "")).strip()
            label = str(node.get("label", nid)).strip()
            desc = str(node.get("description", "") or "").strip()
            if len(desc) > 200:
                desc = desc[:197] + "..."
            etype = str(node.get("entity_type", "other")).strip().lower()
            line = f"  {nid} | type={etype} | name={label}"
            if desc:
                line += f" | description={desc}"
            neighbours = adjacency.get(nid, [])
            if neighbours:
                nb_str = ", ".join(
                    f"{nb['id']}({nb['relation']})"
                    for nb in neighbours[:5]
                )
                line += f" | neighbours=[{nb_str}]"
            lines.append(line)
        return "\n".join(lines)

    def _render_resolution_prompt(self, inventory: str) -> str:
        from .prompt_catalog import get_prompt_catalog

        catalog = get_prompt_catalog(
            path=self._settings.prompt_catalog_path,
            auto_reload=self._settings.prompt_catalog_auto_reload,
        )
        return catalog.render(
            "ingestion.entity_resolution.generic",
            {"entity_inventory": inventory},
        )

    def _create_llm_client(self) -> Any | None:
        try:
            import openai
        except ImportError:
            return None
        api_key = self._settings.llm_provider_api_key
        base_url = self._settings.llm_provider_base_url
        if not api_key:
            return None
        return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def _call_llm(self, client: Any, prompt_text: str) -> str:
        model = self._settings.rag_llm_model
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=min(self._settings.rag_llm_max_tokens, 4000),
            temperature=0,
        )
        choice = response.choices[0]
        return str(choice.message.content or "")

    @staticmethod
    def _extract_json(raw_text: str) -> dict[str, Any]:
        text = str(raw_text or "").strip()
        # The model may wrap JSON in prose, so pull out the first object.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {}

    def _load_case_graph_store(self, case_id: str) -> GraphStore:
        with get_connection(self._settings) as connection:
            case_row = connection.execute(
                'SELECT id, case_slug FROM "case" WHERE id = ?',
                (case_id,),
            ).fetchone()
            if not case_row:
                raise ValueError("Case not found.")
            doc_rows = connection.execute(
                "SELECT id, original_filename, stored_file_path, confidence_code, tags "
                "FROM document WHERE case_id = ?",
                (case_id,),
            ).fetchall()
        case_root = self._settings.cases_root / str(case_row["case_slug"])
        documents = [dict(row) for row in doc_rows]
        return GraphStore(case_root=case_root, case_id=case_id, documents=documents)

    async def _amerge_lightrag_entities(
        self,
        *,
        case_id: str,
        source_entities: list[str],
        target_entity: str,
    ) -> dict[str, Any]:
        with get_connection(self._settings) as connection:
            case_row = connection.execute(
                'SELECT case_slug FROM "case" WHERE id = ?',
                (case_id,),
            ).fetchone()
            if not case_row:
                raise ValueError("Case not found.")
        case_root = self._settings.cases_root / str(case_row["case_slug"])
        storage_dir = resolve_case_lightrag_dir(case_root, case_id)
        if not storage_dir.exists():
            raise ValueError("Case graph workspace does not exist.")

        import numpy as np
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc

        embedding_dim = self._detect_embedding_dim(storage_dir)

        async def _noop_llm(*args, **kwargs) -> str:
            return ""

        async def _noop_embed(texts: list[str]):
            return np.zeros((len(texts), embedding_dim), dtype=float)

        rag = LightRAG(
            working_dir=str(get_case_lightrag_root(case_root)),
            workspace=(
                case_id
                if storage_dir.resolve() != get_case_lightrag_root(case_root).resolve()
                else ""
            ),
            llm_model_func=_noop_llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,
                func=_noop_embed,
            ),
        )
        await rag.initialize_storages()
        try:
            merge_result = await rag.amerge_entities(
                source_entities=source_entities,
                target_entity=target_entity,
                merge_strategy={
                    "description": "concatenate",
                    "entity_type": "keep_first",
                    "source_id": "join_unique",
                },
            )
            self._normalize_merged_entity_type_metadata(storage_dir, target_entity)
            if isinstance(merge_result, dict):
                return merge_result
            return {"result": str(merge_result)}
        finally:
            await rag.finalize_storages()

    @staticmethod
    def _normalize_merged_entity_type_metadata(
        storage_dir: Path, target_entity: str
    ) -> None:
        canonical_type = EntityResolutionService._pole_type_from_entity_id(target_entity)
        if not canonical_type:
            return

        graphml_path = storage_dir / "graph_chunk_entity_relation.graphml"
        if graphml_path.exists():
            try:
                tree = ET.parse(graphml_path)
            except Exception:
                tree = None
            if tree is not None:
                root = tree.getroot()
                key_map: dict[str, str] = {}
                type_key_id: str | None = None
                for key in root.findall(".//{*}key"):
                    key_id = key.attrib.get("id")
                    attr_name = str(key.attrib.get("attr.name") or "").strip().lower()
                    if key_id:
                        key_map[key_id] = key.attrib.get("attr.name", "")
                    if key_id and attr_name in {"entity_type", "type"}:
                        type_key_id = key_id
                updated = False
                for node in root.findall(".//{*}node"):
                    if node.attrib.get("id") != target_entity:
                        continue
                    type_entry = None
                    for entry in node.findall("{*}data"):
                        attr_name = (
                            str(key_map.get(entry.attrib.get("key", "")) or "")
                            .strip()
                            .lower()
                        )
                        if attr_name in {"entity_type", "type"}:
                            type_entry = entry
                            break
                    if type_entry is None and type_key_id:
                        type_entry = ET.SubElement(node, "data", {"key": type_key_id})
                    if type_entry is not None and (type_entry.text or "").strip() != canonical_type:
                        type_entry.text = canonical_type
                        updated = True
                    break
                if updated:
                    tree.write(graphml_path, encoding="utf-8", xml_declaration=True)

        entity_vdb_path = storage_dir / "vdb_entities.json"
        if not entity_vdb_path.exists():
            return
        try:
            payload = json.loads(entity_vdb_path.read_text(encoding="utf-8"))
        except Exception:
            return
        records = payload.get("data")
        if not isinstance(records, list):
            return
        updated = False
        for record in records:
            if not isinstance(record, dict):
                continue
            if str(record.get("entity_name") or "").strip() != target_entity:
                continue
            if str(record.get("entity_type") or "").strip().lower() == canonical_type:
                continue
            record["entity_type"] = canonical_type
            updated = True
        if updated:
            entity_vdb_path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
            )

    @staticmethod
    def _pole_type_from_entity_id(entity_id: str) -> str | None:
        match = _POLE_TYPE_PREFIX_RE.match(str(entity_id or "").strip())
        if not match:
            return None
        return match.group(1).lower()

    @staticmethod
    def _detect_embedding_dim(working_dir: Path) -> int:
        path = working_dir / "vdb_entities.json"
        if not path.exists():
            return 1024
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            dim = payload.get("embedding_dim")
            if isinstance(dim, int) and dim > 0:
                return dim
        except Exception:
            return 1024
        return 1024
