from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any

from .db import get_connection
from .fs import resolve_case_lightrag_dir
from .ontology import (
    CONTROLLED_RELATION_TYPES,
    normalize_entity_type,
    normalize_relation,
    resolve_entity_subtype,
)
from .prompt_catalog import get_prompt_catalog
from .settings import Settings


logger = logging.getLogger(__name__)


GRAPH_FIELD_SEP = "<SEP>"
_POLE_TYPE_PREFIX_RE = re.compile(r"^\[(PERSON|ORGANIZATION|OBJECT|LOCATION|EVENT)\]\s+", re.IGNORECASE)


def _parse_pole_type_prefix(label: str) -> tuple[str, str | None]:
    """If *label* begins with ``[PERSON] `` (or equivalent), return the
    stripped label and the lowercase POLE type.  Otherwise return the
    original label and ``None``."""
    match = _POLE_TYPE_PREFIX_RE.match(label)
    if not match:
        return label, None
    return label[match.end() :], match.group(1).lower()


SEARCH_LIMIT = 25

DATE_PATTERNS = (
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    re.compile(r"\b(\d{4})/(\d{2})/(\d{2})\b"),
    re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b"),
)


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    original_filename: str
    stored_file_path: str
    confidence_code: str | None
    tags: str | None = None


@dataclass
class GraphNode:
    id: str
    label: str
    entity_type: str = "Other"
    summary: str | None = None
    degree: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, str | None]] = field(default_factory=list)


@dataclass
class GraphEdge:
    id: str
    src_id: str
    tgt_id: str
    label: str
    relation_type: str
    relation_raw_phrase: str | None = None
    confidence_score: float | None = None
    confidence_band: str | None = None
    source_ids: list[str] = field(default_factory=list)
    weight: float | None = None
    timestamp: str | None = None
    evidence: list[dict[str, str | None]] = field(default_factory=list)


class GraphStore:
    def __init__(
        self,
        case_root: Path,
        case_id: str,
        documents: list[dict[str, Any]],
    ) -> None:
        self._case_root = case_root.resolve()
        self._case_id = case_id
        self._working_dir = resolve_case_lightrag_dir(case_root, case_id).resolve()
        self._documents = self._build_document_index(documents)
        self._docs_by_id = {doc.id: doc for doc in self._documents}
        self._docs_by_filename = self._build_filename_index(self._documents)
        self._chunk_evidence: dict[str, list[dict[str, str | None]]] = {}
        self._chunk_text_by_id: dict[str, str] = {}
        self._full_doc_text_by_id: dict[str, str] = {}
        self._entity_doc_evidence: dict[str, list[dict[str, str | None]]] = {}
        self._relation_doc_evidence: dict[
            tuple[str, str], list[dict[str, str | None]]
        ] = {}
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._entity_evidence: dict[str, list[dict[str, str | None]]] = {}
        self._relation_evidence: dict[tuple[str, str], list[dict[str, str | None]]] = {}
        self._relation_details: dict[tuple[str, str], dict[str, Any]] = {}
        self._load()

    def _actor_nodes(self) -> dict[str, GraphNode]:
        return self._nodes

    def _actor_edges(self) -> list[GraphEdge]:
        actors = self._actor_nodes()
        return [
            edge
            for edge in self._edges.values()
            if edge.src_id in actors and edge.tgt_id in actors
        ]

    def actor_stats(self) -> dict[str, Any]:
        actors = self._actor_nodes()
        edges = self._actor_edges()
        categories: dict[str, int] = {}
        for edge in edges:
            categories[edge.relation_type] = categories.get(edge.relation_type, 0) + 1
        return {
            "totalDocuments": {"count": len(self._documents)},
            "totalRelationships": {"count": len(edges)},
            "totalActors": {"count": len(actors)},
            "categories": [
                {"category": key, "count": value}
                for key, value in sorted(categories.items())
            ],
        }

    def case_summary_metrics(self) -> dict[str, Any]:
        entity_count = len(self._nodes)
        relationship_count = len(self._edges)
        entity_type_counts: dict[str, int] = {}
        for node in self._nodes.values():
            normalized = normalize_entity_type(node.entity_type) or "other"
            entity_type_counts[normalized] = entity_type_counts.get(normalized, 0) + 1
        top_entity_types = [
            {"entity_type": entity_type, "count": count}
            for entity_type, count in sorted(
                entity_type_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:6]
        ]
        return {
            "entity_count": entity_count,
            "relationship_count": relationship_count,
            "top_entity_types": top_entity_types,
        }

    def summary_context(self) -> dict[str, Any]:
        relation_counts: dict[str, int] = {}
        for edge in self._edges.values():
            relation_counts[edge.relation_type] = (
                relation_counts.get(edge.relation_type, 0) + 1
            )

        top_entities = []
        for node in sorted(
            self._nodes.values(),
            key=lambda n: (-n.degree, n.label.lower()),
        )[:15]:
            top_entities.append({
                "name": node.label,
                "type": node.entity_type or "other",
                "connections": node.degree,
                "summary": (
                    (node.summary or "")[:200] if node.summary else ""
                ),
            })

        poole_counts: dict[str, int] = {"person": 0, "object": 0, "location": 0, "event": 0}
        for node in self._nodes.values():
            t = (node.entity_type or "").lower()
            if t in poole_counts:
                poole_counts[t] += 1

        timestamps = [
            edge.timestamp
            for edge in self._edges.values()
            if edge.timestamp
        ]
        timestamps.sort() if timestamps else None

        return {
            "top_entities": top_entities,
            "top_relation_types": [
                {"type": rtype, "count": count}
                for rtype, count in sorted(
                    relation_counts.items(), key=lambda x: -x[1]
                )[:10]
            ],
            "temporal_scope": {
                "earliest": timestamps[0] if timestamps else None,
                "latest": timestamps[-1] if timestamps else None,
            },
            "pole_breakdown": poole_counts,
            "document_count": len(self._documents),
        }

    def export_entities_csv(self) -> str:
        columns = ("id", "name", "type", "description", "sources")
        rows: list[dict[str, Any]] = []
        for node in sorted(
            self._nodes.values(), key=lambda item: (item.label.lower(), item.id.lower())
        ):
            evidence = self._serialize_relationship_evidence(
                self._dedupe_evidence(
                    list(node.evidence) + self._entity_evidence.get(node.id, [])
                )
            )
            rows.append(
                {
                    "id": node.id,
                    "name": node.label,
                    "type": node.entity_type or "Other",
                    "description": self._clean_entity_description(
                        node.summary, node.label
                    )
                    or "",
                    "sources": self._format_export_sources(evidence),
                }
            )
        return self._write_csv(columns, rows)

    def export_relations_csv(self) -> str:
        columns = (
            "id",
            "source_id",
            "source_name",
            "source_type",
            "target_id",
            "target_name",
            "target_type",
            "relation_type",
            "description",
            "timestamp",
            "confidence_score",
            "confidence_band",
            "sources",
        )
        rows: list[dict[str, Any]] = []
        for edge in sorted(
            self._edges.values(),
            key=lambda item: (
                item.src_id.lower(),
                item.tgt_id.lower(),
                item.relation_type.lower(),
                item.id.lower(),
            ),
        ):
            src = self._nodes.get(edge.src_id)
            tgt = self._nodes.get(edge.tgt_id)
            evidence = self._serialize_relationship_evidence(
                self._dedupe_evidence(
                    list(edge.evidence)
                    + self._relation_evidence.get((edge.src_id, edge.tgt_id), [])
                ),
                source_ids=edge.source_ids,
            )
            rows.append(
                {
                    "id": edge.id,
                    "source_id": edge.src_id,
                    "source_name": src.label if src else "",
                    "source_type": src.entity_type if src else "",
                    "target_id": edge.tgt_id,
                    "target_name": tgt.label if tgt else "",
                    "target_type": tgt.entity_type if tgt else "",
                    "relation_type": edge.relation_type,
                    "description": edge.label,
                    "timestamp": edge.timestamp or "",
                    "confidence_score": (
                        edge.confidence_score
                        if edge.confidence_score is not None
                        else ""
                    ),
                    "confidence_band": edge.confidence_band or "",
                    "sources": self._format_export_sources(evidence),
                }
            )
        return self._write_csv(columns, rows)

    def actor_counts(self, limit: int = 300) -> dict[str, int]:
        actors = self._actor_nodes()
        edges = self._actor_edges()
        degrees: dict[str, int] = {actor_id: 0 for actor_id in actors}
        for edge in edges:
            degrees[edge.src_id] += 1
            degrees[edge.tgt_id] += 1
        ordered = sorted(degrees.items(), key=lambda item: (-item[1], item[0]))[
            : max(1, limit)
        ]
        return {actors[actor_id].label: count for actor_id, count in ordered}

    def search_actors(self, query: str, limit: int = 25) -> list[dict[str, str]]:
        normalized = query.strip().lower()
        if not normalized:
            return []
        actors = self._actor_nodes()
        matches: list[tuple[int, int, str, dict[str, str]]] = []
        for node in actors.values():
            hay = f"{node.label} {node.id}".lower()
            if normalized in hay:
                starts = hay.startswith(normalized)
                idx = hay.find(normalized)
                matches.append(
                    (
                        0 if starts else 1,
                        idx if idx >= 0 else 999999,
                        node.label.lower(),
                        {
                            "name": node.label,
                            "id": node.id,
                            "entity_type": node.entity_type,
                        },
                    )
                )
        matches.sort(key=lambda row: (row[0], row[1], row[2]))
        return [row[3] for row in matches[:limit]]

    def relationships(
        self,
        *,
        limit: int,
        categories: set[str] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        include_undated: bool = True,
        keywords: str | None = None,
        max_hops: int | None = None,
        actor_name: str | None = None,
        clusters: set[str] | None = None,
    ) -> dict[str, Any]:
        actors = self._actor_nodes()
        edges = self._actor_edges()
        raw_node_count = len(self._nodes)
        raw_edge_count = len(self._edges)
        actor_node_count = len(actors)
        actor_edge_count = len(edges)

        cat_filter = {item.upper() for item in categories or set()}
        kw = keywords.strip().lower() if keywords else ""

        if actor_name:
            target_ids = {
                node_id
                for node_id, node in actors.items()
                if node.label == actor_name or node.id == actor_name
            }
            if target_ids:
                allowed = set(target_ids)
                frontier = set(target_ids)
                depth = 0
                max_depth = max(0, max_hops or 0)
                while depth < max_depth:
                    next_frontier: set[str] = set()
                    for edge in edges:
                        if edge.src_id in frontier and edge.tgt_id not in allowed:
                            next_frontier.add(edge.tgt_id)
                        if edge.tgt_id in frontier and edge.src_id not in allowed:
                            next_frontier.add(edge.src_id)
                    if not next_frontier:
                        break
                    allowed.update(next_frontier)
                    frontier = next_frontier
                    depth += 1
                edges = [
                    edge
                    for edge in edges
                    if edge.src_id in allowed and edge.tgt_id in allowed
                ]

        filtered: list[GraphEdge] = []
        for edge in edges:
            if cat_filter and edge.relation_type.upper() not in cat_filter:
                continue
            if kw:
                text = f"{edge.relation_type} {edge.label} {edge.relation_raw_phrase or ''}".lower()
                if kw not in text:
                    continue
            if not include_undated and not edge.timestamp:
                continue
            if edge.timestamp and (year_min or year_max):
                try:
                    year = int(edge.timestamp.split("-")[0])
                except Exception:
                    year = None
                if year is not None:
                    if year_min and year < year_min:
                        continue
                    if year_max and year > year_max:
                        continue
            filtered.append(edge)

        total_before_limit = len(filtered)
        filtered = filtered[: max(1, limit)]

        def _edge_to_relation(edge: GraphEdge) -> dict[str, Any]:
            src = actors.get(edge.src_id)
            tgt = actors.get(edge.tgt_id)
            evidence = self._serialize_relationship_evidence(
                edge.evidence,
                source_ids=edge.source_ids,
            )
            doc_id = next(
                (
                    item.get("document_id")
                    for item in evidence
                    if item.get("document_id")
                ),
                None,
            )
            source_id = next(
                (
                    item.get("source_id") or item.get("reference_id")
                    for item in evidence
                    if item.get("source_id") or item.get("reference_id")
                ),
                None,
            )
            relation_key = (edge.src_id, edge.tgt_id)
            details = self._relation_details.get(relation_key, {})
            description = (
                self._pick_text(details, ("description",))
                or edge.label
                or self._fallback_relationship_description(src, tgt, edge)
            )
            return {
                "id": edge.id,
                "doc_id": doc_id,
                "timestamp": edge.timestamp,
                "actor_id": src.id if src else edge.src_id,
                "actor": src.label if src else edge.src_id,
                "actor_type": src.entity_type if src else "other",
                "actor_summary": src.summary if src else None,
                "action": edge.relation_type,
                "description": description,
                "source_id": source_id,
                "target_id": tgt.id if tgt else edge.tgt_id,
                "target": tgt.label if tgt else edge.tgt_id,
                "target_type": tgt.entity_type if tgt else "other",
                "target_summary": tgt.summary if tgt else None,
                "location": None,
                "tags": self._merge_relationship_tags(edge, evidence),
                "evidence": evidence,
            }

        payload = {
            "relationships": [_edge_to_relation(edge) for edge in filtered],
            "totalBeforeLimit": total_before_limit,
            "totalBeforeFilter": len(edges),
            "actorNodeCount": actor_node_count,
            "actorEdgeCount": actor_edge_count,
            "rawNodeCount": raw_node_count,
            "rawEdgeCount": raw_edge_count,
        }
        if actor_edge_count == 0 and raw_edge_count > 0:
            payload["projectionWarning"] = (
                "Actor-centric relationship projection is empty while the raw graph "
                "still contains nodes/edges. This case may contain usable graph data "
                "outside person-only actor edges."
            )
        return payload

    def _fallback_relationship_description(
        self, src: GraphNode | None, tgt: GraphNode | None, edge: GraphEdge
    ) -> str:
        actor = src.label if src else edge.src_id
        target = tgt.label if tgt else edge.tgt_id
        relation = edge.relation_type.replace("_", " ").lower()
        return f"{actor} {relation} {target}."

    def _merge_relationship_tags(
        self,
        edge: GraphEdge,
        evidence: list[dict[str, str | None]],
    ) -> list[str]:
        values: list[str] = [edge.relation_type]
        for item in evidence:
            document_id = item.get("document_id")
            if not document_id:
                continue
            document = self._docs_by_id.get(document_id)
            if not document or not document.tags:
                continue
            for tag in self._split_tags(document.tags):
                values.append(tag)
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

    def _serialize_relationship_evidence(
        self,
        items: list[dict[str, str | None]],
        *,
        source_ids: list[str] | None = None,
    ) -> list[dict[str, str | None]]:
        output: list[dict[str, str | None]] = []
        ordered_source_ids = [value for value in (source_ids or []) if value]
        for item in self._dedupe_evidence(items):
            source_candidates: list[str] = []
            if item.get("reference_id"):
                source_candidates.append(str(item["reference_id"]))
            source_candidates.extend(ordered_source_ids)

            source_id: str | None = None
            snippet: str | None = None
            for candidate in source_candidates:
                if candidate in self._chunk_text_by_id:
                    source_id = candidate
                    snippet = self._summarize_text(
                        self._chunk_text_by_id[candidate], max_chars=360
                    )
                    break

            if snippet is None:
                document_id = item.get("document_id")
                if document_id and document_id in self._full_doc_text_by_id:
                    source_id = source_id or str(document_id)
                    snippet = self._summarize_text(
                        self._full_doc_text_by_id[document_id], max_chars=360
                    )

            row: dict[str, str | None] = {
                "file_path": item.get("file_path"),
                "reference_id": item.get("reference_id"),
                "document_id": item.get("document_id"),
                "confidence_code": item.get("confidence_code"),
                "source_id": source_id or (item.get("reference_id") or None),
                "snippet": snippet,
            }
            output.append(row)
        return output

    @staticmethod
    def _summarize_text(value: str, *, max_chars: int = 300) -> str | None:
        normalized = " ".join(value.split())
        if not normalized:
            return None
        sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0]
        if len(sentence) > max_chars:
            sentence = sentence[: max_chars - 3].rstrip() + "..."
        return sentence

    @staticmethod
    def _split_tags(value: str) -> list[str]:
        parts = re.split(r"[;,|]", str(value))
        return [part.strip() for part in parts if part.strip()]

    def graph_view(
        self,
        limit: int | None,
        entity_types: set[str] | None = None,
        keyword_filters: set[str] | None = None,
        focus_entity: str | None = None,
        relation_types: set[str] | None = None,
        min_confidence: float | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        max_hops: int | None = None,
    ) -> dict[str, Any]:
        nodes = dict(self._nodes)
        edges = list(self._edges.values())

        if keyword_filters:
            edges = [
                edge
                for edge in edges
                if self._match_keywords(
                    f"{edge.relation_type} {edge.label}", keyword_filters
                )
            ]

        if relation_types:
            allowed_relation_types = {
                value.strip().upper() for value in relation_types if value.strip()
            }
            edges = [
                edge
                for edge in edges
                if edge.relation_type.upper() in allowed_relation_types
            ]

        if min_confidence is not None:
            try:
                threshold = float(min_confidence)
            except (TypeError, ValueError):
                threshold = None
            if threshold is not None:
                edges = [
                    edge
                    for edge in edges
                    if edge.confidence_score is not None
                    and edge.confidence_score >= threshold
                ]

        if date_from or date_to:
            edges = [
                edge
                for edge in edges
                if self._edge_in_date_window(edge, date_from=date_from, date_to=date_to)
            ]

        if entity_types:
            allowed = {value.lower() for value in entity_types}
            nodes = {
                node_id: node
                for node_id, node in nodes.items()
                if node.entity_type.lower() in allowed
            }
            allowed_ids = set(nodes)
            edges = [
                edge
                for edge in edges
                if edge.src_id in allowed_ids and edge.tgt_id in allowed_ids
            ]

        if focus_entity and focus_entity in nodes:
            max_depth = 1 if max_hops is None else max(0, int(max_hops))
            focus_ids = {focus_entity}
            frontier = {focus_entity}
            for _ in range(max_depth):
                next_frontier: set[str] = set()
                for edge in edges:
                    if edge.src_id in frontier and edge.tgt_id not in focus_ids:
                        next_frontier.add(edge.tgt_id)
                    if edge.tgt_id in frontier and edge.src_id not in focus_ids:
                        next_frontier.add(edge.src_id)
                if not next_frontier:
                    break
                focus_ids.update(next_frontier)
                frontier = next_frontier
            nodes = {
                node_id: node for node_id, node in nodes.items() if node_id in focus_ids
            }
            edges = [
                edge
                for edge in edges
                if edge.src_id in focus_ids and edge.tgt_id in focus_ids
            ]

        truncated = False
        effective_limit = DEFAULT_GRAPH_LIMIT if limit is None else max(1, limit)
        if len(nodes) > effective_limit:
            truncated = True
            ordered = sorted(
                nodes.values(),
                key=lambda node: (
                    0 if focus_entity and node.id == focus_entity else 1,
                    -node.degree,
                    node.label.lower(),
                    node.id.lower(),
                ),
            )
            selected_ids = {node.id for node in ordered[:effective_limit]}
            nodes = {
                node_id: node
                for node_id, node in nodes.items()
                if node_id in selected_ids
            }
            edges = [
                edge
                for edge in edges
                if edge.src_id in selected_ids and edge.tgt_id in selected_ids
            ]

        edge_node_ids = {edge.src_id for edge in edges} | {
            edge.tgt_id for edge in edges
        }
        if edge_node_ids:
            for node_id in edge_node_ids:
                if node_id not in nodes and node_id in self._nodes:
                    nodes[node_id] = self._nodes[node_id]

        payload_nodes = [self._serialize_node(node) for node in nodes.values()]
        payload_nodes.sort(
            key=lambda node: (
                -int(node.get("degree", 0)),
                str(node["label"]).lower(),
                str(node["id"]).lower(),
            )
        )
        payload_edges = [self._serialize_edge(edge) for edge in edges]
        payload_edges.sort(
            key=lambda edge: (
                str(edge["src_id"]).lower(),
                str(edge["tgt_id"]).lower(),
                str(edge.get("relation_type", "")).lower(),
                str(edge.get("label", "")).lower(),
            )
        )
        return {
            "nodes": payload_nodes,
            "edges": payload_edges,
            "truncated": truncated,
        }

    def search(self, query: str) -> list[dict[str, Any]]:
        normalized = query.strip().lower()
        if not normalized:
            return []

        default_graph = self.graph_view(limit=DEFAULT_GRAPH_LIMIT)
        matches = []
        for node in default_graph["nodes"]:
            label = str(node.get("label", ""))
            node_id = str(node.get("id", ""))
            haystack = f"{label} {node_id}".lower()
            if normalized not in haystack:
                continue
            label_lower = label.lower()
            node_id_lower = node_id.lower()
            starts = label_lower.startswith(normalized) or node_id_lower.startswith(
                normalized
            )
            idx = haystack.find(normalized)
            matches.append(
                (
                    0 if starts else 1,
                    idx if idx >= 0 else 999999,
                    -int(node.get("degree", 0)),
                    label_lower,
                    {
                        "id": node_id,
                        "label": label,
                        "entity_type": node.get("entity_type", "Other"),
                    },
                )
            )
        matches.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
        return [row[4] for row in matches[:SEARCH_LIMIT]]

    def entity_details(self, entity_id: str) -> dict[str, Any] | None:
        node = self._nodes.get(entity_id)
        if not node:
            return None
        evidence = self._serialize_relationship_evidence(
            self._dedupe_evidence(
                list(node.evidence) + self._entity_evidence.get(entity_id, [])
            )
        )
        payload = self._serialize_node(node)
        payload["evidence"] = evidence
        raw_summary = payload.pop("summary", None)
        related_relationships = self._entity_relationship_context(entity_id)
        payload["related_relationships"] = related_relationships
        payload["raw_description"] = raw_summary
        payload["description"] = self._synthesize_entity_detail_description(
            node=node,
            raw_summary=raw_summary,
            evidence=evidence,
            related_relationships=related_relationships,
        )
        return payload

    def relationship_details(self, src_id: str, tgt_id: str) -> dict[str, Any] | None:
        key = (src_id, tgt_id)
        details = self._relation_details.get(key)
        if not details:
            return None
        edge_source_ids: list[str] = []
        for edge in self._edges.values():
            if edge.src_id == src_id and edge.tgt_id == tgt_id:
                edge_source_ids.extend(edge.source_ids)
        evidence = self._serialize_relationship_evidence(
            self._dedupe_evidence(
                list(details.get("evidence", [])) + self._relation_evidence.get(key, [])
            ),
            source_ids=edge_source_ids,
        )
        relation_types = details.get("relation_types", [])
        description = details.get("description") or ""
        payload: dict[str, Any] = {
            "src_id": src_id,
            "tgt_id": tgt_id,
            "description": description,
            "keywords": relation_types,
            "evidence": evidence,
        }
        if relation_types:
            payload["relation_type"] = relation_types[0]
        if details.get("raw_phrase"):
            payload["relation_raw_phrase"] = details.get("raw_phrase")
        if details.get("confidence_score") is not None:
            payload["confidence_score"] = details.get("confidence_score")
        if details.get("confidence_band"):
            payload["confidence_band"] = details.get("confidence_band")
        payload["raw_description"] = description
        payload["description"] = self._synthesize_relationship_detail_description(
            src_id=src_id,
            tgt_id=tgt_id,
            relation_type=payload.get("relation_type") or "ASSOCIATED_WITH",
            raw_description=description,
            evidence=evidence,
        )
        return payload

    def _entity_relationship_context(self, entity_id: str) -> list[dict[str, str]]:
        related: list[dict[str, str]] = []
        for edge in self._edges.values():
            if edge.src_id != entity_id and edge.tgt_id != entity_id:
                continue
            other_id = edge.tgt_id if edge.src_id == entity_id else edge.src_id
            other = self._nodes.get(other_id)
            related.append(
                {
                    "entity_id": other_id,
                    "entity_label": other.label if other else other_id,
                    "entity_type": other.entity_type if other else "other",
                    "relation_type": edge.relation_type,
                    "description": edge.label,
                }
            )
        related.sort(
            key=lambda item: (
                str(item.get("relation_type") or ""),
                str(item.get("entity_label") or "").lower(),
            )
        )
        return related[:6]

    @staticmethod
    def _synthesize_entity_detail_description(
        *,
        node: "GraphNode",
        raw_summary: str | None,
        evidence: list[dict[str, str | None]],
        related_relationships: list[dict[str, str]],
    ) -> str:
        summary = GraphStore._clean_entity_description(raw_summary, node.label)
        parts: list[str] = []
        if summary:
            parts.append(summary)
        else:
            parts.append(
                f"{node.label} is represented as a {node.entity_type.lower()} entity in this case graph."
            )

        related_labels = [
            item["entity_label"]
            for item in related_relationships
            if item.get("entity_label")
        ]
        if related_labels:
            parts.append(
                "It is linked to "
                + ", ".join(related_labels[:3])
                + (" and others." if len(related_labels) > 3 else ".")
            )

        snippet = next(
            (item.get("snippet") for item in evidence if item.get("snippet")),
            None,
        )
        if snippet:
            parts.append(f"Supporting evidence mentions: {snippet}")
        return " ".join(part.strip() for part in parts if part.strip())

    def _synthesize_relationship_detail_description(
        self,
        *,
        src_id: str,
        tgt_id: str,
        relation_type: str,
        raw_description: str,
        evidence: list[dict[str, str | None]],
    ) -> str:
        src = self._nodes.get(src_id)
        tgt = self._nodes.get(tgt_id)
        base = GraphStore._clean_entity_description(raw_description, None) or ""
        if not base:
            src_label = src.label if src else src_id
            tgt_label = tgt.label if tgt else tgt_id
            base = f"{src_label} is linked to {tgt_label} through {relation_type}."
        snippet = next(
            (item.get("snippet") for item in evidence if item.get("snippet")),
            None,
        )
        if snippet:
            return f"{base} Supporting evidence mentions: {snippet}"
        return base

    def _load(self) -> None:
        if not self._working_dir.exists():
            return
        self._load_chunk_indexes()
        self._load_full_docs_index()
        self._load_full_doc_mappings()
        self._load_nodes()
        self._load_edges()
        self._compute_degrees()

    def _load_nodes(self) -> None:
        node_records: list[dict[str, Any]] = []
        graphml_nodes, _ = self._parse_graphml(
            self._working_dir / "graph_chunk_entity_relation.graphml"
        )
        node_records.extend(graphml_nodes)
        node_records.extend(self._read_records(self._working_dir / "vdb_entities.json"))

        for record in node_records:
            node_id = self._pick_text(
                record, ("entity_name", "id", "node_id", "label", "name")
            )
            if not node_id:
                continue
            label = self._pick_text(record, ("label", "entity_name", "name")) or node_id
            label, prefix_type = _parse_pole_type_prefix(label)
            entity_type = (
                prefix_type
                or normalize_entity_type(
                    self._pick_text(record, ("entity_type", "type"))
                )
                or "object"
            )
            summary = self._extract_node_summary(record)
            summary = self._clean_entity_description(summary, label)
            node = self._nodes.get(node_id)
            if not node:
                node = GraphNode(id=node_id, label=label, entity_type=entity_type)
                self._nodes[node_id] = node
            else:
                if (
                    node.entity_type in {"Other", "UNKNOWN", "unknown", "other"}
                    and entity_type
                ):
                    node.entity_type = entity_type
                if node.label == node.id and label:
                    node.label = label
            if summary and (not node.summary or len(summary) > len(node.summary)):
                node.summary = summary

            evidence = self._collect_evidence(record, entity_id=node_id)
            if evidence:
                node.evidence = self._dedupe_evidence(node.evidence + evidence)
                self._entity_evidence[node_id] = self._dedupe_evidence(
                    self._entity_evidence.get(node_id, []) + evidence
                )

    def _load_edges(self) -> None:
        edge_records: list[dict[str, Any]] = []
        _, graphml_edges = self._parse_graphml(
            self._working_dir / "graph_chunk_entity_relation.graphml"
        )
        edge_records.extend(graphml_edges)
        edge_records.extend(
            self._read_records(self._working_dir / "vdb_relationships.json")
        )

        for record in edge_records:
            src_id = self._pick_text(record, ("src_id", "source", "src", "from"))
            tgt_id = self._pick_text(record, ("tgt_id", "target", "tgt", "to"))
            if not src_id or not tgt_id:
                continue
            relation_type, label = self._extract_relation_type_and_label(
                record, src_id, tgt_id
            )
            normalized = normalize_relation(
                [relation_type, label, record.get("keywords")]
            )
            edge_id = self._edge_id(src_id, tgt_id, normalized.relation_type or label)
            weight = self._to_float(record.get("weight"))
            timestamp = self._extract_timestamp(record)
            evidence = self._collect_evidence(record, relation_pair=(src_id, tgt_id))
            source_ids = self._split_source_ids(record.get("source_id"))

            edge = self._edges.get(edge_id)
            if not edge:
                edge = GraphEdge(
                    id=edge_id,
                    src_id=src_id,
                    tgt_id=tgt_id,
                    label=label,
                    relation_type=normalized.relation_type,
                    relation_raw_phrase=normalized.raw_phrase,
                    confidence_score=normalized.confidence_score,
                    confidence_band=normalized.confidence_band,
                    source_ids=source_ids,
                    weight=weight,
                    timestamp=timestamp,
                    evidence=evidence,
                )
                self._edges[edge_id] = edge
            else:
                if not edge.label and label:
                    edge.label = label
                if edge.relation_type == "ASSOCIATED_WITH" and normalized.relation_type:
                    edge.relation_type = normalized.relation_type
                if edge.relation_raw_phrase is None and normalized.raw_phrase:
                    edge.relation_raw_phrase = normalized.raw_phrase
                if (
                    edge.confidence_score is None
                    and normalized.confidence_score is not None
                ):
                    edge.confidence_score = normalized.confidence_score
                if edge.confidence_band is None and normalized.confidence_band:
                    edge.confidence_band = normalized.confidence_band
                if source_ids:
                    existing = set(edge.source_ids)
                    for source_id in source_ids:
                        if source_id not in existing:
                            edge.source_ids.append(source_id)
                            existing.add(source_id)
                if edge.weight is None and weight is not None:
                    edge.weight = weight
                if edge.timestamp is None and timestamp:
                    edge.timestamp = timestamp
                edge.evidence = self._dedupe_evidence(edge.evidence + evidence)

            for endpoint in (src_id, tgt_id):
                if endpoint not in self._nodes:
                    self._nodes[endpoint] = GraphNode(
                        id=endpoint,
                        label=endpoint,
                        entity_type="Other",
                    )

            pair = (src_id, tgt_id)
            relation_detail = self._relation_details.setdefault(
                pair,
                {
                    "description": "",
                    "relation_types": [],
                    "evidence": [],
                    "raw_phrase": None,
                    "confidence_score": None,
                    "confidence_band": None,
                },
            )
            if label and len(label) > len(relation_detail["description"]):
                relation_detail["description"] = label
            if (
                edge.relation_type
                and edge.relation_type not in relation_detail["relation_types"]
            ):
                relation_detail["relation_types"].append(edge.relation_type)
            if relation_detail.get("raw_phrase") is None and edge.relation_raw_phrase:
                relation_detail["raw_phrase"] = edge.relation_raw_phrase
            if (
                relation_detail.get("confidence_score") is None
                and edge.confidence_score is not None
            ):
                relation_detail["confidence_score"] = edge.confidence_score
            if relation_detail.get("confidence_band") is None and edge.confidence_band:
                relation_detail["confidence_band"] = edge.confidence_band
            relation_detail["evidence"] = self._dedupe_evidence(
                relation_detail["evidence"] + evidence
            )
            self._relation_evidence[pair] = self._dedupe_evidence(
                self._relation_evidence.get(pair, []) + evidence
            )

    def _compute_degrees(self) -> None:
        for node in self._nodes.values():
            node.degree = 0
        for edge in self._edges.values():
            if edge.src_id in self._nodes:
                self._nodes[edge.src_id].degree += 1
            if edge.tgt_id in self._nodes:
                self._nodes[edge.tgt_id].degree += 1

    def _load_chunk_indexes(self) -> None:
        text_chunks = self._read_json(self._working_dir / "kv_store_text_chunks.json")
        if isinstance(text_chunks, dict):
            for chunk_id, raw in text_chunks.items():
                if not isinstance(raw, dict):
                    continue
                chunk_text = self._pick_text(raw, ("content", "text", "summary"))
                if chunk_text:
                    self._chunk_text_by_id[str(chunk_id)] = chunk_text
                evidence = self._collect_evidence(raw, fallback_reference=str(chunk_id))
                if not evidence:
                    continue
                self._append_chunk_evidence(str(chunk_id), evidence)
                chunk_alias = self._pick_text(
                    raw, ("_id", "id", "chunk_id", "source_id")
                )
                if chunk_alias:
                    if chunk_text:
                        self._chunk_text_by_id[chunk_alias] = chunk_text
                    self._append_chunk_evidence(chunk_alias, evidence)

    def _load_full_docs_index(self) -> None:
        payload = self._read_json(self._working_dir / "kv_store_full_docs.json")
        if not isinstance(payload, dict):
            return
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            content = self._pick_text(value, ("content", "text"))
            if content:
                self._full_doc_text_by_id[str(key)] = content
            doc_id = self._pick_text(value, ("document_id", "_id", "id"))
            if doc_id and content:
                self._full_doc_text_by_id[doc_id] = content

    def _load_full_doc_mappings(self) -> None:
        full_entities = self._read_json(
            self._working_dir / "kv_store_full_entities.json"
        )
        if isinstance(full_entities, dict):
            for doc_id, payload in full_entities.items():
                if not isinstance(payload, dict):
                    continue
                evidence = self._evidence_for_document_id(
                    str(doc_id), reference_id=str(doc_id)
                )
                if not evidence:
                    continue
                names = payload.get("entity_names")
                if not isinstance(names, list):
                    continue
                for name in names:
                    entity_name = str(name).strip()
                    if not entity_name:
                        continue
                    self._entity_doc_evidence[entity_name] = self._dedupe_evidence(
                        self._entity_doc_evidence.get(entity_name, []) + evidence
                    )

        full_relations = self._read_json(
            self._working_dir / "kv_store_full_relations.json"
        )
        if isinstance(full_relations, dict):
            for doc_id, payload in full_relations.items():
                if not isinstance(payload, dict):
                    continue
                evidence = self._evidence_for_document_id(
                    str(doc_id), reference_id=str(doc_id)
                )
                if not evidence:
                    continue
                pairs = payload.get("relation_pairs")
                if not isinstance(pairs, list):
                    continue
                for pair in pairs:
                    if not isinstance(pair, list) or len(pair) < 2:
                        continue
                    src = str(pair[0]).strip()
                    tgt = str(pair[1]).strip()
                    if not src or not tgt:
                        continue
                    key = (src, tgt)
                    self._relation_doc_evidence[key] = self._dedupe_evidence(
                        self._relation_doc_evidence.get(key, []) + evidence
                    )

    def _collect_evidence(
        self,
        record: dict[str, Any],
        *,
        entity_id: str | None = None,
        relation_pair: tuple[str, str] | None = None,
        fallback_reference: str | None = None,
    ) -> list[dict[str, str | None]]:
        evidence: list[dict[str, str | None]] = []

        record_reference = (
            self._pick_text(record, ("reference_id",))
            or fallback_reference
            or self._pick_text(record, ("source_id", "__id__", "id"))
        )

        direct = self._evidence_from_record_fields(record, record_reference)
        evidence.extend(direct)

        source_ids = self._split_source_ids(record.get("source_id"))
        for source_id in source_ids:
            evidence.extend(self._chunk_evidence.get(source_id, []))

        if entity_id:
            evidence.extend(self._entity_doc_evidence.get(entity_id, []))
        if relation_pair:
            evidence.extend(self._relation_doc_evidence.get(relation_pair, []))

        return self._dedupe_evidence(evidence)

    def _evidence_from_record_fields(
        self, record: dict[str, Any], reference_id: str | None
    ) -> list[dict[str, str | None]]:
        doc_id = self._pick_text(record, ("document_id", "full_doc_id"))
        file_hint = self._pick_text(record, ("file_path", "original_filename"))
        confidence_hint = self._pick_text(record, ("confidence_code",))
        if doc_id:
            evidence = self._evidence_for_document_id(
                doc_id,
                reference_id=reference_id,
                confidence_hint=confidence_hint,
            )
            if evidence:
                return evidence
        if file_hint:
            evidence = self._evidence_for_file_hint(
                file_hint,
                reference_id=reference_id,
                confidence_hint=confidence_hint,
            )
            if evidence:
                return evidence
        return []

    def _evidence_for_document_id(
        self,
        document_id: str,
        *,
        reference_id: str | None,
        confidence_hint: str | None = None,
    ) -> list[dict[str, str | None]]:
        document = self._docs_by_id.get(document_id)
        if not document:
            return []
        if not self._document_exists(document.stored_file_path):
            return []
        return [
            {
                "file_path": document.stored_file_path,
                "reference_id": reference_id or document_id,
                "document_id": document.id,
                "confidence_code": confidence_hint or document.confidence_code,
            }
        ]

    def _evidence_for_file_hint(
        self,
        file_hint: str,
        *,
        reference_id: str | None,
        confidence_hint: str | None = None,
    ) -> list[dict[str, str | None]]:
        docs = self._docs_by_filename.get(Path(file_hint).name.lower(), [])
        evidence: list[dict[str, str | None]] = []
        for document in docs:
            if not self._document_exists(document.stored_file_path):
                continue
            evidence.append(
                {
                    "file_path": document.stored_file_path,
                    "reference_id": reference_id or file_hint,
                    "document_id": document.id,
                    "confidence_code": confidence_hint or document.confidence_code,
                }
            )
        return evidence

    def _document_exists(self, stored_file_path: str) -> bool:
        try:
            resolved = self._resolve_case_file(stored_file_path)
        except ValueError:
            return False
        return resolved.exists() and resolved.is_file()

    def _resolve_case_file(self, stored_path: str) -> Path:
        candidate = Path(stored_path)
        resolved = candidate if candidate.is_absolute() else self._case_root / candidate
        resolved = resolved.resolve()
        if resolved != self._case_root and self._case_root not in resolved.parents:
            raise ValueError(f"Path escapes case workspace: {stored_path}")
        return resolved

    def _append_chunk_evidence(
        self, chunk_id: str, evidence: list[dict[str, str | None]]
    ) -> None:
        existing = self._chunk_evidence.get(chunk_id, [])
        self._chunk_evidence[chunk_id] = self._dedupe_evidence(existing + evidence)

    @staticmethod
    def _match_keywords(text: str, filters: set[str]) -> bool:
        normalized = text.lower()
        return any(filter_value in normalized for filter_value in filters)

    @staticmethod
    def _extract_node_summary(record: dict[str, Any]) -> str | None:
        description = GraphStore._pick_text(
            record, ("description", "summary", "content")
        )
        if not description:
            return None
        if "\t" in description and "\n" in description:
            return None
        return description

    @staticmethod
    def _clean_entity_description(
        description: str | None, label: str | None
    ) -> str | None:
        if description is None:
            return None
        cleaned = str(description).strip()
        if not cleaned:
            return None
        label_text = (label or "").strip()
        if not label_text:
            return cleaned

        escaped_label = re.escape(label_text)
        separators = r"[\s:;,\-–—]+"

        duplicate_patterns = (
            rf"^(?P<head>{escaped_label}){separators}(?P<rest>{escaped_label}\b.*)$",
            rf"^(?P<head>{escaped_label}){separators}(?P<rest>(?:the|a|an)\s+{escaped_label}\b.*)$",
            rf"^(?P<head>(?:the|a|an)\s+{escaped_label}){separators}(?P<rest>(?:the|a|an)\s+{escaped_label}\b.*)$",
        )
        for pattern in duplicate_patterns:
            match = re.match(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                deduped = match.group("rest").strip()
                return deduped or cleaned
        return cleaned

    @staticmethod
    def _extract_relation_type_and_label(
        record: dict[str, Any], src_id: str, tgt_id: str
    ) -> tuple[str, str]:
        relation_type = GraphStore._pick_text(
            record, ("keywords", "relation_type", "type", "label")
        )
        description = GraphStore._pick_text(record, ("description", "summary"))
        content = GraphStore._pick_text(record, ("content",))
        if content and not relation_type and "\t" in content:
            relation_type = content.split("\t", 1)[0].strip()
        if content and not description:
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            if len(lines) >= 3:
                description = "\n".join(lines[2:])
            elif lines:
                description = lines[-1]
        label = description or relation_type or f"{src_id} -> {tgt_id}"
        return relation_type or "ASSOCIATED_WITH", label

    @staticmethod
    def _extract_timestamp(record: dict[str, Any]) -> str | None:
        direct = GraphStore._pick_text(
            record, ("timestamp", "date", "datetime", "time")
        )
        if direct:
            parsed = GraphStore._extract_date(direct)
            return parsed or direct
        description = GraphStore._pick_text(record, ("description", "content"))
        if not description:
            return None
        return GraphStore._extract_date(description)

    @staticmethod
    def _extract_date(text: str) -> str | None:
        cleaned = text.strip()
        if not cleaned:
            return None
        for pattern in DATE_PATTERNS:
            match = pattern.search(cleaned)
            if not match:
                continue
            parts = match.groups()
            if len(parts) != 3:
                continue
            if len(parts[0]) == 4:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
            else:
                month = int(parts[0])
                day = int(parts[1])
                year = int(parts[2])
                if year < 100:
                    year += 2000 if year < 70 else 1900
            try:
                return datetime(year, month, day).date().isoformat()
            except ValueError:
                continue
        return None

    @staticmethod
    def _edge_in_date_window(
        edge: "GraphEdge", *, date_from: date | None, date_to: date | None
    ) -> bool:
        if not edge.timestamp:
            return True
        try:
            edge_date = datetime.fromisoformat(edge.timestamp).date()
        except ValueError:
            return True
        if date_from and edge_date < date_from:
            return False
        if date_to and edge_date > date_to:
            return False
        return True

    @staticmethod
    def _split_source_ids(source: Any) -> list[str]:
        if source is None:
            return []
        raw = str(source).strip()
        if not raw:
            return []
        if GRAPH_FIELD_SEP in raw:
            parts = raw.split(GRAPH_FIELD_SEP)
        elif "," in raw:
            parts = raw.split(",")
        else:
            parts = [raw]
        return [
            part.strip() for part in parts if part.strip() and part.strip() != "UNKNOWN"
        ]

    @staticmethod
    def _pick_text(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
        for key in keys:
            value = record.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _edge_id(src_id: str, tgt_id: str, relation: str) -> str:
        digest = hashlib.sha1(
            f"{src_id}|{tgt_id}|{relation}".encode("utf-8")
        ).hexdigest()
        return f"edge-{digest[:16]}"

    def _serialize_node(self, node: GraphNode) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": node.id,
            "label": node.label,
            "entity_type": node.entity_type or "Other",
            "degree": node.degree,
        }
        subtype = resolve_entity_subtype(node.entity_type)
        if subtype:
            payload["entity_subtype"] = subtype
        summary = GraphStore._clean_entity_description(node.summary, node.label)
        if summary:
            payload["summary"] = summary
        if node.meta:
            payload["meta"] = node.meta
        evidence = self._serialize_relationship_evidence(node.evidence)
        if evidence:
            payload["evidence"] = evidence
        return payload

    def _serialize_edge(self, edge: GraphEdge) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": edge.id,
            "src_id": edge.src_id,
            "tgt_id": edge.tgt_id,
            "label": edge.label,
            "relation_type": edge.relation_type,
        }
        if edge.relation_raw_phrase:
            payload["relation_raw_phrase"] = edge.relation_raw_phrase
        if edge.confidence_score is not None:
            payload["confidence_score"] = edge.confidence_score
        if edge.confidence_band:
            payload["confidence_band"] = edge.confidence_band
        if edge.weight is not None:
            payload["weight"] = edge.weight
        if edge.timestamp:
            payload["timestamp"] = edge.timestamp
        evidence = self._serialize_relationship_evidence(
            edge.evidence, source_ids=edge.source_ids
        )
        if evidence:
            payload["evidence"] = evidence
        return payload

    @staticmethod
    def _dedupe_evidence(
        items: list[dict[str, str | None]],
    ) -> list[dict[str, str | None]]:
        seen: set[tuple[str, str, str, str]] = set()
        deduped: list[dict[str, str | None]] = []
        for item in items:
            file_path = str(item.get("file_path") or "")
            reference_id = str(item.get("reference_id") or "")
            document_id = str(item.get("document_id") or "")
            confidence_code = str(item.get("confidence_code") or "")
            key = (file_path, reference_id, document_id, confidence_code)
            if not file_path or not reference_id or key in seen:
                continue
            seen.add(key)
            deduped.append(
                {
                    "file_path": file_path,
                    "reference_id": reference_id,
                    "document_id": document_id or None,
                    "confidence_code": confidence_code or None,
                }
            )
        deduped.sort(
            key=lambda row: (
                str(row.get("file_path", "")),
                str(row.get("reference_id", "")),
                str(row.get("document_id", "")),
            )
        )
        return deduped

    @staticmethod
    def _format_export_sources(items: list[dict[str, str | None]]) -> str:
        sources: list[str] = []
        seen: set[str] = set()
        for item in items:
            file_path = str(item.get("file_path") or "").strip()
            reference_id = str(item.get("reference_id") or "").strip()
            if not file_path or not reference_id:
                continue
            source = f"{file_path}#{reference_id}"
            if source in seen:
                continue
            seen.add(source)
            sources.append(source)
        return " | ".join(sources)

    @staticmethod
    def _write_csv(columns: tuple[str, ...], rows: list[dict[str, Any]]) -> str:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
        return buffer.getvalue()

    @staticmethod
    def _build_document_index(documents: list[dict[str, Any]]) -> list[DocumentRecord]:
        indexed: list[DocumentRecord] = []
        for doc in documents:
            doc_id = str(doc.get("id", "")).strip()
            stored_file_path = str(doc.get("stored_file_path", "")).strip()
            if not doc_id or not stored_file_path:
                continue
            indexed.append(
                DocumentRecord(
                    id=doc_id,
                    original_filename=str(doc.get("original_filename", "")).strip(),
                    stored_file_path=stored_file_path,
                    confidence_code=str(doc.get("confidence_code", "")).strip() or None,
                    tags=str(doc.get("tags", "")).strip() or None,
                )
            )
        return indexed

    @staticmethod
    def _build_filename_index(
        documents: list[DocumentRecord],
    ) -> dict[str, list[DocumentRecord]]:
        by_name: dict[str, list[DocumentRecord]] = {}
        for doc in documents:
            filename = Path(doc.original_filename).name.lower()
            if not filename:
                continue
            by_name.setdefault(filename, []).append(doc)
        return by_name

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _read_records(path: Path) -> list[dict[str, Any]]:
        payload = GraphStore._read_json(path)
        if payload is None:
            return []
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                return [item for item in payload["data"] if isinstance(item, dict)]
            rows: list[dict[str, Any]] = []
            for key, value in payload.items():
                if isinstance(value, dict):
                    row = dict(value)
                    row.setdefault("id", key)
                    rows.append(row)
            return rows
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    @staticmethod
    def _parse_graphml(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not path.exists():
            return [], []
        try:
            tree = ET.parse(path)
        except Exception:
            return [], []
        root = tree.getroot()
        key_map: dict[str, str] = {}
        for key in root.findall(".//{*}key"):
            key_id = key.attrib.get("id")
            attr_name = key.attrib.get("attr.name")
            if key_id and attr_name:
                key_map[key_id] = attr_name

        nodes: list[dict[str, Any]] = []
        for node in root.findall(".//{*}node"):
            node_id = node.attrib.get("id")
            if not node_id:
                continue
            data: dict[str, Any] = {"entity_name": node_id}
            for entry in node.findall("{*}data"):
                key = entry.attrib.get("key")
                if not key:
                    continue
                attr = key_map.get(key, key)
                value = (entry.text or "").strip()
                if value:
                    data[attr] = value
            nodes.append(data)

        edges: list[dict[str, Any]] = []
        for edge in root.findall(".//{*}edge"):
            src = edge.attrib.get("source")
            tgt = edge.attrib.get("target")
            if not src or not tgt:
                continue
            data: dict[str, Any] = {"src_id": src, "tgt_id": tgt}
            for entry in edge.findall("{*}data"):
                key = entry.attrib.get("key")
                if not key:
                    continue
                attr = key_map.get(key, key)
                value = (entry.text or "").strip()
                if value:
                    data[attr] = value
            edges.append(data)
        return nodes, edges

    @staticmethod
    def _coerce_string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            values = re.split(r"[,;|]", value)
        elif isinstance(value, list):
            values = value
        else:
            return []
        output: list[str] = []
        for item in values:
            normalized = str(item).strip()
            if normalized and normalized not in output:
                output.append(normalized)
        return output


class GraphInsightService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._prompt_catalog = get_prompt_catalog(
            path=self._settings.prompt_catalog_path,
            auto_reload=self._settings.prompt_catalog_auto_reload,
        )

    def enrich_entity_detail(
        self, case_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        fallback = self._fallback_entity_detail(payload)
        if not self._settings.llm_provider_api_key:
            payload["description"] = fallback
            return payload
        cache_key, case_root = self._cache_context(case_id, payload, kind="entity")
        if cache_key and case_root:
            cached = self._load_cache_value(case_root, cache_key)
            if cached:
                payload["description"] = cached
                payload["generated_detail"] = True
                return payload
        prompt_text = self._prompt_catalog.render(
            "summary.entity_detail",
            {
                "entity_label": str(payload.get("label") or payload.get("id") or ""),
                "entity_type": str(payload.get("entity_type") or "other"),
                "entity_subtype": str(payload.get("entity_subtype") or ""),
                "entity_summary": str(
                    payload.get("raw_description") or payload.get("description") or ""
                ),
                "evidence_json": self._compact_json(payload.get("evidence"), limit=5),
                "related_relationships_json": self._compact_json(
                    payload.get("related_relationships"), limit=6
                ),
            },
        )
        generated = self._run_prompt(prompt_text)
        payload["description"] = generated or fallback
        payload["generated_detail"] = bool(generated)
        if generated and cache_key and case_root:
            self._store_cache_value(case_root, cache_key, generated)
        return payload

    def enrich_relationship_detail(
        self, case_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        fallback = self._fallback_relationship_detail(payload)
        if not self._settings.llm_provider_api_key:
            payload["description"] = fallback
            return payload
        cache_key, case_root = self._cache_context(
            case_id, payload, kind="relationship"
        )
        if cache_key and case_root:
            cached = self._load_cache_value(case_root, cache_key)
            if cached:
                payload["description"] = cached
                payload["generated_detail"] = True
                return payload
        prompt_text = self._prompt_catalog.render(
            "summary.relationship_detail",
            {
                "src_label": str(payload.get("src_id") or ""),
                "tgt_label": str(payload.get("tgt_id") or ""),
                "relation_type": str(payload.get("relation_type") or "ASSOCIATED_WITH"),
                "existing_description": str(
                    payload.get("raw_description") or payload.get("description") or ""
                ),
                "evidence_json": self._compact_json(payload.get("evidence"), limit=5),
            },
        )
        generated = self._run_prompt(prompt_text)
        payload["description"] = generated or fallback
        payload["generated_detail"] = bool(generated)
        if generated and cache_key and case_root:
            self._store_cache_value(case_root, cache_key, generated)
        return payload

    def _cache_context(
        self, case_id: str, payload: dict[str, Any], *, kind: str
    ) -> tuple[str | None, Path | None]:
        case_root = self._case_root(case_id)
        if case_root is None:
            return None, None
        serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
        digest = hashlib.sha1(f"{kind}:{serialized}".encode("utf-8")).hexdigest()
        return f"{kind}:{digest}", case_root

    def _case_root(self, case_id: str) -> Path | None:
        with get_connection(self._settings) as connection:
            row = connection.execute(
                'SELECT case_slug FROM "case" WHERE id = ?',
                (case_id,),
            ).fetchone()
        if not row or not row["case_slug"]:
            return None
        return self._settings.cases_root / str(row["case_slug"])

    @staticmethod
    def _cache_path(case_root: Path) -> Path:
        return case_root / "analysis" / "graph-detail-cache.json"

    @staticmethod
    def _compact_json(value: Any, *, limit: int) -> str:
        trimmed = value[: max(1, limit)] if isinstance(value, list) else value
        return json.dumps(trimmed, ensure_ascii=True, sort_keys=True, default=str)

    def _load_cache_value(self, case_root: Path, key: str) -> str | None:
        path = self._cache_path(case_root)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        value = payload.get(key) if isinstance(payload, dict) else None
        return str(value).strip() if isinstance(value, str) and value.strip() else None

    def _store_cache_value(self, case_root: Path, key: str, value: str) -> None:
        path = self._cache_path(case_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    payload = existing
            except Exception:
                payload = {}
        payload[key] = value
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def _run_prompt(self, prompt_text: str) -> str:
        try:
            from openai import OpenAI
        except Exception:
            return ""
        try:
            client = OpenAI(
                base_url=self._settings.llm_provider_base_url,
                api_key=self._settings.llm_provider_api_key,
                timeout=float(self._settings.rag_llm_timeout_seconds),
                default_headers={
                    key: value
                    for key, value in {
                        "HTTP-Referer": self._settings.llm_provider_site_url,
                        "X-Title": self._settings.llm_provider_app_name,
                    }.items()
                    if value
                }
                or None,
            )
            _, _, messages = self._prompt_catalog.apply_external_overrides(
                messages=[{"role": "user", "content": prompt_text}]
            )
            response = client.chat.completions.create(
                model=self._settings.rag_llm_model,
                messages=messages or [{"role": "user", "content": prompt_text}],
                max_tokens=min(self._settings.rag_llm_max_tokens, 4000),
                temperature=0,
            )
        except Exception:
            logger.debug("Graph insight summary generation failed", exc_info=True)
            return ""
        content = response.choices[0].message.content if response.choices else ""
        return str(content or "").strip()

    @staticmethod
    def _fallback_entity_detail(payload: dict[str, Any]) -> str:
        description = str(
            payload.get("description") or payload.get("raw_description") or ""
        ).strip()
        evidence = payload.get("evidence") or []
        snippets = [
            str(item.get("snippet") or "").strip()
            for item in evidence
            if isinstance(item, dict) and str(item.get("snippet") or "").strip()
        ]
        if snippets:
            description = (
                (description + " ") if description else ""
            ) + f"Supporting evidence mentions: {snippets[0]}"
        return (
            description or str(payload.get("label") or payload.get("id") or "").strip()
        )

    @staticmethod
    def _fallback_relationship_detail(payload: dict[str, Any]) -> str:
        description = str(
            payload.get("description") or payload.get("raw_description") or ""
        ).strip()
        if description:
            return description
        src = str(payload.get("src_id") or "").strip()
        tgt = str(payload.get("tgt_id") or "").strip()
        relation_type = str(payload.get("relation_type") or "ASSOCIATED_WITH").strip()
        return f"{src} is linked to {tgt} through {relation_type}."


def parse_csv_query(value: str | None) -> set[str] | None:
    if value is None:
        return None
    parts = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parts:
        return None
    return set(parts)
