#!/usr/bin/env python3
"""
Minimal thesis-defensible evaluation harness for LightRAG/GraphRAG results.

This version can either:
  1) score cached query-result JSON files, or
  2) query the Rawabit/LightRAG API first, cache those raw query results, then score them.

It supports any LightRAG mode string, including:
  local, global, hybrid, naive, mix.

Metric set implemented:
RQ1 text/source retrieval:
  - deterministic source_file_recall / source_file_precision
  - DeepEval ContextualRecallMetric
  - DeepEval ContextualPrecisionMetric
RQ1 graph retrieval, GraphRAG only:
  - deterministic entity_recall
  - deterministic relation_pair_recall (undirected by default)
RQ2 factual accuracy:
  - DeepEval GEval answer_correctness with a question-conditioned rubric
RQ2 grounding:
  - DeepEval FaithfulnessMetric


By default, the script scores cached query-result JSON files such as:
  queries_hybrid.json
  queries_naive.json
or checkpoint files under:
  results/<run-dir>/checkpoints/<case-id>/<model-slug>/queries_<mode>.json

With --run-queries, it queries the RAG system first and writes raw query results
into the same output folder as the score files:
  <output-dir>/queries_<mode>.json

It writes row-level score checkpoints immediately as JSONL and saves raw query
results after every completed query, so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable
import httpx
import time

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(x * y for x, y in zip(v1, v2))
    mag1 = math.sqrt(sum(x * x for x in v1))
    mag2 = math.sqrt(sum(x * x for x in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

def get_openrouter_embeddings(texts: list[str], api_key: str, model: str = "baai/bge-m3") -> list[list[float]]:
    if not texts:
        return []
    
    if not api_key:
        raise ValueError("OpenRouter API key missing. Set OPENROUTER_API_KEY environment variable.")
    
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"model": model, "input": texts}
    
    last_error = None
    # Embedding calls can race under RQ1 concurrency; retry small transient failures.
    for attempt in range(3):
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
            
    raise RuntimeError(f"Failed to fetch embeddings from OpenRouter: {last_error}")

def retrieved_entities_topk(result: dict[str, Any]) -> list[str]:
    reval = result.get("retrieval_eval") or {}
    vals = reval.get("retrieved_entity_ids_topk")
    if not vals:
        vals = result.get("entities") or result.get("highlight_entities") or []
    if isinstance(vals, str):
        vals = [v.strip() for v in vals.split(",") if v.strip()]
    out, seen = [], set()
    for v in (vals if isinstance(vals, list) else []):
        cand = (v.get("name") or v.get("id") or v.get("entity") or v.get("text")) if isinstance(v, dict) else str(v)
        nv = norm_text(cand)
        if nv and nv not in seen:
            seen.add(nv); out.append(nv)
    return out

def _endpoint_pairs_expected(q: dict[str, Any]) -> list[tuple[str, str]]:
    out = []
    for it in q.get("ground_truth_relationships", []) or []:
        if not isinstance(it, dict):
            continue
        s = norm_text(it.get("src") or it.get("source") or it.get("subject"))
        o = norm_text(it.get("tgt") or it.get("target") or it.get("object"))
        if s and o:
            out.append((s, o))
    return out

def _endpoint_pairs_retrieved(result: dict[str, Any]) -> list[tuple[str, str]]:
    reval = result.get("retrieval_eval") or {}
    out, seen = [], set()
    rels = reval.get("retrieved_relation_ids_topk")
    if rels:
        for r in rels:
            parts = [p.strip() for p in r.split("__")] if "__" in r else []
            if len(parts) >= 3:
                pair = (norm_text(parts[0]), norm_text(parts[2]))
                if all(pair) and pair not in seen:
                    seen.add(pair); out.append(pair)
    else:
        for r in result.get("highlight_relationships") or []:
            if isinstance(r, dict):
                pair = (norm_text(r.get("src_id") or r.get("src")),
                        norm_text(r.get("tgt_id") or r.get("tgt")))
                if all(pair) and pair not in seen:
                    seen.add(pair); out.append(pair)
    return out

def count_semantic_matches(expected_texts, api_key, model, retrieved_texts, threshold):
    """Returns (recall_hits, precision_hits, details)."""
    if not expected_texts:
        return 0, 0, {"matched": [], "missed": []}
    if not retrieved_texts:
        return 0, 0, {"matched": [],
                      "missed": [{"expected": t, "best_retrieved_candidate": None, "similarity": 0.0}
                                 for t in expected_texts]}
    exp_embs = get_openrouter_embeddings(expected_texts, api_key, model)
    ret_embs = get_openrouter_embeddings(retrieved_texts, api_key, model)
    sim = [[cosine_similarity(e, r) for r in ret_embs] for e in exp_embs]
    details = {"matched": [], "missed": []}
    recall_hits = 0
    for i, exp_text in enumerate(expected_texts):
        j = max(range(len(retrieved_texts)), key=lambda k: sim[i][k])
        best = sim[i][j]
        if best >= threshold:
            recall_hits += 1
            details["matched"].append({"expected": exp_text, "retrieved_match": retrieved_texts[j],
                                       "similarity": round(best, 4)})
        else:
            details["missed"].append({"expected": exp_text, "best_retrieved_candidate": retrieved_texts[j],
                                      "similarity": round(best, 4)})
    precision_hits = sum(1 for j in range(len(retrieved_texts))
                         if max(sim[i][j] for i in range(len(expected_texts))) >= threshold)
    return recall_hits, precision_hits, details

# Normalization helpers

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
TYPE_TAG_RE = re.compile(r"^\[[A-Z_]+\]\s*")
UUID_PREFIX_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_(.+)$", re.I)


def norm_text(value: Any) -> str:
    text = str(value or "").strip()
    text = TYPE_TAG_RE.sub("", text)
    text = text.replace("_", " ")
    text = NON_ALNUM_RE.sub(" ", text.lower()).strip()
    return re.sub(r"\s+", " ", text)


def normalize_reference_file(path_value: Any) -> str:
    base = Path(str(path_value or "")).name
    m = UUID_PREFIX_RE.match(base)
    if m:
        base = m.group(1)
    return base


def file_key(path_value: Any) -> str:
    return norm_text(normalize_reference_file(path_value))


def source_match(expected_key: str, returned_key: str) -> bool:
    if not expected_key or not returned_key:
        return False
    return expected_key == returned_key or expected_key in returned_key or returned_key in expected_key


def clean_for_json(value: Any) -> Any:
    """Convert NaN/Inf to None so output is strict JSON."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    return value


# Loading goldens and results


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_goldens(eval_dir: Path) -> Path:
    candidates = [
        eval_dir / "goldens" / "golden_queries.json",
        eval_dir / "golden_queries.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find golden_queries.json under {eval_dir}")


def load_goldens(eval_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    path = find_goldens(eval_dir)
    data = load_json(path)
    queries = data.get("queries", data if isinstance(data, list) else [])
    if isinstance(queries, dict):
        queries = list(queries.values())
    if not isinstance(queries, list):
        raise ValueError(f"Unexpected golden query format in {path}")
    corpus = data.get("metadata", {}).get("corpus", {}) if isinstance(data, dict) else {}
    return queries, corpus


def query_id(q: dict[str, Any]) -> str:
    return str(q.get("id") or q.get("query_id") or "").strip()


def load_result_file(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    # Raw query cache format: {"queries": {"Q01": {...}}}
    if isinstance(data, dict) and isinstance(data.get("queries"), dict):
        return data["queries"]
    # Older scratch runs used a direct mapping: {"Q01": {...}}
    if isinstance(data, dict):
        if all(isinstance(v, dict) for v in data.values()):
            return data
    raise ValueError(f"Unexpected cached result format in {path}")


def result_candidates(eval_dir: Path, case_id: str, model_slug: str, mode: str, run_dir: str | None) -> list[Path]:
    names = [f"queries_{mode}.json"]
    if mode == "hybrid":
        names.append("queries_graphrag_hybrid.json")
    if mode == "naive":
        names.append("queries_vector-only_naive.json")

    candidates: list[Path] = []
    if run_dir:
        rd = Path(run_dir)
        # Allow either full path or run-name relative to eval_dir/results.
        run_paths = [rd]
        if not rd.is_absolute():
            run_paths.append(eval_dir / "results" / run_dir)
        for rp in run_paths:
            for name in names:
                candidates.append(rp / "checkpoints" / case_id / model_slug / name)
                candidates.append(rp / name)
    for name in names:
        candidates.extend([
            eval_dir / name,
            eval_dir / "results" / name,
            Path.cwd() / name,
        ])
    # Keep candidate order while dropping duplicates.
    seen = set()
    unique = []
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def load_cached_results(eval_dir: Path, case_id: str, model_slug: str, mode: str, run_dir: str | None) -> tuple[dict[str, dict[str, Any]], Path]:
    candidates = result_candidates(eval_dir, case_id, model_slug, mode, run_dir)
    for p in candidates:
        if p.exists():
            return load_result_file(p), p
    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No cached results found for mode={mode}. Searched:\n  {searched}")



# Optional Rawabit / LightRAG querying

def safe_mode_name(mode: str) -> str:
    """Sanitize mode names for filenames while preserving common LightRAG modes."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(mode).strip()).strip("_") or "mode"


def query_results_path(output_dir: Path, mode: str) -> Path:
    return output_dir / f"queries_{safe_mode_name(mode)}.json"


def create_chat(case_id: str, *, base_url: str, request_timeout: int) -> str:
    """Create a fresh chat for one query.

    A fresh chat avoids cross-query conversational contamination.
    """
    import httpx

    with httpx.Client(timeout=request_timeout) as client:
        resp = client.post(f"{base_url}/api/cases/{case_id}/chats", json={})
        resp.raise_for_status()
        data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Failed to create chat: {data}")
    return data["data"]["id"]


def _unwrap_api_envelope(envelope: dict[str, Any]) -> dict[str, Any]:
    if envelope.get("status") != "success":
        raise RuntimeError(f"API error: {envelope}")
    return envelope.get("data", {}) or {}


def _extract_query_response(data: dict[str, Any]) -> dict[str, Any]:
    chunks = data.get("chunks", []) or []
    if not isinstance(chunks, list):
        chunks = []
    highlight = data.get("highlight", {}) or {}
    if not isinstance(highlight, dict):
        highlight = {}
    relationships = highlight.get("highlight_relationships", []) or []
    references = data.get("references", []) or []
    return {
        "answer": (data.get("message", {}) or {}).get("content", "") or "",
        "contexts": [c.get("snippet", "") for c in chunks if isinstance(c, dict) and c.get("snippet")],
        "full_texts": [c.get("full_text", "") for c in chunks if isinstance(c, dict) and c.get("full_text")],
        "entities": list(highlight.get("highlight_entities", []) or []),
        "highlight_relationships": relationships if isinstance(relationships, list) else [],
        "references": [r for r in references if isinstance(r, dict)],
        "retrieval_eval": data.get("retrieval_eval", {}) or {},
        "highlight_views": data.get("highlight_views", []) or [],
        "model_name": data.get("model_name", ""),
    }


def query_rag_system(
    question: str,
    case_id: str,
    *,
    mode: str,
    base_url: str,
    top_k: int,
    chunk_top_k: int,
    request_timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    """Query the Rawabit chat API using an arbitrary LightRAG mode string."""
    import httpx

    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            chat_id = create_chat(case_id, base_url=base_url, request_timeout=request_timeout)
            url = f"{base_url}/api/cases/{case_id}/chats/{chat_id}/messages"
            payload = {
                "content": question,
                "mode": mode,
                "options": {
                    "top_k": top_k,
                    "chunk_top_k": chunk_top_k,
                },
            }
            with httpx.Client(timeout=request_timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = _unwrap_api_envelope(resp.json())
            extracted = _extract_query_response(data)
            extracted["chat_id"] = chat_id
            extracted["ok"] = True
            return extracted
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(5 * (attempt + 1))
            else:
                break
    raise RuntimeError(f"Failed to query system after {max_retries + 1} attempts: {last_error}")


def load_query_result_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = load_json(path)
    if isinstance(data, dict) and isinstance(data.get("queries"), dict):
        return data["queries"]
    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        return data
    raise ValueError(f"Unexpected raw query cache format in {path}")


def save_query_result_cache(
    path: Path,
    *,
    case_id: str,
    model_slug: str,
    mode: str,
    queries_by_id: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case_id,
        "model_slug": model_slug,
        "mode": mode,
        "created_or_updated_at": _dt.datetime.now().isoformat(),
        "queries": clean_for_json(queries_by_id),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def query_and_cache_mode(
    *,
    queries: list[dict[str, Any]],
    case_id: str,
    model_slug: str,
    mode: str,
    output_dir: Path,
    base_url: str,
    top_k: int,
    chunk_top_k: int,
    request_timeout: int,
    max_retries: int,
    concurrency: int,
    resume: bool,
    allow_failures: bool,
) -> tuple[dict[str, dict[str, Any]], Path]:
    """Run RAG queries for one mode and persist raw results after each query."""
    out_path = query_results_path(output_dir, mode)
    cached = load_query_result_cache(out_path) if resume and out_path.exists() else {}
    lock = threading.Lock()
    pending = [q for q in queries if not (cached.get(query_id(q), {}).get("ok"))]

    if cached:
        print(f"[Query {mode}] loaded {len(cached)} cached rows from {out_path}")
    if not pending:
        return cached, out_path

    print(f"[Query {mode}] running {len(pending)} queries with concurrency={max(1, concurrency)}")
    failures: dict[str, str] = {}

    def _worker(q: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        qid = query_id(q)
        question = str(q.get("question") or "")
        try:
            entry = query_rag_system(
                question,
                case_id,
                mode=mode,
                base_url=base_url,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                request_timeout=request_timeout,
                max_retries=max_retries,
            )
            entry["question"] = question
            return qid, entry
        except Exception as exc:
            return qid, {
                "question": question,
                "answer": "",
                "contexts": [],
                "full_texts": [],
                "entities": [],
                "highlight_relationships": [],
                "references": [],
                "retrieval_eval": {},
                "model_name": "",
                "ok": False,
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {executor.submit(_worker, q): q for q in pending}
        done_count = 0
        for fut in as_completed(futures):
            q = futures[fut]
            qid = query_id(q)
            done_count += 1
            try:
                returned_qid, entry = fut.result()
            except Exception as exc:
                returned_qid, entry = qid, {"question": q.get("question", ""), "answer": "", "contexts": [], "full_texts": [], "entities": [], "highlight_relationships": [], "references": [], "retrieval_eval": {}, "model_name": "", "ok": False, "error": str(exc)}
            with lock:
                cached[returned_qid] = entry
                save_query_result_cache(out_path, case_id=case_id, model_slug=model_slug, mode=mode, queries_by_id=cached)
            status = "ok" if entry.get("ok") else "FAILED"
            print(f"[Query {mode} {done_count}/{len(pending)}] {returned_qid}: {status}")
            if not entry.get("ok"):
                failures[returned_qid] = str(entry.get("error") or "unknown error")

    if failures and not allow_failures:
        failed = ", ".join(f"{k}: {v}" for k, v in failures.items())
        raise RuntimeError(f"Query execution failed for mode={mode}: {failed}")
    return cached, out_path


# Extract fields from goldens/results


def expected_source_files(q: dict[str, Any], corpus: dict[str, Any]) -> list[str]:
    targets = q.get("citation_targets") or q.get("expected_source_files") or []
    if targets:
        return [str(t) for t in targets]
    out: list[str] = []
    for sid in q.get("source_ids", []) or []:
        meta = corpus.get(str(sid), {}) if isinstance(corpus, dict) else {}
        fn = meta.get("filename") or meta.get("file") or meta.get("title")
        if fn:
            out.append(str(fn))
    return out


def returned_source_files(result: dict[str, Any]) -> list[str]:
    refs = result.get("references") or []
    out: list[str] = []
    if isinstance(refs, list):
        for r in refs:
            if isinstance(r, dict):
                fp = r.get("file_path") or r.get("filename") or r.get("source") or r.get("path")
                if fp:
                    out.append(str(fp))
            elif r:
                out.append(str(r))
    return out


def contexts_from_result(result: dict[str, Any], max_context_chars: int = 8000,
                         include_graph_payload: bool = True) -> list[str]:
    contexts = (result.get("full_texts") or result.get("retrieved_full_texts")
                or result.get("contexts") or result.get("retrieved_contexts") or [])
    if not isinstance(contexts, list):
        return []
    cleaned: list[str] = []
    for c in contexts:
        if c is None:
            continue
        s = str(c).strip()
        if not s:
            continue
        if max_context_chars and len(s) > max_context_chars:
            s = s[:max_context_chars]
        cleaned.append(s)

    if not include_graph_payload:
        return cleaned   # chunks-only: used for the fair retrieval COMPARISON

    entities = result.get("entities") or []
    rels = result.get("highlight_relationships") or []
    if entities:
        entity_block = "Graph entities: " + "; ".join(
            (str(e.get("name") or e) + (f" ({e.get('description','')})" if e.get("description") else "")
             if isinstance(e, dict) else str(e)) for e in entities if e)
        cleaned.append(entity_block[:max_context_chars])
    if rels:
        rel_block = "Graph relationships: " + "; ".join(
            f"{r.get('src_id','?')} → {r.get('relation_type','?')} → {r.get('tgt_id','?')}"
            + (f" ({r.get('description','')})" if r.get("description") else "")
            for r in rels if isinstance(r, dict))
        cleaned.append(rel_block[:max_context_chars])
    return cleaned


def answer_from_result(result: dict[str, Any]) -> str:
    return str(result.get("answer") or result.get("actual_output") or result.get("response") or "").strip()


def reference_answer(q: dict[str, Any]) -> str:
    return str(q.get("reference_answer") or q.get("expected_output") or q.get("answer") or "").strip()

def evidence_reference(q: dict[str, Any]) -> str:
    """Recall basis = curated gold evidence spans, NOT the reasoning answer.
    For TRUE/FALSE and flow queries the verdict is inferred and is not
    retrievable, so scoring recall against it is invalid."""
    chunks = [str(c).strip() for c in (q.get("ground_truth_chunks") or []) if str(c).strip()]
    return "\n".join(chunks) if chunks else reference_answer(q)

def expected_entities(q: dict[str, Any]) -> set[str]:
    vals = []
    for item in q.get("ground_truth_entities", []) or []:
        if isinstance(item, dict):
            vals.append(item.get("name") or item.get("id") or item.get("text"))
        else:
            vals.append(item)
    return {norm_text(v) for v in vals if norm_text(v)}


def retrieved_entities(result: dict[str, Any]) -> set[str]:
    vals = result.get("entities") or result.get("highlight_entities") or []
    if not vals and isinstance(result.get("retrieval_eval"), dict):
        vals = result["retrieval_eval"].get("retrieved_entity_ids_topk") or []
    
    if isinstance(vals, str):
        vals = [v.strip() for v in vals.split(",") if v.strip()]
        
    out: set[str] = set()
    if isinstance(vals, list):
        for v in vals:
            if isinstance(v, dict):
                cand = v.get("name") or v.get("id") or v.get("entity") or v.get("text")
            else:
                cand = str(v) # Ensure it's treated as a string
            nv = norm_text(cand)
            if nv:
                out.add(nv)
    return out


def expected_relation_pairs(q: dict[str, Any], undirected: bool = False) -> set[tuple[str, str, str]]:
    pairs: set[tuple[str, str, str]] = set()
    for item in q.get("ground_truth_relationships", []) or []:
        if not isinstance(item, dict):
            continue
        s = norm_text(item.get("src") or item.get("source") or item.get("subject"))
        o = norm_text(item.get("tgt") or item.get("target") or item.get("object"))
        r = norm_text(item.get("relation") or item.get("predicate") or item.get("type") or "")
        if not s or not o:
            continue
        pair = tuple(sorted([s, o]) + [r]) if undirected else (s, r, o) 
        pairs.add(pair)  # type: ignore[arg-type]
    return pairs

def retrieved_relation_pairs(result: dict[str, Any], undirected: bool = False) -> set[tuple[str, str, str]]:
    rels = result.get("highlight_relationships") or result.get("relationships") or []
    if not rels and isinstance(result.get("retrieval_eval"), dict):
        rels = result["retrieval_eval"].get("retrieved_relationships_topk") or []
    pairs: set[tuple[str, str, str]] = set()
    
    if isinstance(rels, list):
        for rel_item in rels:
            
            if isinstance(rel_item, str):
                parts = [p.strip() for p in rel_item.split("->")]
                if len(parts) >= 3:
                    s, r, o = parts[0], parts[1], parts[2]
                elif len(parts) == 2:
                    s, o = parts[0], parts[1]
                    r = ""
                else:
                    continue
            
            elif isinstance(rel_item, dict):
                s = rel_item.get("src_id") or rel_item.get("src") or rel_item.get("source") or rel_item.get("subject")
                o = rel_item.get("tgt_id") or rel_item.get("tgt") or rel_item.get("target") or rel_item.get("object")
                r = rel_item.get("relation_type") or rel_item.get("description") or rel_item.get("relation") or rel_item.get("keywords") or ""
            else:
                continue
                
            s, o, r = norm_text(s), norm_text(o), norm_text(r)
            if not s or not o:
                continue
            pair = tuple(sorted([s, o]) + [r]) if undirected else (s, r, o) 
            pairs.add(pair)  # type: ignore[arg-type]
    return pairs

def relation_connectivity_recall(exp_pairs, ret_entities, ret_rel_pairs, api_key, model, threshold):
    """Lenient relationship coverage: a gold relation is 'covered' if BOTH endpoints
    appear anywhere in the retrieved subgraph (entities ∪ relation endpoints),
    regardless of whether a single direct edge links them. Captures multi-hop
    reasoning capacity, which hub-and-spoke graphs support via 2-hop paths."""
    if not exp_pairs:
        return None, {"covered": [], "uncovered": []}
    nodes = sorted(set(ret_entities) | {e for p in ret_rel_pairs for e in p})
    if not nodes:
        return 0.0, {"covered": [], "uncovered": [f"{s} ~ {o}" for s, o in exp_pairs]}
    gold_ents = sorted({e for p in exp_pairs for e in p})
    embs = get_openrouter_embeddings(gold_ents + nodes, api_key, model)
    gi = {t: i for i, t in enumerate(gold_ents)}
    node_embs = embs[len(gold_ents):]
    def present(ent):
        ev = embs[gi[ent]]
        return max(cosine_similarity(ev, nv) for nv in node_embs) >= threshold
    presence = {e: present(e) for e in gold_ents}
    covered, details = 0, {"covered": [], "uncovered": []}
    for s, o in exp_pairs:
        if presence[s] and presence[o]:
            covered += 1; details["covered"].append(f"{s} ~ {o}")
        else:
            details["uncovered"].append(f"{s} ~ {o}")
    return round(covered / len(exp_pairs), 4), details

# Deterministic metrics


def source_overlap_counts(q: dict[str, Any], result: dict[str, Any], corpus: dict[str, Any]) -> dict[str, Any]:
    """Count overlap between expected source files and returned references.

    This is deterministic and uses filename normalization to handle UUID prefixes,
    underscores, punctuation, and spacing differences.
    """
    expected = [file_key(x) for x in expected_source_files(q, corpus)]
    returned = [file_key(x) for x in returned_source_files(result)]
    expected = sorted({x for x in expected if x})
    returned = sorted({x for x in returned if x})
    matched = set()
    for e in expected:
        for r in returned:
            if source_match(e, r):
                matched.add(e)
                break
    matched_count = len(matched)
    return {
        "expected_source_count": len(expected),
        "returned_source_count": len(returned),
        "matched_source_count": matched_count,
    }


def source_file_metrics(q: dict[str, Any], result: dict[str, Any], corpus: dict[str, Any]) -> dict[str, Any]:
    """RQ1 source-document retrieval metrics."""
    counts = source_overlap_counts(q, result, corpus)
    exp = counts["expected_source_count"]
    ret = counts["returned_source_count"]
    mat = counts["matched_source_count"]
    return {
        **counts,
        "source_file_recall": round(mat / exp, 4) if exp else None,
        "source_file_precision": round(mat / ret, 4) if ret else None,
        "source_file_hit": 1.0 if mat > 0 else 0.0,
    }

def relation_endpoint_match(exp_pairs, ret_pairs, api_key, model, undirected, threshold):
    if not exp_pairs:
        return {"recall_hits": 0, "precision_hits": 0, "exact_hits": 0, "details": {"matched": [], "missed": []}}
    if not ret_pairs:
        return {"recall_hits": 0, "precision_hits": 0, "exact_hits": 0,
                "details": {"matched": [],
                            "missed": [{"expected": f"{s} ~ {o}", "best_retrieved_candidate": None, "score": 0.0}
                                       for s, o in exp_pairs]}}
    vocab = sorted({e for p in (exp_pairs + ret_pairs) for e in p})
    embs = get_openrouter_embeddings(vocab, api_key, model)
    idx = {t: i for i, t in enumerate(vocab)}
    def sim(a, b): return cosine_similarity(embs[idx[a]], embs[idx[b]])
    def score(eg, rg):
        (sg, og), (sr, orr) = eg, rg
        direct = min(sim(sg, sr), sim(og, orr))      # BOTH endpoints must clear threshold
        if undirected:
            return max(direct, min(sim(sg, orr), sim(og, sr)))
        return direct

    details = {"matched": [], "missed": []}
    recall_hits = 0
    for eg in exp_pairs:
        best_rg, best = None, 0.0
        for rg in ret_pairs:
            sc = score(eg, rg)
            if sc > best:
                best, best_rg = sc, rg
        tag = {"expected": f"{eg[0]} ~ {eg[1]}", "score": round(best, 4)}
        if best >= threshold:
            recall_hits += 1
            tag["retrieved_match"] = f"{best_rg[0]} ~ {best_rg[1]}"
            details["matched"].append(tag)
        else:
            tag["best_retrieved_candidate"] = f"{best_rg[0]} ~ {best_rg[1]}" if best_rg else None
            details["missed"].append(tag)
    precision_hits = sum(1 for rg in ret_pairs if any(score(eg, rg) >= threshold for eg in exp_pairs))
    ret_set = set(ret_pairs)
    exact_hits = sum(1 for s, o in exp_pairs
                     if (s, o) in ret_set or (undirected and (o, s) in ret_set))
    return {"recall_hits": recall_hits, "precision_hits": precision_hits,
            "exact_hits": exact_hits, "details": details}



def graph_metrics(q, api_key, model, result, *, mode, undirected=True,
                  entity_threshold=0.80, relation_threshold=0.80):
    exp_ent = list(expected_entities(q))
    exp_rel = _endpoint_pairs_expected(q)
    base = {"expected_entity_count": len(exp_ent), "expected_relation_count": len(exp_rel)}

    if mode == "naive":
        return {**base,
                "retrieved_entity_count": None,
                "entity_recall_at_k": None,
                "entity_precision_at_k": None,
                "entity_recall_exact_at_k": None,
                "entity_precision_exact_at_k": None,
                "retrieved_relation_count": None,
                "relation_recall_at_k": None,
                "relation_precision_at_k": None,
                "relation_recall_exact_at_k": None,
                "relation_precision_exact_at_k": None,
                "relation_connectivity_recall_at_k": None,
                }

    ret_ent = retrieved_entities_topk(result)
    ret_rel = _endpoint_pairs_retrieved(result)

    ent_rec, ent_prec, ent_det = count_semantic_matches(exp_ent, api_key, model, ret_ent, entity_threshold)
    ent_exact = len(set(exp_ent) & set(ret_ent))
    rel = relation_endpoint_match(exp_rel, ret_rel, api_key, model, undirected, relation_threshold)
    conn_recall, conn_det = relation_connectivity_recall(exp_rel, ret_ent, ret_rel, api_key, model, relation_threshold)
    return {
        **base,
        "retrieved_entity_count": len(ret_ent),
        "matched_entity_count": ent_rec,
        "entity_recall_at_k": round(ent_rec / len(exp_ent), 4) if exp_ent else None,
        "entity_precision_at_k": round(ent_prec / len(ret_ent), 4) if ret_ent else None,
        "entity_recall_exact_at_k": round(ent_exact / len(exp_ent), 4) if exp_ent else None,
        "entity_precision_exact_at_k": round(ent_exact / len(ret_ent), 4) if ret_ent else None,
        "entity_match_details": ent_det,
        
        "retrieved_relation_count": len(ret_rel),
        "matched_relation_count": rel["recall_hits"],
        "relation_recall_at_k": round(rel["recall_hits"] / len(exp_rel), 4) if exp_rel else None,
        "relation_precision_at_k": round(rel["precision_hits"] / len(ret_rel), 4) if ret_rel else None,
        "relation_recall_exact_at_k": round(rel["exact_hits"] / len(exp_rel), 4) if exp_rel else None,
        "relation_precision_exact_at_k": round(rel["exact_hits"] / len(ret_rel), 4) if ret_rel else None,
        "relation_match_details": rel["details"],
        "relation_connectivity_recall_at_k": conn_recall,
        "relation_connectivity_details": conn_det,
    }
# DeepEval setup and wrappers


class OpenAICompatibleDeepEvalLLM:  # real base class is mixed in at runtime if available
    pass


def build_deepeval_model(model_name: str, api_key: str, base_url: str, timeout: int = 300):
    try:
        from deepeval.models import DeepEvalBaseLLM
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "DeepEval/OpenAI dependencies are not importable. Install deepeval and openai in this environment."
        ) from exc

    class _OpenAICompatibleLLM(DeepEvalBaseLLM):
        def __init__(self, model: str, key: str, url: str, request_timeout: int):
            self.model_name = model
            self._api_key = key
            self._base_url = url
            self._timeout = request_timeout
            self._client = None

        def load_model(self):
            if self._client is None:
                self._client = OpenAI(api_key=self._api_key, base_url=self._base_url, timeout=self._timeout)
            return self._client

        def generate(self, prompt: str) -> str:
            client = self.load_model()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content or ""

        async def a_generate(self, prompt: str) -> str:
            return self.generate(prompt)

        def get_model_name(self) -> str:
            return self.model_name

    return _OpenAICompatibleLLM(model_name, api_key, base_url, timeout)


def deepeval_test_case(question: str, answer: str, expected: str, contexts: list[str]):
    from deepeval.test_case import LLMTestCase
    return LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected,
        retrieval_context=contexts,
    )


def instantiate_metric(cls, **kwargs):
    """DeepEval versions differ slightly; try preferred kwargs, then fall back."""
    try:
        return cls(**kwargs)
    except TypeError:
        kwargs2 = dict(kwargs)
        kwargs2.pop("async_mode", None)
        try:
            return cls(**kwargs2)
        except TypeError:
            kwargs3 = dict(kwargs2)
            kwargs3.pop("include_reason", None)
            return cls(**kwargs3)


def metric_score_reason(metric: Any) -> tuple[float | None, str | None, str | None]:
    score = getattr(metric, "score", None)
    reason = getattr(metric, "reason", None)
    error = getattr(metric, "error", None)
    try:
        if score is not None:
            score = float(score)
    except Exception:
        score = None
    return score, reason, error


def run_contextual_metrics(test_case: Any, model: Any, threshold: float) -> dict[str, Any]:
    from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

    out: dict[str, Any] = {}
    metric_specs = [
        ("contextual_recall", ContextualRecallMetric),
        ("contextual_precision", ContextualPrecisionMetric),
    ]
    for name, cls in metric_specs:
        m = instantiate_metric(cls, threshold=threshold, model=model, include_reason=True, async_mode=False)
        try:
            m.measure(test_case)
            score, reason, error = metric_score_reason(m)
            out[name] = score
            out[f"{name}_reason"] = reason
            out[f"{name}_error"] = error
        except Exception as exc:
            out[name] = None
            out[f"{name}_reason"] = None
            out[f"{name}_error"] = str(exc)
    return out



def answer_cited_files(result: dict[str, Any], answer: str) -> set[str]:
    INLINE_CITE_RE = re.compile(r"\[(\d+)\]")
    """Map inline [n] markers in the answer to reference file_paths."""
    id_to_file: dict[str, str] = {}
    for r in result.get("references") or []:
        if isinstance(r, dict):
            rid = str(r.get("reference_id") or "").strip()
            fp = r.get("file_path") or r.get("filename") or r.get("source") or ""
            if rid and fp:
                id_to_file[rid] = file_key(fp)
    cited_ids = set(INLINE_CITE_RE.findall(answer or ""))
    return {id_to_file[c] for c in cited_ids if c in id_to_file}

def source_attribution_metrics(q: dict, result: dict, corpus: dict, mode: str = "") -> dict:
    """RQ2 evidence traceability diagnostics.

    Two related but separate concepts are measured:
      1) inline_citation_*: source files explicitly cited in the generated answer
         through [n]-style markers mapped to result.references.
      2) source_provenance_*: expected source files surfaced anywhere in the
         system result references. For hybrid, this includes evidence reachable
         through retrieved chunks and highlighted graph/subgraph provenance; for
         naive, it includes evidence reachable through retrieved chunk references.

    Report source_provenance_* in the main thesis. Keep inline_citation_* as a
    diagnostic only unless the UI/output explicitly requires inline citations.
    """
    answer = answer_from_result(result)
    expected = sorted(file_key(x) for x in expected_source_files(q, corpus) if file_key(x))
    inline_cited = sorted(answer_cited_files(result, answer))

    # Inline citation diagnostics: strict [n]-style citations in the answer text.
    inline_matched = set(
        e for e in expected
        if any(source_match(e, c) for c in inline_cited)
    )

    # Source provenance diagnostics: source references surfaced by the system.
    # Hybrid and naive expose provenance differently, but the source overlap
    # calculation is shared.
    provenance_counts = source_overlap_counts(q, result, corpus)
    exp = provenance_counts["expected_source_count"]
    ret = provenance_counts["returned_source_count"]
    mat = provenance_counts["matched_source_count"]

    if not expected:  # out-of-scope / no expected source target
        return {
            "inline_cited_source_count": len(inline_cited),
            "inline_matched_citation_count": 0,
            "inline_citation_source_recall": None,
            "inline_citation_source_precision": 1.0 if not inline_cited else 0.0,
            "inline_citation_source_hit": None,
            "provenance_expected_source_count": exp,
            "provenance_returned_source_count": ret,
            "provenance_matched_source_count": mat,
            "source_provenance_recall": None,
            "source_provenance_precision": 1.0 if ret == 0 else 0.0,
            "source_provenance_hit": None,
        }

    return {
        "inline_cited_source_count": len(inline_cited),
        "inline_matched_citation_count": len(inline_matched),
        "inline_citation_source_recall": round(len(inline_matched) / len(expected), 4),
        "inline_citation_source_precision": round(len(inline_matched) / len(inline_cited), 4) if inline_cited else 0.0,
        "inline_citation_source_hit": 1.0 if inline_matched else 0.0,
        "provenance_expected_source_count": exp,
        "provenance_returned_source_count": ret,
        "provenance_matched_source_count": mat,
        "source_provenance_recall": round(mat / exp, 4) if exp else None,
        "source_provenance_precision": round(mat / ret, 4) if ret else 0.0,
        "source_provenance_hit": 1.0 if mat > 0 else 0.0,
    }

def run_generation_metrics(test_case: Any, model: Any, threshold: float) -> dict[str, Any]:
    from deepeval.metrics import FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCaseParams

    out: dict[str, Any] = {}

    correctness_criteria = (
        "Evaluate whether the actual output correctly answers the input question. "
        "Use the expected output as the authoritative source for the correct answer, but do NOT treat every detail in the expected output as mandatory. "
        "First infer the minimal facts required by the input question. Score the actual output against those required facts only. "
        "Do not penalize omissions of background details, affiliations, locations, dates, actions, or corroborating context unless the input question explicitly asks for them. "
        "Penalize wrong entities, wrong relationships, contradictions of the expected output, incorrect refusals when the expected output answers the question, and unsupported extra claims that change the answer. "
        "If the expected output states that the information is not available in the source documents, then a refusal or 'not found' response from the actual output is correct and should receive a high score. "
        "A concise answer that directly gives the requested names/facts should receive a high score even if it omits explanatory background from the expected output."
    )
    try:
        g = instantiate_metric(
            GEval,
            name="Answer Correctness",
            criteria=correctness_criteria,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=threshold,
            model=model,
            async_mode=False,
        )
        g.measure(test_case)
        score, reason, error = metric_score_reason(g)
        out["answer_correctness_geval"] = score
        out["answer_correctness_reason"] = reason
        out["answer_correctness_error"] = error
    except Exception as exc:
        out["answer_correctness_geval"] = None
        out["answer_correctness_reason"] = None
        out["answer_correctness_error"] = str(exc)

    try:
        f = instantiate_metric(FaithfulnessMetric, threshold=threshold, model=model, include_reason=True, async_mode=False)
        f.measure(test_case)
        score, reason, error = metric_score_reason(f)
        out["faithfulness"] = score
        out["faithfulness_reason"] = reason
        out["faithfulness_error"] = error
    except Exception as exc:
        out["faithfulness"] = None
        out["faithfulness_reason"] = None
        out["faithfulness_error"] = str(exc)

    return out


# Output helpers


def read_done_keys(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    keys = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = obj.get("_key")
                if key:
                    keys.add(key)
            except Exception:
                continue
    return keys


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean_for_json(row), ensure_ascii=False) + "\n")


def jsonl_to_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # Stable field order: first row order, then any later fields.
    fields: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields and not k.startswith("_"):
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})


def mean_numeric(values: Iterable[Any]) -> float | None:
    vals: list[float] = []
    for v in values:
        if v is None or v == "":
            continue
        try:
            x = float(v)
        except Exception:
            continue
        if math.isnan(x) or math.isinf(x):
            continue
        vals.append(x)
    if not vals:
        return None
    return round(statistics.mean(vals), 4)


def write_summary(path: Path, rows: list[dict[str, Any]], metrics: list[str]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get("mode", "unknown")), []).append(r)
    out: list[dict[str, Any]] = []
    for mode, group in groups.items():
        row = {"mode": mode, "n": len(group)}
        for m in metrics:
            row[m] = mean_numeric(r.get(m) for r in group)
        out.append(row)
    write_csv(path, out)


# Main evaluation


def mode_label(mode: str) -> str:
    if mode == "naive":
        return "Vector-only (naive)"
    if mode in {"local", "global", "hybrid", "mix"}:
        return f"GraphRAG ({mode})"
    return str(mode)


def parse_csv_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [x.strip() for x in value.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minimal DeepEval + deterministic thesis evaluation for cached RAG outputs.")
    parser.add_argument("--eval-dir", required=True, help="Evaluation/casepack directory containing goldens and cached result JSONs.")
    parser.add_argument("--case-id", required=True, help="Case ID used in checkpoint paths.")
    parser.add_argument("--model-slug", required=True, help="RAG model slug used in checkpoint paths, e.g. gpt-5.4-mini.")
    parser.add_argument("--run-dir", default="", help="Optional run folder name under eval-dir/results, or absolute path to a run folder.")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to eval-dir/results/thesis_<timestamp>.")
    parser.add_argument("--RQs", default="1,2", help="Comma-separated RQs to run: 1,2.")
    parser.add_argument("--modes", default="hybrid,naive", help="Comma-separated modes to score: hybrid,naive.")
    parser.add_argument("--query-ids", default="", help="Optional comma-separated query IDs, e.g. Q01,Q02.")
    parser.add_argument("--resume", action="store_true", help="Resume from JSONL checkpoints in output-dir.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing checkpoints in output-dir before running.")
    parser.add_argument("--undirected-relation-pairs", action="store_true", help="Deprecated/no-op: relation pairs are undirected by default.")
    parser.add_argument("--directed-relation-pairs", action="store_true", help="Match graph relation pairs with direction. Default is undirected.")
    parser.add_argument("--max-context-chars", type=int, default=8000, help="Truncate each retrieved context to this many chars before DeepEval.")

    parser.add_argument("--judge-model", default=os.getenv("EVALUATOR_MODEL", "deepseek/deepseek-v4-pro"), help="DeepEval judge model")
    parser.add_argument("--model-provider-api-key", default="", help="API key. Defaults to OPENAI_API_KEY or RAWABIT_* provider env vars.")
    parser.add_argument("--model-provider-base-url", default="", help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL/RAWABIT_* or OpenRouter.")
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"), help="Embedding model for semantic similarity calculations.")
    parser.add_argument("--judge-timeout", type=int, default=300)
    parser.add_argument("--deepeval-threshold", type=float, default=0.5, help="DeepEval pass threshold. Scores are still recorded continuously.")

    parser.add_argument("--run-queries", action="store_true", help="Query the RAG system first, cache raw results in output-dir, then score them.")
    parser.add_argument("--only-query", action="store_true", help="Only query/cache raw RAG results; skip DeepEval scoring.")
    parser.add_argument("--base-url", default=os.getenv("RAWABIT_BASE_URL", "http://localhost:8000"), help="Rawabit API base URL.")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("RAWABIT_RAG_TOP_K", "10")), help="top_k option sent to the RAG API.")
    parser.add_argument("--chunk-top-k", type=int, default=int(os.getenv("RAWABIT_RAG_CHUNK_TOP_K", "10")), help="chunk_top_k option sent to the RAG API.")
    parser.add_argument("--request-timeout", type=int, default=int(os.getenv("RAWABIT_REQUEST_TIMEOUT", "120")), help="Rawabit API request timeout in seconds.")
    parser.add_argument("--query-retries", type=int, default=int(os.getenv("RAWABIT_MAX_RETRIES", "2")), help="Retries for failed Rawabit query requests.")
    parser.add_argument("--query-concurrency", type=int, default=2, help="Parallel RAG queries per mode. Start with 1-2 if the backend is fragile.")
    parser.add_argument("--score-concurrency", type=int, default=1, help="Parallel DeepEval scoring workers. Default 1 is safest; increase cautiously.")
    parser.add_argument("--allow-query-failures", action="store_true", help="Do not abort if some RAG queries fail; failed rows are cached with ok=false.")
    parser.add_argument("--entity-sim-threshold", type=float, default=0.80)
    parser.add_argument("--relation-sim-threshold", type=float, default=0.80)

    parser.add_argument(
        "--label", default="",
        help="Optional label appended to the auto-generated output folder name, e.g. 'threshold070' → thesis_xxxxx_threshold070."
    )
    args = parser.parse_args(argv)

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(f"eval-dir does not exist: {eval_dir}")

    RQs = parse_csv_list(args.RQs, ["1", "2"])
    modes = parse_csv_list(args.modes, ["hybrid", "naive"])
    selected_ids = set(parse_csv_list(args.query_ids, []))
    undirected_pairs = not bool(args.directed_relation_pairs)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"thesis_{ts}_{args.label}" if args.label else f"thesis_{ts}"
    output_dir = Path(args.output_dir) if args.output_dir else eval_dir / "results" / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    RQ1_ckpt = ckpt_dir / "RQ1.jsonl"
    RQ2_ckpt = ckpt_dir / "RQ2.jsonl"
    if args.overwrite:
        for p in [RQ1_ckpt, RQ2_ckpt]:
            if p.exists():
                p.unlink()

    # If requested, remove raw query caches too. This makes --overwrite a true fresh run.
    if args.overwrite and args.run_queries:
        for mode in modes:
            raw_path = query_results_path(output_dir, mode)
            if raw_path.exists():
                raw_path.unlink()

    queries, corpus = load_goldens(eval_dir)
    if selected_ids:
        queries = [q for q in queries if query_id(q) in selected_ids]
    queries = [q for q in queries if query_id(q)]
    if not queries:
        raise RuntimeError("No queries selected.")

    results_by_mode: dict[str, dict[str, dict[str, Any]]] = {}
    result_paths: dict[str, str] = {}

    if args.run_queries:
        for mode in modes:
            results, path = query_and_cache_mode(
                queries=queries,
                case_id=args.case_id,
                model_slug=args.model_slug,
                mode=mode,
                output_dir=output_dir,
                base_url=args.base_url.rstrip("/"),
                top_k=args.top_k,
                chunk_top_k=args.chunk_top_k,
                request_timeout=args.request_timeout,
                max_retries=args.query_retries,
                concurrency=args.query_concurrency,
                resume=args.resume,
                allow_failures=args.allow_query_failures,
            )
            results_by_mode[mode] = results
            result_paths[mode] = str(path)
    else:
        for mode in modes:
            results, path = load_cached_results(eval_dir, args.case_id, args.model_slug, mode, args.run_dir or None)
            results_by_mode[mode] = results
            result_paths[mode] = str(path)

    model_provider_api_key = (
        args.model_provider_api_key
        or os.getenv("OPENAI_API_KEY", "")
        or os.getenv("RAWABIT_LLM_PROVIDER_API_KEY", "")
        or os.getenv("RAWABIT_OPENROUTER_API_KEY", "")
    )
    model_provider_base_url = (
        args.model_provider_base_url
        or os.getenv("OPENAI_BASE_URL", "")
        or os.getenv("RAWABIT_LLM_PROVIDER_BASE_URL", "")
        or os.getenv("RAWABIT_OPENROUTER_BASE_URL", "")
        or "https://openrouter.ai/api/v1"
    )

    if args.only_query:
        # Still write a manifest for query-only runs.
        model = None
    else:
        # DeepEval / OpenAI-compatible judge API config
        if not model_provider_api_key:
            raise RuntimeError("No judge API key found. Pass --judge-api-key or set OPENAI_API_KEY / RAWABIT_LLM_PROVIDER_API_KEY.")
        # Shared model is safe for score_concurrency=1. For parallel scoring, each
        # worker thread gets its own model instance below.
        model = build_deepeval_model(args.judge_model, model_provider_api_key, model_provider_base_url, args.judge_timeout)
    manifest = {
        "created_at": _dt.datetime.now().isoformat(),
        "eval_dir": str(eval_dir),
        "case_id": args.case_id,
        "model_slug": args.model_slug,
        "run_dir": args.run_dir or None,
        "output_dir": str(output_dir),
        "RQs": RQs,
        "modes": modes,
        "query_ids": [query_id(q) for q in queries],
        "judge_model": args.judge_model,
        "model_provider_base_url": model_provider_base_url if not args.only_query else None,
        "run_queries": bool(args.run_queries),
        "raw_api_base_url": args.base_url.rstrip("/") if args.run_queries else None,
        "top_k": args.top_k if args.run_queries else None,
        "chunk_top_k": args.chunk_top_k if args.run_queries else None,
        "query_concurrency": args.query_concurrency if args.run_queries else None,
        "score_concurrency": args.score_concurrency if not args.only_query else None,
        "deepeval_threshold": args.deepeval_threshold,
        "undirected_relation_pairs": undirected_pairs,
        "max_context_chars": args.max_context_chars,
        "entity_sim_threshold": args.entity_sim_threshold, 
        "relation_sim_threshold": args.relation_sim_threshold, 
        "embedding_model": args.embedding_model,
        "result_paths": result_paths,
        "metrics": {
            "RQ1_main_reported": [
                "source_file_recall", "source_file_precision",
                "contextual_recall", "contextual_precision",
                "entity_recall_at_k", "relation_connectivity_recall_at_k",
            ],
            "RQ1_diagnostics": [
                "source_file_hit",
                "entity_precision_at_k", "entity_recall_exact_at_k", "entity_precision_exact_at_k",
                "relation_recall_at_k", "relation_precision_at_k",
                "relation_recall_exact_at_k", "relation_precision_exact_at_k",
            ],
            "RQ2_main_reported": [
                "answer_correctness_geval", "faithfulness",
                "source_provenance_hit", "source_provenance_recall", "source_provenance_precision",
            ],
            "RQ2_diagnostics": [
                "inline_citation_source_hit", "inline_citation_source_recall", "inline_citation_source_precision",
            ],
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(clean_for_json(manifest), indent=2, ensure_ascii=False), encoding="utf-8")

    if args.only_query:
        print(f"\nQuerying complete. Raw query results written to: {output_dir}")
        return 0

    # Thread-local judge model for optional parallel scoring.
    score_model_tls = threading.local()

    def get_score_model():
        if max(1, args.score_concurrency) <= 1:
            return model
        if not hasattr(score_model_tls, "model"):
            score_model_tls.model = build_deepeval_model(args.judge_model, model_provider_api_key, model_provider_base_url, args.judge_timeout)
        return score_model_tls.model

    # RQ 1
    if "1" in RQs:
        done = read_done_keys(RQ1_ckpt) if args.resume else set()
        jobs: list[tuple[int, str, dict[str, Any], dict[str, Any], str]] = []
        total = len(queries) * len(modes)
        idx = 0
        for mode in modes:
            mode_results = results_by_mode[mode]
            for q in queries:
                idx += 1
                qid = query_id(q)
                key = f"{mode}::{qid}"
                if key in done:
                    print(f"[RQ 1 {idx}/{total}] skip {key}")
                    continue
                result = mode_results.get(qid)
                if not result:
                    print(f"[RQ 1 {idx}/{total}] missing result for {key}", file=sys.stderr)
                    continue
                jobs.append((idx, mode, q, result, key))

        def _score_RQ1_job(job: tuple[int, str, dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
            idx, mode, q, result, key = job
            qid = query_id(q)
            question = str(q.get("question") or result.get("question") or "")
            answer = answer_from_result(result)
            expected = reference_answer(q)
            evidence = evidence_reference(q)
            contexts_full = contexts_from_result(result, max_context_chars=args.max_context_chars,
                                                 include_graph_payload=True)
            contexts_chunks = contexts_from_result(result, max_context_chars=args.max_context_chars,
                                                  include_graph_payload=False)

            print(f"[RQ 1 {idx}/{total}] scoring {key}: {len(contexts_chunks)} chunk-contexts")
            tc_ctx = deepeval_test_case(question, answer, evidence, contexts_chunks)   # chunks only
            row: dict[str, Any] = {
                "_key": key,
                "query_id": qid,
                "question": question,
                "analysis_type": q.get("analysis_type"),
                "mode": mode_label(mode),
                "raw_mode": mode,
                "out_of_scope": bool(q.get("out_of_scope", False)),
                "retrieved_context_count": len(contexts_full),
                "retrieved_chunk_count": len(contexts_chunks),
                "answer_length": len(answer),
            }
            
            if not row["out_of_scope"]:
                row.update(source_file_metrics(q, result, corpus))
                row.update(run_contextual_metrics(tc_ctx, get_score_model(), args.deepeval_threshold))
                row.update(graph_metrics(q, args.model_provider_api_key, args.embedding_model,
                                         result, mode=mode, undirected=undirected_pairs,
                                         entity_threshold=args.entity_sim_threshold,
                                         relation_threshold=args.relation_sim_threshold))
            else:
                null_metrics = ["source_file_recall", "source_file_precision", "source_file_hit",
                                "contextual_recall", "contextual_precision", 
                                "entity_recall_at_k", "entity_precision_at_k", "entity_recall_exact_at_k", "entity_precision_exact_at_k", 
                                "relation_connectivity_recall_at_k", "relation_recall_at_k", "relation_precision_at_k",
                                "relation_recall_exact_at_k", "relation_precision_exact_at_k"]
                row.update({m: None for m in null_metrics})
            return row

        if jobs:
            with ThreadPoolExecutor(max_workers=max(1, args.score_concurrency)) as executor:
                futures = [executor.submit(_score_RQ1_job, job) for job in jobs]
                for fut in as_completed(futures):
                    row = fut.result()
                    append_jsonl(RQ1_ckpt, row)
                    print(f"[RQ 1] saved {row.get('_key')}")

    # RQ 2
    if "2" in RQs:
        done = read_done_keys(RQ2_ckpt) if args.resume else set()
        jobs: list[tuple[int, str, dict[str, Any], dict[str, Any], str]] = []
        total = len(queries) * len(modes)
        idx = 0
        for mode in modes:
            mode_results = results_by_mode[mode]
            for q in queries:
                idx += 1
                qid = query_id(q)
                key = f"{mode}::{qid}"
                if key in done:
                    print(f"[RQ 2 {idx}/{total}] skip {key}")
                    continue
                result = mode_results.get(qid)
                if not result:
                    print(f"[RQ 2 {idx}/{total}] missing result for {key}", file=sys.stderr)
                    continue
                jobs.append((idx, mode, q, result, key))

        def _score_RQ2_job(job: tuple[int, str, dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
            idx, mode, q, result, key = job
            qid = query_id(q)
            question = str(q.get("question") or result.get("question") or "")
            answer = answer_from_result(result)
            expected = reference_answer(q)
            contexts = contexts_from_result(result, max_context_chars=args.max_context_chars, include_graph_payload=True)
            print(f"[RQ 2 {idx}/{total}] scoring {key}: answer_len={len(answer)} contexts={len(contexts)}")
            tc = deepeval_test_case(question, answer, expected, contexts)
            row = {
                "_key": key,
                "query_id": qid,
                "question": question,
                "analysis_type": q.get("analysis_type"),
                "difficulty": q.get("difficulty"),
                "mode": mode_label(mode),
                "raw_mode": mode,
                "out_of_scope": bool(q.get("out_of_scope", False)),
                "answer_length": len(answer),
                "retrieved_context_count": len(contexts),
            }
            row.update(run_generation_metrics(tc, get_score_model(), args.deepeval_threshold))
            row.update(source_attribution_metrics(q, result, corpus, mode=mode))
            return row

        if jobs:
            with ThreadPoolExecutor(max_workers=max(1, args.score_concurrency)) as executor:
                futures = [executor.submit(_score_RQ2_job, job) for job in jobs]
                for fut in as_completed(futures):
                    row = fut.result()
                    append_jsonl(RQ2_ckpt, row)
                    print(f"[RQ 2] saved {row.get('_key')}")

    # Final score files come from the checkpoint rows.
    if RQ1_ckpt.exists():
        rows1 = jsonl_to_rows(RQ1_ckpt)
        write_csv(output_dir / "RQ1.csv", rows1)
        (output_dir / "score_RQ1.json").write_text(
            json.dumps(clean_for_json({
                r["_key"]: {k: v for k, v in r.items() if not k.startswith("_")}
                for r in rows1
            }), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # In-scope only for recall/precision aggregate (out_of_scope queries
        # always score 0 on contextual recall by design — they should not drag the mean)
        rows1_in_scope = [r for r in rows1 if not r.get("out_of_scope", False)]

        write_summary(
            output_dir / "RQ1_summary.csv",
            rows1_in_scope,   
            ["source_file_recall", "source_file_precision", "source_file_hit",
             "contextual_recall", "contextual_precision",
             "entity_recall_at_k", "entity_precision_at_k",
             "entity_recall_exact_at_k", "entity_precision_exact_at_k",
             "relation_connectivity_recall_at_k",
             "relation_recall_at_k", "relation_precision_at_k",
             "relation_recall_exact_at_k", "relation_precision_exact_at_k"],
        )
        rows1_oos = [r for r in rows1 if r.get("out_of_scope", False)]
        if rows1_oos:
            write_summary(
                output_dir / "RQ1_retrieval_honesty_summary.csv",
                rows1_oos,
                [
                    "source_file_hit",
                    "contextual_recall", "contextual_precision",
                    "entity_recall_at_k", "entity_precision_at_k",
                    "entity_recall_exact_at_k", "entity_precision_exact_at_k",
                    "relation_connectivity_recall_at_k",
                    "relation_recall_at_k", "relation_precision_at_k",
                    "relation_recall_exact_at_k",
                    "relation_precision_exact_at_k",
                ]
            )
            
            
    if RQ2_ckpt.exists():
        rows2 = jsonl_to_rows(RQ2_ckpt)
        write_csv(output_dir / "RQ2.csv", rows2)
        (output_dir / "score_RQ2.json").write_text(
            json.dumps(clean_for_json({
                r["_key"]: {k: v for k, v in r.items() if not k.startswith("_")}
                for r in rows2
            }), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        rows2_in_scope = [r for r in rows2 if not r.get("out_of_scope", False)]
        write_summary(
            output_dir / "RQ2_summary.csv",
            rows2_in_scope,
            ["answer_correctness_geval", "faithfulness",
             "source_provenance_hit", "source_provenance_recall", "source_provenance_precision",
             "inline_citation_source_hit", "inline_citation_source_recall", "inline_citation_source_precision"],
        )
        rows2_oos = [r for r in rows2 if r.get("out_of_scope", False)]
        if rows2_oos:
            write_summary(
                output_dir / "RQ2_retrieval_honesty_summary.csv",
                rows2_oos,
                ["answer_correctness_geval"],  # only correctness makes sense for honesty queries
            )
        

    print(f"\nDone. Outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
