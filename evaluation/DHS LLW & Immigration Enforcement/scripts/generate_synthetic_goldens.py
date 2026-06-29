#!/usr/bin/env python
"""Generate goldens from casepack PDFs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parents[1]
CASEPACK_DIR = EVAL_DIR / "casepack"
GENERATED_ROOT = EVAL_DIR / "generated_goldens"
CANONICAL_GOLDENS = (EVAL_DIR / "goldens" / "golden_queries.json").resolve()

DEFAULT_SPEC: dict[str, Any] = {
    "version": "1.0",
    "thesis_title": "From Evidence to Insight: A GraphRAG Approach to Multimodal Investigative Intelligence",
    "analysis_type_distribution": {
        "link": 6,
        "temporal": 5,
        "flow": 5,
        "factual_verification": 7,
        "retrieval_honesty": 7,
    },
    # This per-type allocation sums to the requested 24/5/1 source-span split.
    "source_spans_by_analysis_type": {
        "link": {"1": 4, "2": 1, "3": 1},
        "temporal": {"1": 4, "2": 1},
        "flow": {"1": 3, "2": 2},
        "factual_verification": {"1": 6, "2": 1},
        "retrieval_honesty": {"1": 7},
    },
    "source_span_distribution": {"1": 24, "2": 5, "3": 1},
    "entity_types": ["person", "organization", "object", "location", "event"],
    "model": "gpt-5.4",
    "critic_model": None,
    "seed": 20260521,
    "temperature": 0,
    "request_timeout_seconds": 300,
    "chunk_size_chars": 4000,
    "chunk_overlap_chars": 400,
    "context_chunks_per_source": 2,
    "filtration_quality_threshold": 0.7,
    "filtration_max_retries": 2,
    "generation_max_retries": 3,
    "num_evolutions": 1,
    "evolutions": {
        "MULTICONTEXT": 0.25,
        "CONCRETIZING": 0.25,
        "CONSTRAINED": 0.25,
        "COMPARATIVE": 0.25,
    },
    "verify_grounding_with_llm": True,
    "honesty_validation_max_chars": 500000,
}

STYLES = {
    "link": {
        "scenario": "An investigative analyst examining actors, aliases, organizations, events, objects, and locations.",
        "task": "Identify explicit links that require connecting at least two entities in the supplied evidence.",
        "input_format": "One precise English investigative question asking which entities are connected and how. It must be answerable only from the supplied evidence.",
    },
    "temporal": {
        "scenario": "An investigative analyst reconstructing an evidence-based chronology.",
        "task": "Reconstruct dates, ordering, or changes over time from the supplied evidence.",
        "input_format": "One precise English question asking for a sequence or timeline containing at least two evidence-supported temporal points.",
    },
    "flow": {
        "scenario": "An investigative analyst tracing actions, transfers, escalation, identification, or consequences.",
        "task": "Trace a multi-step process or causal chain supported by the supplied evidence.",
        "input_format": "One precise English question beginning with 'How' or 'Trace' and requiring an evidence-supported multi-step explanation.",
    },
    "factual_verification": {
        "scenario": "An investigative analyst verifying a concrete claim against documentary evidence.",
        "task": "Test a specific claim whose truth value and explanation are present in the supplied evidence.",
        "input_format": "One unambiguous English question beginning exactly with 'TRUE or FALSE:' and containing a claim that can be verified from the supplied evidence.",
    },
    "retrieval_honesty": {
        "scenario": "An investigative analyst testing whether a system refuses to invent unavailable evidence.",
        "task": "Ask for a plausible, specific detail related to the supplied evidence but not stated or inferable in it.",
        "input_format": "One precise English question about an absent outcome, private attribute, exact specification, price, batch number, or investigation result. The supplied evidence must not answer it.",
    },
}

QUERY_FIELDS = {
    "id",
    "question",
    "analysis_type",
    "phases",
    "ground_truth_entities",
    "ground_truth_relationships",
    "reference_answer",
    "source_ids",
    "citation_targets",
    "ground_truth_chunks",
    "out_of_scope",
    "notes",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_spec(path: str | None, model: str | None, seed: int | None) -> dict[str, Any]:
    spec = deepcopy(DEFAULT_SPEC)
    if path:
        spec = deep_update(spec, json.loads(Path(path).read_text(encoding="utf-8")))
    if model:
        spec["model"] = model
    if seed is not None:
        spec["seed"] = seed
    validate_spec(spec)
    return spec


def validate_spec(spec: dict[str, Any]) -> None:
    analysis = spec["analysis_type_distribution"]
    per_type = spec["source_spans_by_analysis_type"]
    if set(analysis) != set(STYLES) or set(per_type) != set(STYLES):
        raise ValueError(f"Analysis types must be exactly: {', '.join(STYLES)}")
    for analysis_type, expected in analysis.items():
        actual = sum(int(count) for count in per_type[analysis_type].values())
        if actual != int(expected):
            raise ValueError(f"{analysis_type} source-span counts sum to {actual}, expected {expected}")
    actual_spans = Counter()
    for spans in per_type.values():
        actual_spans.update({str(span): int(count) for span, count in spans.items()})
    expected_spans = Counter({str(k): int(v) for k, v in spec["source_span_distribution"].items()})
    if actual_spans != expected_spans:
        raise ValueError(f"Source-span allocation {dict(actual_spans)} != {dict(expected_spans)}")
    if spec["chunk_overlap_chars"] >= spec["chunk_size_chars"]:
        raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")
    if any(entity_type not in {"person", "organization", "object", "location", "event"} for entity_type in spec["entity_types"]):
        raise ValueError("Unsupported entity type in entity_types")


def build_schedule(spec: dict[str, Any]) -> list[dict[str, Any]]:
    schedule = []
    for analysis_type in STYLES:
        for source_span, count in spec["source_spans_by_analysis_type"][analysis_type].items():
            schedule.extend(
                {"analysis_type": analysis_type, "source_span": int(source_span)}
                for _ in range(int(count))
            )
    for index, item in enumerate(schedule, 1):
        item["id"] = f"Q{index:02d}"
    return schedule


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_documents(spec: dict[str, Any]) -> list[dict[str, Any]]:
    from pypdf import PdfReader

    pdfs = sorted(CASEPACK_DIR.glob("*.pdf"), key=lambda path: path.name.casefold())
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {CASEPACK_DIR}")
    documents = []
    for index, path in enumerate(pdfs, 1):
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "").replace("\r\n", "\n").strip() for page in reader.pages]
        text = "\n\n".join(page for page in pages if page)
        if not text:
            raise ValueError(f"No extractable text in {path.name}")
        metadata = reader.metadata or {}
        creation_date = str(metadata.get("/CreationDate") or "")
        date_match = re.search(r"(\d{4})(\d{2})(\d{2})", creation_date)
        chunks = []
        for page_number, page_text in enumerate(pages, 1):
            for chunk in chunk_text(page_text, spec["chunk_size_chars"], spec["chunk_overlap_chars"]):
                chunks.append({"page": page_number, "text": chunk})
        documents.append(
            {
                "id": f"doc_{index}",
                "path": path,
                "filename": path.name,
                "title": str(metadata.get("/Title") or path.stem),
                "url": None,
                "publisher": None,
                "author": str(metadata.get("/Author") or "") or None,
                "date": (
                    f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                    if date_match
                    else None
                ),
                "page_count": len(reader.pages),
                "text": text,
                "chunks": chunks,
                "sha256": sha256_file(path),
            }
        )
    return documents


def choose_sources(documents: list[dict[str, Any]], span: int, seed: int) -> list[dict[str, Any]]:
    if span > len(documents):
        raise ValueError(f"Requested {span} sources from {len(documents)} documents")
    return random.Random(seed).sample(documents, span)


def choose_context(
    sources: list[dict[str, Any]], spec: dict[str, Any], seed: int
) -> tuple[list[str], str]:
    rng = random.Random(seed)
    contexts = []
    digest_parts = []
    per_source = int(spec["context_chunks_per_source"])
    for source in sources:
        chunks = source["chunks"]
        if not chunks:
            raise ValueError(f"No chunks available for {source['filename']}")
        count = min(per_source, len(chunks))
        start = rng.randrange(len(chunks) - count + 1)
        selected = chunks[start : start + count]
        for chunk in selected:
            contexts.append(f"SOURCE: {source['filename']}\nPAGE: {chunk['page']}\n{chunk['text']}")
            digest_parts.append(chunk["text"])
    return contexts, sha256_bytes("\n".join(digest_parts).encode("utf-8"))


def model_environment() -> tuple[str, str]:
    api_key = os.getenv("RAWABIT_LLM_PROVIDER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = (
        os.getenv("RAWABIT_LLM_PROVIDER_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    if not api_key:
        raise RuntimeError("Set RAWABIT_LLM_PROVIDER_API_KEY or OPENAI_API_KEY")
    return api_key, base_url


def build_model(model_name: str, seed: int, spec: dict[str, Any]):
    from deepeval.models import DeepEvalBaseLLM
    from openai import OpenAI

    api_key, base_url = model_environment()

    class OpenAICompatibleLLM(DeepEvalBaseLLM):
        def __init__(self):
            self.model_name = model_name
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=int(spec["request_timeout_seconds"]),
            )
            self.system_fingerprints: set[str] = set()

        def load_model(self):
            return self.client

        def generate(self, prompt: str) -> str:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(spec["temperature"]),
                seed=seed,
            )
            fingerprint = getattr(response, "system_fingerprint", None)
            if fingerprint:
                self.system_fingerprints.add(fingerprint)
            return response.choices[0].message.content or ""

        async def a_generate(self, prompt: str) -> str:
            return await asyncio.to_thread(self.generate, prompt)

        def get_model_name(self) -> str:
            return self.model_name

    return OpenAICompatibleLLM()


def parse_json_object(value: str) -> dict[str, Any]:
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response did not contain a JSON object")
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response was not a JSON object")
    return parsed


def evolution_config(spec: dict[str, Any], honesty: bool):
    from deepeval.synthesizer import Evolution
    from deepeval.synthesizer.config import EvolutionConfig

    evolutions = {getattr(Evolution, name): float(weight) for name, weight in spec["evolutions"].items()}
    return EvolutionConfig(num_evolutions=0 if honesty else int(spec["num_evolutions"]), evolutions=evolutions)


def synthesize_question(
    analysis_type: str,
    contexts: list[str],
    source_names: list[str],
    model: Any,
    critic_model: Any,
    spec: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    from deepeval.synthesizer import Synthesizer
    from deepeval.synthesizer.config import FiltrationConfig, StylingConfig

    honesty = analysis_type == "retrieval_honesty"
    style = STYLES[analysis_type]
    synthesizer = Synthesizer(
        model=model,
        async_mode=False,
        max_concurrent=1,
        filtration_config=FiltrationConfig(
            synthetic_input_quality_threshold=0.0 if honesty else float(spec["filtration_quality_threshold"]),
            max_quality_retries=0 if honesty else int(spec["filtration_max_retries"]),
            critic_model=critic_model,
        ),
        evolution_config=evolution_config(spec, honesty),
        styling_config=StylingConfig(**style),
    )
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=[contexts],
        include_expected_output=False,
        max_goldens_per_context=1,
        source_files=[" | ".join(source_names)],
        _send_data=False,
    )
    if not goldens:
        raise ValueError("DeepEval returned no golden")
    golden = goldens[0]
    return golden.input.strip(), golden.additional_metadata or {}


def enrichment_prompt(
    question: str,
    analysis_type: str,
    contexts: list[str],
    allowed_entity_types: list[str],
) -> str:
    honesty = analysis_type == "retrieval_honesty"
    mode_rules = (
        """This is an intentionally unanswerable retrieval-honesty query.
Set ground_truth_entities, ground_truth_relationships, and ground_truth_chunks to [].
The reference_answer must state only that the requested information is not contained in the provided documents."""
        if honesty
        else """This query must be fully answerable from the evidence.
Provide 1-8 named entities, at least one relationship, and 1-4 short verbatim evidence excerpts.
Each evidence excerpt must be copied as an exact contiguous substring from the supplied evidence.
For factual_verification, the reference answer must begin TRUE. or FALSE. and explain why."""
    )
    return f"""Create the project-specific annotation for this DeepEval-generated question.
Use only the supplied evidence. Do not use outside knowledge or infer facts not stated in the evidence.
Keep the question unchanged.

ANALYSIS TYPE: {analysis_type}
QUESTION: {json.dumps(question, ensure_ascii=False)}
ALLOWED ENTITY TYPES: {json.dumps(allowed_entity_types)}

{mode_rules}

Return JSON only with this shape:
{{
  "question": "exact unchanged question",
  "ground_truth_entities": [{{"name": "human-readable source name", "type": "allowed type"}}],
  "ground_truth_relationships": [{{"src": "entity name", "relation": "UPPER_SNAKE_CASE", "tgt": "entity name"}}],
  "reference_answer": "concise source-grounded answer",
  "ground_truth_chunks": ["exact verbatim excerpt"],
  "notes": "brief explanation of the analytical skill tested"
}}

EVIDENCE:
{chr(10).join(contexts)}
"""


def grounding_prompt(question: str, annotation: dict[str, Any], evidence: str, honesty: bool) -> str:
    if honesty:
        instruction = """Determine whether the complete corpus explicitly contains enough information to answer the question.
Return {"answerable": false, "evidence": []} only if the requested detail is absent from every supplied document.
If any document answers it, return answerable true and quote the supporting passage."""
    else:
        instruction = """Determine whether every factual claim, named entity, relationship, and quoted evidence span in the annotation is supported by the supplied evidence.
Return grounded false if the annotation adds outside knowledge, changes a quotation, or overstates an inference."""
    shape = (
        '{"answerable": false, "evidence": ["exact supporting excerpt if answerable"]}'
        if honesty
        else '{"grounded": true, "unsupported_claims": []}'
    )
    return f"""{instruction}
Return JSON only in this shape: {shape}

QUESTION: {question}
ANNOTATION: {json.dumps(annotation, ensure_ascii=False)}

EVIDENCE:
{evidence}
"""


def normalize_annotation(
    raw: dict[str, Any],
    question: str,
    analysis_type: str,
    sources: list[dict[str, Any]],
    query_id: str,
) -> dict[str, Any]:
    honesty = analysis_type == "retrieval_honesty"
    return {
        "id": query_id,
        "question": question,
        "analysis_type": analysis_type,
        "phases": ["retrieval", "generation"],
        "ground_truth_entities": [] if honesty else raw.get("ground_truth_entities", []),
        "ground_truth_relationships": [] if honesty else raw.get("ground_truth_relationships", []),
        "reference_answer": str(raw.get("reference_answer", "")).strip(),
        "source_ids": [source["id"] for source in sources],
        "citation_targets": [] if honesty else [source["filename"] for source in sources],
        "ground_truth_chunks": [] if honesty else raw.get("ground_truth_chunks", []),
        "out_of_scope": honesty,
        "notes": str(raw.get("notes", "")).strip(),
    }


def normalized_span_in_text(span: str, text: str, allow_ellipses: bool = False) -> bool:
    punctuation = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"})
    normalized_text = " ".join(unicodedata.normalize("NFKC", text).translate(punctuation).split())
    normalized_span = " ".join(unicodedata.normalize("NFKC", span).translate(punctuation).strip().split())
    if normalized_span in normalized_text:
        return True
    if not allow_ellipses or "..." not in normalized_span:
        return False
    position = 0
    for part in (" ".join(piece.split()) for piece in normalized_span.split("...")):
        if not part:
            continue
        position = normalized_text.find(part, position)
        if position < 0:
            return False
        position += len(part)
    return True


def validate_query(
    query: dict[str, Any],
    spec: dict[str, Any],
    source_texts: dict[str, str],
    allow_ellipses: bool = False,
    check_evidence: bool = True,
) -> list[str]:
    errors = []
    missing = QUERY_FIELDS - set(query)
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    if query.get("analysis_type") not in STYLES:
        errors.append("invalid analysis_type")
    if query.get("phases") != ["retrieval", "generation"]:
        errors.append("phases must be ['retrieval', 'generation']")
    if not str(query.get("question", "")).strip() or not str(query.get("reference_answer", "")).strip():
        errors.append("question and reference_answer are required")
    entities = query.get("ground_truth_entities", [])
    relations = query.get("ground_truth_relationships", [])
    chunks = query.get("ground_truth_chunks", [])
    if query.get("out_of_scope"):
        if entities or relations or chunks or query.get("citation_targets"):
            errors.append("retrieval-honesty queries must have empty entities, relationships, chunks, and citations")
    else:
        if not entities or not relations or not chunks:
            errors.append("supported queries require entities, relationships, and evidence chunks")
    entity_names = {str(entity.get("name", "")).strip() for entity in entities if isinstance(entity, dict)}
    for entity in entities:
        if not isinstance(entity, dict) or entity.get("type") not in spec["entity_types"] or not entity.get("name"):
            errors.append(f"invalid entity: {entity!r}")
    for relation in relations:
        if not isinstance(relation, dict):
            errors.append(f"invalid relationship: {relation!r}")
            continue
        if relation.get("src") not in entity_names or relation.get("tgt") not in entity_names:
            errors.append(f"relationship endpoint missing from entities: {relation!r}")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", str(relation.get("relation", ""))):
            errors.append(f"invalid relationship label: {relation!r}")
    if check_evidence:
        combined_source = "\n".join(source_texts.get(source_id, "") for source_id in query.get("source_ids", []))
        for chunk in chunks:
            if (
                not isinstance(chunk, str)
                or not chunk.strip()
                or not normalized_span_in_text(chunk, combined_source, allow_ellipses)
            ):
                errors.append(f"evidence chunk is not verbatim in selected sources: {chunk!r}")
    return errors


def validate_dataset(
    data: dict[str, Any],
    spec: dict[str, Any],
    source_texts: dict[str, str],
    expected_schedule: list[dict[str, Any]] | None = None,
    allow_ellipses: bool = False,
    check_evidence: bool = True,
) -> dict[str, Any]:
    queries = data.get("queries")
    errors = []
    if not isinstance(queries, list):
        return {"valid": False, "errors": ["queries must be a list"]}
    questions = Counter(str(query.get("question", "")).strip().casefold() for query in queries)
    duplicates = [question for question, count in questions.items() if question and count > 1]
    if duplicates:
        errors.append(f"duplicate questions: {duplicates}")
    for index, query in enumerate(queries, 1):
        if query.get("id") != f"Q{index:02d}":
            errors.append(f"query {index} has non-sequential id {query.get('id')!r}")
        errors.extend(
            f"{query.get('id', index)}: {error}"
            for error in validate_query(query, spec, source_texts, allow_ellipses, check_evidence)
        )
    actual_types = Counter(query.get("analysis_type") for query in queries)
    actual_spans = Counter(str(len(query.get("source_ids", []))) for query in queries)
    if expected_schedule is not None:
        expected_types = Counter(item["analysis_type"] for item in expected_schedule)
        expected_spans = Counter(str(item["source_span"]) for item in expected_schedule)
        if actual_types != expected_types:
            errors.append(f"analysis distribution {dict(actual_types)} != {dict(expected_types)}")
        if actual_spans != expected_spans:
            errors.append(f"source-span distribution {dict(actual_spans)} != {dict(expected_spans)}")
    metadata = data.get("metadata", {})
    if metadata.get("total_queries") != len(queries):
        errors.append("metadata.total_queries does not match query count")
    return {
        "valid": not errors,
        "errors": errors,
        "query_count": len(queries),
        "analysis_type_distribution": dict(actual_types),
        "source_span_distribution": dict(actual_spans),
    }


def corpus_metadata(documents: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        document["id"]: {
            "id": document["id"],
            "filename": document["filename"],
            "title": document["title"],
            "url": document["url"],
            "publisher": document["publisher"],
            "author": document["author"],
            "date": document["date"],
        }
        for document in documents
    }


def build_dataset(queries: list[dict[str, Any]], documents: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": {
            "version": spec["version"],
            "created": datetime.now(timezone.utc).date().isoformat(),
            "purpose": "Synthetic golden query set for GraphRAG system evaluation",
            "thesis_title": spec["thesis_title"],
            "total_queries": len(queries),
            "analysis_type_distribution": dict(Counter(query["analysis_type"] for query in queries)),
            "entity_types": spec["entity_types"],
            "corpus": corpus_metadata(documents),
        },
        "queries": queries,
    }


def safe_run_dir(resume: str | None, model: str) -> Path:
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    if resume:
        if resume == "latest":
            candidates = sorted(path for path in GENERATED_ROOT.iterdir() if path.is_dir())
            if not candidates:
                raise FileNotFoundError(f"No generated runs under {GENERATED_ROOT}")
            run_dir = candidates[-1]
        else:
            candidate = Path(resume)
            run_dir = candidate if candidate.is_absolute() else EVAL_DIR / candidate
    else:
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model).strip("-")
        run_dir = GENERATED_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug}"
    resolved = run_dir.resolve()
    if GENERATED_ROOT.resolve() not in resolved.parents:
        raise ValueError(f"Run directory must be inside {GENERATED_ROOT}")
    if (resolved / "golden_queries.json").resolve() == CANONICAL_GOLDENS:
        raise ValueError("Refusing to overwrite canonical goldens")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_checkpoints(path: Path) -> dict[str, dict[str, Any]]:
    completed = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        query = record.get("query", {})
        query_id = query.get("id")
        if not query_id:
            raise ValueError(f"Checkpoint line {line_number} has no query id")
        completed[query_id] = record
    return completed


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in ("deepeval", "pypdf", "openai"):
        versions[package] = importlib.metadata.version(package)
    return versions


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def generate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec_json, args.model, args.seed)
    documents = extract_documents(spec)
    schedule = build_schedule(spec)
    if args.max_queries is not None:
        if args.max_queries < 1:
            raise ValueError("--max-queries must be positive")
        schedule = schedule[: args.max_queries]
    run_dir = safe_run_dir(args.resume, spec["model"])
    checkpoint_path = run_dir / "checkpoints.jsonl"
    completed = read_checkpoints(checkpoint_path)
    document_by_id = {document["id"]: document for document in documents}
    source_texts = {document["id"]: document["text"] for document in documents}
    all_corpus = "\n\n".join(
        f"DOCUMENT: {document['filename']}\n{document['text']}" for document in documents
    )
    if len(all_corpus) > int(spec["honesty_validation_max_chars"]):
        raise ValueError(
            f"Corpus has {len(all_corpus)} characters, above honesty_validation_max_chars="
            f"{spec['honesty_validation_max_chars']}; raise the explicit limit rather than silently truncating"
        )
    source_manifest = [
        {
            "id": document["id"],
            "filename": document["filename"],
            "sha256": document["sha256"],
            "pages": document["page_count"],
        }
        for document in documents
    ]
    manifest_path = run_dir / "run_manifest.json"
    previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if args.resume and manifest_path.exists() else None
    if previous_manifest:
        if previous_manifest.get("spec") != spec:
            raise ValueError("Resume spec differs from the original run_manifest.json")
        previous_sources = [(item["filename"], item["sha256"]) for item in previous_manifest.get("sources", [])]
        current_sources = [(item["filename"], item["sha256"]) for item in source_manifest]
        if previous_sources != current_sources:
            raise ValueError("Casepack files changed since the original run")
    manifest = {
        "protocol": "rawabit-deepeval-synthetic-goldens-v1",
        "status": "running",
        "started_at": previous_manifest.get("started_at") if previous_manifest else utc_now(),
        "run_directory": str(run_dir),
        "spec": spec,
        "provider_base_url": model_environment()[1],
        "packages": package_versions(),
        "sources": source_manifest,
        "completed_queries": sorted(completed),
        "validation": None,
        "system_fingerprints": previous_manifest.get("system_fingerprints", []) if previous_manifest else [],
        "reproducibility_note": (
            "Temperature and seeds reduce variation but do not guarantee byte-identical LLM output. "
            "checkpoints.jsonl is the exact replay artifact."
        ),
    }
    write_json(manifest_path, manifest)

    fingerprints: set[str] = set(manifest["system_fingerprints"])
    for index, item in enumerate(schedule):
        if item["id"] in completed:
            continue
        last_error = None
        for attempt in range(int(spec["generation_max_retries"])):
            call_seed = int(spec["seed"]) + index * 100 + attempt
            random.seed(call_seed)
            sources = choose_sources(documents, item["source_span"], call_seed)
            contexts, context_hash = choose_context(sources, spec, call_seed)
            model = build_model(spec["model"], call_seed, spec)
            critic_model = (
                model
                if not spec.get("critic_model") or spec["critic_model"] == spec["model"]
                else build_model(spec["critic_model"], call_seed, spec)
            )
            try:
                question, deepeval_metadata = synthesize_question(
                    item["analysis_type"],
                    contexts,
                    [source["filename"] for source in sources],
                    model,
                    critic_model,
                    spec,
                )
                raw_annotation = parse_json_object(
                    model.generate(
                        enrichment_prompt(question, item["analysis_type"], contexts, spec["entity_types"])
                    )
                )
                if raw_annotation.get("question") != question:
                    raise ValueError("Annotation changed the DeepEval-generated question")
                query = normalize_annotation(raw_annotation, question, item["analysis_type"], sources, item["id"])
                query_errors = validate_query(query, spec, source_texts)
                if query_errors:
                    raise ValueError("; ".join(query_errors))
                verification = None
                if spec["verify_grounding_with_llm"]:
                    honesty = item["analysis_type"] == "retrieval_honesty"
                    evidence = all_corpus if honesty else "\n".join(contexts)
                    verification = parse_json_object(
                        critic_model.generate(grounding_prompt(question, raw_annotation, evidence, honesty))
                    )
                    if honesty and verification.get("answerable") is not False:
                        raise ValueError("Honesty query is answerable somewhere in the complete corpus")
                    if not honesty and verification.get("grounded") is not True:
                        raise ValueError(f"Annotation is not fully grounded: {verification}")
                fingerprints.update(model.system_fingerprints)
                fingerprints.update(critic_model.system_fingerprints)
                record = {
                    "query": query,
                    "generation": {
                        "attempt": attempt + 1,
                        "seed": call_seed,
                        "source_ids": [source["id"] for source in sources],
                        "context_sha256": context_hash,
                        "deepeval_metadata": deepeval_metadata,
                        "verification": verification,
                    },
                }
                append_checkpoint(checkpoint_path, record)
                completed[item["id"]] = record
                print(f"{item['id']}: generated {item['analysis_type']}")
                break
            except Exception as exc:
                last_error = exc
                print(f"{item['id']}: attempt {attempt + 1} failed: {exc}", file=sys.stderr)
        else:
            manifest["status"] = "failed"
            manifest["failed_query"] = item["id"]
            manifest["error"] = str(last_error)
            manifest["completed_queries"] = sorted(completed)
            write_json(manifest_path, manifest)
            raise RuntimeError(f"Failed to generate {item['id']} after retries") from last_error

    ordered_records = [completed[item["id"]] for item in schedule if item["id"] in completed]
    queries = [record["query"] for record in ordered_records]
    dataset = build_dataset(queries, documents, spec)
    validation = validate_dataset(dataset, spec, source_texts, schedule)
    write_json(run_dir / "golden_queries.json", dataset)
    manifest.update(
        {
            "status": "complete" if validation["valid"] else "invalid",
            "finished_at": utc_now(),
            "completed_queries": [query["id"] for query in queries],
            "validation": validation,
            "system_fingerprints": sorted(fingerprints),
        }
    )
    write_json(manifest_path, manifest)
    if not validation["valid"]:
        raise ValueError("Generated dataset failed validation:\n" + "\n".join(validation["errors"]))
    print(run_dir / "golden_queries.json")
    return 0


def validate_file(path: str, spec: dict[str, Any]) -> int:
    documents = extract_documents(spec)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    documents_by_filename = {document["filename"]: document for document in documents}
    source_texts = {
        source_id: documents_by_filename.get(source.get("filename"), {}).get("text", "")
        for source_id, source in data.get("metadata", {}).get("corpus", {}).items()
    }
    # Historical goldens contain lightly edited/truncated web-article quotations;
    # --validate-file checks their schema without pretending those are exact PDF spans.
    result = validate_dataset(data, spec, source_texts, allow_ellipses=True, check_evidence=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["valid"] else 1


def self_check() -> int:
    spec = load_spec(None, None, None)
    schedule = build_schedule(spec)
    assert len(schedule) == 30
    assert Counter(item["analysis_type"] for item in schedule) == Counter(spec["analysis_type_distribution"])
    assert Counter(str(item["source_span"]) for item in schedule) == Counter(spec["source_span_distribution"])
    assert importlib.metadata.version("deepeval") == "3.9.7"
    try:
        resolved = (CANONICAL_GOLDENS.parent / "golden_queries.json").resolve()
        if resolved == CANONICAL_GOLDENS:
            raise ValueError("Refusing to overwrite canonical goldens")
    except ValueError as exc:
        assert "Refusing" in str(exc)
    print("self-check passed")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate source-only GraphRAG goldens with DeepEval 3.9.7."
    )
    parser.add_argument("--model", help="Generator model identifier; default comes from DEFAULT_SPEC")
    parser.add_argument("--seed", type=int, help="Base random/API seed")
    parser.add_argument("--spec-json", help="JSON file that recursively overrides DEFAULT_SPEC")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        metavar="RUN_DIR",
        help="Resume a run directory under generated_goldens; omit the value to use the latest run",
    )
    parser.add_argument("--max-queries", type=int, help="Generate only the first N scheduled queries")
    parser.add_argument("--validate-file", help="Validate an existing golden JSON without generation")
    parser.add_argument("--self-check", action="store_true", help="Run no-cost protocol checks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_check:
        return self_check()
    spec = load_spec(args.spec_json, args.model, args.seed)
    if args.validate_file:
        return validate_file(args.validate_file, spec)
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
