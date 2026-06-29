from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import types
import uuid
import xml.etree.ElementTree as ET

import httpx
import pytest

from backend.app.db import get_connection, init_db
from backend.app.ingestion_pipeline import IngestionPipeline
from backend.app.settings import get_settings


def _configure_env(temp_dir: Path) -> None:
    os.environ["RAWABIT_DB_PATH"] = str(temp_dir / "db.sqlite")
    os.environ["RAWABIT_CASES_ROOT"] = str(temp_dir / "cases")


def test_ingestion_pipeline_rejects_path_escape() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)

        case_root = settings.cases_root / "case-a"
        case_root.mkdir(parents=True, exist_ok=True)

        safe = pipeline._resolve_case_file(case_root, "raw/evidence.pdf")  # noqa: SLF001
        assert safe == (case_root / "raw" / "evidence.pdf").resolve()

        with pytest.raises(RuntimeError):
            pipeline._resolve_case_file(case_root, "..\\other-case\\secret.pdf")  # noqa: SLF001
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_effective_config_includes_model_metadata() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-model-config-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)

        effective = pipeline._build_effective_ingestion_config(  # noqa: SLF001
            ingest_profile="balanced_fast_intel",
            processing_mode="multimodal",
            advanced_overrides={},
            preflight={
                "source_kind": "image",
                "complexity_class": "medium",
                "eta_seconds": 120,
            },
        )

        assert effective["llm_model"] == settings.rag_llm_model
        assert effective["vlm_model"] == settings.rag_vlm_model
        assert effective["embedding_model"] == settings.rag_embedding_model
        assert effective["used_vlm"] is False
        assert effective["primary_ingest_model"] == settings.rag_llm_model
        assert effective["enable_preinsert_summary"] is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_prefers_vlm_model_when_vlm_was_used() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-primary-model-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)

        primary_model = pipeline._resolve_primary_ingest_model(  # noqa: SLF001
            effective_config={
                "llm_model": settings.rag_llm_model,
                "vlm_model": settings.rag_vlm_model,
            },
            used_vlm=True,
        )

        assert primary_model == settings.rag_vlm_model
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_prepends_document_notes_to_ingest_text() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-notes-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)

        content_list = pipeline._prepend_document_notes(  # noqa: SLF001
            [{"type": "text", "text": "Evidence body."}],
            "Handle as source-context note.",
        )
        text = pipeline._compose_ingest_text_from_content_list(content_list)  # noqa: SLF001

        assert text.startswith("Analyst notes for this evidence:")
        assert "Handle as source-context note." in text
        assert "Evidence body." in text
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_adds_generalized_extraction_prompt_rules() -> None:
    prompts = {
        "entity_extraction_user_prompt": "Extract facts.",
        "entity_continue_extraction_user_prompt": "Continue extraction.",
    }

    IngestionPipeline._apply_extraction_prompt_enhancements(  # noqa: SLF001
        "lightrag",
        prompts,
    )

    assert "Preserve source-specific relationship wording" in prompts["entity_extraction_user_prompt"]
    assert "missed relationships" in prompts["entity_continue_extraction_user_prompt"]


def test_ingestion_pipeline_parses_multiline_vlm_sections() -> None:
    parsed = IngestionPipeline._parse_vlm_analysis_text(  # noqa: SLF001
        "Description:\nAerial view of a street scene.\n\nVisible text:\nICE shooter\nPORTLAND AVE"
    )

    assert parsed == (
        "Aerial view of a street scene.",
        "ICE shooter\nPORTLAND AVE",
    )


def test_ingestion_pipeline_filters_low_value_image_blocks() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-low-value-image-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)

        text = pipeline._compose_ingest_text_from_content_list(  # noqa: SLF001
            [
                {
                    "type": "image",
                    "vlm_description": "A dark, circular object with a lighter, curved shape within it, resembling a question mark.",
                    "vlm_visible_text": "none",
                    "bbox": [0, 0, 20, 20],
                },
                {
                    "type": "image",
                    "vlm_description": "An annotated aerial street scene with labeled federal vehicles and other federal agents.",
                    "vlm_visible_text": "Federal vehicles, Other federal agents, LOCATION: 44.9, -93.2",
                    "bbox": [0, 0, 400, 400],
                },
            ]
        )

        assert "dark, circular object" not in text
        assert "Federal vehicles" in text
        assert "Visual depiction:" in text
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_builds_graph_retry_text_for_screenshot() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-graph-retry-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)

        retry_text = pipeline._build_graph_retry_text(  # noqa: SLF001
            "Bellingcat @bellingcat.com\n\nUsing imagery online of the shooting by an ICE agent in Minneapolis.",
            [
                {"type": "text", "text": "Bellingcat @bellingcat.com"},
                {
                    "type": "text",
                    "text": "Using imagery online of the shooting by an ICE agent in Minneapolis.",
                },
                {
                    "type": "image",
                    "vlm_description": "An annotated aerial and street-level composite depicts the reported shooting scene.",
                    "vlm_visible_text": "Federal vehicles, Victim's vehicle, ICE shooter, Other federal agents",
                    "bbox": [0, 0, 500, 500],
                },
            ],
            effective_config={"source_kind": "image"},
        )

        assert "Evidence type: social media screenshot." in retry_text
        assert "Source account: Bellingcat (@bellingcat.com)." in retry_text
        assert "Post text:" in retry_text
        assert "Image labels:" in retry_text
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_renders_normalized_evidence_text_without_low_value() -> (
    None
):
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-normalized-text-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)

        rendered = pipeline._render_normalized_evidence_text(  # noqa: SLF001
            {
                "primary_evidence_text": [
                    "Federal agents deployed pepper ball launchers against protesters in Illinois."
                ],
                "entities_and_roles": ["Bellingcat documented the incidents."],
                "events_and_timeline": ["A TRO hearing was scheduled for Nov. 5."],
                "movements_and_transfers": [],
                "source_and_reporting_context": [
                    "The article cites videos and court filings."
                ],
                "supporting_observations": [
                    "Agents pointed launchers from moving vehicles."
                ],
                "low_value_content": ["Newsletter", "Donate"],
            }
        )

        assert "pepper ball launchers" in rendered
        assert "Newsletter" not in rendered
        assert "Donate" not in rendered
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_extracts_fenced_json_object() -> None:
    payload = IngestionPipeline._extract_json_object(  # noqa: SLF001
        '```json\n{"primary_evidence_text":["A"],"low_value_content":[]}\n```'
    )

    assert payload["primary_evidence_text"] == ["A"]





class _RecorderRag:
    def __init__(self) -> None:
        self.kwargs = None

    async def parse_document(self, **kwargs):
        self.kwargs = kwargs
        return [], "doc-id"


class _FallbackLightRag:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def ainsert(self, input, ids=None, file_paths=None, track_id=None):
        self.calls.append(
            {
                "input": input,
                "ids": ids,
                "file_paths": file_paths,
            }
        )
        return "ok"

    async def _insert_done(self):
        return None


class _FallbackRag:
    def __init__(self) -> None:
        self.lightrag = _FallbackLightRag()
        self.insert_calls = 0

    async def insert_content_list(self, **kwargs):
        _ = kwargs
        self.insert_calls += 1
        return None


class _LlmErrorRag:
    def __init__(self) -> None:
        self._rawabit_llm_error_state = {"count": 0, "samples": []}
        self._rawabit_capture_llm_errors = False
        self.lightrag = self

    async def ainsert(self, *args, **kwargs):
        _ = args
        _ = kwargs
        if self._rawabit_capture_llm_errors:
            self._rawabit_llm_error_state["count"] = 1
            self._rawabit_llm_error_state["samples"] = ["Connection error."]
        return None


class _ParallelCaptionPipeline(IngestionPipeline):
    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.active_captions = 0
        self.peak_captions = 0

    async def _caption_image(self, image_path: str, rag, effective_config=None):  # type: ignore[override]
        _ = rag
        _ = effective_config
        self.active_captions += 1
        self.peak_captions = max(self.peak_captions, self.active_captions)
        await asyncio.sleep(0.03)
        self.active_captions -= 1
        return f"Description: {image_path}\nVisible text: none"


class _PreInsertEnrichmentPipeline(IngestionPipeline):
    async def _analyze_images_batch(  # type: ignore[override]
        self, image_paths, rag, ingest_profile, effective_config=None
    ):
        _ = rag
        _ = ingest_profile
        _ = effective_config
        return [
            {
                "description": f"desc-{idx}",
                "visible_text": f"text-{idx}",
                "combined_text": f"Description: desc-{idx}\nVisible text: text-{idx}",
            }
            for idx, _path in enumerate(image_paths, start=1)
        ]

    async def _summarize_ingestion_text(  # type: ignore[override]
        self, source_text: str, rag, summary_max_tokens: int | None = None
    ):
        _ = rag
        _ = summary_max_tokens
        if not source_text:
            return ""
        return "Summary: preinsert"


class _InitLightRag:
    def __init__(self) -> None:
        self.workspace = "case-workspace"
        self.initialize_calls = 0

    async def initialize_storages(self):
        self.initialize_calls += 1


class _InitRagAnything:
    def __init__(self) -> None:
        self.ensure_calls = 0
        self.lightrag = _InitLightRag()

    async def _ensure_lightrag_initialized(self):
        self.ensure_calls += 1
        return {"success": True}


class _FinalizeOrderingClient:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def close(self):
        self._events.append("client-close")


class _FinalizeOrderingWrapper:
    def __init__(self, events: list[str], event_name: str) -> None:
        self._events = events
        self._event_name = event_name

    async def shutdown(self):
        self._events.append(self._event_name)


class _FinalizeOrderingEmbedding:
    def __init__(self, events: list[str]) -> None:
        self.func = _FinalizeOrderingWrapper(events, "embedding-shutdown")


class _FinalizeOrderingLightRag:
    def __init__(self, events: list[str]) -> None:
        self.llm_model_func = _FinalizeOrderingWrapper(events, "llm-shutdown")
        self.embedding_func = _FinalizeOrderingEmbedding(events)
        self._events = events

    async def finalize_storages(self):
        self._events.append("finalize-storages")


class _FinalizeOrderingRag:
    def __init__(self, events: list[str]) -> None:
        self.lightrag = _FinalizeOrderingLightRag(events)
        self._rawabit_openai_client = _FinalizeOrderingClient(events)


def test_ingestion_pipeline_parse_does_not_pass_runtime_parser_kwarg() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-parse-kwargs-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        rag = _RecorderRag()

        import asyncio

        asyncio.run(
            pipeline._parse_document(  # noqa: SLF001 - targeted regression test
                rag=rag,
                file_path=(temp_dir / "sample.pdf"),
                output_dir=(temp_dir / "out"),
            )
        )
        assert isinstance(rag.kwargs, dict)
        assert "parser" not in rag.kwargs
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_initializes_lightrag_pipeline_status(monkeypatch) -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-init-rag-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        rag = _InitRagAnything()
        pipeline_status_calls: list[str | None] = []

        async def _fake_initialize_pipeline_status(workspace=None):
            pipeline_status_calls.append(workspace)

        lightrag_module = types.ModuleType("lightrag")
        kg_module = types.ModuleType("lightrag.kg")
        shared_storage_module = types.ModuleType("lightrag.kg.shared_storage")
        shared_storage_module.initialize_pipeline_status = (
            _fake_initialize_pipeline_status
        )
        monkeypatch.setitem(sys.modules, "lightrag", lightrag_module)
        monkeypatch.setitem(sys.modules, "lightrag.kg", kg_module)
        monkeypatch.setitem(
            sys.modules, "lightrag.kg.shared_storage", shared_storage_module
        )

        asyncio.run(
            pipeline._ensure_rag_initialized(rag)  # noqa: SLF001 - targeted regression test
        )

        assert rag.ensure_calls == 1
        assert rag.lightrag.initialize_calls == 1
        assert pipeline_status_calls == ["case-workspace"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_finalize_shuts_down_worker_queues_before_client_close() -> (
    None
):
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-finalize-order-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        events: list[str] = []
        rag = _FinalizeOrderingRag(events)

        asyncio.run(
            pipeline._finalize_rag(rag)  # noqa: SLF001 - targeted lifecycle regression test
        )

        assert events == [
            "llm-shutdown",
            "embedding-shutdown",
            "finalize-storages",
            "client-close",
        ]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_canonical_text_insert_uses_lightrag() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-multimodal-insert-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        rag = _FallbackRag()

        asyncio.run(
            pipeline._insert_content(  # noqa: SLF001 - targeted unit test
                rag=rag,
                content_list=[{"type": "text", "text": "tweet content"}],
                document_id="doc-xyz",
                citation_file_path="raw/doc-xyz_tweet.png",
            )
        )

        assert rag.insert_calls == 0
        assert len(rag.lightrag.calls) == 1
        assert rag.lightrag.calls[0]["file_paths"] == "raw/doc-xyz_tweet.png"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_text_first_skips_multimodal_insert() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-text-first-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        rag = _FallbackRag()

        asyncio.run(
            pipeline._insert_content(  # noqa: SLF001 - targeted unit test
                rag=rag,
                content_list=[{"type": "text", "text": "plain text"}],
                document_id="doc-text-first",
                citation_file_path="raw/doc-text-first_plain.txt",
                ingest_profile="balanced_fast_intel",
                processing_mode="text_first",
                precomputed_fallback_text="plain text",
            )
        )

        assert rag.insert_calls == 0
        assert len(rag.lightrag.calls) == 1
        assert rag.lightrag.calls[0]["ids"] == "doc-text-first"
        assert rag.lightrag.calls[0]["file_paths"] == "raw/doc-text-first_plain.txt"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_network_retry_succeeds_after_transient_error() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-network-retry-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        os.environ["RAWABIT_RAG_NETWORK_RETRY_WINDOW_SECONDS"] = "15"
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        attempts = {"count": 0}
        pipeline._network_retry_delay_seconds = lambda *_args, **_kwargs: 0.0  # type: ignore[method-assign]  # noqa: SLF001

        async def _flaky_network_call() -> str:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise httpx.ConnectError("transient connection drop")
            return "ok"

        result = asyncio.run(
            pipeline._run_network_call_with_retry(  # noqa: SLF001 - targeted unit test
                operation="test network call",
                call_factory=_flaky_network_call,
            )
        )

        assert result == "ok"
        assert attempts["count"] == 3
    finally:
        os.environ.pop("RAWABIT_RAG_NETWORK_RETRY_WINDOW_SECONDS", None)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_fails_when_llm_errors_detected() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-llm-errors-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        rag = _LlmErrorRag()

        with pytest.raises(
            RuntimeError, match="LLM errors detected during canonical text insertion"
        ):
            asyncio.run(
                pipeline._insert_content(  # noqa: SLF001 - targeted unit test
                    rag=rag,
                    content_list=[{"type": "text", "text": "tweet content"}],
                    document_id="doc-xyz",
                    citation_file_path="raw/doc-xyz_tweet.png",
                )
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_network_retry_window_exhaustion_fails() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = (
        temp_root / f"ingestion-pipeline-network-retry-exhaust-{uuid.uuid4().hex}"
    )
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        os.environ["RAWABIT_RAG_NETWORK_RETRY_WINDOW_SECONDS"] = "0"
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        pipeline._network_retry_delay_seconds = lambda *_args, **_kwargs: 0.0  # type: ignore[method-assign]  # noqa: SLF001

        async def _always_fail() -> str:
            raise httpx.ConnectError("network unavailable")

        with pytest.raises(RuntimeError, match="failed due to network error"):
            asyncio.run(
                pipeline._run_network_call_with_retry(  # noqa: SLF001 - targeted unit test
                    operation="test network call",
                    call_factory=_always_fail,
                )
            )
    finally:
        os.environ.pop("RAWABIT_RAG_NETWORK_RETRY_WINDOW_SECONDS", None)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_non_network_error_is_not_retried() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-non-network-error-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        os.environ["RAWABIT_RAG_NETWORK_RETRY_WINDOW_SECONDS"] = "60"
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        attempts = {"count": 0}
        pipeline._network_retry_delay_seconds = lambda *_args, **_kwargs: 0.0  # type: ignore[method-assign]  # noqa: SLF001

        async def _raise_value_error() -> str:
            attempts["count"] += 1
            raise ValueError("non-network failure")

        with pytest.raises(ValueError, match="non-network failure"):
            asyncio.run(
                pipeline._run_network_call_with_retry(  # noqa: SLF001 - targeted unit test
                    operation="test non-network call",
                    call_factory=_raise_value_error,
                )
            )

        assert attempts["count"] == 1
    finally:
        os.environ.pop("RAWABIT_RAG_NETWORK_RETRY_WINDOW_SECONDS", None)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_route_text_direct_for_text_mime() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-route-text-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)

        route = pipeline._choose_ingestion_route(  # noqa: SLF001
            mime_type="text/plain",
            source_path=temp_dir / "note.txt",
            ingest_profile="balanced_fast",
        )
        assert route.route_type == "text_direct"
        assert route.direct_text is True
        assert route.parse_method == "txt-direct"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_route_image_defaults_to_auto_when_ocr_mode_is_off() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-route-image-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)

        route = pipeline._choose_ingestion_route(  # noqa: SLF001
            mime_type="image/png",
            source_path=temp_dir / "img.png",
            ingest_profile="balanced_fast",
        )
        assert route.route_type == "image_auto"
        assert route.parse_method == "auto"
        assert route.parser_kwargs["table"] is False
        assert route.parser_kwargs["formula"] is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_route_image_can_force_ocr_via_override() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-route-image-force-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)

        route = pipeline._choose_ingestion_route(  # noqa: SLF001
            mime_type="image/png",
            source_path=temp_dir / "img.png",
            ingest_profile="balanced_fast_intel",
            effective_config={"ocr_mode": "force", "source_kind": "image"},
        )
        assert route.route_type == "image_ocr"
        assert route.parse_method == "ocr"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_route_pdf_balanced_prefers_txt_when_probe_dense() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-route-pdf-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)

        pipeline._estimate_pdf_chars_per_page = lambda *_args, **_kwargs: 1000.0  # type: ignore[method-assign]  # noqa: SLF001
        pipeline._estimate_pdf_page_count = lambda *_args, **_kwargs: 5  # type: ignore[method-assign]  # noqa: SLF001

        route = pipeline._choose_ingestion_route(  # noqa: SLF001
            mime_type="application/pdf",
            source_path=temp_dir / "doc.pdf",
            ingest_profile="balanced_fast",
        )
        assert route.route_type == "pdf_txt"
        assert route.parse_method == "txt"
        assert route.parser_kwargs["table"] is True
        assert route.parser_kwargs["formula"] is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_route_pdf_full_enrichment_still_disables_formula() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-route-pdf-full-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)

        route = pipeline._choose_ingestion_route(  # noqa: SLF001
            mime_type="application/pdf",
            source_path=temp_dir / "doc.pdf",
            ingest_profile="full_enrichment",
        )
        assert route.route_type == "pdf_auto"
        assert route.parse_method == "auto"
        assert route.parser_kwargs["formula"] is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_route_office_document_uses_auto_parse() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-route-office-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)

        route = pipeline._choose_ingestion_route(  # noqa: SLF001
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            source_path=temp_dir / "report.docx",
            ingest_profile="balanced_fast_intel",
        )
        assert route.route_type == "office_document_auto"
        assert route.parse_method == "auto"
        assert route.direct_text is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_balanced_profile_force_vlm_caption_even_when_text_is_present() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-balanced-vlm-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        os.environ["RAWABIT_RAG_BALANCED_FORCE_VLM_IMAGE_CAPTION"] = "true"
        settings = get_settings()
        pipeline = IngestionPipeline(settings)

        should_caption = pipeline._should_caption_image_with_vlm(  # noqa: SLF001
            ingest_profile="balanced_fast",
            image_path="some-image.png",
            extracted_text="already extracted text that is long enough to bypass threshold",
        )
        assert should_caption is True
    finally:
        os.environ.pop("RAWABIT_RAG_BALANCED_FORCE_VLM_IMAGE_CAPTION", None)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_profile_specific_parallel_insert_settings() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-parallel-insert-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        os.environ["RAWABIT_RAG_MAX_PARALLEL_INSERT"] = "2"
        os.environ["RAWABIT_RAG_MAX_PARALLEL_INSERT_BALANCED"] = "5"
        os.environ["RAWABIT_RAG_MAX_PARALLEL_INSERT_FULL"] = "3"
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        assert pipeline._max_parallel_insert_for_profile("balanced_fast") == 5  # noqa: SLF001
        assert pipeline._max_parallel_insert_for_profile("full_enrichment") == 3  # noqa: SLF001
    finally:
        os.environ.pop("RAWABIT_RAG_MAX_PARALLEL_INSERT", None)
        os.environ.pop("RAWABIT_RAG_MAX_PARALLEL_INSERT_BALANCED", None)
        os.environ.pop("RAWABIT_RAG_MAX_PARALLEL_INSERT_FULL", None)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_fallback_text_uses_parallel_vlm_captioning_for_images() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-vlm-parallel-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        os.environ["RAWABIT_RAG_ENABLE_VLM"] = "true"
        os.environ["RAWABIT_RAG_BALANCED_FORCE_VLM_IMAGE_CAPTION"] = "true"
        os.environ["RAWABIT_RAG_VLM_PARALLEL_CAPTIONS_BALANCED"] = "4"
        settings = get_settings()
        pipeline = _ParallelCaptionPipeline(settings)

        fallback_text, meta = asyncio.run(
            pipeline._build_fallback_text(  # noqa: SLF001
                content_list=[
                    {"type": "image", "img_path": "img-1.png", "text": ""},
                    {"type": "image", "img_path": "img-2.png", "text": ""},
                    {"type": "image", "img_path": "img-3.png", "text": ""},
                    {"type": "image", "img_path": "img-4.png", "text": ""},
                ],
                rag=object(),
                ingest_profile="balanced_fast",
            )
        )

        assert meta["images_for_vlm"] == 4
        assert meta["vlm_parallelism"] == 4
        assert meta["vlm_success"] == 4
        assert "Visible text:" in fallback_text
        assert pipeline.peak_captions >= 2
    finally:
        os.environ.pop("RAWABIT_RAG_ENABLE_VLM", None)
        os.environ.pop("RAWABIT_RAG_BALANCED_FORCE_VLM_IMAGE_CAPTION", None)
        os.environ.pop("RAWABIT_RAG_VLM_PARALLEL_CAPTIONS_BALANCED", None)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_preinsert_enrichment_adds_vlm_analysis_and_summary() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-preinsert-enrich-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = _PreInsertEnrichmentPipeline(settings)

        enriched, meta = asyncio.run(
            pipeline._enrich_content_for_ingestion(  # noqa: SLF001
                content_list=[
                    {"type": "text", "text": "base text"},
                    {"type": "image", "img_path": "a.png", "text": ""},
                    {"type": "image", "img_path": "b.png", "text": ""},
                ],
                rag=object(),
                ingest_profile="balanced_fast",
                effective_config={"enable_preinsert_summary": True},
            )
        )

        assert meta["images_analyzed"] == 2
        assert meta["visible_text_hits"] == 2
        assert meta["summary_added"] == 1
        assert enriched[0]["generated_by"] == "preinsert_summary"
        image_rows = [row for row in enriched if row.get("type") == "image"]
        assert image_rows[0]["vlm_description"] == "desc-1"
        assert image_rows[0]["vlm_visible_text"] == "text-1"
        assert image_rows[0]["text"] == ""
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_preinsert_artifacts_are_persisted() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-preinsert-artifacts-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        processed_dir = temp_dir / "cases" / "demo" / "processed" / "doc-1"
        processed_dir.mkdir(parents=True, exist_ok=True)
        content_list = [
            {"type": "text", "text": "alpha paragraph"},
            {
                "type": "image",
                "vlm_description": "package photo",
                "vlm_visible_text": "tracking id 123",
                "text": "",
            },
        ]
        preinsert_text = pipeline._compose_ingest_text_from_content_list(content_list)  # noqa: SLF001
        pipeline._persist_preinsert_artifacts(  # noqa: SLF001
            processed_dir=processed_dir,
            content_list=content_list,
            preinsert_text=preinsert_text,
        )

        enriched_path = processed_dir / "content_list_enriched.json"
        canonical_path = processed_dir / "canonical_text.txt"
        preinsert_path = processed_dir / "preinsert_text.txt"
        assert enriched_path.exists()
        assert canonical_path.exists()
        assert preinsert_path.exists()

        persisted_content = json.loads(enriched_path.read_text(encoding="utf-8"))
        persisted_canonical = canonical_path.read_text(encoding="utf-8")
        persisted_text = preinsert_path.read_text(encoding="utf-8")
        assert len(persisted_content) == 2
        assert persisted_canonical == persisted_text
        assert "package photo" not in persisted_text
        assert "Image labels: tracking id 123" in persisted_text
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_writes_manifest() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-manifest-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        processed_dir = temp_dir / "cases" / "demo" / "processed" / "doc-1"
        processed_dir.mkdir(parents=True, exist_ok=True)
        context = {
            "document_id": "doc-1",
            "case_id": "case-1",
            "case_slug": "demo",
            "original_filename": "tweet.png",
        }
        pipeline._write_manifest(  # noqa: SLF001 - targeted helper test
            processed_dir=processed_dir,
            context=context,
            started_at="2026-02-08T12:00:00+00:00",
            finished_at="2026-02-08T12:00:10+00:00",
            status="complete",
            parse_method="txt-direct",
            stats={"content_blocks": 5},
            error=None,
        )
        manifest = json.loads(
            (processed_dir / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["document_id"] == "doc-1"
        assert manifest["case_id"] == "case-1"
        assert manifest["status"] == "complete"
        assert manifest["parse_method"] == "txt-direct"
        assert manifest["stats"]["content_blocks"] == 5
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_annotates_confidence_in_lightrag_files() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-provenance-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        working_dir = temp_dir / "cases" / "demo" / "lightrag"
        working_dir.mkdir(parents=True, exist_ok=True)
        document_id = "doc-1"
        chunk_id = "chunk-1"

        (working_dir / "kv_store_text_chunks.json").write_text(
            json.dumps(
                {
                    chunk_id: {
                        "full_doc_id": document_id,
                        "file_path": "tweet.png",
                        "_id": chunk_id,
                    }
                }
            ),
            encoding="utf-8",
        )
        (working_dir / "kv_store_full_docs.json").write_text(
            json.dumps({document_id: {"file_path": "tweet.png"}}), encoding="utf-8"
        )
        (working_dir / "kv_store_doc_status.json").write_text(
            json.dumps({document_id: {"status": "processed"}}), encoding="utf-8"
        )
        (working_dir / "kv_store_full_entities.json").write_text(
            json.dumps({document_id: {"entity_names": ["A"]}}), encoding="utf-8"
        )
        (working_dir / "kv_store_full_relations.json").write_text(
            json.dumps({document_id: {"relation_pairs": [["A", "B"]]}}),
            encoding="utf-8",
        )
        (working_dir / "kv_store_entity_chunks.json").write_text(
            json.dumps({"A": {"chunk_ids": [chunk_id]}}), encoding="utf-8"
        )
        (working_dir / "kv_store_relation_chunks.json").write_text(
            json.dumps({"A<SEP>B": {"chunk_ids": [chunk_id]}}), encoding="utf-8"
        )
        (working_dir / "vdb_chunks.json").write_text(
            json.dumps({"embedding_dim": 1024, "data": [{"full_doc_id": document_id}]}),
            encoding="utf-8",
        )
        (working_dir / "vdb_entities.json").write_text(
            json.dumps(
                {
                    "embedding_dim": 1024,
                    "data": [{"source_id": chunk_id, "entity_name": "A"}],
                }
            ),
            encoding="utf-8",
        )
        (working_dir / "vdb_relationships.json").write_text(
            json.dumps(
                {
                    "embedding_dim": 1024,
                    "data": [{"source_id": chunk_id, "src_id": "A", "tgt_id": "B"}],
                }
            ),
            encoding="utf-8",
        )

        pipeline._annotate_lightrag_provenance(  # noqa: SLF001 - targeted helper test
            working_dir=working_dir,
            document_id=document_id,
            confidence_code="A3",
            stored_file_path="raw/doc-1_tweet.png",
        )

        text_chunks = json.loads(
            (working_dir / "kv_store_text_chunks.json").read_text(encoding="utf-8")
        )
        assert text_chunks[chunk_id]["document_id"] == document_id
        assert text_chunks[chunk_id]["confidence_code"] == "A3"

        full_docs = json.loads(
            (working_dir / "kv_store_full_docs.json").read_text(encoding="utf-8")
        )
        assert full_docs[document_id]["confidence_code"] == "A3"

        vdb_entities = json.loads(
            (working_dir / "vdb_entities.json").read_text(encoding="utf-8")
        )
        assert vdb_entities["data"][0]["document_id"] == document_id
        assert vdb_entities["data"][0]["confidence_code"] == "A3"
        assert vdb_entities["data"][0]["file_path"] == "raw/doc-1_tweet.png"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_entity_resolution_logs_success(monkeypatch) -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-er-success-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        log_rows: list[tuple[str, str]] = []

        class _FakeResolver:
            async def arun_generic_resolution(self, **kwargs):
                _ = kwargs
                return {
                    "merged_count": 2,
                    "proposed_count": 2,
                    "merges": [{"source": "A", "target": "B"}],
                }

        pipeline._entity_resolution_service = _FakeResolver()  # type: ignore[attr-defined]

        def _fake_log(
            case_id: str, job_id: str, message: str, level: str = "info"
        ) -> None:
            _ = case_id
            _ = job_id
            log_rows.append((level, message))

        monkeypatch.setattr(pipeline, "_log_job", _fake_log)
        asyncio.run(
            pipeline._run_entity_resolution_after_ingest(  # noqa: SLF001 - targeted helper test
                case_id="case-1",
                job_id="job-1",
                rag=types.SimpleNamespace(_rawabit_openai_client=None),
            )
        )
        assert log_rows
        assert log_rows[-1][0] == "info"
        assert "merged=2" in log_rows[-1][1]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_entity_resolution_logs_warning_on_failure(
    monkeypatch,
) -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-er-failure-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        log_rows: list[tuple[str, str]] = []

        class _FailingResolver:
            async def arun_generic_resolution(self, **kwargs):
                _ = kwargs
                raise RuntimeError("forced-er-failure")

        pipeline._entity_resolution_service = _FailingResolver()  # type: ignore[attr-defined]

        def _fake_log(
            case_id: str, job_id: str, message: str, level: str = "info"
        ) -> None:
            _ = case_id
            _ = job_id
            log_rows.append((level, message))

        monkeypatch.setattr(pipeline, "_log_job", _fake_log)
        asyncio.run(
            pipeline._run_entity_resolution_after_ingest(  # noqa: SLF001 - targeted helper test
                case_id="case-1",
                job_id="job-1",
                rag=types.SimpleNamespace(_rawabit_openai_client=None),
            )
        )
        assert log_rows
        assert log_rows[-1][0] == "warning"
        assert "forced-er-failure" in log_rows[-1][1]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_pipeline_detects_pending_case_jobs() -> None:
    temp_root = Path.cwd() / "data" / "pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"ingestion-pipeline-pending-jobs-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _configure_env(temp_dir)
        settings = get_settings()
        init_db(settings)
        pipeline = IngestionPipeline(settings)

        with get_connection(settings) as connection:
            connection.execute(
                'INSERT INTO "case" (id, name, description, status, case_slug, created_at, updated_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?)',
                (
                    "case-1",
                    "Case 1",
                    None,
                    "open",
                    "case-1",
                    "2026-05-07T00:00:00+00:00",
                    "2026-05-07T00:00:00+00:00",
                ),
            )
            for doc_id in ("doc-1", "doc-2"):
                connection.execute(
                    "INSERT INTO document (id, case_id, original_filename, stored_file_path, mime_type, size_bytes, confidence_source_reliability, confidence_information_validity, confidence_code, ingestion_status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        doc_id,
                        "case-1",
                        f"{doc_id}.txt",
                        f"raw/{doc_id}.txt",
                        "text/plain",
                        10,
                        "A",
                        "1",
                        "A1",
                        "queued",
                        "2026-05-07T00:00:00+00:00",
                        "2026-05-07T00:00:00+00:00",
                    ),
                )
            connection.execute(
                "INSERT INTO ingestion_job (id, case_id, document_id, status, queue_priority) VALUES (?, ?, ?, ?, ?)",
                ("job-1", "case-1", "doc-1", "indexing", "normal"),
            )
            connection.execute(
                "INSERT INTO ingestion_job (id, case_id, document_id, status, queue_priority) VALUES (?, ?, ?, ?, ?)",
                ("job-2", "case-1", "doc-2", "queued", "normal"),
            )

        assert (
            pipeline._case_has_pending_ingestion_jobs("case-1", exclude_job_id="job-1")
            is True
        )

        with get_connection(settings) as connection:
            connection.execute(
                "UPDATE ingestion_job SET status = ? WHERE id = ?", ("complete", "job-2")
            )

        assert (
            pipeline._case_has_pending_ingestion_jobs("case-1", exclude_job_id="job-1")
            is False
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
