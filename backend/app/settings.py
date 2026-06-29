from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

_SETTINGS_OVERRIDES_PATH = Path("settings_overrides.json")

USER_MUTABLE_SETTINGS = {
    "rag_llm_model",
    "rag_vlm_model",
    "rag_embedding_model",
    "rag_embedding_dim_hint",
    "rag_llm_max_tokens",
    "rag_llm_temperature",
    "rag_llm_timeout_seconds",
    "rag_llm_max_async",
    "rag_embedding_max_async",
    "rag_cosine_threshold",
    "rag_naive_cosine_threshold",
    "rag_default_top_k",
    "rag_default_chunk_top_k",
    "ingestion_worker_concurrency",
    "rag_lightrag_max_parallel_insert",
}


def load_settings_overrides() -> dict[str, object] | None:
    if not _SETTINGS_OVERRIDES_PATH.exists():
        return None
    try:
        raw = json.loads(_SETTINGS_OVERRIDES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    return {k: v for k, v in raw.items() if k in USER_MUTABLE_SETTINGS}


def save_settings_overrides(overrides: dict[str, object]) -> None:
    cleaned = {
        k: v
        for k, v in overrides.items()
        if k in USER_MUTABLE_SETTINGS and (not isinstance(v, str) or v.strip())
    }
    _SETTINGS_OVERRIDES_PATH.write_text(
        json.dumps(cleaned, indent=2, sort_keys=True), encoding="utf-8"
    )


@dataclass
class Settings:
    db_path: Path
    cases_root: Path
    ingestion_enabled: bool
    ingestion_poll_interval_seconds: float
    ingestion_worker_concurrency: int
    ingestion_timeout_seconds: float
    ingestion_parse_timeout_seconds: float
    ingest_profile_default: str
    llm_provider_base_url: str
    llm_provider_api_key: str | None
    embedding_provider_base_url: str
    embedding_provider_api_key: str | None
    llm_provider_site_url: str | None
    llm_provider_app_name: str | None
    rag_llm_model: str
    rag_vlm_model: str
    rag_embedding_model: str
    rag_embedding_dim_hint: int
    rag_llm_max_tokens: int
    rag_llm_temperature: float
    rag_llm_timeout_seconds: int
    rag_embedding_timeout_seconds: int
    rag_network_retry_window_seconds: int
    rag_llm_max_async: int
    rag_embedding_max_async: int
    rag_lightrag_max_parallel_insert: int
    rag_lightrag_max_parallel_insert_balanced: int
    rag_lightrag_max_parallel_insert_full_enrichment: int
    rag_parser: str
    rag_parse_method: str
    rag_ocr_mode_default: str
    rag_parser_lang: str | None
    rag_parser_device: str | None
    rag_parser_backend: str | None
    rag_parser_start_page: int | None
    rag_parser_end_page: int | None
    rag_mineru_inter_op_threads: int | None
    rag_mineru_intra_op_threads: int | None
    rag_enable_vlm: bool
    rag_enable_vlm_image_analysis: bool
    rag_enable_vlm_visible_text_extraction: bool
    rag_enable_preinsert_summary: bool
    rag_preinsert_summary_max_input_chars: int
    rag_preinsert_summary_max_tokens: int
    rag_evidence_normalization_max_tokens: int
    rag_chunk_token_size: int
    rag_chunk_overlap_token_size: int
    rag_balanced_force_vlm_image_caption: bool
    rag_vlm_parallel_captions_balanced: int
    rag_vlm_parallel_captions_full_enrichment: int
    rag_image_vlm_min_ocr_chars: int
    rag_pdf_probe_min_chars_per_page: int
    rag_balanced_table_max_pages: int
    rag_balanced_enable_equation_processing: bool
    rag_enable_image_processing: bool
    rag_enable_table_processing: bool
    rag_enable_equation_processing: bool
    rag_display_stats: bool
    rag_cosine_threshold: float
    rag_naive_cosine_threshold: float
    rag_resolution_auto_trigger: bool
    rag_resolution_confidence_threshold: float
    rag_default_top_k: int
    rag_default_chunk_top_k: int
    rag_rerank_model: str
    rag_rerank_provider_base_url: str
    rag_rerank_provider_api_key: str | None
    rag_rerank_enabled: bool
    prompt_catalog_path: Path
    prompt_catalog_auto_reload: bool

    _OVERRIDE_CONVERTERS: dict[str, type[int] | type[float] | type[bool] | type[str]] = field(default_factory=lambda: {
        "rag_llm_model": str,
        "rag_vlm_model": str,
        "rag_embedding_model": str,
        "rag_embedding_dim_hint": int,
        "rag_llm_max_tokens": int,
        "rag_llm_temperature": float,
        "rag_llm_timeout_seconds": int,
        "rag_llm_max_async": int,
        "rag_embedding_max_async": int,
        "rag_cosine_threshold": float,
        "rag_naive_cosine_threshold": float,
        "rag_resolution_confidence_threshold": float,
        "rag_default_top_k": int,
        "rag_default_chunk_top_k": int,
        "ingestion_worker_concurrency": int,
        "rag_lightrag_max_parallel_insert": int,
    })

    def reload_overrides(self) -> None:
        overrides = load_settings_overrides()
        if not overrides:
            return
        for key, converter in self._OVERRIDE_CONVERTERS.items():
            if key in overrides:
                value = overrides[key]
                if converter is bool:
                    value = str(value).strip().lower() in {"1", "true", "yes", "on"}
                else:
                    value = converter(value)
                    if converter is str and not str(value).strip():
                        continue
                setattr(self, key, value)


def _load_dotenv() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def _env_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def get_settings() -> Settings:
    _load_dotenv()
    db_path = Path(os.environ.get("RAWABIT_DB_PATH", "data/rawabit.db"))
    cases_root = Path(os.environ.get("RAWABIT_CASES_ROOT", "cases"))
    ingestion_enabled = _env_bool(os.environ.get("RAWABIT_ENABLE_INGESTION"), False)
    poll_interval = _env_float(os.environ.get("RAWABIT_INGEST_POLL_SECONDS"), 2.0)
    ingestion_worker_concurrency = _env_int(
        os.environ.get("RAWABIT_INGEST_WORKER_CONCURRENCY"), 2
    )
    ingestion_timeout_seconds = _env_float(
        os.environ.get("RAWABIT_INGEST_TIMEOUT_SECONDS"), 900.0
    )
    ingestion_parse_timeout_seconds = _env_float(
        os.environ.get("RAWABIT_INGEST_PARSE_TIMEOUT_SECONDS"), 900.0
    )
    ingest_profile_default = os.environ.get(
        "RAWABIT_INGEST_PROFILE_DEFAULT", "balanced_fast_intel"
    ).strip()
    llm_provider_base_url = os.environ.get(
        "RAWABIT_LLM_PROVIDER_BASE_URL",
        os.environ.get(
            "RAWABIT_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
    ).strip()
    llm_provider_api_key = _env_optional_str(
        os.environ.get("RAWABIT_LLM_PROVIDER_API_KEY")
        or os.environ.get("RAWABIT_OPENROUTER_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    embedding_provider_base_url = os.environ.get(
        "RAWABIT_EMBEDDING_PROVIDER_BASE_URL", llm_provider_base_url
    ).strip()
    embedding_provider_api_key = _env_optional_str(
        os.environ.get("RAWABIT_EMBEDDING_PROVIDER_API_KEY")
        or llm_provider_api_key
    )
    llm_provider_site_url = _env_optional_str(
        os.environ.get("RAWABIT_LLM_PROVIDER_SITE_URL")
        or os.environ.get("RAWABIT_OPENROUTER_SITE_URL")
    )
    llm_provider_app_name = _env_optional_str(
        os.environ.get("RAWABIT_LLM_PROVIDER_APP_NAME")
        or os.environ.get("RAWABIT_OPENROUTER_APP_NAME", "Rawabit GraphRAG MVP")
    )
    rag_llm_model = os.environ.get(
        "RAWABIT_RAG_LLM_MODEL", "google/gemini-2.0-flash-001"
    ).strip()
    rag_vlm_model = os.environ.get(
        "RAWABIT_RAG_VLM_MODEL", "google/gemini-2.0-flash-001"
    ).strip()
    rag_embedding_model = os.environ.get(
        "RAWABIT_RAG_EMBEDDING_MODEL", "openai/text-embedding-3-small"
    ).strip()
    rag_embedding_dim_hint = _env_int(
        os.environ.get("RAWABIT_RAG_EMBEDDING_DIM_HINT"), 1536
    )
    rag_llm_max_tokens = _env_int(os.environ.get("RAWABIT_RAG_LLM_MAX_TOKENS"), 4000)
    rag_llm_temperature = _env_float(os.environ.get("RAWABIT_RAG_LLM_TEMPERATURE"), 0.1)
    rag_llm_timeout_seconds = _env_int(
        os.environ.get("RAWABIT_RAG_LLM_TIMEOUT_SECONDS"), 90
    )
    rag_embedding_timeout_seconds = _env_int(
        os.environ.get("RAWABIT_RAG_EMBEDDING_TIMEOUT_SECONDS"), 45
    )
    rag_network_retry_window_seconds = _env_int(
        os.environ.get("RAWABIT_RAG_NETWORK_RETRY_WINDOW_SECONDS"), 1200
    )
    rag_llm_max_async = _env_int(os.environ.get("RAWABIT_RAG_LLM_MAX_ASYNC"), 2)
    rag_embedding_max_async = _env_int(
        os.environ.get("RAWABIT_RAG_EMBEDDING_MAX_ASYNC"), 4
    )
    rag_lightrag_max_parallel_insert = _env_int(
        os.environ.get("RAWABIT_RAG_MAX_PARALLEL_INSERT"), 2
    )
    rag_lightrag_max_parallel_insert_balanced = _env_int(
        os.environ.get("RAWABIT_RAG_MAX_PARALLEL_INSERT_BALANCED"),
        rag_lightrag_max_parallel_insert,
    )
    rag_lightrag_max_parallel_insert_full_enrichment = _env_int(
        os.environ.get("RAWABIT_RAG_MAX_PARALLEL_INSERT_FULL"),
        rag_lightrag_max_parallel_insert,
    )
    rag_parser = os.environ.get("RAWABIT_RAG_PARSER", "mineru").strip()
    rag_parse_method = os.environ.get("RAWABIT_RAG_PARSE_METHOD", "auto").strip()
    rag_ocr_mode_default = (
        os.environ.get("RAWABIT_RAG_OCR_MODE_DEFAULT", "off").strip().lower()
    )
    if rag_ocr_mode_default not in {"off", "auto", "force"}:
        rag_ocr_mode_default = "off"
    rag_parser_lang = _env_optional_str(os.environ.get("RAWABIT_RAG_PARSER_LANG", "en"))
    rag_parser_device = _env_optional_str(
        os.environ.get("RAWABIT_RAG_PARSER_DEVICE", "cpu")
    )
    rag_parser_backend = _env_optional_str(
        os.environ.get("RAWABIT_RAG_PARSER_BACKEND", "pipeline")
    )
    rag_parser_start_page = _env_optional_int(
        os.environ.get("RAWABIT_RAG_PARSER_START_PAGE")
    )
    rag_parser_end_page = _env_optional_int(
        os.environ.get("RAWABIT_RAG_PARSER_END_PAGE")
    )
    rag_mineru_inter_op_threads = _env_optional_int(
        os.environ.get("RAWABIT_MINERU_INTER_OP_NUM_THREADS")
    )
    rag_mineru_intra_op_threads = _env_optional_int(
        os.environ.get("RAWABIT_MINERU_INTRA_OP_NUM_THREADS")
    )
    rag_enable_vlm = _env_bool(os.environ.get("RAWABIT_RAG_ENABLE_VLM"), True)
    rag_enable_vlm_image_analysis = _env_bool(
        os.environ.get("RAWABIT_RAG_ENABLE_VLM_IMAGE_ANALYSIS"), True
    )
    rag_enable_vlm_visible_text_extraction = _env_bool(
        os.environ.get("RAWABIT_RAG_ENABLE_VLM_VISIBLE_TEXT_EXTRACTION"), True
    )
    rag_enable_preinsert_summary = _env_bool(
        os.environ.get("RAWABIT_RAG_ENABLE_PREINSERT_SUMMARY"), False
    )
    rag_preinsert_summary_max_input_chars = _env_int(
        os.environ.get("RAWABIT_RAG_PREINSERT_SUMMARY_MAX_INPUT_CHARS"), 24000
    )
    rag_preinsert_summary_max_tokens = _env_int(
        os.environ.get("RAWABIT_RAG_PREINSERT_SUMMARY_MAX_TOKENS"), 400
    )
    rag_evidence_normalization_max_tokens = _env_int(
        os.environ.get("RAWABIT_RAG_EVIDENCE_NORMALIZATION_MAX_TOKENS"),
        max(8000, rag_llm_max_tokens),
    )
    rag_chunk_token_size = _env_int(
        os.environ.get("RAWABIT_RAG_CHUNK_TOKEN_SIZE"), 600
    )
    rag_chunk_overlap_token_size = _env_int(
        os.environ.get("RAWABIT_RAG_CHUNK_OVERLAP_TOKEN_SIZE"), 50
    )
    rag_balanced_force_vlm_image_caption = _env_bool(
        os.environ.get("RAWABIT_RAG_BALANCED_FORCE_VLM_IMAGE_CAPTION"), False
    )
    rag_vlm_parallel_captions_balanced = _env_int(
        os.environ.get("RAWABIT_RAG_VLM_PARALLEL_CAPTIONS_BALANCED"), 4
    )
    rag_vlm_parallel_captions_full_enrichment = _env_int(
        os.environ.get("RAWABIT_RAG_VLM_PARALLEL_CAPTIONS_FULL"), 6
    )
    rag_image_vlm_min_ocr_chars = _env_int(
        os.environ.get("RAWABIT_IMAGE_VLM_MIN_OCR_CHARS"), 120
    )
    rag_pdf_probe_min_chars_per_page = _env_int(
        os.environ.get("RAWABIT_PDF_PROBE_MIN_CHARS_PER_PAGE"), 180
    )
    rag_balanced_table_max_pages = _env_int(
        os.environ.get("RAWABIT_RAG_ENABLE_TABLE_BALANCED_MAX_PAGES"), 40
    )
    rag_balanced_enable_equation_processing = _env_bool(
        os.environ.get("RAWABIT_RAG_ENABLE_FORMULA_BALANCED"), False
    )
    rag_enable_image_processing = _env_bool(
        os.environ.get("RAWABIT_RAG_ENABLE_IMAGE_PROCESSING"), True
    )
    rag_enable_table_processing = _env_bool(
        os.environ.get("RAWABIT_RAG_ENABLE_TABLE_PROCESSING"), True
    )
    rag_enable_equation_processing = _env_bool(
        os.environ.get("RAWABIT_RAG_ENABLE_EQUATION_PROCESSING"), False
    )
    rag_display_stats = _env_bool(os.environ.get("RAWABIT_RAG_DISPLAY_STATS"), False)
    rag_cosine_threshold = _env_float(
        os.environ.get("RAWABIT_RAG_COSINE_THRESHOLD"), 0.45
    )
    rag_naive_cosine_threshold = _env_float(
        os.environ.get("RAWABIT_RAG_NAIVE_COSINE_THRESHOLD"), 0.2
    )
    rag_resolution_auto_trigger = _env_bool(
        os.environ.get("RAWABIT_RAG_RESOLUTION_AUTO_TRIGGER"), True
    )
    rag_resolution_confidence_threshold = _env_float(
        os.environ.get("RAWABIT_RAG_RESOLUTION_CONFIDENCE_THRESHOLD"), 0.85
    )
    rag_default_top_k = _env_int(
        os.environ.get("RAWABIT_RAG_TOP_K"), 10
    )
    rag_default_chunk_top_k = _env_int(
        os.environ.get("RAWABIT_RAG_CHUNK_TOP_K"), 10
    )
    rag_rerank_provider_base_url = os.environ.get(
        "RAWABIT_RERANK_PROVIDER_BASE_URL",
        os.environ.get(
            "RAWABIT_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
    ).strip()
    rag_rerank_provider_api_key = _env_optional_str(
        os.environ.get("RAWABIT_RERANK_PROVIDER_API_KEY")
        or os.environ.get("RAWABIT_OPENROUTER_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    rag_rerank_model = os.environ.get(
        "RAWABIT_RAG_RERANK_MODEL"
    ).strip()
    rag_rerank_enabled = _env_bool(
        os.environ.get("RAWABIT_RAG_RERANK_ENABLED"), True
    )
    prompt_catalog_path = Path(
        os.environ.get("RAWABIT_PROMPT_CATALOG_PATH", "backend/config/prompts.json")
    )
    prompt_catalog_auto_reload = _env_bool(
        os.environ.get("RAWABIT_PROMPT_CATALOG_AUTO_RELOAD"), True
    )
    _overrides = load_settings_overrides()
    if _overrides:
        if "rag_llm_model" in _overrides:
            val = str(_overrides["rag_llm_model"]).strip()
            if val:
                rag_llm_model = val
        if "rag_vlm_model" in _overrides:
            val = str(_overrides["rag_vlm_model"]).strip()
            if val:
                rag_vlm_model = val
        if "rag_embedding_model" in _overrides:
            val = str(_overrides["rag_embedding_model"]).strip()
            if val:
                rag_embedding_model = val
        if "rag_embedding_dim_hint" in _overrides:
            rag_embedding_dim_hint = int(_overrides["rag_embedding_dim_hint"])
        if "rag_llm_max_tokens" in _overrides:
            rag_llm_max_tokens = int(_overrides["rag_llm_max_tokens"])
        if "rag_llm_temperature" in _overrides:
            rag_llm_temperature = float(_overrides["rag_llm_temperature"])
        if "rag_llm_timeout_seconds" in _overrides:
            rag_llm_timeout_seconds = int(_overrides["rag_llm_timeout_seconds"])
        if "rag_llm_max_async" in _overrides:
            rag_llm_max_async = int(_overrides["rag_llm_max_async"])
        if "rag_embedding_max_async" in _overrides:
            rag_embedding_max_async = int(_overrides["rag_embedding_max_async"])
        if "rag_cosine_threshold" in _overrides:
            rag_cosine_threshold = float(_overrides["rag_cosine_threshold"])
        if "rag_resolution_confidence_threshold" in _overrides:
            rag_resolution_confidence_threshold = float(_overrides["rag_resolution_confidence_threshold"])
        if "rag_default_top_k" in _overrides:
            rag_default_top_k = int(_overrides["rag_default_top_k"])
        if "rag_default_chunk_top_k" in _overrides:
            rag_default_chunk_top_k = int(_overrides["rag_default_chunk_top_k"])
        if "ingestion_worker_concurrency" in _overrides:
            ingestion_worker_concurrency = int(_overrides["ingestion_worker_concurrency"])
        if "rag_lightrag_max_parallel_insert" in _overrides:
            rag_lightrag_max_parallel_insert = int(_overrides["rag_lightrag_max_parallel_insert"])
    return Settings(
        db_path=db_path,
        cases_root=cases_root,
        ingestion_enabled=ingestion_enabled,
        ingestion_poll_interval_seconds=poll_interval,
        ingestion_worker_concurrency=ingestion_worker_concurrency,
        ingestion_timeout_seconds=ingestion_timeout_seconds,
        ingestion_parse_timeout_seconds=ingestion_parse_timeout_seconds,
        ingest_profile_default=ingest_profile_default,
        llm_provider_base_url=llm_provider_base_url,
        llm_provider_api_key=llm_provider_api_key,
        embedding_provider_base_url=embedding_provider_base_url,
        embedding_provider_api_key=embedding_provider_api_key,
        llm_provider_site_url=llm_provider_site_url,
        llm_provider_app_name=llm_provider_app_name,
        rag_llm_model=rag_llm_model,
        rag_vlm_model=rag_vlm_model,
        rag_embedding_model=rag_embedding_model,
        rag_embedding_dim_hint=rag_embedding_dim_hint,
        rag_llm_max_tokens=rag_llm_max_tokens,
        rag_llm_temperature=rag_llm_temperature,
        rag_llm_timeout_seconds=rag_llm_timeout_seconds,
        rag_embedding_timeout_seconds=rag_embedding_timeout_seconds,
        rag_network_retry_window_seconds=rag_network_retry_window_seconds,
        rag_llm_max_async=rag_llm_max_async,
        rag_embedding_max_async=rag_embedding_max_async,
        rag_lightrag_max_parallel_insert=rag_lightrag_max_parallel_insert,
        rag_lightrag_max_parallel_insert_balanced=rag_lightrag_max_parallel_insert_balanced,
        rag_lightrag_max_parallel_insert_full_enrichment=rag_lightrag_max_parallel_insert_full_enrichment,
        rag_parser=rag_parser,
        rag_parse_method=rag_parse_method,
        rag_ocr_mode_default=rag_ocr_mode_default,
        rag_parser_lang=rag_parser_lang,
        rag_parser_device=rag_parser_device,
        rag_parser_backend=rag_parser_backend,
        rag_parser_start_page=rag_parser_start_page,
        rag_parser_end_page=rag_parser_end_page,
        rag_mineru_inter_op_threads=rag_mineru_inter_op_threads,
        rag_mineru_intra_op_threads=rag_mineru_intra_op_threads,
        rag_enable_vlm=rag_enable_vlm,
        rag_enable_vlm_image_analysis=rag_enable_vlm_image_analysis,
        rag_enable_vlm_visible_text_extraction=rag_enable_vlm_visible_text_extraction,
        rag_enable_preinsert_summary=rag_enable_preinsert_summary,
        rag_preinsert_summary_max_input_chars=rag_preinsert_summary_max_input_chars,
        rag_preinsert_summary_max_tokens=rag_preinsert_summary_max_tokens,
        rag_evidence_normalization_max_tokens=rag_evidence_normalization_max_tokens,
        rag_chunk_token_size=rag_chunk_token_size,
        rag_chunk_overlap_token_size=rag_chunk_overlap_token_size,
        rag_balanced_force_vlm_image_caption=rag_balanced_force_vlm_image_caption,
        rag_vlm_parallel_captions_balanced=rag_vlm_parallel_captions_balanced,
        rag_vlm_parallel_captions_full_enrichment=rag_vlm_parallel_captions_full_enrichment,
        rag_image_vlm_min_ocr_chars=rag_image_vlm_min_ocr_chars,
        rag_pdf_probe_min_chars_per_page=rag_pdf_probe_min_chars_per_page,
        rag_balanced_table_max_pages=rag_balanced_table_max_pages,
        rag_balanced_enable_equation_processing=rag_balanced_enable_equation_processing,
        rag_enable_image_processing=rag_enable_image_processing,
        rag_enable_table_processing=rag_enable_table_processing,
        rag_enable_equation_processing=rag_enable_equation_processing,
        rag_display_stats=rag_display_stats,
        rag_cosine_threshold=rag_cosine_threshold,
        rag_naive_cosine_threshold=rag_naive_cosine_threshold,
        rag_resolution_auto_trigger=rag_resolution_auto_trigger,
        rag_resolution_confidence_threshold=rag_resolution_confidence_threshold,
        rag_default_top_k=rag_default_top_k,
        rag_default_chunk_top_k=rag_default_chunk_top_k,
        rag_rerank_model=rag_rerank_model,
        rag_rerank_provider_base_url=rag_rerank_provider_base_url,
        rag_rerank_provider_api_key=rag_rerank_provider_api_key,
        rag_rerank_enabled=rag_rerank_enabled,
        prompt_catalog_path=prompt_catalog_path,
        prompt_catalog_auto_reload=prompt_catalog_auto_reload,
    )
