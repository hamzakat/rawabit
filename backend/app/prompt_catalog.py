from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RUNTIME_REQUIRED_PROMPT_KEYS = frozenset(
    {
        "ingestion.preinsert_summary",
        "ingestion.evidence_normalization",
        "ingestion.image_caption.with_visible_text",
        "ingestion.image_caption.description_only",
        "ingestion.embedding.dimension_probe_input",
        "analysis.link_projection",
        "analysis.event_projection",
        "analysis.flow_projection",
        "analysis.bundle_repair",
        "analysis.mermaid_repair",
        "summary.entity_detail",
        "summary.relationship_detail",
    }
)


@dataclass(frozen=True)
class PromptDefinition:
    key: str
    text: str
    description: str
    placeholders: tuple[str, ...]
    domain: str
    default_model_hint: str | None
    default_params: dict[str, Any]
    enabled: bool


@dataclass(frozen=True)
class OverrideRule:
    rule_id: str
    enabled: bool
    target: str
    match_type: str
    pattern: str
    replace: str
    priority: int


@dataclass(frozen=True)
class PromptCatalogSnapshot:
    schema_version: str
    metadata: dict[str, Any]
    prompts: dict[str, PromptDefinition]
    override_rules: tuple[OverrideRule, ...]
    library_prompt_catalog: dict[str, dict[str, dict[str, Any]]]


class PromptCatalog:
    def __init__(self, path: Path, auto_reload: bool = True) -> None:
        self._path = path.resolve()
        self._auto_reload = auto_reload
        self._lock = threading.RLock()
        self._mtime: float | None = None
        self._snapshot: PromptCatalogSnapshot | None = None

    def validate_required_keys(self, keys: set[str] | frozenset[str]) -> None:
        snapshot = self._get_snapshot()
        missing = sorted([key for key in keys if key not in snapshot.prompts])
        if missing:
            raise RuntimeError(
                "Prompt catalog is missing required keys: " + ", ".join(missing)
            )

    def render(self, key: str, values: dict[str, Any] | None = None) -> str:
        snapshot = self._get_snapshot()
        definition = snapshot.prompts.get(key)
        if definition is None:
            raise KeyError(f"Prompt key not found: {key}")
        if not definition.enabled:
            raise RuntimeError(f"Prompt key is disabled: {key}")
        payload = values or {}
        missing = [name for name in definition.placeholders if name not in payload]
        if missing:
            raise RuntimeError(
                f"Prompt '{key}' is missing placeholders: {', '.join(sorted(missing))}"
            )
        unknown = [name for name in payload if name not in definition.placeholders]
        if unknown:
            raise RuntimeError(
                f"Prompt '{key}' received unknown placeholders: {', '.join(sorted(unknown))}"
            )
        rendered = definition.text
        for name in definition.placeholders:
            rendered = rendered.replace("{" + name + "}", str(payload[name]))
        return rendered

    def apply_external_overrides(
        self,
        *,
        prompt: str | None = None,
        system_prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> tuple[str | None, str | None, list[dict[str, Any]] | None]:
        snapshot = self._get_snapshot()
        if not snapshot.override_rules:
            return prompt, system_prompt, messages

        next_prompt = prompt
        next_system_prompt = system_prompt
        next_messages = self._clone_messages(messages)
        for rule in snapshot.override_rules:
            if not rule.enabled:
                continue
            if rule.target == "prompt":
                next_prompt = self._apply_rule(next_prompt, rule)
            elif rule.target == "system_prompt":
                next_system_prompt = self._apply_rule(next_system_prompt, rule)
            elif rule.target == "messages":
                next_messages = self._apply_rule_to_messages(next_messages, rule)
        return next_prompt, next_system_prompt, next_messages

    def get_library_prompt_descriptions(self, provider: str) -> dict[str, str]:
        snapshot = self._get_snapshot()
        provider_rows = snapshot.library_prompt_catalog.get(provider, {})
        descriptions: dict[str, str] = {}
        for key, row in provider_rows.items():
            description = row.get("description")
            if isinstance(description, str):
                descriptions[key] = description
        return descriptions

    def get_library_prompt_overrides(self, provider: str) -> dict[str, Any]:
        snapshot = self._get_snapshot()
        provider_rows = snapshot.library_prompt_catalog.get(provider, {})
        overrides: dict[str, Any] = {}
        for key, row in provider_rows.items():
            if "value" not in row:
                continue
            value = row.get("value")
            if value is not None:
                overrides[key] = value
        return overrides

    def _get_snapshot(self) -> PromptCatalogSnapshot:
        with self._lock:
            if self._snapshot is None:
                self._snapshot = self._load_from_disk()
                self._mtime = self._read_mtime()
                return self._snapshot

            if not self._auto_reload:
                return self._snapshot

            current_mtime = self._read_mtime()
            if current_mtime == self._mtime:
                return self._snapshot

            try:
                updated = self._load_from_disk()
            except Exception:
                logger.exception(
                    "Prompt catalog reload failed; keeping previous valid snapshot at %s",
                    self._path,
                )
                self._mtime = current_mtime
                return self._snapshot

            self._snapshot = updated
            self._mtime = current_mtime
            return self._snapshot

    def _read_mtime(self) -> float | None:
        try:
            return self._path.stat().st_mtime
        except FileNotFoundError:
            return None

    def _load_from_disk(self) -> PromptCatalogSnapshot:
        if not self._path.exists():
            raise RuntimeError(f"Prompt catalog file not found: {self._path}")
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Prompt catalog JSON is invalid at {self._path}: {exc}"
            ) from exc
        return self._parse_payload(payload)

    def _parse_payload(self, payload: Any) -> PromptCatalogSnapshot:
        if not isinstance(payload, dict):
            raise RuntimeError("Prompt catalog payload must be a JSON object.")
        schema_version = payload.get("schema_version")
        if not isinstance(schema_version, str) or not schema_version.strip():
            raise RuntimeError("Prompt catalog requires a non-empty 'schema_version'.")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("Prompt catalog requires object 'metadata'.")

        prompts_section = payload.get("prompts")
        if not isinstance(prompts_section, dict) or not prompts_section:
            raise RuntimeError("Prompt catalog requires non-empty object 'prompts'.")
        prompts: dict[str, PromptDefinition] = {}
        for key, row in prompts_section.items():
            if not isinstance(key, str) or not key.strip():
                raise RuntimeError("Prompt key must be a non-empty string.")
            if not isinstance(row, dict):
                raise RuntimeError(f"Prompt '{key}' must be an object.")
            text = row.get("text")
            description = row.get("description")
            placeholders = row.get("placeholders")
            domain = row.get("domain")
            enabled = row.get("enabled")
            default_params = row.get("default_params", {})
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(f"Prompt '{key}' requires non-empty 'text'.")
            if not isinstance(description, str) or not description.strip():
                raise RuntimeError(f"Prompt '{key}' requires non-empty 'description'.")
            if not isinstance(placeholders, list) or not all(
                isinstance(item, str) and item.strip() for item in placeholders
            ):
                raise RuntimeError(
                    f"Prompt '{key}' requires string array 'placeholders'."
                )
            if not isinstance(domain, str) or not domain.strip():
                raise RuntimeError(f"Prompt '{key}' requires non-empty 'domain'.")
            if not isinstance(enabled, bool):
                raise RuntimeError(f"Prompt '{key}' requires boolean 'enabled'.")
            if not isinstance(default_params, dict):
                raise RuntimeError(f"Prompt '{key}' has invalid 'default_params'.")
            default_model_hint = row.get("default_model_hint")
            if default_model_hint is not None and not isinstance(
                default_model_hint, str
            ):
                raise RuntimeError(f"Prompt '{key}' has invalid 'default_model_hint'.")
            prompts[key] = PromptDefinition(
                key=key,
                text=text,
                description=description,
                placeholders=tuple(placeholders),
                domain=domain,
                default_model_hint=default_model_hint,
                default_params=default_params,
                enabled=enabled,
            )

        overrides_section = payload.get("external_overrides", {})
        if not isinstance(overrides_section, dict):
            raise RuntimeError("'external_overrides' must be an object.")
        rules_data = overrides_section.get("rules", [])
        if not isinstance(rules_data, list):
            raise RuntimeError("'external_overrides.rules' must be an array.")
        rules: list[OverrideRule] = []
        for idx, row in enumerate(rules_data):
            if not isinstance(row, dict):
                raise RuntimeError("Each override rule must be an object.")
            target = row.get("target")
            match_type = row.get("match_type")
            pattern = row.get("pattern")
            replace = row.get("replace")
            if target not in {"prompt", "system_prompt", "messages"}:
                raise RuntimeError(f"Override rule {idx} has invalid target.")
            if match_type not in {"exact", "contains", "regex"}:
                raise RuntimeError(f"Override rule {idx} has invalid match_type.")
            if not isinstance(pattern, str):
                raise RuntimeError(f"Override rule {idx} has invalid pattern.")
            if not isinstance(replace, str):
                raise RuntimeError(f"Override rule {idx} has invalid replace.")
            if match_type == "regex":
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise RuntimeError(
                        f"Override rule {idx} has invalid regex pattern: {exc}"
                    ) from exc
            enabled = row.get("enabled", True)
            if not isinstance(enabled, bool):
                raise RuntimeError(f"Override rule {idx} has invalid enabled flag.")
            priority = row.get("priority", idx)
            if not isinstance(priority, int):
                raise RuntimeError(f"Override rule {idx} has invalid priority.")
            rules.append(
                OverrideRule(
                    rule_id=str(row.get("id") or f"rule_{idx}"),
                    enabled=enabled,
                    target=target,
                    match_type=match_type,
                    pattern=pattern,
                    replace=replace,
                    priority=priority,
                )
            )

        library_prompt_catalog_raw = payload.get("library_prompt_catalog", {})
        if not isinstance(library_prompt_catalog_raw, dict):
            raise RuntimeError("'library_prompt_catalog' must be an object.")

        library_prompt_catalog: dict[str, dict[str, dict[str, Any]]] = {}
        for provider, provider_rows in library_prompt_catalog_raw.items():
            if provider not in {"lightrag", "raganything"}:
                raise RuntimeError(
                    f"'library_prompt_catalog' has unsupported provider '{provider}'."
                )
            if not isinstance(provider_rows, dict):
                raise RuntimeError(
                    f"'library_prompt_catalog.{provider}' must be an object."
                )
            parsed_rows: dict[str, dict[str, Any]] = {}
            for key, row in provider_rows.items():
                if not isinstance(key, str) or not key.strip():
                    raise RuntimeError(
                        f"'library_prompt_catalog.{provider}' has invalid prompt key."
                    )
                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"'library_prompt_catalog.{provider}.{key}' must be an object."
                    )
                description = row.get("description")
                if not isinstance(description, str) or not description.strip():
                    raise RuntimeError(
                        f"'library_prompt_catalog.{provider}.{key}.description' "
                        "must be a non-empty string."
                    )
                if "value" in row:
                    value = row.get("value")
                    if value is not None and not isinstance(value, (str, list)):
                        raise RuntimeError(
                            f"'library_prompt_catalog.{provider}.{key}.value' "
                            "must be string, array, or null."
                        )
                    if isinstance(value, list) and not all(
                        isinstance(item, str) for item in value
                    ):
                        raise RuntimeError(
                            f"'library_prompt_catalog.{provider}.{key}.value' "
                            "array must contain only strings."
                        )
                parsed_row = {"description": description}
                if "value" in row:
                    parsed_row["value"] = row.get("value")
                parsed_rows[key] = parsed_row
            library_prompt_catalog[provider] = parsed_rows

        ordered_rules = tuple(sorted(rules, key=lambda item: item.priority))
        return PromptCatalogSnapshot(
            schema_version=schema_version,
            metadata=metadata,
            prompts=prompts,
            override_rules=ordered_rules,
            library_prompt_catalog=library_prompt_catalog,
        )

    def _clone_messages(
        self, messages: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        if messages is None:
            return None
        cloned: list[dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            row = dict(msg)
            content = row.get("content")
            if isinstance(content, list):
                cloned_parts = []
                for part in content:
                    if isinstance(part, dict):
                        cloned_parts.append(dict(part))
                    else:
                        cloned_parts.append(part)
                row["content"] = cloned_parts
            cloned.append(row)
        return cloned

    def _apply_rule(self, text: str | None, rule: OverrideRule) -> str | None:
        if text is None:
            return None
        if rule.match_type == "exact":
            if text == rule.pattern:
                return rule.replace
            return text
        if rule.match_type == "contains":
            if rule.pattern in text:
                return text.replace(rule.pattern, rule.replace)
            return text
        return re.sub(rule.pattern, rule.replace, text)

    def _apply_rule_to_messages(
        self, messages: list[dict[str, Any]] | None, rule: OverrideRule
    ) -> list[dict[str, Any]] | None:
        if messages is None:
            return None
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = self._apply_rule(content, rule)
                continue
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        part["text"] = self._apply_rule(part.get("text"), rule)
        return messages


_catalog_cache: dict[tuple[str, bool], PromptCatalog] = {}
_catalog_cache_lock = threading.Lock()


def get_prompt_catalog(path: Path, auto_reload: bool = True) -> PromptCatalog:
    key = (str(path.resolve()), auto_reload)
    with _catalog_cache_lock:
        catalog = _catalog_cache.get(key)
        if catalog is None:
            catalog = PromptCatalog(path=path, auto_reload=auto_reload)
            _catalog_cache[key] = catalog
        return catalog
