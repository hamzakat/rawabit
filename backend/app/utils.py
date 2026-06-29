from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    cleaned = cleaned.strip("-")
    return cleaned or "case"


def ensure_unique_slug(base_slug: str, existing_slugs: Iterable[str]) -> str:
    if base_slug not in existing_slugs:
        return base_slug
    index = 2
    while True:
        candidate = f"{base_slug}-{index}"
        if candidate not in existing_slugs:
            return candidate
        index += 1
