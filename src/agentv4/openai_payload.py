from __future__ import annotations

from typing import Any


def add_completion_token_limit(payload: dict[str, Any], *, model: str, max_tokens: int) -> None:
    key = "max_completion_tokens" if _uses_completion_token_limit(model) else "max_tokens"
    payload[key] = max_tokens


def _uses_completion_token_limit(model: str) -> bool:
    normalized = model.lower().strip()
    return normalized.startswith(("gpt-5", "o1", "o3", "o4"))
