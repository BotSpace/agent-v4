from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from .api.auth import CurrentUser
from .backend import get_backend_client
from .config import Settings


def extract_ai_token_usage(plan_usage: dict[str, Any]) -> dict[str, Any]:
    return plan_usage.get("ai_token_usage") or {}


async def require_available_ai_tokens(user: CurrentUser, settings: Settings) -> dict[str, Any]:
    if not settings.enforce_usage_limits:
        return {}

    try:
        backend = get_backend_client(settings)
        plan_usage = await backend.subscription_usage(auth_claims=user.claims)
        usage = extract_ai_token_usage(plan_usage)
        remaining = usage.get("remaining")
        if remaining is None:
            raise HTTPException(
                status_code=503,
                detail={"message": "Token limitini backend orqali tekshirib bo'lmadi."},
            )
        try:
            remaining_tokens = int(remaining)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=503,
                detail={"message": "Backend token limit javobi noto'g'ri."},
            ) from exc
        if remaining_tokens <= 0:
            raise HTTPException(
                status_code=429,
                detail={
                    "message": "Oylik token limiti tugadi. Tarifni yangilang.",
                    "tokens_used": usage.get("used", 0),
                    "tokens_limit": usage.get("limit", 0),
                    "tokens_remaining": remaining_tokens,
                },
            )
        return usage
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"message": "Token limitini backend orqali tekshirib bo'lmadi."},
        ) from exc
