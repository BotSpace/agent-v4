from __future__ import annotations

import os
from functools import lru_cache


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_redis_url() -> str | None:
    explicit = os.getenv("AGENTV4_REDIS_URL")
    if explicit:
        return explicit

    fallback = os.getenv("REDIS_URL")
    if fallback:
        return fallback

    host = os.getenv("REDIS_HOST")
    port = os.getenv("REDIS_PORT", "6379")
    password = os.getenv("REDIS_PASSWORD")
    addr = os.getenv("REDIS_ADDR")
    if host:
        if password:
            return f"redis://:{password}@{host}:{port}/0"
        return f"redis://{host}:{port}/0"
    if addr:
        if password:
            return f"redis://:{password}@{addr}/0"
        return f"redis://{addr}/0"
    return None


class Settings:
    def __init__(self) -> None:
        self.host = os.getenv("AGENTV4_HOST", "0.0.0.0")
        self.port = int(os.getenv("AGENTV4_PORT", "8094"))
        self.workers = int(os.getenv("AGENTV4_WORKERS", "2"))

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("AGENTV4_OPENAI_MODEL", "gpt-5.2")
        self.openai_timeout = float(os.getenv("AGENTV4_OPENAI_TIMEOUT", "45"))
        self.openai_max_tokens = int(os.getenv("AGENTV4_OPENAI_MAX_TOKENS", "1800"))

        self.backend_base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
        self.backend_internal_token = os.getenv("BACKEND_INTERNAL_TOKEN")
        self.backend_timeout = float(os.getenv("AGENTV4_BACKEND_TIMEOUT", "30"))
        self.enforce_usage_limits = _get_bool("AGENTV4_ENFORCE_USAGE_LIMITS", True)
        self.report_token_usage = _get_bool("AGENTV4_REPORT_TOKEN_USAGE", True)

        self.session_secret = os.getenv("SESSION_SECRET", "change-me-to-a-32-char-secret-string")
        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "RS256")
        self.jwt_public_key_path = os.getenv("JWT_PUBLIC_KEY_PATH", "keys/public.pem")
        self.internal_api_token = os.getenv("AGENTV4_INTERNAL_API_TOKEN") or os.getenv("AGENT_INTERNAL_API_TOKEN")
        self.auth_required = _get_bool("AGENTV4_AUTH_REQUIRED", True)
        self.redis_url = _get_redis_url()
        self.redis_prefix = os.getenv("AGENTV4_REDIS_PREFIX", "agentv4")
        self.run_ttl_seconds = int(os.getenv("AGENTV4_RUN_TTL_SECONDS", "86400"))
        self.thread_ttl_seconds = int(os.getenv("AGENTV4_THREAD_TTL_SECONDS", "604800"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
