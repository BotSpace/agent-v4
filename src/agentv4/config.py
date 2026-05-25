from __future__ import annotations

import os
from functools import lru_cache


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    def __init__(self) -> None:
        self.host = os.getenv("AGENTV4_HOST", "0.0.0.0")
        self.port = int(os.getenv("AGENTV4_PORT", "8094"))

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("AGENTV4_OPENAI_MODEL", "gpt-4o-mini")
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
