from __future__ import annotations

import asyncio

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from .. import __version__
from ..ai import AIBuildRequest, AIBuildResponse, AIConfigurationError, build_bot_with_ai
from ..backend import get_backend_client
from ..builder.commands import BotCommand
from ..builder.compiler import compile_spec
from ..builder.models import BotSpec, CompileResult
from ..builder.session import SessionStore, build_from_commands
from ..config import Settings, get_settings
from ..engine_catalog import ENGINE_NODE_SPECS
from ..limits import extract_ai_token_usage, require_available_ai_tokens
from .auth import CurrentUser, get_current_user
from .compat import create_compat_router
from .run_store import create_run_store


class CreateSessionResponse(BaseModel):
    session_id: str
    spec: BotSpec


class CommandRequest(BaseModel):
    command: BotCommand


class CommandResponse(BaseModel):
    session_id: str
    result: CompileResult


class BuildRequest(BaseModel):
    commands: list[BotCommand]


class CompileRequest(BaseModel):
    spec: BotSpec


class EngineNodeCatalogItem(BaseModel):
    node_type: str
    example: dict[str, object]
    required: list[str]
    allowed_branches: list[str] | None
    branch_prefixes: list[str]


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    app = FastAPI(title="Botmother AgentV4 BotBuilder", version=__version__)
    app.dependency_overrides[get_settings] = lambda: s
    store = SessionStore()
    compat_store = create_run_store(s)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "agentv4", "version": __version__}

    @app.get("/v1/engine/catalog", response_model=list[EngineNodeCatalogItem])
    async def engine_catalog() -> list[EngineNodeCatalogItem]:
        return [
            EngineNodeCatalogItem(
                node_type=node_type,
                example=spec.example,
                required=sorted(spec.required),
                allowed_branches=sorted(spec.allowed_branches) if spec.allowed_branches is not None else None,
                branch_prefixes=list(spec.branch_prefixes),
            )
            for node_type, spec in sorted(ENGINE_NODE_SPECS.items())
        ]

    @app.get("/usage")
    async def usage(user: CurrentUser = Depends(get_current_user)) -> dict[str, object]:
        if not s.enforce_usage_limits:
            return {
                "user_id": user.user_id,
                "limit": 0,
                "used": 0,
                "remaining": 0,
                "input_used": 0,
                "output_used": 0,
                "plan": None,
                "status": "disabled",
                "token_reset_at": None,
            }
        backend = get_backend_client(s)
        plan_usage = await backend.subscription_usage(auth_claims=user.claims)
        token_usage = extract_ai_token_usage(plan_usage)
        return {
            "user_id": user.user_id,
            "limit": token_usage.get("limit", 0),
            "used": token_usage.get("used", 0),
            "remaining": token_usage.get("remaining", 0),
            "input_used": token_usage.get("input_used", 0),
            "output_used": token_usage.get("output_used", 0),
            "plan": token_usage.get("plan"),
            "status": token_usage.get("status"),
            "token_reset_at": token_usage.get("token_reset_at"),
        }

    @app.post("/v1/sessions", response_model=CreateSessionResponse)
    async def create_session(_: CurrentUser = Depends(get_current_user)) -> CreateSessionResponse:
        session = store.create()
        return CreateSessionResponse(session_id=session.id, spec=session.spec)

    @app.get("/v1/sessions/{session_id}", response_model=CreateSessionResponse)
    async def get_session(
        session_id: str,
        _: CurrentUser = Depends(get_current_user),
    ) -> CreateSessionResponse:
        session = store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return CreateSessionResponse(session_id=session.id, spec=session.spec)

    @app.post("/v1/sessions/{session_id}/commands", response_model=CommandResponse)
    async def apply_session_command(
        session_id: str,
        body: CommandRequest,
        _: CurrentUser = Depends(get_current_user),
    ) -> CommandResponse:
        session = store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        result = session.apply(body.command)
        return CommandResponse(session_id=session.id, result=result)

    @app.post("/v1/build", response_model=CompileResult)
    async def build(
        body: BuildRequest,
        _: CurrentUser = Depends(get_current_user),
    ) -> CompileResult:
        return build_from_commands(body.commands)

    @app.post("/v1/compile", response_model=CompileResult)
    async def compile_bot(
        body: CompileRequest,
        _: CurrentUser = Depends(get_current_user),
    ) -> CompileResult:
        return compile_spec(body.spec)

    @app.post("/v1/ai/build", response_model=AIBuildResponse)
    async def ai_build(
        body: AIBuildRequest,
        user: CurrentUser = Depends(get_current_user),
    ) -> AIBuildResponse:
        try:
            await require_available_ai_tokens(user, s)
            result = await build_bot_with_ai(body)
            if s.report_token_usage and (result.input_tokens or result.output_tokens):
                try:
                    backend = get_backend_client(s)
                    await backend.report_token_usage(
                        user_id=user.user_id,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                    )
                except Exception:
                    pass
            return result
        except AIConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="OpenAI request timed out") from exc
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="OpenAI request timed out") from exc
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    compat_router = create_compat_router(compat_store, settings=s)
    app.include_router(compat_router)

    return app
