from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .. import __version__
from ..ai import AIBuildRequest, AIBuildResponse, AIConfigurationError, build_bot_with_ai
from ..builder.commands import BotCommand
from ..builder.compiler import compile_spec
from ..builder.models import BotSpec, CompileResult
from ..builder.session import SessionStore, build_from_commands
from ..engine_catalog import ENGINE_NODE_SPECS
from .compat import CompatRunStore, create_compat_router


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


def create_app() -> FastAPI:
    app = FastAPI(title="Botmother AgentV4 BotBuilder", version=__version__)
    store = SessionStore()
    compat_store = CompatRunStore()

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
    async def usage() -> dict[str, object]:
        return {
            "user_id": "compat",
            "project_id": None,
            "limit": 0,
            "used": 0,
            "remaining": 0,
            "input_used": 0,
            "output_used": 0,
            "plan": None,
            "status": "unknown",
            "token_reset_at": None,
        }

    @app.post("/v1/sessions", response_model=CreateSessionResponse)
    async def create_session() -> CreateSessionResponse:
        session = store.create()
        return CreateSessionResponse(session_id=session.id, spec=session.spec)

    @app.get("/v1/sessions/{session_id}", response_model=CreateSessionResponse)
    async def get_session(session_id: str) -> CreateSessionResponse:
        session = store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return CreateSessionResponse(session_id=session.id, spec=session.spec)

    @app.post("/v1/sessions/{session_id}/commands", response_model=CommandResponse)
    async def apply_session_command(session_id: str, body: CommandRequest) -> CommandResponse:
        session = store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        result = session.apply(body.command)
        return CommandResponse(session_id=session.id, result=result)

    @app.post("/v1/build", response_model=CompileResult)
    async def build(body: BuildRequest) -> CompileResult:
        return build_from_commands(body.commands)

    @app.post("/v1/compile", response_model=CompileResult)
    async def compile_bot(body: CompileRequest) -> CompileResult:
        return compile_spec(body.spec)

    @app.post("/v1/ai/build", response_model=AIBuildResponse)
    async def ai_build(body: AIBuildRequest) -> AIBuildResponse:
        try:
            return await build_bot_with_ai(body)
        except AIConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="OpenAI request timed out") from exc
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="OpenAI request timed out") from exc
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    compat_router = create_compat_router(compat_store)
    app.include_router(compat_router)

    return app
