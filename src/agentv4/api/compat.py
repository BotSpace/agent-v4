from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..ai import AIBuildRequest, build_bot_with_ai_stream
from ..backend import get_backend_client
from ..compat_agent import (
    BOTBUILDER_TOOL_NAME,
    PROJECT_CREATE_TOOL_NAME,
    PROJECT_DELETE_TOOL_NAME,
    PROJECT_GET_TOOL_NAME,
    PROJECT_LIST_TOOL_NAME,
    PROJECT_UPDATE_TOOL_NAME,
    TOOL_EVENT_NAMES,
)
from ..compat_agent import run_compat_agent_final, run_compat_agent_turn
from ..config import Settings
from ..limits import require_available_ai_tokens
from .auth import CurrentUser, get_current_user


class RunCreateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    thread_id: str | None = None
    baseline_flow: dict[str, Any] | None = None


class RunCreateResponse(BaseModel):
    request_id: str
    status: str


class RunStatusResponse(BaseModel):
    request_id: str
    status: str
    final_response: str | None = None
    error: str | None = None
    pending_approval: dict[str, Any] | None = None
    pending_clarification: dict[str, Any] | None = None
    current_plan: list[dict[str, Any]] | None = None


class ClarifyRequest(BaseModel):
    answer: str = Field(min_length=1)


class ApproveRequest(BaseModel):
    approved: bool
    reason: str | None = None


@dataclass
class CompatRun:
    request_id: str
    project_id: str
    user_id: str
    prompt: str
    thread_id: str | None = None
    auth_claims: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    final_response: str | None = None
    error: str | None = None
    current_plan: list[dict[str, Any]] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue[dict[str, Any] | None]] = field(default_factory=list)
    task: asyncio.Task[None] | None = None


class CompatRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, CompatRun] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        *,
        project_id: str,
        user_id: str,
        prompt: str,
        thread_id: str | None,
        auth_claims: dict[str, Any],
    ) -> CompatRun:
        async with self._lock:
            run = CompatRun(
                request_id=str(uuid4()),
                project_id=project_id,
                user_id=user_id,
                prompt=prompt,
                thread_id=thread_id,
                auth_claims=auth_claims,
            )
            self._runs[run.request_id] = run
            return run

    async def get(self, request_id: str, *, project_id: str, user_id: str) -> CompatRun | None:
        async with self._lock:
            run = self._runs.get(request_id)
            if run is None or run.project_id != project_id or run.user_id != user_id:
                return None
            return run

    async def set_status(
        self,
        request_id: str,
        status: str,
        *,
        final_response: str | None = None,
        error: str | None = None,
    ) -> None:
        async with self._lock:
            run = self._runs.get(request_id)
            if run is None:
                return
            run.status = status
            if final_response is not None:
                run.final_response = final_response
            if error is not None:
                run.error = error

    async def add_event(self, request_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            run = self._runs[request_id]
            event = {
                "seq": len(run.events) + 1,
                "type": event_type,
                "payload": payload,
                "at": datetime.now(UTC).isoformat(),
            }
            run.events.append(event)
            subscribers = list(run.subscribers)
        for queue in subscribers:
            await queue.put(event)
        return event

    async def list_events(self, request_id: str, *, since_seq: int = 0) -> list[dict[str, Any]]:
        async with self._lock:
            run = self._runs.get(request_id)
            if run is None:
                return []
            return [event for event in run.events if int(event["seq"]) > since_seq]

    async def subscribe(self, request_id: str) -> asyncio.Queue[dict[str, Any] | None]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        async with self._lock:
            run = self._runs[request_id]
            run.subscribers.append(queue)
            if run.status in {"completed", "failed", "cancelled"}:
                await queue.put(None)
        return queue

    async def unsubscribe(self, request_id: str, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        async with self._lock:
            run = self._runs.get(request_id)
            if run and queue in run.subscribers:
                run.subscribers.remove(queue)

    async def finish_streams(self, request_id: str) -> None:
        async with self._lock:
            run = self._runs.get(request_id)
            subscribers = list(run.subscribers) if run else []
        for queue in subscribers:
            await queue.put(None)

    async def cancel(self, request_id: str, *, project_id: str, user_id: str) -> bool:
        async with self._lock:
            run = self._runs.get(request_id)
            if run is None or run.project_id != project_id or run.user_id != user_id:
                return False
            if run.task and not run.task.done():
                run.task.cancel()
            run.status = "cancelled"
        await self.add_event(request_id, "run_failed", {"error": "Run cancelled"})
        await self.finish_streams(request_id)
        return True


def create_compat_router(store: CompatRunStore, *, settings: Settings) -> APIRouter:
    router = APIRouter()

    @router.post("/projects/{project_id}/runs", response_model=RunCreateResponse)
    async def create_run(
        project_id: str,
        body: RunCreateRequest,
        user: CurrentUser = Depends(get_current_user),
    ) -> RunCreateResponse:
        await require_available_ai_tokens(user, settings)
        run = await store.create(
            project_id=project_id,
            user_id=user.user_id,
            prompt=body.prompt,
            thread_id=body.thread_id,
            auth_claims=user.claims,
        )
        run.task = asyncio.create_task(_execute_run(store, run.request_id, body.prompt, settings=settings))
        return RunCreateResponse(request_id=run.request_id, status="queued")

    @router.get("/projects/{project_id}/runs/{request_id}", response_model=RunStatusResponse)
    async def get_run(
        project_id: str,
        request_id: str,
        user: CurrentUser = Depends(get_current_user),
    ) -> RunStatusResponse:
        run = await store.get(request_id, project_id=project_id, user_id=user.user_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return RunStatusResponse(
            request_id=run.request_id,
            status=run.status,
            final_response=run.final_response,
            error=run.error,
            current_plan=run.current_plan,
        )

    @router.get("/projects/{project_id}/runs/{request_id}/events")
    async def stream_events(
        project_id: str,
        request_id: str,
        since_seq: int = 0,
        user: CurrentUser = Depends(get_current_user),
    ) -> StreamingResponse:
        run = await store.get(request_id, project_id=project_id, user_id=user.user_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return StreamingResponse(
            _event_stream(store, request_id, since_seq=since_seq),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/projects/{project_id}/runs/{request_id}/clarify")
    async def clarify_run(
        project_id: str,
        request_id: str,
        _: ClarifyRequest,
        user: CurrentUser = Depends(get_current_user),
    ) -> dict[str, str]:
        run = await store.get(request_id, project_id=project_id, user_id=user.user_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"status": run.status}

    @router.post("/projects/{project_id}/runs/{request_id}/approve")
    async def approve_run(
        project_id: str,
        request_id: str,
        body: ApproveRequest,
        user: CurrentUser = Depends(get_current_user),
    ) -> dict[str, str]:
        run = await store.get(request_id, project_id=project_id, user_id=user.user_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"status": run.status, "approved": "true" if body.approved else "false"}

    @router.post("/projects/{project_id}/runs/{request_id}/cancel")
    async def cancel_run(
        project_id: str,
        request_id: str,
        user: CurrentUser = Depends(get_current_user),
    ) -> dict[str, str]:
        if not await store.cancel(request_id, project_id=project_id, user_id=user.user_id):
            raise HTTPException(status_code=404, detail="Run not found")
        return {"status": "cancelling"}

    return router


async def _execute_run(
    store: CompatRunStore,
    request_id: str,
    prompt: str,
    *,
    settings: Settings,
) -> None:
    await store.set_status(request_id, "running")
    await store.add_event(request_id, "run_started", {"request_id": request_id})
    try:
        first_turn = await run_compat_agent_turn(prompt, settings=settings)
        total_input_tokens = first_turn.input_tokens
        total_output_tokens = first_turn.output_tokens
        if first_turn.text:
            await store.add_event(
                request_id,
                "assistant_message",
                {
                    "text": first_turn.text,
                    "usage": {
                        "input_tokens": first_turn.input_tokens,
                        "output_tokens": first_turn.output_tokens,
                    },
                },
            )

        async with store._lock:
            run = store._runs.get(request_id)
            project_id = run.project_id if run else None
            user_id = run.user_id if run else ""

        tool_call = next((call for call in first_turn.tool_calls if call.name in TOOL_EVENT_NAMES), None)
        if tool_call is None:
            message = first_turn.text or "Tushundim."
            await store.set_status(request_id, "completed", final_response=message)
            await report_usage_if_needed(
                settings=settings,
                user_id=user_id,
                project_id=project_id,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            )
            await store.add_event(request_id, "run_completed", {"request_id": request_id})
            return

        await store.add_event(
            request_id,
            "tool_call",
            {
                "id": tool_call.id,
                "name": TOOL_EVENT_NAMES[tool_call.name],
                "arguments": tool_call.arguments,
            },
        )

        if tool_call.name != BOTBUILDER_TOOL_NAME:
            tool_content, tool_data, is_error = await execute_backend_tool(
                tool_call.name,
                tool_call.arguments,
                settings=settings,
                auth_claims=run.auth_claims if run else {},
            )
            await store.add_event(
                request_id,
                "tool_result",
                {
                    "tool_call_id": tool_call.id,
                    "content": tool_content,
                    "is_error": is_error,
                    "data": tool_data,
                },
            )
            final_turn = await run_compat_agent_final(
                prompt,
                assistant_text=first_turn.text,
                tool_call=tool_call,
                tool_content=tool_content,
                tool_data=tool_data,
                settings=settings,
            )
            total_input_tokens += final_turn.input_tokens
            total_output_tokens += final_turn.output_tokens
            message = final_turn.text or tool_content
            status = "failed" if is_error else "completed"
            await store.set_status(request_id, status, final_response=message, error=message if is_error else None)
            await store.add_event(
                request_id,
                "assistant_message",
                {
                    "text": message,
                    "usage": {
                        "input_tokens": final_turn.input_tokens,
                        "output_tokens": final_turn.output_tokens,
                    },
                },
            )
            await report_usage_if_needed(
                settings=settings,
                user_id=user_id,
                project_id=project_id,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            )
            await store.add_event(
                request_id,
                "run_failed" if is_error else "run_completed",
                {"error": message} if is_error else {"request_id": request_id},
            )
            return

        tool_prompt = str(tool_call.arguments.get("prompt") or prompt)
        stream_buffer = ""
        stream_started = False

        async def emit_stream_progress(delta: str) -> None:
            nonlocal stream_buffer, stream_started
            if not stream_started:
                stream_started = True
                await store.add_event(
                    request_id,
                    "thinking_delta",
                    {"delta": "Model javobni real time tayyorlayapti...\n"},
                )
            stream_buffer += delta
            if len(stream_buffer) < 120:
                return
            stream_buffer = ""
            await store.add_event(
                request_id,
                "thinking_delta",
                {"delta": "Yangi chunk olindi, bot komandalar yig'ilmoqda...\n"},
            )

        result = await build_bot_with_ai_stream(
            AIBuildRequest(prompt=tool_prompt, language="uz", max_repair_attempts=2),
            on_delta=emit_stream_progress,
        )
        if stream_started:
            await store.add_event(
                request_id,
                "thinking_delta",
                {"delta": "Model javobi olindi, flow compile qilinmoqda...\n"},
            )
        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens
        flow = result.result.flow.model_dump(mode="json") if result.result.flow else None
        issues = [issue.model_dump(mode="json") for issue in result.result.issues]
        tool_content = "Flow compiled" if result.ok else "Flow validation failed"
        tool_data = {
            "ok": result.ok,
            "attempts": result.attempts,
            "commands": result.commands,
            "flow": flow,
            "issues": issues,
        }
        await store.add_event(
            request_id,
            "tool_result",
            {
                "tool_call_id": tool_call.id,
                "content": tool_content,
                "is_error": not result.ok,
                "data": tool_data,
            },
        )
        if result.ok:
            final_turn = await run_compat_agent_final(
                prompt,
                assistant_text=first_turn.text,
                tool_call=tool_call,
                tool_content=tool_content,
                tool_data=tool_data,
                settings=settings,
            )
            total_input_tokens += final_turn.input_tokens
            total_output_tokens += final_turn.output_tokens
            message = final_turn.text or "Flow tayyor. Natija `tool_result.data.flow` ichida."
            await store.set_status(request_id, "completed", final_response=message)
            await store.add_event(
                request_id,
                "assistant_message",
                {
                    "text": message,
                    "usage": {
                        "input_tokens": final_turn.input_tokens,
                        "output_tokens": final_turn.output_tokens,
                    },
                },
            )
            await report_usage_if_needed(
                settings=settings,
                user_id=user_id,
                project_id=project_id,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            )
            await store.add_event(request_id, "run_completed", {"request_id": request_id})
        else:
            message = "Bot flow to'liq compile bo'lmadi. Issues: " + ", ".join(issue["code"] for issue in issues)
            await store.set_status(request_id, "failed", error=message)
            await store.add_event(request_id, "assistant_message", {"text": message, "usage": {}})
            await store.add_event(request_id, "run_failed", {"error": message})
    except asyncio.CancelledError:
        await store.set_status(request_id, "cancelled", error="Run cancelled")
        raise
    except Exception as exc:
        message = str(exc)
        await store.set_status(request_id, "failed", error=message)
        await store.add_event(request_id, "run_failed", {"error": message})
    finally:
        await store.finish_streams(request_id)


async def _event_stream(
    store: CompatRunStore,
    request_id: str,
    *,
    since_seq: int,
    heartbeat_seconds: float = 15.0,
) -> AsyncIterator[str]:
    for event in await store.list_events(request_id, since_seq=since_seq):
        yield _sse_event(event["type"], event)

    queue = await store.subscribe(request_id)
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
                continue
            if event is None:
                yield _sse_event("close", {"reason": "run_finished"})
                return
            yield _sse_event(event["type"], event)
    finally:
        await store.unsubscribe(request_id, queue)


def _sse_event(event_type: str, payload: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def report_usage_if_needed(
    *,
    settings: Settings,
    user_id: str,
    project_id: str | None,
    input_tokens: int,
    output_tokens: int,
) -> None:
    if not settings.report_token_usage or not (input_tokens or output_tokens):
        return
    try:
        backend = get_backend_client(settings)
        await backend.report_token_usage(
            user_id=user_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            project_id=project_id,
        )
    except Exception:
        pass


async def execute_backend_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    settings: Settings,
    auth_claims: dict[str, Any],
) -> tuple[str, dict[str, Any], bool]:
    backend = get_backend_client(settings)
    try:
        if name == PROJECT_LIST_TOOL_NAME:
            projects = await backend.list_projects(search=arguments.get("search"), auth_claims=auth_claims)
            summary = [
                {"id": item.get("id"), "name": item.get("name"), "status": item.get("status")}
                for item in projects
            ]
            names = ", ".join(f"{item['name']} (id={item['id']})" for item in summary) or "(none)"
            return f"Found {len(summary)} project(s): {names}", {"projects": summary}, False

        if name == PROJECT_GET_TOOL_NAME:
            project_id = arguments.get("id")
            project = await backend.get_project(project_id, auth_claims=auth_claims)
            if not arguments.get("include_flow") and isinstance(project.get("flow"), dict):
                project = {key: value for key, value in project.items() if key != "flow"}
            return f"Project `{project.get('name', project_id)}` loaded.", {"project": project}, False

        if name == PROJECT_CREATE_TOOL_NAME:
            body = {key: value for key, value in arguments.items() if key in {"name", "description", "token"}}
            project = await backend.create_project(body, auth_claims=auth_claims)
            return (
                f"Project `{project.get('name')}` created with id `{project.get('id')}`.",
                {"project": project},
                False,
            )

        if name == PROJECT_UPDATE_TOOL_NAME:
            project_id = arguments.get("id")
            body = {
                key: value
                for key, value in arguments.items()
                if key in {"name", "description", "token", "flow"} and value is not None
            }
            if not body:
                return "Nothing to update.", {"project_id": project_id}, True
            project = await backend.update_project(project_id, body, auth_claims=auth_claims)
            return f"Project `{project_id}` updated.", {"project": project}, False

        if name == PROJECT_DELETE_TOOL_NAME:
            project_id = arguments.get("id")
            await backend.delete_project(project_id, auth_claims=auth_claims)
            return f"Project `{project_id}` deleted.", {"project_id": project_id}, False

        return f"Unknown tool `{name}`.", {"tool": name}, True
    except Exception as exc:
        return f"Backend error: {exc}", {"tool": name, "error": str(exc)}, True
