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
    CompatAgentTurn,
    CompatToolCall,
    PROJECT_CREATE_TOOL_NAME,
    PROJECT_DELETE_TOOL_NAME,
    PROJECT_GET_TOOL_NAME,
    PROJECT_LIST_TOOL_NAME,
    PROJECT_UPDATE_TOOL_NAME,
    TOOL_EVENT_NAMES,
    compat_system_prompt,
)
from ..compat_agent import run_compat_agent_final, run_compat_agent_messages
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
        self._thread_messages: dict[tuple[str, str], list[dict[str, str]]] = {}
        self._thread_project_ids: dict[tuple[str, str], str] = {}
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

    async def thread_context(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        fallback_project_id: str,
    ) -> tuple[list[dict[str, str]], str]:
        if thread_id is None:
            return [], fallback_project_id
        key = (user_id, thread_id)
        async with self._lock:
            messages = list(self._thread_messages.get(key, []))
            project_id = self._thread_project_ids.get(key) or fallback_project_id
        return messages, project_id

    async def append_thread_message(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        role: str,
        content: str,
        limit: int = 20,
    ) -> None:
        if thread_id is None or not content:
            return
        key = (user_id, thread_id)
        async with self._lock:
            messages = self._thread_messages.setdefault(key, [])
            messages.append({"role": role, "content": content})
            if len(messages) > limit:
                del messages[: len(messages) - limit]

    async def set_thread_project_id(self, *, user_id: str, thread_id: str | None, project_id: str | int | None) -> None:
        if thread_id is None or project_id is None:
            return
        async with self._lock:
            self._thread_project_ids[(user_id, thread_id)] = str(project_id)

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
    final_message: str | None = None
    completed_successfully = False
    user_id = ""
    thread_id: str | None = None
    try:
        async with store._lock:
            run = store._runs.get(request_id)
            project_id = run.project_id if run else None
            user_id = run.user_id if run else ""
            thread_id = run.thread_id if run else None
            auth_claims = run.auth_claims if run else {}
        history, current_project_id = await store.thread_context(
            user_id=user_id,
            thread_id=thread_id,
            fallback_project_id=project_id or "",
        )
        await store.append_thread_message(user_id=user_id, thread_id=thread_id, role="user", content=prompt)

        first_turn = await run_compat_agent_messages(
            _initial_agent_messages(prompt, history=history, current_project_id=current_project_id),
            settings=settings,
            allow_tools=True,
        )
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

        tool_call = next((call for call in first_turn.tool_calls if call.name in TOOL_EVENT_NAMES), None)
        if tool_call is None:
            message = first_turn.text or "Tushundim."
            await store.set_status(request_id, "completed", final_response=message)
            final_message = message
            completed_successfully = True
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
            message, is_error, loop_input, loop_output = await execute_backend_tool_loop(
                store,
                request_id,
                first_turn=first_turn,
                first_tool_call=tool_call,
                settings=settings,
                auth_claims=auth_claims,
                current_project_id=current_project_id,
                base_messages=_initial_agent_messages(prompt, history=history, current_project_id=current_project_id),
                user_id=user_id,
                thread_id=thread_id,
            )
            total_input_tokens += loop_input
            total_output_tokens += loop_output
            await store.set_status(
                request_id,
                "failed" if is_error else "completed",
                final_response=message,
                error=message if is_error else None,
            )
            final_message = message
            completed_successfully = not is_error
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
            saved_project_id = current_project_id
            if flow and saved_project_id:
                save_content, save_data, save_error = await save_flow_to_project(
                    settings=settings,
                    auth_claims=auth_claims,
                    project_id=saved_project_id,
                    flow=flow,
                )
                await store.add_event(
                    request_id,
                    "tool_call",
                    {
                        "id": f"{tool_call.id}:project_update",
                        "name": TOOL_EVENT_NAMES[PROJECT_UPDATE_TOOL_NAME],
                        "arguments": {"id": saved_project_id, "flow": "<compiled_flow>"},
                    },
                )
                await store.add_event(
                    request_id,
                    "tool_result",
                    {
                        "tool_call_id": f"{tool_call.id}:project_update",
                        "content": save_content,
                        "is_error": save_error,
                        "data": save_data,
                    },
                )
                tool_data["saved_project_id"] = saved_project_id
                tool_data["save"] = save_data
                if save_error:
                    message = f"Flow compile bo'ldi, lekin project `{saved_project_id}` ga saqlanmadi: {save_content}"
                    await store.set_status(request_id, "failed", error=message)
                    final_message = message
                    await store.add_event(request_id, "assistant_message", {"text": message, "usage": {}})
                    await store.add_event(request_id, "run_failed", {"error": message})
                    return
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
            final_message = message
            completed_successfully = True
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
            final_message = message
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
        if completed_successfully and final_message:
            await store.append_thread_message(user_id=user_id, thread_id=thread_id, role="assistant", content=final_message)
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


def _initial_agent_messages(
    prompt: str,
    *,
    history: list[dict[str, str]],
    current_project_id: str | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": compat_system_prompt()}]
    if current_project_id:
        messages.append(
            {
                "role": "system",
                "content": f"Current Botmother project_id is `{current_project_id}`. Use it for 'this project' requests.",
            }
        )
    messages.extend(history[-20:])
    messages.append({"role": "user", "content": prompt})
    return messages


async def save_flow_to_project(
    *,
    settings: Settings,
    auth_claims: dict[str, Any],
    project_id: str | int,
    flow: dict[str, Any],
) -> tuple[str, dict[str, Any], bool]:
    try:
        project = await get_backend_client(settings).update_project(project_id, {"flow": flow}, auth_claims=auth_claims)
        return f"Flow saved to project `{project_id}`.", {"project": project, "project_id": project_id}, False
    except Exception as exc:
        return f"Backend error: {exc}", {"project_id": project_id, "error": str(exc)}, True


async def execute_backend_tool_loop(
    store: CompatRunStore,
    request_id: str,
    *,
    first_turn: CompatAgentTurn,
    first_tool_call: CompatToolCall,
    settings: Settings,
    auth_claims: dict[str, Any],
    current_project_id: str | None,
    base_messages: list[dict[str, Any]],
    user_id: str,
    thread_id: str | None,
    max_tool_rounds: int = 6,
) -> tuple[str, bool, int, int]:
    messages = list(base_messages)
    current_turn = first_turn
    current_tool_call = first_tool_call
    input_tokens = 0
    output_tokens = 0
    last_content = ""

    for round_index in range(max_tool_rounds):
        tool_content, tool_data, is_error = await execute_backend_tool(
            current_tool_call.name,
            current_tool_call.arguments,
            settings=settings,
            auth_claims=auth_claims,
            current_project_id=current_project_id,
        )
        last_content = tool_content
        await store.add_event(
            request_id,
            "tool_result",
            {
                "tool_call_id": current_tool_call.id,
                "content": tool_content,
                "is_error": is_error,
                "data": tool_data,
            },
        )
        next_project_id = _project_id_from_tool_result(current_tool_call.name, current_tool_call.arguments, tool_data)
        if next_project_id is not None:
            current_project_id = str(next_project_id)
            await store.set_thread_project_id(user_id=user_id, thread_id=thread_id, project_id=current_project_id)
        messages.extend(_tool_exchange_messages(current_turn, current_tool_call, tool_content, tool_data))

        next_turn = await run_compat_agent_messages(
            messages,
            settings=settings,
            allow_tools=not is_error and round_index + 1 < max_tool_rounds,
        )
        input_tokens += next_turn.input_tokens
        output_tokens += next_turn.output_tokens
        if next_turn.text:
            await store.add_event(
                request_id,
                "assistant_message",
                {
                    "text": next_turn.text,
                    "usage": {
                        "input_tokens": next_turn.input_tokens,
                        "output_tokens": next_turn.output_tokens,
                    },
                },
            )

        next_tool_call = next((call for call in next_turn.tool_calls if call.name in TOOL_EVENT_NAMES), None)
        if is_error or next_tool_call is None:
            return next_turn.text or tool_content, is_error, input_tokens, output_tokens
        if next_tool_call.name == BOTBUILDER_TOOL_NAME:
            message = next_turn.text or "Bot flow qurish uchun botbuilder tool tayyor, lekin project tool zanjiri yakunlandi."
            return message, False, input_tokens, output_tokens

        await store.add_event(
            request_id,
            "tool_call",
            {
                "id": next_tool_call.id,
                "name": TOOL_EVENT_NAMES[next_tool_call.name],
                "arguments": next_tool_call.arguments,
            },
        )
        current_turn = next_turn
        current_tool_call = next_tool_call

    return last_content or "Tool limit reached.", False, input_tokens, output_tokens


def _tool_exchange_messages(
    turn: CompatAgentTurn,
    tool_call: CompatToolCall,
    tool_content: str,
    tool_data: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": turn.text or None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps({"content": tool_content, "data": tool_data}, ensure_ascii=False),
        },
    ]


def _tool_event(name: str) -> str:
    return TOOL_EVENT_NAMES.get(name, name)


def _project_id_arg(arguments: dict[str, Any], current_project_id: str | None) -> str | int | None:
    return arguments.get("project_id") or current_project_id


def _limit_arg(arguments: dict[str, Any], default: int, maximum: int) -> int:
    try:
        value = int(arguments.get("limit") or default)
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, maximum))


def _project_id_from_tool_result(
    tool_name: str,
    arguments: dict[str, Any],
    tool_data: dict[str, Any],
) -> str | int | None:
    if tool_name == PROJECT_CREATE_TOOL_NAME:
        project = tool_data.get("project")
        if isinstance(project, dict):
            return project.get("id")
    if tool_name == PROJECT_GET_TOOL_NAME:
        return arguments.get("id")
    if tool_name == PROJECT_UPDATE_TOOL_NAME:
        project = tool_data.get("project")
        if isinstance(project, dict):
            return project.get("id") or arguments.get("id")
        return arguments.get("id")
    return None


async def execute_backend_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    settings: Settings,
    auth_claims: dict[str, Any],
    current_project_id: str | None = None,
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

        event_name = _tool_event(name)

        if name == "env_list":
            project_id = _project_id_arg(arguments, current_project_id)
            if project_id is None:
                return "No project_id in context or input.", {"tool": event_name}, True
            items = await backend.list_env(project_id, auth_claims=auth_claims)
            safe = [{"id": item.get("id"), "key": item.get("key")} for item in items]
            keys = ", ".join(str(item.get("key")) for item in safe if item.get("key")) or "(none)"
            return f"{len(safe)} env var(s). Keys: {keys}", {"env": safe}, False

        if name == "env_set":
            project_id = _project_id_arg(arguments, current_project_id)
            if project_id is None:
                return "No project_id in context or input.", {"tool": event_name}, True
            await backend.create_env(
                {"project": project_id, "key": arguments.get("key"), "value": arguments.get("value")},
                auth_claims=auth_claims,
            )
            return f"Env var `{arguments.get('key')}` set for project `{project_id}`.", {"key": arguments.get("key")}, False

        if name == "env_unset":
            env_id = arguments.get("env_id")
            await backend.delete_env(env_id, auth_claims=auth_claims)
            return f"Env var `{env_id}` removed.", {"env_id": env_id}, False

        if name == "release_list":
            project_id = _project_id_arg(arguments, current_project_id)
            if project_id is None:
                return "No project_id available.", {"tool": event_name}, True
            items = await backend.list_releases(project_id, auth_claims=auth_claims)
            payload = [
                {
                    "id": item.get("id"),
                    "version": item.get("version"),
                    "status": item.get("status"),
                    "created_at": item.get("created_at"),
                }
                for item in items[: _limit_arg(arguments, 10, 50)]
            ]
            summary = ", ".join(f"v{item['version']}({item['status']})" for item in payload if item.get("version")) or "(none)"
            return f"{len(payload)} release(s): {summary}", {"releases": payload}, False

        if name == "release_deploy":
            project_id = _project_id_arg(arguments, current_project_id)
            if project_id is None:
                return "No project_id available.", {"tool": event_name}, True
            release = await backend.create_release(
                project_id,
                plan_id=arguments.get("plan_id"),
                months=arguments.get("months"),
                auth_claims=auth_claims,
            )
            return (
                f"Deploy queued (release id={release.get('id')}, version={release.get('version')}).",
                {"release": release},
                False,
            )

        if name == "release_status":
            project_id = _project_id_arg(arguments, current_project_id)
            if project_id is None:
                return "No project_id available.", {"tool": event_name}, True
            release_id = arguments.get("release_id")
            items = await backend.list_releases(project_id, auth_claims=auth_claims)
            release = next((item for item in items if str(item.get("id")) == str(release_id)), None)
            if release is None:
                return f"Release `{release_id}` not found.", {"release_id": release_id}, True
            return f"Release `{release_id}` status: {release.get('status')}", {"release": release}, False

        if name == "plans_list":
            items = await backend.list_plans(auth_claims=auth_claims)
            plans = [
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "price": item.get("price"),
                    "currency": item.get("currency"),
                    "is_default": item.get("is_default"),
                }
                for item in items
            ]
            summary = ", ".join(f"{item['name']}#{item['id']}({item['price']})" for item in plans if item.get("name")) or "(none)"
            return f"{len(plans)} plan(s): {summary}", {"plans": plans}, False

        if name == "file_list":
            items = await backend.list_files(_project_id_arg(arguments, current_project_id), auth_claims=auth_claims)
            files = [
                {"id": item.get("id"), "name": item.get("name"), "url": item.get("url"), "size": item.get("size")}
                for item in items[: _limit_arg(arguments, 50, 200)]
            ]
            names = ", ".join(str(item.get("name") or item.get("id")) for item in files) or "(none)"
            return f"{len(files)} file(s): {names}", {"files": files}, False

        if name == "file_delete":
            file_id = arguments.get("file_id")
            await backend.delete_file(file_id, auth_claims=auth_claims)
            return f"File `{file_id}` deleted.", {"file_id": file_id}, False

        if name == "collection_list":
            project_id = _project_id_arg(arguments, current_project_id)
            if project_id is None:
                return "No project_id available.", {"tool": event_name}, True
            collections = await backend.list_collections(project_id, auth_claims=auth_claims)
            return f"{len(collections)} collection(s).", {"collections": collections}, False

        if name == "collection_get":
            collection_id = arguments.get("collection_id")
            collection = await backend.get_collection(collection_id, auth_claims=auth_claims)
            return f"Collection `{collection_id}` loaded.", {"collection": collection}, False

        if name == "chat_list_users":
            project_id = _project_id_arg(arguments, current_project_id)
            if project_id is None:
                return "No project_id.", {"tool": event_name}, True
            users = await backend.list_chat_users(project_id, auth_claims=auth_claims)
            users = users[: _limit_arg(arguments, 50, 500)]
            return f"{len(users)} user(s).", {"users": users}, False

        if name == "chat_user_history":
            project_id = _project_id_arg(arguments, current_project_id)
            if project_id is None:
                return "No project_id.", {"tool": event_name}, True
            history = await backend.user_chat_history(project_id, arguments.get("chat_id"), auth_claims=auth_claims)
            return f"{len(history)} message(s).", {"history": history}, False

        if name == "broadcast_preview":
            target = arguments.get("target") or "all"
            ids = list(arguments.get("telegram_ids") or []) if target == "custom" else []
            return (
                f"Broadcast preview: target={target}, recipients={len(ids) if ids else 'all'}",
                {
                    "text": arguments.get("text"),
                    "target": target,
                    "recipient_count": len(ids) if ids else None,
                    "sample": ids[:5],
                },
                False,
            )

        if name == "broadcast_send":
            project_id = _project_id_arg(arguments, current_project_id)
            text = arguments.get("text")
            if not text:
                return "No broadcast text.", {"tool": event_name}, True
            if project_id is None:
                return "No project_id available.", {"tool": event_name}, True
            body: dict[str, Any] = {"project": project_id, "caption": text, "type": "text"}
            ids = arguments.get("telegram_ids")
            if ids:
                body["target_telegram_ids"] = ids
            message = await backend.create_message(body, auth_claims=auth_claims)
            return (
                f"Broadcast queued (message id={message.get('id')}). Total recipients: {message.get('total') or 'pending'}.",
                {"message": message},
                False,
            )

        if name == "broadcast_status":
            messages = await backend.list_messages(_project_id_arg(arguments, current_project_id), auth_claims=auth_claims)
            broadcasts = [
                {
                    "id": item.get("id"),
                    "status": item.get("status"),
                    "total": item.get("total"),
                    "done": item.get("done"),
                    "fail": item.get("fail"),
                }
                for item in messages[: _limit_arg(arguments, 10, 50)]
            ]
            return f"{len(broadcasts)} broadcast(s).", {"broadcasts": broadcasts}, False

        if name == "message_delete":
            message_id = arguments.get("message_id")
            await backend.delete_message(message_id, auth_claims=auth_claims)
            return f"Message `{message_id}` deleted.", {"message_id": message_id}, False

        if name.startswith("cms_"):
            return await execute_cms_tool(backend, name, arguments, auth_claims=auth_claims, current_project_id=current_project_id)

        return f"Unknown tool `{name}`.", {"tool": name}, True
    except Exception as exc:
        return f"Backend error: {exc}", {"tool": name, "error": str(exc)}, True


async def execute_cms_tool(
    backend: Any,
    name: str,
    arguments: dict[str, Any],
    *,
    auth_claims: dict[str, Any],
    current_project_id: str | None,
) -> tuple[str, dict[str, Any], bool]:
    project_id = _project_id_arg(arguments, current_project_id)
    if project_id is None:
        return "No project_id.", {"tool": _tool_event(name)}, True

    if name == "cms_list_collections":
        items = await backend.cms_list_collections(project_id, auth_claims=auth_claims)
        payload = [
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "label": item.get("label"),
                "fields_count": item.get("fields_count"),
                "system": item.get("system"),
            }
            for item in items
        ]
        names = ", ".join(str(item.get("name")) for item in payload if item.get("name")) or "(none)"
        return f"{len(payload)} collection(s): {names}", {"collections": payload}, False

    if name == "cms_get_collection":
        collection = await backend.cms_get_collection(project_id, arguments.get("name"), auth_claims=auth_claims)
        return f"Collection `{arguments.get('name')}` loaded.", {"collection": collection}, False

    if name == "cms_create_collection":
        body = {
            key: value
            for key, value in arguments.items()
            if key in {"name", "label", "description", "fields"} and value is not None
        }
        collection = await backend.cms_create_collection(project_id, body, auth_claims=auth_claims)
        return f"Collection `{collection.get('name')}` created (id={collection.get('id')}).", {"collection": collection}, False

    if name == "cms_delete_collection":
        collection_name = arguments.get("name")
        await backend.cms_delete_collection(project_id, collection_name, auth_claims=auth_claims)
        return f"Collection `{collection_name}` deleted.", {"collection": collection_name}, False

    if name == "cms_add_field":
        field = await backend.cms_add_field(
            project_id,
            arguments.get("collection"),
            arguments.get("field") or {},
            auth_claims=auth_claims,
        )
        return f"Field `{field.get('name')}` added.", {"field": field}, False

    if name == "cms_update_field":
        field = await backend.cms_update_field(
            project_id,
            arguments.get("collection"),
            arguments.get("field_name"),
            arguments.get("patch") or {},
            auth_claims=auth_claims,
        )
        return f"Field `{arguments.get('field_name')}` updated.", {"field": field}, False

    if name == "cms_remove_field":
        await backend.cms_remove_field(
            project_id,
            arguments.get("collection"),
            arguments.get("field_name"),
            auth_claims=auth_claims,
        )
        return (
            f"Field `{arguments.get('field_name')}` removed from `{arguments.get('collection')}`.",
            {"collection": arguments.get("collection"), "field_name": arguments.get("field_name")},
            False,
        )

    if name == "cms_find":
        result = await backend.cms_list_documents(
            project_id,
            arguments.get("collection"),
            filter=arguments.get("filter"),
            sort=arguments.get("sort"),
            limit=_limit_arg(arguments, 50, 500),
            skip=int(arguments.get("skip") or 0),
            auth_claims=auth_claims,
        )
        rows = result.get("results", [])
        return (
            f"{len(rows)} of {result.get('total', '?')} document(s).",
            {"results": rows, "total": result.get("total"), "limit": result.get("limit"), "skip": result.get("skip")},
            False,
        )

    if name == "cms_get_document":
        document = await backend.cms_get_document(
            project_id,
            arguments.get("collection"),
            arguments.get("id"),
            auth_claims=auth_claims,
        )
        return f"Document `{arguments.get('id')}` loaded.", {"document": document}, False

    if name == "cms_insert":
        document = await backend.cms_insert(
            project_id,
            arguments.get("collection"),
            arguments.get("document") or {},
            auth_claims=auth_claims,
        )
        return f"Inserted, _id={document.get('_id')}", {"document": document}, False

    if name == "cms_bulk_insert":
        documents = list(arguments.get("documents") or [])
        result = await backend.cms_bulk_insert(project_id, arguments.get("collection"), documents, auth_claims=auth_claims)
        return f"Inserted {result.get('count', len(documents))} document(s).", result, False

    if name == "cms_update":
        patch = dict(arguments.get("patch") or {})
        if arguments.get("expected_version") is not None:
            patch["_version"] = arguments.get("expected_version")
        document = await backend.cms_update(
            project_id,
            arguments.get("collection"),
            arguments.get("id"),
            patch,
            auth_claims=auth_claims,
        )
        return f"Document `{arguments.get('id')}` updated.", {"document": document}, False

    if name == "cms_delete":
        await backend.cms_delete(project_id, arguments.get("collection"), arguments.get("id"), auth_claims=auth_claims)
        return f"Document `{arguments.get('id')}` soft-deleted.", {"id": arguments.get("id")}, False

    if name == "cms_restore":
        result = await backend.cms_restore(project_id, arguments.get("collection"), arguments.get("id"), auth_claims=auth_claims)
        return f"Document `{arguments.get('id')}` restored.", result, False

    if name == "cms_aggregate":
        result = await backend.cms_aggregate(
            project_id,
            arguments.get("collection"),
            list(arguments.get("pipeline") or []),
            limit=_limit_arg(arguments, 500, 5000),
            auth_claims=auth_claims,
        )
        return f"{result.get('count', 0)} row(s).", result, False

    if name == "cms_audit_log":
        items = await backend.cms_audit(
            project_id,
            arguments.get("collection"),
            action=arguments.get("action"),
            document_id=arguments.get("document_id"),
            auth_claims=auth_claims,
        )
        return f"{len(items)} audit entries.", {"audit": items}, False

    return f"Unknown tool `{name}`.", {"tool": name}, True
