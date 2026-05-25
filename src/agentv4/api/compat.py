from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..ai import AIBuildRequest, build_bot_with_ai


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
    prompt: str
    thread_id: str | None = None
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

    async def create(self, *, project_id: str, prompt: str, thread_id: str | None) -> CompatRun:
        async with self._lock:
            run = CompatRun(request_id=str(uuid4()), project_id=project_id, prompt=prompt, thread_id=thread_id)
            self._runs[run.request_id] = run
            return run

    async def get(self, request_id: str, *, project_id: str) -> CompatRun | None:
        async with self._lock:
            run = self._runs.get(request_id)
            if run is None or run.project_id != project_id:
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

    async def cancel(self, request_id: str, *, project_id: str) -> bool:
        async with self._lock:
            run = self._runs.get(request_id)
            if run is None or run.project_id != project_id:
                return False
            if run.task and not run.task.done():
                run.task.cancel()
            run.status = "cancelled"
        await self.add_event(request_id, "run_failed", {"error": "Run cancelled"})
        await self.finish_streams(request_id)
        return True


def create_compat_router(store: CompatRunStore) -> APIRouter:
    router = APIRouter()

    @router.post("/projects/{project_id}/runs", response_model=RunCreateResponse)
    async def create_run(project_id: str, body: RunCreateRequest) -> RunCreateResponse:
        run = await store.create(project_id=project_id, prompt=body.prompt, thread_id=body.thread_id)
        run.task = asyncio.create_task(_execute_run(store, run.request_id, body.prompt))
        return RunCreateResponse(request_id=run.request_id, status="queued")

    @router.get("/projects/{project_id}/runs/{request_id}", response_model=RunStatusResponse)
    async def get_run(project_id: str, request_id: str) -> RunStatusResponse:
        run = await store.get(request_id, project_id=project_id)
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
    async def stream_events(project_id: str, request_id: str, since_seq: int = 0) -> StreamingResponse:
        run = await store.get(request_id, project_id=project_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return StreamingResponse(
            _event_stream(store, request_id, since_seq=since_seq),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/projects/{project_id}/runs/{request_id}/clarify")
    async def clarify_run(project_id: str, request_id: str, _: ClarifyRequest) -> dict[str, str]:
        run = await store.get(request_id, project_id=project_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"status": run.status}

    @router.post("/projects/{project_id}/runs/{request_id}/approve")
    async def approve_run(project_id: str, request_id: str, body: ApproveRequest) -> dict[str, str]:
        run = await store.get(request_id, project_id=project_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"status": run.status, "approved": "true" if body.approved else "false"}

    @router.post("/projects/{project_id}/runs/{request_id}/cancel")
    async def cancel_run(project_id: str, request_id: str) -> dict[str, str]:
        if not await store.cancel(request_id, project_id=project_id):
            raise HTTPException(status_code=404, detail="Run not found")
        return {"status": "cancelling"}

    return router


async def _execute_run(store: CompatRunStore, request_id: str, prompt: str) -> None:
    await store.set_status(request_id, "running")
    await store.add_event(request_id, "run_started", {"request_id": request_id})
    tool_call_id = f"tool-{request_id[:8]}"
    await store.add_event(
        request_id,
        "tool_call",
        {"id": tool_call_id, "name": "botbuilder.ai_build", "arguments": {"prompt": prompt}},
    )
    try:
        result = await build_bot_with_ai(AIBuildRequest(prompt=prompt, language="uz", max_repair_attempts=2))
        flow = result.result.flow.model_dump(mode="json") if result.result.flow else None
        issues = [issue.model_dump(mode="json") for issue in result.result.issues]
        await store.add_event(
            request_id,
            "tool_result",
            {
                "tool_call_id": tool_call_id,
                "content": "Flow compiled" if result.ok else "Flow validation failed",
                "is_error": not result.ok,
                "data": {
                    "ok": result.ok,
                    "attempts": result.attempts,
                    "commands": result.commands,
                    "flow": flow,
                    "issues": issues,
                },
            },
        )
        if result.ok:
            node_count = len(flow["nodes"]) if flow else 0
            edge_count = len(flow["edges"]) if flow else 0
            message = (
                f"Bot flow tayyor. {node_count} ta node va {edge_count} ta edge yaratildi.\n\n"
                "Flow JSON `tool_result.data.flow` ichida qaytdi."
            )
            await store.set_status(request_id, "completed", final_response=message)
            await store.add_event(request_id, "assistant_message", {"text": message, "usage": {}})
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
        await store.add_event(request_id, "tool_result", {"tool_call_id": tool_call_id, "content": message, "is_error": True})
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
