from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from redis import asyncio as redis

from ..config import Settings


@dataclass
class CompatRunRecord:
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


class RunStore(Protocol):
    async def create(
        self,
        *,
        project_id: str,
        user_id: str,
        prompt: str,
        thread_id: str | None,
        auth_claims: dict[str, Any],
    ) -> CompatRunRecord: ...

    async def get(self, request_id: str, *, project_id: str, user_id: str) -> CompatRunRecord | None: ...

    async def get_execution_context(self, request_id: str) -> CompatRunRecord | None: ...

    async def thread_context(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        fallback_project_id: str,
    ) -> tuple[list[dict[str, str]], str]: ...

    async def append_thread_message(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        role: str,
        content: str,
        limit: int = 20,
    ) -> None: ...

    async def set_thread_project_id(self, *, user_id: str, thread_id: str | None, project_id: str | int | None) -> None: ...

    async def last_failed_tool(self, *, user_id: str, thread_id: str | None) -> dict[str, Any] | None: ...

    async def set_last_failed_tool(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        name: str,
        arguments: dict[str, Any],
        content: str,
    ) -> None: ...

    async def clear_last_failed_tool(self, *, user_id: str, thread_id: str | None) -> None: ...

    async def set_status(
        self,
        request_id: str,
        status: str,
        *,
        final_response: str | None = None,
        error: str | None = None,
    ) -> None: ...

    async def add_event(self, request_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def list_events(self, request_id: str, *, since_seq: int = 0) -> list[dict[str, Any]]: ...

    def register_task(self, request_id: str, task: asyncio.Task[None]) -> None: ...

    async def cancel(self, request_id: str, *, project_id: str, user_id: str) -> bool: ...

    async def is_cancel_requested(self, request_id: str) -> bool: ...


class InMemoryRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, CompatRunRecord] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._thread_messages: dict[tuple[str, str], list[dict[str, str]]] = {}
        self._thread_project_ids: dict[tuple[str, str], str] = {}
        self._thread_failed_tools: dict[tuple[str, str], dict[str, Any]] = {}
        self._cancel_requested: set[str] = set()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        *,
        project_id: str,
        user_id: str,
        prompt: str,
        thread_id: str | None,
        auth_claims: dict[str, Any],
    ) -> CompatRunRecord:
        async with self._lock:
            run = CompatRunRecord(
                request_id=str(uuid4()),
                project_id=project_id,
                user_id=user_id,
                prompt=prompt,
                thread_id=thread_id,
                auth_claims=auth_claims,
            )
            self._runs[run.request_id] = run
            self._events[run.request_id] = []
            return run

    async def get(self, request_id: str, *, project_id: str, user_id: str) -> CompatRunRecord | None:
        async with self._lock:
            run = self._runs.get(request_id)
            if run is None or run.project_id != project_id or run.user_id != user_id:
                return None
            return CompatRunRecord(**run.__dict__)

    async def get_execution_context(self, request_id: str) -> CompatRunRecord | None:
        async with self._lock:
            run = self._runs.get(request_id)
            return CompatRunRecord(**run.__dict__) if run else None

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

    async def last_failed_tool(self, *, user_id: str, thread_id: str | None) -> dict[str, Any] | None:
        if thread_id is None:
            return None
        async with self._lock:
            failed_tool = self._thread_failed_tools.get((user_id, thread_id))
            return dict(failed_tool) if failed_tool else None

    async def set_last_failed_tool(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        name: str,
        arguments: dict[str, Any],
        content: str,
    ) -> None:
        if thread_id is None:
            return
        async with self._lock:
            self._thread_failed_tools[(user_id, thread_id)] = {
                "name": name,
                "arguments": dict(arguments),
                "content": content,
            }

    async def clear_last_failed_tool(self, *, user_id: str, thread_id: str | None) -> None:
        if thread_id is None:
            return
        async with self._lock:
            self._thread_failed_tools.pop((user_id, thread_id), None)

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
            events = self._events[request_id]
            event = {
                "seq": len(events) + 1,
                "type": event_type,
                "payload": payload,
                "at": datetime.now(UTC).isoformat(),
            }
            events.append(event)
        return event

    async def list_events(self, request_id: str, *, since_seq: int = 0) -> list[dict[str, Any]]:
        async with self._lock:
            return [event for event in self._events.get(request_id, []) if int(event["seq"]) > since_seq]

    def register_task(self, request_id: str, task: asyncio.Task[None]) -> None:
        self._tasks[request_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(request_id, None))

    async def cancel(self, request_id: str, *, project_id: str, user_id: str) -> bool:
        async with self._lock:
            run = self._runs.get(request_id)
            if run is None or run.project_id != project_id or run.user_id != user_id:
                return False
            self._cancel_requested.add(request_id)
            task = self._tasks.get(request_id)
            if task and not task.done():
                task.cancel()
            run.status = "cancelled"
        return True

    async def is_cancel_requested(self, request_id: str) -> bool:
        async with self._lock:
            return request_id in self._cancel_requested


class RedisRunStore:
    def __init__(self, redis_url: str, *, prefix: str, run_ttl_seconds: int, thread_ttl_seconds: int) -> None:
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._prefix = prefix
        self._run_ttl_seconds = run_ttl_seconds
        self._thread_ttl_seconds = thread_ttl_seconds
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def _run_key(self, request_id: str) -> str:
        return f"{self._prefix}:run:{request_id}"

    def _run_events_key(self, request_id: str) -> str:
        return f"{self._prefix}:run:{request_id}:events"

    def _run_seq_key(self, request_id: str) -> str:
        return f"{self._prefix}:run:{request_id}:seq"

    def _thread_prefix(self, user_id: str, thread_id: str) -> str:
        return f"{self._prefix}:thread:{user_id}:{thread_id}"

    def _thread_messages_key(self, user_id: str, thread_id: str) -> str:
        return f"{self._thread_prefix(user_id, thread_id)}:messages"

    def _thread_project_key(self, user_id: str, thread_id: str) -> str:
        return f"{self._thread_prefix(user_id, thread_id)}:project_id"

    def _thread_failed_tool_key(self, user_id: str, thread_id: str) -> str:
        return f"{self._thread_prefix(user_id, thread_id)}:failed_tool"

    async def _expire_run_keys(self, request_id: str) -> None:
        await self._redis.expire(self._run_key(request_id), self._run_ttl_seconds)
        await self._redis.expire(self._run_events_key(request_id), self._run_ttl_seconds)
        await self._redis.expire(self._run_seq_key(request_id), self._run_ttl_seconds)

    async def create(
        self,
        *,
        project_id: str,
        user_id: str,
        prompt: str,
        thread_id: str | None,
        auth_claims: dict[str, Any],
    ) -> CompatRunRecord:
        run = CompatRunRecord(
            request_id=str(uuid4()),
            project_id=project_id,
            user_id=user_id,
            prompt=prompt,
            thread_id=thread_id,
            auth_claims=auth_claims,
        )
        await self._redis.hset(
            self._run_key(run.request_id),
            mapping=_encode_run(run),
        )
        await self._expire_run_keys(run.request_id)
        return run

    async def get(self, request_id: str, *, project_id: str, user_id: str) -> CompatRunRecord | None:
        run = await self.get_execution_context(request_id)
        if run is None or run.project_id != project_id or run.user_id != user_id:
            return None
        return run

    async def get_execution_context(self, request_id: str) -> CompatRunRecord | None:
        payload = await self._redis.hgetall(self._run_key(request_id))
        if not payload:
            return None
        return _decode_run(request_id, payload)

    async def thread_context(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        fallback_project_id: str,
    ) -> tuple[list[dict[str, str]], str]:
        if thread_id is None:
            return [], fallback_project_id
        raw_messages = await self._redis.lrange(self._thread_messages_key(user_id, thread_id), 0, -1)
        messages = []
        for raw_message in raw_messages:
            try:
                parsed = json.loads(raw_message)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                role = str(parsed.get("role") or "")
                content = str(parsed.get("content") or "")
                if role and content:
                    messages.append({"role": role, "content": content})
        project_id = await self._redis.get(self._thread_project_key(user_id, thread_id))
        return messages, project_id or fallback_project_id

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
        key = self._thread_messages_key(user_id, thread_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, json.dumps({"role": role, "content": content}, ensure_ascii=False))
            pipe.ltrim(key, -limit, -1)
            pipe.expire(key, self._thread_ttl_seconds)
            await pipe.execute()

    async def set_thread_project_id(self, *, user_id: str, thread_id: str | None, project_id: str | int | None) -> None:
        if thread_id is None or project_id is None:
            return
        await self._redis.set(self._thread_project_key(user_id, thread_id), str(project_id), ex=self._thread_ttl_seconds)

    async def last_failed_tool(self, *, user_id: str, thread_id: str | None) -> dict[str, Any] | None:
        if thread_id is None:
            return None
        payload = await self._redis.get(self._thread_failed_tool_key(user_id, thread_id))
        if not payload:
            return None
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, dict) else None

    async def set_last_failed_tool(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        name: str,
        arguments: dict[str, Any],
        content: str,
    ) -> None:
        if thread_id is None:
            return
        await self._redis.set(
            self._thread_failed_tool_key(user_id, thread_id),
            json.dumps(
                {
                    "name": name,
                    "arguments": dict(arguments),
                    "content": content,
                },
                ensure_ascii=False,
            ),
            ex=self._thread_ttl_seconds,
        )

    async def clear_last_failed_tool(self, *, user_id: str, thread_id: str | None) -> None:
        if thread_id is None:
            return
        await self._redis.delete(self._thread_failed_tool_key(user_id, thread_id))

    async def set_status(
        self,
        request_id: str,
        status: str,
        *,
        final_response: str | None = None,
        error: str | None = None,
    ) -> None:
        mapping: dict[str, str] = {"status": status}
        if final_response is not None:
            mapping["final_response"] = final_response
        if error is not None:
            mapping["error"] = error
        await self._redis.hset(self._run_key(request_id), mapping=mapping)
        await self._expire_run_keys(request_id)

    async def add_event(self, request_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        seq = int(await self._redis.incr(self._run_seq_key(request_id)))
        event = {
            "seq": seq,
            "type": event_type,
            "payload": payload,
            "at": datetime.now(UTC).isoformat(),
        }
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(self._run_events_key(request_id), json.dumps(event, ensure_ascii=False))
            pipe.expire(self._run_events_key(request_id), self._run_ttl_seconds)
            pipe.expire(self._run_seq_key(request_id), self._run_ttl_seconds)
            pipe.expire(self._run_key(request_id), self._run_ttl_seconds)
            await pipe.execute()
        return event

    async def list_events(self, request_id: str, *, since_seq: int = 0) -> list[dict[str, Any]]:
        payloads = await self._redis.lrange(self._run_events_key(request_id), since_seq, -1)
        events: list[dict[str, Any]] = []
        for payload in payloads:
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def register_task(self, request_id: str, task: asyncio.Task[None]) -> None:
        self._tasks[request_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(request_id, None))

    async def cancel(self, request_id: str, *, project_id: str, user_id: str) -> bool:
        run = await self.get(request_id, project_id=project_id, user_id=user_id)
        if run is None:
            return False
        await self._redis.hset(self._run_key(request_id), mapping={"status": "cancelled", "cancel_requested": "1"})
        await self._expire_run_keys(request_id)
        task = self._tasks.get(request_id)
        if task and not task.done():
            task.cancel()
        return True

    async def is_cancel_requested(self, request_id: str) -> bool:
        value = await self._redis.hget(self._run_key(request_id), "cancel_requested")
        return value == "1"


def create_run_store(settings: Settings) -> RunStore:
    if settings.redis_url:
        return RedisRunStore(
            settings.redis_url,
            prefix=settings.redis_prefix,
            run_ttl_seconds=settings.run_ttl_seconds,
            thread_ttl_seconds=settings.thread_ttl_seconds,
        )
    return InMemoryRunStore()


def _encode_run(run: CompatRunRecord) -> dict[str, str]:
    return {
        "request_id": run.request_id,
        "project_id": run.project_id,
        "user_id": run.user_id,
        "prompt": run.prompt,
        "thread_id": run.thread_id or "",
        "auth_claims": json.dumps(run.auth_claims, ensure_ascii=False),
        "status": run.status,
        "final_response": run.final_response or "",
        "error": run.error or "",
        "current_plan": json.dumps(run.current_plan or [], ensure_ascii=False),
    }


def _decode_run(request_id: str, payload: dict[str, str]) -> CompatRunRecord:
    auth_claims: dict[str, Any] = {}
    current_plan: list[dict[str, Any]] | None = None
    raw_auth_claims = payload.get("auth_claims")
    if raw_auth_claims:
        try:
            decoded = json.loads(raw_auth_claims)
        except json.JSONDecodeError:
            decoded = {}
        if isinstance(decoded, dict):
            auth_claims = decoded
    raw_plan = payload.get("current_plan")
    if raw_plan:
        try:
            decoded_plan = json.loads(raw_plan)
        except json.JSONDecodeError:
            decoded_plan = []
        if isinstance(decoded_plan, list):
            current_plan = decoded_plan
    return CompatRunRecord(
        request_id=request_id,
        project_id=payload.get("project_id", ""),
        user_id=payload.get("user_id", ""),
        prompt=payload.get("prompt", ""),
        thread_id=payload.get("thread_id") or None,
        auth_claims=auth_claims,
        status=payload.get("status", "queued"),
        final_response=payload.get("final_response") or None,
        error=payload.get("error") or None,
        current_plan=current_plan,
    )
