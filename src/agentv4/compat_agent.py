from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .ai import AIConfigurationError
from .config import Settings, get_settings

BOTBUILDER_TOOL_NAME = "botbuilder_ai_build"
BOTBUILDER_EVENT_NAME = "botbuilder.ai_build"
PROJECT_LIST_TOOL_NAME = "project_list"
PROJECT_GET_TOOL_NAME = "project_get"
PROJECT_CREATE_TOOL_NAME = "project_create"
PROJECT_UPDATE_TOOL_NAME = "project_update"
PROJECT_DELETE_TOOL_NAME = "project_delete"

TOOL_EVENT_NAMES = {
    BOTBUILDER_TOOL_NAME: BOTBUILDER_EVENT_NAME,
    PROJECT_LIST_TOOL_NAME: "project.list",
    PROJECT_GET_TOOL_NAME: "project.get",
    PROJECT_CREATE_TOOL_NAME: "project.create",
    PROJECT_UPDATE_TOOL_NAME: "project.update",
    PROJECT_DELETE_TOOL_NAME: "project.delete",
}


@dataclass(frozen=True)
class CompatToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class CompatAgentTurn:
    text: str
    tool_calls: list[CompatToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


async def run_compat_agent_turn(prompt: str, *, settings: Settings | None = None) -> CompatAgentTurn:
    s = settings or get_settings()
    return await _call_chat_completions(
        settings=s,
        messages=[
            {"role": "system", "content": compat_system_prompt()},
            {"role": "user", "content": prompt},
        ],
        tools=compat_tool_specs(),
        tool_choice="auto",
    )


async def run_compat_agent_final(
    prompt: str,
    *,
    assistant_text: str,
    tool_call: CompatToolCall,
    tool_content: str,
    tool_data: dict[str, Any],
    settings: Settings | None = None,
) -> CompatAgentTurn:
    s = settings or get_settings()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": compat_system_prompt()},
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": assistant_text or None,
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
    return await _call_chat_completions(settings=s, messages=messages)


async def _call_chat_completions(
    *,
    settings: Settings,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> CompatAgentTurn:
    if not settings.openai_api_key:
        raise AIConfigurationError("OPENAI_API_KEY is not configured")

    return await asyncio.wait_for(
        asyncio.to_thread(
            _call_chat_completions_sync,
            settings=settings,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        ),
        timeout=settings.openai_timeout + 5,
    )


def _call_chat_completions_sync(
    *,
    settings: Settings,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> CompatAgentTurn:
    payload: dict[str, Any] = {
        "model": settings.openai_model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": settings.openai_max_tokens,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=settings.openai_timeout) as res:
            body = json.loads(res.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"OpenAI HTTP {exc.code}: {detail}") from exc

    message = body["choices"][0]["message"]
    usage = body.get("usage") if isinstance(body, dict) else None
    return CompatAgentTurn(
        text=str(message.get("content") or ""),
        tool_calls=parse_tool_calls(message.get("tool_calls") or []),
        input_tokens=int((usage or {}).get("prompt_tokens") or 0),
        output_tokens=int((usage or {}).get("completion_tokens") or 0),
    )


def parse_tool_calls(raw_calls: list[Any]) -> list[CompatToolCall]:
    calls: list[CompatToolCall] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "")
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
        except Exception:
            arguments = {}
        calls.append(CompatToolCall(id=str(item.get("id") or ""), name=name, arguments=arguments))
    return calls


def botbuilder_tool_spec() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": BOTBUILDER_TOOL_NAME,
            "description": (
                "Build or update a Botmother visual bot flow. Use this only when the user asks to create, "
                "generate, edit, connect, or configure a bot/flow."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The user's flow-building request, preserving their language and requirements.",
                    }
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        },
    }


def compat_tool_specs() -> list[dict[str, Any]]:
    return [
        botbuilder_tool_spec(),
        {
            "type": "function",
            "function": {
                "name": PROJECT_LIST_TOOL_NAME,
                "description": "List bot projects belonging to the current user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search": {"type": "string", "description": "Optional project name search filter."}
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": PROJECT_GET_TOOL_NAME,
                "description": "Fetch one bot project by id. Optionally include its saved flow JSON.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": ["string", "integer"], "description": "Project id."},
                        "include_flow": {"type": "boolean", "description": "Whether to include saved flow JSON."},
                    },
                    "required": ["id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": PROJECT_CREATE_TOOL_NAME,
                "description": "Create a new bot project.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Bot project name."},
                        "description": {"type": "string", "description": "Optional project description."},
                        "token": {"type": "string", "description": "Optional Telegram bot token."},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": PROJECT_UPDATE_TOOL_NAME,
                "description": "Update a bot project's name, description, token, or saved flow.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": ["string", "integer"], "description": "Project id."},
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "token": {"type": "string"},
                        "flow": {"type": "object", "description": "Saved flow JSON."},
                    },
                    "required": ["id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": PROJECT_DELETE_TOOL_NAME,
                "description": "Delete a bot project permanently. Use only when the user explicitly asks.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": ["string", "integer"], "description": "Project id."}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def compat_system_prompt() -> str:
    return """
You are Botmother AgentV4, a conversational agent for building Botmother visual bot flows.

Work like a coding agent with tools:
- If the user is chatting, greeting you, asking who you are, or asking a general question, answer directly.
- If the user asks to create, generate, change, connect, deploy, or configure a bot/flow, call the botbuilder_ai_build tool.
- If the user asks about their projects, list/fetch/create/update/delete projects with project_* tools.
- Do not call tools for casual chat.
- After a tool result, explain briefly what happened and point to the returned flow data when relevant.
- Reply in the user's language. Uzbek Latin is preferred when the user writes Uzbek.
""".strip()
