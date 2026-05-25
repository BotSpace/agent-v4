from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .ai import AIConfigurationError
from .config import Settings, get_settings
from .openai_payload import add_completion_token_limit

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
    "env_list": "env.list",
    "env_set": "env.set",
    "env_unset": "env.unset",
    "release_list": "release.list",
    "release_deploy": "release.deploy",
    "release_status": "release.status",
    "plans_list": "plans.list",
    "subscription_current": "subscription.current",
    "subscription_upgrade": "subscription.upgrade",
    "payment_link_create": "payment.link.create",
    "file_list": "file.list",
    "file_delete": "file.delete",
    "collection_list": "collection.list",
    "collection_get": "collection.get",
    "chat_list_users": "chat.list_users",
    "chat_user_history": "chat.user_history",
    "broadcast_preview": "broadcast.preview",
    "broadcast_send": "broadcast.send",
    "broadcast_status": "broadcast.status",
    "message_delete": "message.delete",
    "cms_list_collections": "cms.list_collections",
    "cms_get_collection": "cms.get_collection",
    "cms_create_collection": "cms.create_collection",
    "cms_delete_collection": "cms.delete_collection",
    "cms_add_field": "cms.add_field",
    "cms_update_field": "cms.update_field",
    "cms_remove_field": "cms.remove_field",
    "cms_find": "cms.find",
    "cms_get_document": "cms.get_document",
    "cms_insert": "cms.insert",
    "cms_bulk_insert": "cms.bulk_insert",
    "cms_update": "cms.update",
    "cms_delete": "cms.delete",
    "cms_restore": "cms.restore",
    "cms_aggregate": "cms.aggregate",
    "cms_audit_log": "cms.audit_log",
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


async def run_compat_agent_messages(
    messages: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
    allow_tools: bool = True,
) -> CompatAgentTurn:
    s = settings or get_settings()
    return await _call_chat_completions(
        settings=s,
        messages=messages,
        tools=compat_tool_specs() if allow_tools else None,
        tool_choice="auto" if allow_tools else None,
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
    }
    add_completion_token_limit(payload, model=settings.openai_model, max_tokens=settings.openai_max_tokens)
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
        *_project_tool_specs(),
        *_ops_tool_specs(),
        *_cms_tool_specs(),
    ]


def _tool_spec(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    *,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


def _id(description: str = "Project id.") -> dict[str, Any]:
    return {"type": ["string", "integer"], "description": description}


def _project_id_optional() -> dict[str, Any]:
    return {"type": ["string", "integer"], "description": "Project id. Defaults to the current project."}


def _limit(default: int, maximum: int) -> dict[str, Any]:
    return {"type": "integer", "default": default, "minimum": 1, "maximum": maximum}


def _project_tool_specs() -> list[dict[str, Any]]:
    return [
        _tool_spec(
            PROJECT_LIST_TOOL_NAME,
            "List bot projects belonging to the current user.",
            {"search": {"type": "string", "description": "Optional project name search filter."}},
        ),
        _tool_spec(
            PROJECT_GET_TOOL_NAME,
            "Fetch one bot project by id. Optionally include its saved flow JSON.",
            {"id": _id(), "include_flow": {"type": "boolean"}},
            required=["id"],
        ),
        _tool_spec(
            PROJECT_CREATE_TOOL_NAME,
            "Create a new bot project.",
            {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "token": {"type": "string", "description": "Optional Telegram bot token."},
            },
            required=["name"],
        ),
        _tool_spec(
            PROJECT_UPDATE_TOOL_NAME,
            "Update a bot project's name, description, token, or saved flow.",
            {"id": _id(), "name": {"type": "string"}, "description": {"type": "string"}, "token": {"type": "string"}, "flow": {"type": "object"}},
            required=["id"],
        ),
        _tool_spec(
            PROJECT_DELETE_TOOL_NAME,
            "Delete a bot project permanently. Use only when the user explicitly asks.",
            {"id": _id()},
            required=["id"],
        ),
    ]


def _ops_tool_specs() -> list[dict[str, Any]]:
    return [
        _tool_spec("env_list", "List environment variables for a project.", {"project_id": _project_id_optional()}),
        _tool_spec(
            "env_set",
            "Create or replace an environment variable. Treat value as secret and never echo it back.",
            {"key": {"type": "string"}, "value": {"type": "string"}, "project_id": _project_id_optional()},
            required=["key", "value"],
        ),
        _tool_spec("env_unset", "Remove an environment variable by id.", {"env_id": _id("Env var id.")}, required=["env_id"]),
        _tool_spec("release_list", "List recent releases for a project.", {"project_id": _project_id_optional(), "limit": _limit(10, 50)}),
        _tool_spec(
            "release_deploy",
            "Deploy the saved flow of the project to production. First release may require plan_id from plans_list.",
            {"project_id": _project_id_optional(), "plan_id": {"type": "integer"}, "months": {"type": "integer", "enum": [1, 3, 6, 12]}},
        ),
        _tool_spec("release_status", "Get status of a release by id.", {"release_id": _id("Release id."), "project_id": _project_id_optional()}, required=["release_id"]),
        _tool_spec("plans_list", "List available pricing plans."),
        _tool_spec("subscription_current", "Get current user's active subscription plan."),
        _tool_spec(
            "subscription_upgrade",
            "Upgrade or change the current user's subscription plan.",
            {"plan_id": {"type": "integer"}, "months": {"type": "integer", "enum": [1, 3, 6, 12], "default": 1}},
            required=["plan_id"],
        ),
        _tool_spec(
            "payment_link_create",
            "Create a payment top-up link for the current user.",
            {"method": {"type": "string", "enum": ["click", "payme"]}, "amount": {"type": "integer", "minimum": 1}},
            required=["method", "amount"],
        ),
        _tool_spec("file_list", "List files belonging to a project.", {"project_id": _project_id_optional(), "limit": _limit(50, 200)}),
        _tool_spec("file_delete", "Delete a file by id.", {"file_id": _id("File id.")}, required=["file_id"]),
        _tool_spec("collection_list", "List legacy backend collection metadata for a project.", {"project_id": _project_id_optional()}),
        _tool_spec("collection_get", "Get legacy backend collection metadata by id.", {"collection_id": _id("Collection id.")}, required=["collection_id"]),
        _tool_spec("chat_list_users", "List Telegram users who interacted with the bot.", {"project_id": _project_id_optional(), "limit": _limit(50, 500)}),
        _tool_spec("chat_user_history", "Get message history with one Telegram user.", {"chat_id": _id("Telegram chat/user id."), "project_id": _project_id_optional()}, required=["chat_id"]),
        _tool_spec(
            "broadcast_preview",
            "Preview a broadcast before sending.",
            {"text": {"type": "string"}, "project_id": _project_id_optional(), "target": {"type": "string", "enum": ["all", "custom"]}, "telegram_ids": {"type": "array", "items": {"type": "integer"}}},
            required=["text"],
        ),
        _tool_spec(
            "broadcast_send",
            "Send a broadcast message. Use after preview or when the user explicitly confirms.",
            {"text": {"type": "string"}, "project_id": _project_id_optional(), "telegram_ids": {"type": "array", "items": {"type": "integer"}}},
        ),
        _tool_spec("broadcast_status", "List recent broadcasts and delivery status.", {"project_id": _project_id_optional(), "limit": _limit(10, 50)}),
        _tool_spec("message_delete", "Delete a broadcast/message by id.", {"message_id": _id("Message id.")}, required=["message_id"]),
    ]


def _field_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "kind": {
                "type": "string",
                "enum": [
                    "text", "longtext", "int", "float", "bool", "datetime", "date", "email", "url", "phone",
                    "enum", "json", "array", "relation", "file", "image", "audio", "video", "geo",
                ],
            },
            "label": {"type": "string"},
            "required": {"type": "boolean"},
            "default": {},
            "validators": {"type": "object"},
            "relation_collection": {"type": "integer"},
            "index": {"type": "string", "enum": ["none", "asc", "desc", "text", "hashed", "unique", "ttl"]},
            "order": {"type": "integer"},
        },
        "required": ["name", "kind"],
        "additionalProperties": False,
    }


def _cms_tool_specs() -> list[dict[str, Any]]:
    field = _field_schema()
    return [
        _tool_spec("cms_list_collections", "List CMS collections for the current project.", {"project_id": _project_id_optional()}),
        _tool_spec("cms_get_collection", "Get one CMS collection schema.", {"name": {"type": "string"}, "project_id": _project_id_optional()}, required=["name"]),
        _tool_spec(
            "cms_create_collection",
            "Create a CMS collection with optional initial schema.",
            {"name": {"type": "string"}, "label": {"type": "string"}, "description": {"type": "string"}, "fields": {"type": "array", "items": field}, "project_id": _project_id_optional()},
            required=["name"],
        ),
        _tool_spec("cms_delete_collection", "Soft-delete a CMS collection schema.", {"name": {"type": "string"}, "project_id": _project_id_optional()}, required=["name"]),
        _tool_spec("cms_add_field", "Add a field to a CMS collection.", {"collection": {"type": "string"}, "field": field, "project_id": _project_id_optional()}, required=["collection", "field"]),
        _tool_spec("cms_update_field", "Patch a CMS collection field.", {"collection": {"type": "string"}, "field_name": {"type": "string"}, "patch": {"type": "object"}, "project_id": _project_id_optional()}, required=["collection", "field_name", "patch"]),
        _tool_spec("cms_remove_field", "Remove a field from a CMS collection.", {"collection": {"type": "string"}, "field_name": {"type": "string"}, "project_id": _project_id_optional()}, required=["collection", "field_name"]),
        _tool_spec("cms_find", "Query CMS documents in a collection.", {"collection": {"type": "string"}, "filter": {"type": "object"}, "sort": {"type": "string"}, "limit": _limit(50, 500), "skip": {"type": "integer", "minimum": 0}, "project_id": _project_id_optional()}, required=["collection"]),
        _tool_spec("cms_get_document", "Fetch one CMS document by id.", {"collection": {"type": "string"}, "id": {"type": "string"}, "project_id": _project_id_optional()}, required=["collection", "id"]),
        _tool_spec("cms_insert", "Insert one CMS document.", {"collection": {"type": "string"}, "document": {"type": "object"}, "project_id": _project_id_optional()}, required=["collection", "document"]),
        _tool_spec("cms_bulk_insert", "Insert many CMS documents.", {"collection": {"type": "string"}, "documents": {"type": "array", "items": {"type": "object"}}, "project_id": _project_id_optional()}, required=["collection", "documents"]),
        _tool_spec("cms_update", "Patch one CMS document by id.", {"collection": {"type": "string"}, "id": {"type": "string"}, "patch": {"type": "object"}, "expected_version": {"type": "integer"}, "project_id": _project_id_optional()}, required=["collection", "id", "patch"]),
        _tool_spec("cms_delete", "Soft-delete one CMS document.", {"collection": {"type": "string"}, "id": {"type": "string"}, "project_id": _project_id_optional()}, required=["collection", "id"]),
        _tool_spec("cms_restore", "Restore one soft-deleted CMS document.", {"collection": {"type": "string"}, "id": {"type": "string"}, "project_id": _project_id_optional()}, required=["collection", "id"]),
        _tool_spec("cms_aggregate", "Run a MongoDB aggregation pipeline on a CMS collection.", {"collection": {"type": "string"}, "pipeline": {"type": "array", "items": {"type": "object"}}, "limit": _limit(500, 5000), "project_id": _project_id_optional()}, required=["collection", "pipeline"]),
        _tool_spec("cms_audit_log", "View CMS collection audit log.", {"collection": {"type": "string"}, "action": {"type": "string"}, "document_id": {"type": "string"}, "project_id": _project_id_optional()}, required=["collection"]),
    ]


def compat_system_prompt() -> str:
    return """
You are Botmother AgentV4, a conversational agent for building Botmother visual bot flows.

Work like a coding agent with tools:
- If the user is chatting, greeting you, asking who you are, or asking a general question, answer directly.
- If the user asks to create, generate, change, connect, deploy, or configure a bot/flow, call the botbuilder_ai_build tool.
- If the user asks about their projects, files, environment variables, releases/deploys, pricing plans,
  current subscription, payment link, chat users/history, broadcasts/messages, collections, or CMS data,
  use the matching management tools.
- If the user asks "qaysi plandaman" (which plan am I on), use subscription_current (not release_list).
- Use project_id from the current run when the user does not provide one. If an operation cannot be done
  without choosing a project, list projects or ask the user to choose.
- For destructive tools (delete, deploy, broadcast_send), call them only after the user clearly asks or confirms.
- Do not call tools for casual chat.
- After a tool result, explain briefly what happened and point to the returned flow data when relevant.
- Reply in the user's language. Uzbek Latin is preferred when the user writes Uzbek.
""".strip()
