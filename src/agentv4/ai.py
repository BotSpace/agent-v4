from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from pydantic import BaseModel, Field, TypeAdapter

from .builder.commands import BotCommand
from .builder.models import CompileResult
from .builder.session import build_from_commands
from .config import get_settings
from .engine_catalog import engine_catalog_prompt
from .openai_payload import add_completion_token_limit

COMMANDS_ADAPTER = TypeAdapter(list[BotCommand])


class AIBuildRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)
    language: str = Field(default="uz", min_length=2, max_length=12)
    model: str | None = None
    max_repair_attempts: int = Field(default=2, ge=0, le=3)


class AIBuildResponse(BaseModel):
    ok: bool
    model: str
    attempts: int
    commands: list[dict[str, Any]]
    result: CompileResult
    input_tokens: int = 0
    output_tokens: int = 0


class AIConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAIResult:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0


async def build_bot_with_ai(request: AIBuildRequest) -> AIBuildResponse:
    settings = get_settings()
    api_key = settings.openai_api_key
    if not api_key:
        raise AIConfigurationError("OPENAI_API_KEY is not configured")

    model = request.model or settings.openai_model

    attempts = 0
    feedback = ""
    last_commands: list[BotCommand] = []
    last_result: CompileResult | None = None
    last_error: Exception | None = None
    input_tokens = 0
    output_tokens = 0

    while attempts <= request.max_repair_attempts:
        attempts += 1
        response = await call_openai(api_key=api_key, model=model, request=request, feedback=feedback)
        raw = response.content
        input_tokens += response.input_tokens
        output_tokens += response.output_tokens
        try:
            commands = parse_commands(raw)
        except Exception as exc:
            last_error = exc
            feedback = (
                "Command JSON parse error: "
                f"{exc}. Return exact BotBuilder command objects. Use `action`, not `command`."
            )
            continue
        result = build_from_commands(commands)
        last_commands = commands
        last_result = result
        if result.ok:
            return AIBuildResponse(
                ok=True,
                model=model,
                attempts=attempts,
                commands=[command.model_dump(mode="json") for command in commands],
                result=result,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        feedback = format_issues(result)

    if last_result is None:
        raise ValueError(f"AI did not produce valid BotBuilder commands: {last_error}")
    return AIBuildResponse(
        ok=False,
        model=model,
        attempts=attempts,
        commands=[command.model_dump(mode="json") for command in last_commands],
        result=last_result,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


async def build_bot_with_ai_stream(
    request: AIBuildRequest,
    *,
    on_delta: Callable[[str], Awaitable[None]] | None = None,
) -> AIBuildResponse:
    settings = get_settings()
    api_key = settings.openai_api_key
    if not api_key:
        raise AIConfigurationError("OPENAI_API_KEY is not configured")

    model = request.model or settings.openai_model

    attempts = 0
    feedback = ""
    last_commands: list[BotCommand] = []
    last_result: CompileResult | None = None
    last_error: Exception | None = None
    input_tokens = 0
    output_tokens = 0

    while attempts <= request.max_repair_attempts:
        attempts += 1
        response = await call_openai_stream(
            api_key=api_key,
            model=model,
            request=request,
            feedback=feedback,
            on_delta=on_delta,
        )
        raw = response.content
        input_tokens += response.input_tokens
        output_tokens += response.output_tokens
        try:
            commands = parse_commands(raw)
        except Exception as exc:
            last_error = exc
            feedback = (
                "Command JSON parse error: "
                f"{exc}. Return exact BotBuilder command objects. Use `action`, not `command`."
            )
            continue
        result = build_from_commands(commands)
        last_commands = commands
        last_result = result
        if result.ok:
            return AIBuildResponse(
                ok=True,
                model=model,
                attempts=attempts,
                commands=[command.model_dump(mode="json") for command in commands],
                result=result,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        feedback = format_issues(result)

    if last_result is None:
        raise ValueError(f"AI did not produce valid BotBuilder commands: {last_error}")
    return AIBuildResponse(
        ok=False,
        model=model,
        attempts=attempts,
        commands=[command.model_dump(mode="json") for command in last_commands],
        result=last_result,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


async def call_openai(
    *,
    api_key: str,
    model: str,
    request: AIBuildRequest,
    feedback: str,
) -> OpenAIResult:
    import asyncio

    timeout = get_settings().openai_timeout
    return await asyncio.wait_for(
        asyncio.to_thread(
            call_openai_sync,
            api_key=api_key,
            model=model,
            request=request,
            feedback=feedback,
        ),
        timeout=timeout + 5,
    )


async def call_openai_stream(
    *,
    api_key: str,
    model: str,
    request: AIBuildRequest,
    feedback: str,
    on_delta: Callable[[str], Awaitable[None]] | None = None,
) -> OpenAIResult:
    settings = get_settings()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": user_prompt(request.prompt, language=request.language, feedback=feedback),
            },
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    add_completion_token_limit(payload, model=model, max_tokens=settings.openai_max_tokens)

    body_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0
    timeout = httpx.Timeout(settings.openai_timeout + 5, connect=settings.openai_timeout)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = await response.aread()
                raise ValueError(
                    f"OpenAI HTTP {response.status_code}: {detail.decode('utf-8', errors='replace')}"
                ) from exc

            async for line in response.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                if choices:
                    delta = ((choices[0] or {}).get("delta") or {}).get("content")
                    if isinstance(delta, str) and delta:
                        body_parts.append(delta)
                        if on_delta is not None:
                            await on_delta(delta)

                usage = chunk.get("usage") or {}
                input_tokens = int(usage.get("prompt_tokens") or input_tokens or 0)
                output_tokens = int(usage.get("completion_tokens") or output_tokens or 0)

    content = "".join(body_parts)
    if not content:
        raise ValueError("OpenAI returned an empty response")
    return OpenAIResult(content=content, input_tokens=input_tokens, output_tokens=output_tokens)


def call_openai_sync(
    *,
    api_key: str,
    model: str,
    request: AIBuildRequest,
    feedback: str,
) -> OpenAIResult:
    settings = get_settings()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": user_prompt(request.prompt, language=request.language, feedback=feedback),
            },
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    add_completion_token_limit(payload, model=model, max_tokens=settings.openai_max_tokens)
    timeout = settings.openai_timeout
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            body = json.loads(res.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"OpenAI HTTP {exc.code}: {detail}") from exc

    content = body["choices"][0]["message"].get("content")
    if not content:
        raise ValueError("OpenAI returned an empty response")
    usage = body.get("usage") if isinstance(body, dict) else None
    input_tokens = int((usage or {}).get("prompt_tokens") or 0)
    output_tokens = int((usage or {}).get("completion_tokens") or 0)
    return OpenAIResult(content=str(content), input_tokens=input_tokens, output_tokens=output_tokens)


def parse_commands(raw: str) -> list[BotCommand]:
    payload = json.loads(raw)
    commands_raw = payload.get("commands")
    if not isinstance(commands_raw, list):
        raise ValueError("AI response must contain a `commands` array")
    normalized = [normalize_command(item) for item in commands_raw]
    return COMMANDS_ADAPTER.validate_python(repair_command_sequence(normalized))


def normalize_command(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"Command must be an object, got {type(item).__name__}")
    normalized = dict(item)
    if "action" not in normalized and "command" in normalized:
        normalized["action"] = normalized.pop("command")
    if "action" not in normalized and "tool" in normalized:
        normalized["action"] = normalized.pop("tool")

    action = normalized.get("action")
    if action == "add_message":
        normalized["action"] = "add_text"
    if action in {"add_raw_node", "add_node", "engine_node"}:
        normalized["action"] = "add_engine_node"
    if normalized.get("action") == "add_engine_node":
        if "node_type" not in normalized and "type" in normalized:
            normalized["node_type"] = normalized.pop("type")

    if normalized.get("action") == "add_buttons":
        buttons = normalized.get("buttons")
        if isinstance(buttons, list):
            normalized["buttons"] = [
                {"text": button} if isinstance(button, str) else button
                for button in buttons
            ]

    if normalized.get("action") == "add_question":
        answer_type = normalized.get("answer_type")
        aliases = {
            "phone_number": "phone",
            "tel": "phone",
            "integer": "number",
            "date": "text",
            "time": "text",
            "button": "text",
            "buttons": "text",
            "choice": "text",
        }
        if isinstance(answer_type, str) and answer_type in aliases:
            normalized["answer_type"] = aliases[answer_type]

    if normalized.get("action") == "add_condition":
        if "conditions" not in normalized and "condition" in normalized:
            normalized["conditions"] = normalize_condition_input(normalized.pop("condition"))
        else:
            normalized["conditions"] = normalize_condition_input(normalized.get("conditions", []))

    return normalized


def repair_command_sequence(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    commands = [dict(command) for command in commands]
    rename_reserved_start(commands)
    remove_future_after_references(commands)
    commands = collapse_button_membership_conditions(commands)
    commands = collapse_button_choice_conditions(commands)
    synthesize_condition_after_branches(commands)
    commands = remove_orphan_finish_commands(commands)
    return commands


def rename_reserved_start(commands: list[dict[str, Any]]) -> None:
    renamed_to: str | None = None
    for command in commands:
        if command.get("action") == "create_bot":
            continue
        if command.get("key") == "start":
            action = str(command.get("action", "step")).replace("add_", "")
            renamed_to = f"start_{action}"
            command["key"] = renamed_to
            break

    if renamed_to is None:
        return

    for command in commands:
        if command.get("source") == "start":
            command["source"] = renamed_to
        if command.get("target") == "start":
            command["target"] = renamed_to


def remove_future_after_references(commands: list[dict[str, Any]]) -> None:
    positions = {
        command["key"]: index
        for index, command in enumerate(commands)
        if isinstance(command.get("key"), str)
    }
    by_key = {
        command["key"]: command
        for command in commands
        if isinstance(command.get("key"), str)
    }
    incoming_targets = {
        command.get("target")
        for command in commands
        if command.get("action") == "connect" and isinstance(command.get("target"), str)
    }

    for index, command in enumerate(commands):
        after = command.get("after")
        if not isinstance(after, str) or after not in positions or positions[after] <= index:
            continue

        after_command = by_key.get(after)
        if after_command is not None and after_command.get("action") == "add_finish":
            after_command["after"] = command.get("key")
        if command.get("key") in incoming_targets:
            command["after"] = None
        else:
            command["after"] = "start"


def collapse_button_membership_conditions(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buttons_by_choices: list[tuple[set[str], str]] = []
    for command in commands:
        if command.get("action") != "add_buttons":
            continue
        choices = {
            str(button.get("text") if isinstance(button, dict) else button)
            for button in command.get("buttons", [])
        }
        if choices and isinstance(command.get("key"), str):
            buttons_by_choices.append((choices, command["key"]))

    condition_to_buttons: dict[str, str] = {}
    for command in commands:
        if command.get("action") != "add_condition" or not isinstance(command.get("key"), str):
            continue
        conditions = command.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != 1:
            continue
        condition = conditions[0]
        if not isinstance(condition, dict) or condition.get("operator") != "in":
            continue
        raw_values = condition.get("value")
        if not isinstance(raw_values, list):
            continue
        values = {str(value) for value in raw_values}
        for choices, buttons_key in buttons_by_choices:
            if values and values.issubset(choices):
                condition_to_buttons[command["key"]] = buttons_key
                break

    if not condition_to_buttons:
        return commands

    repaired: list[dict[str, Any]] = []
    for command in commands:
        key = command.get("key")
        if command.get("action") == "add_condition" and key in condition_to_buttons:
            continue
        if command.get("action") == "connect":
            source = command.get("source")
            target = command.get("target")
            if target in condition_to_buttons:
                continue
            if source in condition_to_buttons:
                command = {**command, "source": condition_to_buttons[str(source)]}
        repaired.append(command)
    return repaired


def collapse_button_choice_conditions(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    button_choices: dict[str, set[str]] = {}
    for command in commands:
        if command.get("action") != "add_buttons" or not isinstance(command.get("key"), str):
            continue
        button_choices[command["key"]] = {
            str(button.get("text") if isinstance(button, dict) else button)
            for button in command.get("buttons", [])
        }

    condition_to_branch: dict[str, tuple[str, str]] = {}
    for command in commands:
        if command.get("action") != "add_condition" or not isinstance(command.get("key"), str):
            continue
        conditions = command.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != 1:
            continue
        condition = conditions[0]
        if not isinstance(condition, dict) or condition.get("operator") != "equals":
            continue
        variable = condition.get("variable")
        value = str(condition.get("value"))
        if isinstance(variable, str) and variable in button_choices and value in button_choices[variable]:
            condition_to_branch[command["key"]] = (variable, value)

    if not condition_to_branch:
        return commands

    repaired: list[dict[str, Any]] = []
    for command in commands:
        key = command.get("key")
        if command.get("action") == "add_condition" and key in condition_to_branch:
            continue
        if command.get("action") == "connect" and command.get("source") in condition_to_branch:
            buttons_key, branch = condition_to_branch[str(command["source"])]
            if command.get("on") in {"true", True, None, ""}:
                command = {**command, "source": buttons_key, "on": branch}
        if command.get("after") in condition_to_branch:
            buttons_key, branch = condition_to_branch[str(command["after"])]
            command = {**command, "after": None}
            repaired.append(command)
            if isinstance(command.get("key"), str):
                repaired.append({"action": "connect", "source": buttons_key, "target": command["key"], "on": branch})
            continue
        repaired.append(command)
    return repaired


def synthesize_condition_after_branches(commands: list[dict[str, Any]]) -> None:
    condition_keys = {
        command["key"]
        for command in commands
        if command.get("action") == "add_condition" and isinstance(command.get("key"), str)
    }
    outgoing_conditions = {
        command.get("source")
        for command in commands
        if command.get("action") == "connect" and command.get("source") in condition_keys
    }
    for condition_key in sorted(condition_keys - outgoing_conditions):
        followers = [
            command
            for command in commands
            if command.get("action") != "connect"
            and command.get("after") == condition_key
            and isinstance(command.get("key"), str)
        ]
        if not followers:
            continue
        commands.append({"action": "connect", "source": condition_key, "target": followers[0]["key"], "on": "true"})
        followers[0]["after"] = None
        if len(followers) > 1:
            commands.append({"action": "connect", "source": condition_key, "target": followers[1]["key"], "on": "false"})
            followers[1]["after"] = None


def remove_orphan_finish_commands(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    incoming_targets = {
        command.get("target")
        for command in commands
        if command.get("action") == "connect" and isinstance(command.get("target"), str)
    }
    repaired: list[dict[str, Any]] = []
    seen_real_steps = 0
    for command in commands:
        if command.get("action") not in {"create_bot", "connect"}:
            seen_real_steps += 1
        if (
            command.get("action") == "add_finish"
            and command.get("after") is None
            and command.get("key") not in incoming_targets
            and seen_real_steps > 1
        ):
            continue
        repaired.append(command)
    return repaired


def normalize_condition_input(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        return [normalize_condition_object(raw)]
    if isinstance(raw, list):
        return [normalize_condition_object(item) for item in raw]
    if isinstance(raw, str):
        parsed = parse_condition_string(raw)
        if parsed is not None:
            return [parsed]
    raise ValueError("Condition must be an object, array, or simple expression like `age >= 18`")


def normalize_condition_object(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Condition item must be an object")
    condition = dict(raw)
    aliases = {
        "==": "equals",
        "=": "equals",
        "!=": "not_equals",
        ">": "greater_than",
        "<": "less_than",
        ">=": ">=",
        "<=": "<=",
        "exists": "not_empty",
        "present": "not_empty",
    }
    operator = condition.get("operator")
    if isinstance(operator, str) and operator in aliases:
        condition["operator"] = aliases[operator]
    return condition


def parse_condition_string(raw: str) -> dict[str, Any] | None:
    match = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(>=|<=|==|=|!=|>|<)\s*('?\"?)(.+?)\3\s*$", raw)
    if not match:
        return None
    variable, operator, _, value = match.groups()
    parsed_value: str | int | float = value
    if re.fullmatch(r"-?\d+", value):
        parsed_value = int(value)
    elif re.fullmatch(r"-?\d+\.\d+", value):
        parsed_value = float(value)
    return normalize_condition_object({"variable": variable, "operator": operator, "value": parsed_value})


def format_issues(result: CompileResult) -> str:
    return "\n".join(f"- {issue.code}: {issue.message}" for issue in result.issues)


def system_prompt() -> str:
    return f"""
You are Botmother AgentV4. You build Telegram bot flows using only validated BotBuilder commands.

Return only valid JSON:
{{"commands":[...]}}

Allowed commands:
- create_bot(name, language)
- add_text(key, text, after?)
- add_question(key, text, variable, answer_type?, after?)
- add_buttons(key, text, buttons, mode?, after?)
- add_condition(key, conditions, after?)
- add_finish(key, text, after?)
- add_engine_node(key, node_type, data, after?, width?, height?) for direct engine-golang nodes
- connect(source, target, on?)

Rules:
- Always start with create_bot.
- Every command object must use the key `action`. Do not use `command` or `tool`.
- The start step is created automatically as key "start".
- Use snake_case keys matching ^[a-z][a-z0-9_]{1,63}$.
- For a linear step, prefer the `after` field instead of manual connect.
- For buttons, add one connect per button branch. The `on` value must equal the exact button text.
- For conditions, add both true and false branches with connect(..., on="true") and connect(..., on="false").
- Conditions must be objects like {{"variable":"age","operator":">=","value":18}}.
- For collected answers, conditions may use the plain variable name; the compiler will map it to state variables.
- For direct engine nodes, node_type must be one of the engine catalog entries and data must match the handler keys.
- Branching engine nodes use connect(..., on="handle"): IfConditionNode true/false, RandomNode option_0/option_1, CheckMembershipNode is-member/not-member, ForLoopNode loop-body/no-items, ForLoopContinueNode loop-continue/loop-done.
- Do not output nodes, edges, code, markdown, comments, or explanations.

{engine_catalog_prompt()}
""".strip()


def user_prompt(prompt: str, *, language: str, feedback: str) -> str:
    repair = f"\nCompiler feedback to fix:\n{feedback}\n" if feedback else ""
    return f"""
Build a useful bot in language `{language}` for this request:
{prompt}

Keep the bot practical and not too large. Use 4 to 12 commands when possible.
{repair}
""".strip()
