import json

import pytest

from agentv4.ai import AIBuildRequest, parse_commands, user_prompt
from agentv4.builder.session import build_from_commands


def test_parse_ai_commands_json() -> None:
    raw = json.dumps(
        {
            "commands": [
                {"action": "create_bot", "name": "Cafe bot", "language": "uz"},
                {"action": "add_text", "key": "welcome", "text": "Salom", "after": "start"},
            ]
        }
    )

    commands = parse_commands(raw)
    result = build_from_commands(commands)

    assert result.ok is True
    assert commands[0].action == "create_bot"


def test_parse_tolerates_command_key_and_string_buttons() -> None:
    raw = json.dumps(
        {
            "commands": [
                {"command": "create_bot", "name": "Cafe bot", "language": "uz"},
                {
                    "command": "add_buttons",
                    "key": "menu",
                    "text": "Tanlang",
                    "buttons": ["Pizza", "Burger"],
                    "after": "start",
                },
                {"command": "add_finish", "key": "pizza", "text": "Pizza tanlandi."},
                {"command": "connect", "source": "menu", "target": "pizza", "on": "Pizza"},
            ]
        }
    )

    commands = parse_commands(raw)
    result = build_from_commands(commands)

    assert result.ok is True
    assert commands[1].buttons[0].text == "Pizza"


def test_parse_simple_condition_string() -> None:
    raw = json.dumps(
        {
            "commands": [
                {"command": "create_bot", "name": "Age bot", "language": "uz"},
                {
                    "command": "add_condition",
                    "key": "check_age",
                    "condition": "age >= 18",
                    "after": "start",
                },
                {"command": "add_finish", "key": "ok", "text": "OK"},
                {"command": "add_finish", "key": "no", "text": "NO"},
                {"command": "connect", "source": "check_age", "target": "ok", "on": "true"},
                {"command": "connect", "source": "check_age", "target": "no", "on": "false"},
            ]
        }
    )

    commands = parse_commands(raw)
    assert commands[1].conditions[0].operator == ">="


def test_parse_repairs_common_ai_menu_condition_mistake() -> None:
    raw = json.dumps(
        {
            "commands": [
                {"action": "create_bot", "name": "Kafe Bot", "language": "uz"},
                {
                    "action": "add_buttons",
                    "key": "start",
                    "text": "Menyudan tanlang",
                    "buttons": ["Pizza", "Burger", "Ichimlik"],
                    "after": "menu_selection",
                },
                {
                    "action": "add_condition",
                    "key": "menu_selection",
                    "conditions": {
                        "variable": "menu_item",
                        "operator": "in",
                        "value": ["Pizza", "Burger", "Ichimlik"],
                    },
                    "after": "ask_phone",
                },
                {
                    "action": "add_question",
                    "key": "ask_phone",
                    "text": "Telefon raqamingizni kiriting:",
                    "variable": "phone_number",
                    "answer_type": "text",
                    "after": "order_confirmation",
                },
                {"action": "add_finish", "key": "order_confirmation", "text": "Buyurtmangiz qabul qilindi!"},
                {"action": "connect", "source": "start", "target": "menu_selection", "on": ""},
                {"action": "connect", "source": "menu_selection", "target": "ask_phone", "on": "Pizza"},
                {"action": "connect", "source": "menu_selection", "target": "ask_phone", "on": "Burger"},
                {"action": "connect", "source": "menu_selection", "target": "ask_phone", "on": "Ichimlik"},
            ]
        }
    )

    commands = parse_commands(raw)
    result = build_from_commands(commands)

    assert result.ok is True
    assert result.flow is not None
    assert all(command.action != "add_condition" for command in commands)
    assert any(edge.source == "start_buttons" and edge.target == "ask_phone_prompt" for edge in result.flow.edges)


def test_parse_maps_button_answer_type_and_exists_operator() -> None:
    raw = json.dumps(
        {
            "commands": [
                {"action": "create_bot", "name": "Support bot", "language": "uz"},
                {
                    "action": "add_question",
                    "key": "operator_message",
                    "text": "Xabar yozing",
                    "variable": "message",
                    "answer_type": "button",
                    "after": "start",
                },
                {
                    "action": "add_condition",
                    "key": "has_message",
                    "conditions": [{"variable": "message", "operator": "exists", "value": True}],
                    "after": "operator_message",
                },
                {"action": "add_finish", "key": "done", "text": "Qabul qilindi", "after": "has_message"},
            ]
        }
    )

    commands = parse_commands(raw)
    result = build_from_commands(commands)

    assert result.ok is True
    assert commands[1].answer_type == "text"
    assert commands[2].conditions[0].operator == "not_empty"


def test_parse_synthesizes_condition_after_branches() -> None:
    raw = json.dumps(
        {
            "commands": [
                {"action": "create_bot", "name": "Booking bot", "language": "uz"},
                {
                    "action": "add_question",
                    "key": "ask_phone",
                    "text": "Telefon?",
                    "variable": "phone",
                    "after": "start",
                },
                {
                    "action": "add_condition",
                    "key": "check_phone",
                    "conditions": [{"variable": "phone", "operator": "not_empty", "value": True}],
                    "after": "ask_phone",
                },
                {"action": "add_finish", "key": "ok", "text": "OK", "after": "check_phone"},
                {"action": "add_finish", "key": "no", "text": "NO", "after": "check_phone"},
            ]
        }
    )

    commands = parse_commands(raw)
    result = build_from_commands(commands)

    assert result.ok is True
    assert result.flow is not None
    assert {edge.sourceHandle for edge in result.flow.edges if edge.source == "check_phone"} == {"true", "false"}


def test_parse_collapses_button_choice_conditions() -> None:
    raw = json.dumps(
        {
            "commands": [
                {"action": "create_bot", "name": "Support bot", "language": "uz"},
                {"action": "add_text", "key": "start_text", "text": "Tanlang"},
                {
                    "action": "add_buttons",
                    "key": "main_menu",
                    "text": "Tanlang",
                    "buttons": ["To'lov", "Tariflar", "Operator"],
                    "after": "start_text",
                },
                {
                    "action": "add_condition",
                    "key": "payment_condition",
                    "conditions": [{"variable": "main_menu", "operator": "equals", "value": "To'lov"}],
                },
                {"action": "add_text", "key": "payment_response", "text": "To'lov info", "after": "payment_condition"},
                {"action": "add_finish", "key": "end", "text": "Rahmat"},
            ]
        }
    )

    commands = parse_commands(raw)
    result = build_from_commands(commands)

    assert result.ok is True
    assert all(command.action != "add_condition" for command in commands)
    assert all(getattr(command, "key", None) != "end" for command in commands)
    assert result.flow is not None
    assert any(edge.source == "main_menu" and edge.sourceHandle == "target-handler-main_menu-To'lov" for edge in result.flow.edges)


def test_parse_rejects_non_commands_json() -> None:
    with pytest.raises(ValueError):
        parse_commands(json.dumps({"nodes": [], "edges": []}))


def test_user_prompt_includes_repair_feedback() -> None:
    request = AIBuildRequest(prompt="support bot", language="uz")
    prompt = user_prompt(request.prompt, language=request.language, feedback="- unknown_button: bad")

    assert "Compiler feedback" in prompt
    assert "unknown_button" in prompt
