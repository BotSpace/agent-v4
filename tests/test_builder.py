from agentv4.builder.commands import (
    AddButtonsCommand,
    AddConditionCommand,
    AddEngineNodeCommand,
    AddFinishCommand,
    AddQuestionCommand,
    AddTextCommand,
    ConnectCommand,
    CreateBotCommand,
)
from agentv4.builder.session import build_from_commands


def test_build_simple_button_flow() -> None:
    result = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="Order bot", language="uz"),
            AddTextCommand(action="add_text", key="welcome", text="Assalomu alaykum!", after="start"),
            AddButtonsCommand(
                action="add_buttons",
                key="menu",
                text="Nima buyurtma qilasiz?",
                buttons=[{"text": "Pizza"}, {"text": "Burger"}],
                after="welcome",
            ),
            AddFinishCommand(action="add_finish", key="pizza_done", text="Pizza tanlandi."),
            ConnectCommand(action="connect", source="menu", target="pizza_done", on="Pizza"),
        ]
    )

    assert result.ok is True
    assert result.flow is not None
    assert [node.id for node in result.flow.nodes] == ["start", "welcome", "menu", "pizza_done"]
    assert any(edge.sourceHandle == "target-handler-menu-Pizza" for edge in result.flow.edges)


def test_rejects_unknown_button_branch() -> None:
    result = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="Order bot", language="uz"),
            AddButtonsCommand(
                action="add_buttons",
                key="menu",
                text="Tanlang",
                buttons=[{"text": "Pizza"}],
                after="start",
            ),
            AddFinishCommand(action="add_finish", key="burger_done", text="Burger tanlandi."),
            ConnectCommand(action="connect", source="menu", target="burger_done", on="Burger"),
        ]
    )

    assert result.ok is False
    assert any(issue.code == "unknown_button" for issue in result.issues)


def test_question_compiles_to_message_trigger_state() -> None:
    result = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="Lead bot", language="uz"),
            AddQuestionCommand(
                action="add_question",
                key="ask_phone",
                text="Telefon raqamingizni yuboring",
                variable="phone",
                answer_type="phone",
                after="start",
            ),
            AddFinishCommand(action="add_finish", key="done", text="Rahmat, {{state_phone}}."),
            ConnectCommand(action="connect", source="ask_phone", target="done"),
        ]
    )

    assert result.ok is True
    assert result.flow is not None
    prompt = next(node for node in result.flow.nodes if node.id == "ask_phone_prompt")
    assert prompt.type == "SendTextMessageNode"
    assert prompt.data["messageText"] == "Telefon raqamingizni yuboring"
    ask_phone = next(node for node in result.flow.nodes if node.id == "ask_phone")
    assert ask_phone.type == "MessageTriggerNode"
    assert ask_phone.data["state"] == {"key": "phone", "type": "phone"}
    assert any(edge.source == "ask_phone_prompt" and edge.target == "ask_phone" for edge in result.flow.edges)


def test_order_bot_with_two_button_branches() -> None:
    result = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="Cafe bot", language="uz"),
            AddTextCommand(action="add_text", key="welcome", text="Xush kelibsiz!", after="start"),
            AddButtonsCommand(
                action="add_buttons",
                key="menu",
                text="Nima buyurtma qilasiz?",
                buttons=[{"text": "Pizza"}, {"text": "Ichimlik"}],
                after="welcome",
            ),
            AddFinishCommand(action="add_finish", key="pizza", text="Pizza bo'limi."),
            AddFinishCommand(action="add_finish", key="drink", text="Ichimlik bo'limi."),
            ConnectCommand(action="connect", source="menu", target="pizza", on="Pizza"),
            ConnectCommand(action="connect", source="menu", target="drink", on="Ichimlik"),
        ]
    )

    assert result.ok is True
    assert result.flow is not None
    handles = {edge.sourceHandle for edge in result.flow.edges if edge.source == "menu"}
    assert handles == {"target-handler-menu-Pizza", "target-handler-menu-Ichimlik"}


def test_condition_bot_normalizes_question_variable() -> None:
    result = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="Age gate", language="uz"),
            AddQuestionCommand(
                action="add_question",
                key="ask_age",
                text="Yoshingiz nechida?",
                variable="age",
                answer_type="number",
                after="start",
            ),
            AddConditionCommand(
                action="add_condition",
                key="check_age",
                conditions=[{"variable": "age", "operator": "greater_than", "value": 17}],
                after="ask_age",
            ),
            AddFinishCommand(action="add_finish", key="adult", text="Marhamat."),
            AddFinishCommand(action="add_finish", key="minor", text="Kechirasiz, ruxsat yo'q."),
            ConnectCommand(action="connect", source="check_age", target="adult", on="true"),
            ConnectCommand(action="connect", source="check_age", target="minor", on="false"),
        ]
    )

    assert result.ok is True
    assert result.flow is not None
    check_age = next(node for node in result.flow.nodes if node.id == "check_age")
    assert check_age.data["conditions"][0]["variable"] == "state_age"
    handles = {edge.sourceHandle for edge in result.flow.edges if edge.source == "check_age"}
    assert handles == {"true", "false"}


def test_support_faq_bot() -> None:
    result = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="Support bot", language="uz"),
            AddButtonsCommand(
                action="add_buttons",
                key="faq",
                text="Qaysi savol bo'yicha yordam kerak?",
                buttons=[{"text": "To'lov"}, {"text": "Operator"}],
                after="start",
            ),
            AddFinishCommand(action="add_finish", key="payment_help", text="To'lov bo'yicha yo'riqnoma."),
            AddQuestionCommand(
                action="add_question",
                key="operator_message",
                text="Operatorga xabaringizni yozing.",
                variable="operator_text",
            ),
            AddFinishCommand(action="add_finish", key="operator_done", text="Xabaringiz qabul qilindi."),
            ConnectCommand(action="connect", source="faq", target="payment_help", on="To'lov"),
            ConnectCommand(action="connect", source="faq", target="operator_message", on="Operator"),
            ConnectCommand(action="connect", source="operator_message", target="operator_done"),
        ]
    )

    assert result.ok is True
    assert result.flow is not None
    assert any(edge.source == "faq" and edge.target == "operator_message_prompt" for edge in result.flow.edges)


def test_unbranched_button_edge_expands_to_all_buttons() -> None:
    result = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="Lead bot", language="uz"),
            AddButtonsCommand(
                action="add_buttons",
                key="service",
                text="Xizmat turini tanlang",
                buttons=[{"text": "Remont"}, {"text": "Diagnostika"}],
                after="start",
            ),
            AddQuestionCommand(
                action="add_question",
                key="ask_name",
                text="Ismingiz?",
                variable="name",
                after="service",
            ),
        ]
    )

    assert result.ok is True
    assert result.flow is not None
    service_edges = [edge for edge in result.flow.edges if edge.source == "service"]
    assert {edge.sourceHandle for edge in service_edges} == {
        "target-handler-service-Remont",
        "target-handler-service-Diagnostika",
    }
    assert {edge.target for edge in service_edges} == {"ask_name_prompt"}


def test_first_step_without_after_connects_to_start() -> None:
    result = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="Welcome bot", language="uz"),
            AddTextCommand(action="add_text", key="welcome", text="Salom"),
            AddFinishCommand(action="add_finish", key="done", text="Tayyor", after="welcome"),
        ]
    )

    assert result.ok is True
    assert result.flow is not None
    assert any(edge.source == "start" and edge.target == "welcome" for edge in result.flow.edges)


def test_unknown_after_reference_is_ignored() -> None:
    result = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="Loose bot", language="uz"),
            AddTextCommand(action="add_text", key="welcome", text="Salom"),
            AddFinishCommand(action="add_finish", key="done", text="Tayyor", after="missing_step"),
        ]
    )

    assert result.ok is False
    assert not any(issue.code == "unknown_source" for issue in result.issues)
    assert any(issue.code == "unreachable_step" for issue in result.issues)


def test_after_condition_does_not_create_ambiguous_edge() -> None:
    result = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="Age bot", language="uz"),
            AddConditionCommand(
                action="add_condition",
                key="check_age",
                conditions=[{"variable": "age", "operator": ">=", "value": 18}],
                after="start",
            ),
            AddFinishCommand(action="add_finish", key="ok", text="OK", after="check_age"),
            AddFinishCommand(action="add_finish", key="no", text="NO", after="check_age"),
            ConnectCommand(action="connect", source="check_age", target="ok", on="allowed"),
            ConnectCommand(action="connect", source="check_age", target="no", on="denied"),
        ]
    )

    assert result.ok is True
    assert result.flow is not None
    assert {edge.sourceHandle for edge in result.flow.edges if edge.source == "check_age"} == {"true", "false"}


def test_engine_node_compiles_raw_engine_data() -> None:
    result = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="API bot", language="uz"),
            AddEngineNodeCommand(
                action="add_engine_node",
                key="fetch_status",
                node_type="HTTPRequestNode",
                data={
                    "method": "GET",
                    "url": "https://api.example.com/status",
                    "responseVariable": "status_response",
                    "timeout": 10,
                },
                after="start",
            ),
            AddEngineNodeCommand(
                action="add_engine_node",
                key="send_photo",
                node_type="SendPhotoNode",
                data={"photoSourceType": "url", "photoUrl": "https://example.com/a.jpg", "caption": "Rasm"},
                after="fetch_status",
            ),
        ]
    )

    assert result.ok is True
    assert result.flow is not None
    fetch = next(node for node in result.flow.nodes if node.id == "fetch_status")
    assert fetch.type == "HTTPRequestNode"
    assert fetch.data["responseVariable"] == "status_response"
    photo = next(node for node in result.flow.nodes if node.id == "send_photo")
    assert photo.type == "SendPhotoNode"
    assert photo.data["photoUrl"] == "https://example.com/a.jpg"


def test_engine_condition_uses_branch_handles() -> None:
    result = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="Condition bot", language="uz"),
            AddEngineNodeCommand(
                action="add_engine_node",
                key="check_age",
                node_type="IfConditionNode",
                data={
                    "operator": "AND",
                    "conditions": [{"variable": "state_age", "operator": ">=", "value": 18}],
                },
                after="start",
            ),
            AddFinishCommand(action="add_finish", key="ok", text="OK"),
            AddFinishCommand(action="add_finish", key="no", text="NO"),
            ConnectCommand(action="connect", source="check_age", target="ok", on="true"),
            ConnectCommand(action="connect", source="check_age", target="no", on="false"),
        ]
    )

    assert result.ok is True
    assert result.flow is not None
    assert {edge.sourceHandle for edge in result.flow.edges if edge.source == "check_age"} == {"true", "false"}


def test_engine_random_requires_valid_branch_handle() -> None:
    result = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="Random bot", language="uz"),
            AddEngineNodeCommand(
                action="add_engine_node",
                key="random_offer",
                node_type="RandomNode",
                data={"options": ["A", "B"]},
                after="start",
            ),
            AddFinishCommand(action="add_finish", key="offer_a", text="A"),
            ConnectCommand(action="connect", source="random_offer", target="offer_a", on="option-0"),
        ]
    )

    assert result.ok is False
    assert any(issue.code == "unknown_engine_branch" for issue in result.issues)


def test_engine_random_accepts_engine_branch_handle() -> None:
    result = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="Random bot", language="uz"),
            AddEngineNodeCommand(
                action="add_engine_node",
                key="random_offer",
                node_type="RandomNode",
                data={"options": ["A", "B"]},
                after="start",
            ),
            AddFinishCommand(action="add_finish", key="offer_a", text="A"),
            ConnectCommand(action="connect", source="random_offer", target="offer_a", on="option_0"),
        ]
    )

    assert result.ok is True
    assert result.flow is not None
    assert any(edge.source == "random_offer" and edge.sourceHandle == "option_0" for edge in result.flow.edges)


def test_engine_node_requires_required_data() -> None:
    result = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="Broken API bot", language="uz"),
            AddEngineNodeCommand(
                action="add_engine_node",
                key="fetch_status",
                node_type="HTTPRequestNode",
                data={"method": "GET"},
                after="start",
            ),
        ]
    )

    assert result.ok is False
    assert any(issue.code == "invalid_engine_node_data" for issue in result.issues)


def test_rejects_unknown_engine_node() -> None:
    result = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="Bad bot", language="uz"),
            AddEngineNodeCommand(
                action="add_engine_node",
                key="bad_node",
                node_type="NotRealNode",
                data={},
                after="start",
            ),
        ]
    )

    assert result.ok is False
    assert any(issue.code == "unknown_engine_node" for issue in result.issues)
