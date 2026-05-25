from fastapi.testclient import TestClient
import pytest

from agentv4.api.app import create_app
from agentv4.ai import AIBuildResponse
from agentv4.backend import get_backend_client
from agentv4.builder.commands import CreateBotCommand, AddTextCommand
from agentv4.builder.session import build_from_commands
from agentv4.compat_agent import BOTBUILDER_TOOL_NAME, PROJECT_LIST_TOOL_NAME, CompatAgentTurn, CompatToolCall
from agentv4.config import get_settings


@pytest.fixture(autouse=True)
def compat_auth(monkeypatch):
    monkeypatch.setenv("AGENTV4_AUTH_REQUIRED", "false")
    monkeypatch.setenv("AGENTV4_ENFORCE_USAGE_LIMITS", "false")
    monkeypatch.setenv("AGENTV4_REPORT_TOKEN_USAGE", "false")
    get_settings.cache_clear()
    get_backend_client.cache_clear()
    yield
    get_settings.cache_clear()
    get_backend_client.cache_clear()


def test_health() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "agentv4"


def test_build_endpoint() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/v1/build",
        json={
            "commands": [
                {"action": "create_bot", "name": "Support bot", "language": "uz"},
                {"action": "add_text", "key": "welcome", "text": "Salom", "after": "start"},
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["flow"]["nodes"][0]["type"] == "CommandTriggerNode"


def test_engine_catalog_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/v1/engine/catalog")

    assert response.status_code == 200
    body = response.json()
    http_node = next(item for item in body if item["node_type"] == "HTTPRequestNode")
    assert "url" in http_node["required"]
    random_node = next(item for item in body if item["node_type"] == "RandomNode")
    assert random_node["branch_prefixes"] == ["option_"]


def test_build_rejects_invalid_engine_node_data() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/v1/build",
        json={
            "commands": [
                {"action": "create_bot", "name": "API bot", "language": "uz"},
                {
                    "action": "add_engine_node",
                    "key": "fetch_status",
                    "node_type": "HTTPRequestNode",
                    "data": {"method": "GET"},
                    "after": "start",
                },
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["issues"][0]["code"] == "invalid_engine_node_data"


def test_session_commands() -> None:
    client = TestClient(create_app())
    created = client.post("/v1/sessions").json()

    response = client.post(
        f"/v1/sessions/{created['session_id']}/commands",
        json={"command": {"action": "create_bot", "name": "Support bot", "language": "uz"}},
    )

    assert response.status_code == 200
    assert response.json()["result"]["spec"]["name"] == "Support bot"


def test_usage_compat_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/usage")

    assert response.status_code == 200
    body = response.json()
    assert {"limit", "used", "remaining", "input_used", "output_used"}.issubset(body)


def test_default_agent_model_is_gpt_5_2(monkeypatch) -> None:
    monkeypatch.delenv("AGENTV4_OPENAI_MODEL", raising=False)
    get_settings.cache_clear()

    assert get_settings().openai_model == "gpt-5.2"


def test_compat_run_reports_usage_once_and_emits_update(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeBackend:
        async def report_token_usage(self, **kwargs):
            calls.append(kwargs)
            return {
                "usage": {
                    "used": 12,
                    "limit": 100,
                    "remaining": 88,
                    "input_used": 5,
                    "output_used": 7,
                    "token_reset_at": None,
                }
            }

    async def fake_agent_turn(_messages, *, settings=None, allow_tools=True):
        return CompatAgentTurn(text="Salom!", input_tokens=5, output_tokens=7)

    monkeypatch.setenv("AGENTV4_REPORT_TOKEN_USAGE", "true")
    get_settings.cache_clear()
    monkeypatch.setattr("agentv4.api.compat.run_compat_agent_messages", fake_agent_turn)
    monkeypatch.setattr("agentv4.api.compat.get_backend_client", lambda settings: FakeBackend())
    client = TestClient(create_app())

    created = client.post(
        "/projects/project_1/runs",
        json={"prompt": "salom", "thread_id": "usage_thread"},
    )
    assert created.status_code == 200
    request_id = created.json()["request_id"]

    with client.stream("GET", f"/projects/project_1/runs/{request_id}/events") as response:
        assert response.status_code == 200
        stream_text = "".join(response.iter_text())

    assert len(calls) == 1
    assert calls[0]["input_tokens"] == 5
    assert calls[0]["output_tokens"] == 7
    assert "event: usage_updated" in stream_text
    assert stream_text.index("event: usage_updated") < stream_text.index("event: run_completed")


def test_agentv3_compat_run_lifecycle(monkeypatch) -> None:
    class FakeBackend:
        async def update_project(self, project_id, body, *, auth_claims=None):
            return {"id": project_id, **body}

    async def fake_agent_turn(_messages, *, settings=None, allow_tools=True):
        return CompatAgentTurn(
            text="Bot yaratish uchun tool chaqiraman.",
            tool_calls=[
                CompatToolCall(
                    id="tool-call-1",
                    name=BOTBUILDER_TOOL_NAME,
                    arguments={"prompt": "Kafe bot yarat"},
                )
            ],
            input_tokens=5,
            output_tokens=7,
        )

    async def fake_agent_final(*_args, **_kwargs):
        return CompatAgentTurn(
            text="Bot flow tayyor. Flow JSON `tool_result.data.flow` ichida qaytdi.",
            input_tokens=3,
            output_tokens=4,
        )

    async def fake_build_bot_with_ai_stream(request, *, on_delta=None):
        if on_delta is not None:
            await on_delta('{"commands":[')
            await on_delta('{"action":"create_bot"}]}')
        commands = [
            CreateBotCommand(action="create_bot", name="Compat bot", language="uz"),
            AddTextCommand(action="add_text", key="welcome", text=request.prompt, after="start"),
        ]
        result = build_from_commands(commands)
        return AIBuildResponse(
            ok=True,
            model="fake",
            attempts=1,
            commands=[command.model_dump(mode="json") for command in commands],
            result=result,
        )

    monkeypatch.setattr("agentv4.api.compat.run_compat_agent_messages", fake_agent_turn)
    monkeypatch.setattr("agentv4.api.compat.run_compat_agent_final", fake_agent_final)
    monkeypatch.setattr("agentv4.api.compat.build_bot_with_ai_stream", fake_build_bot_with_ai_stream)
    monkeypatch.setattr("agentv4.api.compat.get_backend_client", lambda settings: FakeBackend())
    client = TestClient(create_app())

    created = client.post(
        "/projects/project_1/runs",
        json={"prompt": "Kafe bot yarat", "thread_id": "thread_1"},
    )
    assert created.status_code == 200
    request_id = created.json()["request_id"]

    with client.stream("GET", f"/projects/project_1/runs/{request_id}/events") as response:
        assert response.status_code == 200
        stream_text = "".join(response.iter_text())

    assert "event: run_started" in stream_text
    assert "event: tool_call" in stream_text
    assert "event: tool_result" in stream_text
    assert "project.update" in stream_text
    assert "event: assistant_message" in stream_text
    assert "event: run_completed" in stream_text

    status = client.get(f"/projects/project_1/runs/{request_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert "Bot flow tayyor" in status.json()["final_response"]


def test_agentv3_compat_casual_chat_does_not_build_flow(monkeypatch) -> None:
    async def fake_agent_turn(_messages, *, settings=None, allow_tools=True):
        return CompatAgentTurn(
            text="Salom! Men Botmother AgentV4man. Bot flow tuzishda yordam beraman.",
            input_tokens=4,
            output_tokens=6,
        )

    async def fail_build_bot_with_ai_stream(_request, *, on_delta=None):
        raise AssertionError("casual chat should not call botbuilder")

    monkeypatch.setattr("agentv4.api.compat.run_compat_agent_messages", fake_agent_turn)
    monkeypatch.setattr("agentv4.api.compat.build_bot_with_ai_stream", fail_build_bot_with_ai_stream)
    client = TestClient(create_app())

    created = client.post(
        "/projects/project_1/runs",
        json={"prompt": "salom isming nima", "thread_id": "thread_1"},
    )
    assert created.status_code == 200
    request_id = created.json()["request_id"]

    with client.stream("GET", f"/projects/project_1/runs/{request_id}/events") as response:
        assert response.status_code == 200
        stream_text = "".join(response.iter_text())

    assert "event: assistant_message" in stream_text
    assert "botbuilder.ai_build" not in stream_text
    assert "Men Botmother AgentV4man" in stream_text

    status = client.get(f"/projects/project_1/runs/{request_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert "Men Botmother AgentV4man" in status.json()["final_response"]


def test_agentv3_compat_can_list_projects_with_tool(monkeypatch) -> None:
    class FakeBackend:
        async def list_projects(self, *, search=None, auth_claims=None):
            return [
                {"id": 1, "name": "Kafe bot", "status": "active"},
                {"id": 2, "name": "Support bot", "status": "draft"},
            ]

    calls = 0

    async def fake_agent_messages(_messages, *, settings=None, allow_tools=True):
        nonlocal calls
        calls += 1
        if calls > 1:
            return CompatAgentTurn(
                text="2 ta loyiha topildi: Kafe bot va Support bot.",
                input_tokens=3,
                output_tokens=4,
            )
        return CompatAgentTurn(
            text="Loyihalarni tekshiraman.",
            tool_calls=[
                CompatToolCall(
                    id="tool-call-projects",
                    name=PROJECT_LIST_TOOL_NAME,
                    arguments={},
                )
            ],
            input_tokens=5,
            output_tokens=6,
        )

    async def fail_build_bot_with_ai_stream(_request, *, on_delta=None):
        raise AssertionError("project listing should not call botbuilder")

    monkeypatch.setattr("agentv4.api.compat.run_compat_agent_messages", fake_agent_messages)
    monkeypatch.setattr("agentv4.api.compat.build_bot_with_ai_stream", fail_build_bot_with_ai_stream)
    monkeypatch.setattr("agentv4.api.compat.get_backend_client", lambda settings: FakeBackend())
    client = TestClient(create_app())

    created = client.post(
        "/projects/project_1/runs",
        json={"prompt": "loyihalarimni ko'rsat", "thread_id": "thread_1"},
    )
    assert created.status_code == 200
    request_id = created.json()["request_id"]

    with client.stream("GET", f"/projects/project_1/runs/{request_id}/events") as response:
        assert response.status_code == 200
        stream_text = "".join(response.iter_text())

    assert "event: tool_call" in stream_text
    assert "project.list" in stream_text
    assert "event: tool_result" in stream_text
    assert "Kafe bot" in stream_text
    assert "botbuilder.ai_build" not in stream_text

    status = client.get(f"/projects/project_1/runs/{request_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert "2 ta loyiha topildi" in status.json()["final_response"]


def test_agentv3_compat_can_chain_project_management_tools(monkeypatch) -> None:
    class FakeBackend:
        async def list_plans(self, *, auth_claims=None):
            return [{"id": 11, "name": "Start", "price": "0", "currency": "UZS", "is_default": True}]

        async def create_release(self, project_id, *, plan_id=None, months=None, auth_claims=None):
            return {"id": 77, "project": project_id, "plan_id": plan_id, "months": months, "version": 1}

    calls = 0

    async def fake_agent_messages(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return CompatAgentTurn(
                text="Avval tariflarni tekshiraman.",
                tool_calls=[CompatToolCall(id="plans-call", name="plans_list", arguments={})],
                input_tokens=5,
                output_tokens=6,
            )
        if calls == 2:
            return CompatAgentTurn(
                text="Start tarif topildi, deploy qilaman.",
                tool_calls=[
                    CompatToolCall(
                        id="deploy-call",
                        name="release_deploy",
                        arguments={"plan_id": 11, "months": 1},
                    )
                ],
                input_tokens=7,
                output_tokens=8,
            )
        return CompatAgentTurn(
            text="Deploy navbatga qo'yildi: release id=77.",
            input_tokens=3,
            output_tokens=4,
        )

    monkeypatch.setattr("agentv4.api.compat.run_compat_agent_messages", fake_agent_messages)
    monkeypatch.setattr("agentv4.api.compat.get_backend_client", lambda settings: FakeBackend())
    client = TestClient(create_app())

    created = client.post(
        "/projects/project_9/runs",
        json={"prompt": "loyihani deploy qil", "thread_id": "thread_1"},
    )
    assert created.status_code == 200
    request_id = created.json()["request_id"]

    with client.stream("GET", f"/projects/project_9/runs/{request_id}/events") as response:
        assert response.status_code == 200
        stream_text = "".join(response.iter_text())

    assert "plans.list" in stream_text
    assert "release.deploy" in stream_text
    assert "release id=77" in stream_text

    status = client.get(f"/projects/project_9/runs/{request_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"


def test_agentv3_compat_remembers_thread_messages(monkeypatch) -> None:
    seen_second_messages = []

    async def fake_agent_messages(messages, *, settings=None, allow_tools=True):
        if messages[-1]["content"] == "meni ismim samandar":
            return CompatAgentTurn(text="Tanishganimdan xursandman, Samandar.", input_tokens=2, output_tokens=3)
        seen_second_messages.extend(messages)
        return CompatAgentTurn(text="Sizning ismingiz Samandar.", input_tokens=4, output_tokens=5)

    monkeypatch.setattr("agentv4.api.compat.run_compat_agent_messages", fake_agent_messages)
    client = TestClient(create_app())

    first = client.post(
        "/projects/project_1/runs",
        json={"prompt": "meni ismim samandar", "thread_id": "memory_thread"},
    )
    assert first.status_code == 200
    with client.stream("GET", f"/projects/project_1/runs/{first.json()['request_id']}/events") as response:
        assert response.status_code == 200
        _ = "".join(response.iter_text())

    second = client.post(
        "/projects/project_1/runs",
        json={"prompt": "meni ismim nima", "thread_id": "memory_thread"},
    )
    assert second.status_code == 200
    with client.stream("GET", f"/projects/project_1/runs/{second.json()['request_id']}/events") as response:
        assert response.status_code == 200
        stream_text = "".join(response.iter_text())

    assert "Sizning ismingiz Samandar" in stream_text
    assert any(item["role"] == "user" and item["content"] == "meni ismim samandar" for item in seen_second_messages)


def test_agentv3_compat_uses_created_project_for_next_flow_save(monkeypatch) -> None:
    saved: dict[str, object] = {}

    class FakeBackend:
        async def create_project(self, body, *, auth_claims=None):
            return {"id": 981, "name": body["name"], "status": "draft"}

        async def update_project(self, project_id, body, *, auth_claims=None):
            saved["project_id"] = project_id
            saved["body"] = body
            return {"id": project_id, **body}

    calls = 0

    async def fake_agent_messages(messages, *, settings=None, allow_tools=True):
        nonlocal calls
        calls += 1
        current_prompt = next(item["content"] for item in reversed(messages) if item["role"] == "user")
        if current_prompt == "loyiha yarat ai agent test nomli":
            if messages[-1]["role"] != "tool":
                return CompatAgentTurn(
                    text="Loyiha yarataman.",
                    tool_calls=[
                        CompatToolCall(
                            id="create-project",
                            name="project_create",
                            arguments={"name": "ai agent test"},
                        )
                    ],
                )
            return CompatAgentTurn(text='Loyiha "ai agent test" yaratildi. ID: 981.')
        return CompatAgentTurn(
            text="Flow yarataman.",
            tool_calls=[
                CompatToolCall(
                    id="build-flow",
                    name=BOTBUILDER_TOOL_NAME,
                    arguments={"prompt": current_prompt},
                )
            ],
        )

    async def fake_agent_final(*_args, **_kwargs):
        return CompatAgentTurn(text="Flow yaratildi va loyihaga saqlandi.")

    async def fake_build_bot_with_ai_stream(request, *, on_delta=None):
        commands = [
            CreateBotCommand(action="create_bot", name="Compat bot", language="uz"),
            AddTextCommand(action="add_text", key="welcome", text="Salom", after="start"),
        ]
        return AIBuildResponse(
            ok=True,
            model="fake",
            attempts=1,
            commands=[command.model_dump(mode="json") for command in commands],
            result=build_from_commands(commands),
        )

    monkeypatch.setattr("agentv4.api.compat.run_compat_agent_messages", fake_agent_messages)
    monkeypatch.setattr("agentv4.api.compat.run_compat_agent_final", fake_agent_final)
    monkeypatch.setattr("agentv4.api.compat.build_bot_with_ai_stream", fake_build_bot_with_ai_stream)
    monkeypatch.setattr("agentv4.api.compat.get_backend_client", lambda settings: FakeBackend())
    client = TestClient(create_app())

    first = client.post(
        "/projects/project_1/runs",
        json={"prompt": "loyiha yarat ai agent test nomli", "thread_id": "project_thread"},
    )
    assert first.status_code == 200
    with client.stream("GET", f"/projects/project_1/runs/{first.json()['request_id']}/events") as response:
        assert response.status_code == 200
        _ = "".join(response.iter_text())

    second = client.post(
        "/projects/project_1/runs",
        json={"prompt": "endi shu loyihaga start bossa salom deydigan qilib ber", "thread_id": "project_thread"},
    )
    assert second.status_code == 200
    with client.stream("GET", f"/projects/project_1/runs/{second.json()['request_id']}/events") as response:
        assert response.status_code == 200
        stream_text = "".join(response.iter_text())

    assert "project.update" in stream_text
    assert saved["project_id"] == "981"
    assert isinstance(saved["body"], dict)
    assert "flow" in saved["body"]


def test_agentv3_compat_subscription_current_handles_int_plan(monkeypatch) -> None:
    class FakeBackend:
        async def subscription_current(self, *, auth_claims=None):
            return {"id": 5, "plan": 1, "status": "active"}

    calls = 0

    async def fake_agent_messages(_messages, *, settings=None, allow_tools=True):
        nonlocal calls
        calls += 1
        if calls == 1:
            return CompatAgentTurn(
                text="Planingizni tekshiraman.",
                tool_calls=[CompatToolCall(id="subscription-call", name="subscription_current", arguments={})],
            )
        return CompatAgentTurn(text="Siz plan#1 tarifidasiz, status active.")

    monkeypatch.setattr("agentv4.api.compat.run_compat_agent_messages", fake_agent_messages)
    monkeypatch.setattr("agentv4.api.compat.get_backend_client", lambda settings: FakeBackend())
    client = TestClient(create_app())

    created = client.post(
        "/projects/global/runs",
        json={"prompt": "meni planim qanday", "thread_id": "subscription_thread"},
    )
    assert created.status_code == 200

    with client.stream("GET", f"/projects/global/runs/{created.json()['request_id']}/events") as response:
        assert response.status_code == 200
        stream_text = "".join(response.iter_text())

    assert "subscription.current" in stream_text
    assert "plan#1" in stream_text
    assert "Backend error" not in stream_text


def test_agentv3_compat_retry_reuses_failed_tool(monkeypatch) -> None:
    class FakeBackend:
        attempts = 0

        async def subscription_current(self, *, auth_claims=None):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("temporary subscription failure")
            return {"plan": 2, "status": "active"}

        async def create_release(self, *_args, **_kwargs):
            raise AssertionError("retry should not deploy")

    backend = FakeBackend()
    calls = 0

    async def fake_agent_messages(_messages, *, settings=None, allow_tools=True):
        nonlocal calls
        calls += 1
        if calls in {1, 3}:
            return CompatAgentTurn(
                text="Planingizni tekshiraman.",
                tool_calls=[CompatToolCall(id=f"subscription-call-{calls}", name="subscription_current", arguments={})],
            )
        if calls == 2:
            return CompatAgentTurn(text="Planingizni olishda xatolik bo'ldi.")
        return CompatAgentTurn(text="Qayta urinish muvaffaqiyatli: plan#2 active.")

    monkeypatch.setattr("agentv4.api.compat.run_compat_agent_messages", fake_agent_messages)
    monkeypatch.setattr("agentv4.api.compat.get_backend_client", lambda settings: backend)
    client = TestClient(create_app())

    first = client.post(
        "/projects/global/runs",
        json={"prompt": "meni planim qanday", "thread_id": "retry_subscription_thread"},
    )
    assert first.status_code == 200
    with client.stream("GET", f"/projects/global/runs/{first.json()['request_id']}/events") as response:
        assert response.status_code == 200
        first_stream = "".join(response.iter_text())
    assert "subscription.current" in first_stream
    assert "run_failed" in first_stream

    second = client.post(
        "/projects/global/runs",
        json={"prompt": "qayta urun", "thread_id": "retry_subscription_thread"},
    )
    assert second.status_code == 200
    with client.stream("GET", f"/projects/global/runs/{second.json()['request_id']}/events") as response:
        assert response.status_code == 200
        second_stream = "".join(response.iter_text())

    assert "subscription.current" in second_stream
    assert "release.deploy" not in second_stream
    assert "plan#2" in second_stream


def test_run_requires_auth_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AGENTV4_AUTH_REQUIRED", "true")
    monkeypatch.setenv("AGENTV4_ENFORCE_USAGE_LIMITS", "false")
    get_settings.cache_clear()

    client = TestClient(create_app())
    response = client.post("/projects/project_1/runs", json={"prompt": "Bot yarat"})

    assert response.status_code == 401


def test_run_rejects_exhausted_token_limit(monkeypatch) -> None:
    class FakeBackend:
        async def subscription_usage(self, *, auth_claims=None):
            return {"ai_token_usage": {"remaining": 0, "used": 10, "limit": 10}}

    monkeypatch.setenv("AGENTV4_AUTH_REQUIRED", "true")
    monkeypatch.setenv("AGENTV4_INTERNAL_API_TOKEN", "secret")
    monkeypatch.setenv("AGENTV4_ENFORCE_USAGE_LIMITS", "true")
    get_settings.cache_clear()
    monkeypatch.setattr("agentv4.limits.get_backend_client", lambda settings: FakeBackend())

    client = TestClient(create_app())
    response = client.post(
        "/projects/project_1/runs",
        headers={"X-Internal-Token": "secret"},
        json={"prompt": "Bot yarat"},
    )

    assert response.status_code == 429
    assert response.json()["detail"]["tokens_remaining"] == 0
