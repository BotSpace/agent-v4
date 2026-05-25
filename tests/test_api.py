from fastapi.testclient import TestClient

from agentv4.api.app import create_app
from agentv4.ai import AIBuildResponse
from agentv4.builder.commands import CreateBotCommand, AddTextCommand
from agentv4.builder.session import build_from_commands


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


def test_agentv3_compat_run_lifecycle(monkeypatch) -> None:
    async def fake_build_bot_with_ai(request):
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

    monkeypatch.setattr("agentv4.api.compat.build_bot_with_ai", fake_build_bot_with_ai)
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
    assert "event: assistant_message" in stream_text
    assert "event: run_completed" in stream_text

    status = client.get(f"/projects/project_1/runs/{request_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert "Bot flow tayyor" in status.json()["final_response"]
