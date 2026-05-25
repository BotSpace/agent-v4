# AgentV4 BotBuilder

AgentV4 is a controlled BotBuilder compiler for Botmother flows.

The main idea is simple: the AI does not write the final `nodes/edges` flow JSON directly. It sends validated commands such as `add_text`, `add_question`, `add_buttons`, `add_condition`, `add_engine_node`, and `connect`. AgentV4 validates those commands and compiles them into the engine-compatible flow format.

## Run

```bash
pip install -e ".[dev]"
uvicorn agentv4.main:app --reload --port 8094
```

Production container:

```bash
docker build -t agentv4 .
docker run --rm -p 8094:8094 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e AGENTV4_OPENAI_MODEL=gpt-4o-mini \
  -e AGENTV4_OPENAI_TIMEOUT=45 \
  -e AGENTV4_OPENAI_MAX_TOKENS=1800 \
  -e AGENTV4_INTERNAL_API_TOKEN="$AGENT_INTERNAL_API_TOKEN" \
  -e BACKEND_BASE_URL="http://backend:8000" \
  -e BACKEND_INTERNAL_TOKEN="$BACKEND_INTERNAL_TOKEN" \
  agentv4
```

## Build A Bot In One Request

```bash
curl -X POST http://localhost:8094/v1/build \
  -H 'Authorization: Bearer <access-token>' \
  -H 'content-type: application/json' \
  -d '{
    "commands": [
      {"action":"create_bot","name":"Order bot","language":"uz"},
      {"action":"add_text","key":"welcome","text":"Assalomu alaykum!","after":"start"},
      {"action":"add_buttons","key":"menu","text":"Nima buyurtma qilasiz?","after":"welcome","buttons":[{"text":"Pizza"},{"text":"Burger"}]},
      {"action":"add_finish","key":"pizza_done","text":"Pizza tanlandi."},
      {"action":"connect","source":"menu","target":"pizza_done","on":"Pizza"}
    ]
  }'
```

## Session API

```bash
curl -X POST http://localhost:8094/v1/sessions \
  -H 'Authorization: Bearer <access-token>'
curl -X POST http://localhost:8094/v1/sessions/{session_id}/commands \
  -H 'Authorization: Bearer <access-token>' \
  -H 'content-type: application/json' \
  -d '{"command":{"action":"create_bot","name":"Support bot","language":"uz"}}'
```

## AI Build API

`/v1/ai/build` uses OpenAI to generate BotBuilder commands, then validates and compiles them. The API key is read only from the runtime environment.

```bash
export OPENAI_API_KEY=...
export AGENTV4_OPENAI_MODEL=gpt-4o-mini
export AGENTV4_OPENAI_TIMEOUT=45
export AGENTV4_OPENAI_MAX_TOKENS=1800

curl -X POST http://localhost:8094/v1/ai/build \
  -H 'Authorization: Bearer <access-token>' \
  -H 'content-type: application/json' \
  -d '{"prompt":"pizza buyurtma qabul qiladigan bot yarat","language":"uz"}'
```

## Auth And Limits

AgentV4 carries forward AgentV3-style request protection:

- `Authorization: Bearer <jwt>` validates the user token with `JWT_PUBLIC_KEY` or `JWT_PUBLIC_KEY_PATH`.
- `X-Internal-Token` is accepted when it matches `AGENTV4_INTERNAL_API_TOKEN` or `AGENT_INTERNAL_API_TOKEN`.
- `/usage`, `/v1/*`, and `/projects/{project_id}/runs/*` require auth by default.
- AI endpoints and AgentV3-compatible run creation check `/api/subscription/usage/` before spending tokens.
- Successful AI runs report token usage to `/api/subscription/usage/report/` when provider usage is available.

Useful runtime envs:

```bash
AGENTV4_AUTH_REQUIRED=true
AGENTV4_ENFORCE_USAGE_LIMITS=true
AGENTV4_REPORT_TOKEN_USAGE=true
BACKEND_BASE_URL=http://localhost:8000
BACKEND_INTERNAL_TOKEN=...
JWT_ALGORITHM=RS256
JWT_PUBLIC_KEY_PATH=keys/public.pem
```

## Supported Commands

- `create_bot`: initializes the bot and creates the global `/start` trigger.
- `add_text`: adds a text message step.
- `add_question`: adds a message trigger step that saves user input into `state_{variable}`.
- `add_buttons`: adds an inline or reply keyboard message.
- `add_condition`: adds an `IfConditionNode`.
- `add_finish`: adds a final text message step.
- `add_engine_node`: adds a direct `engine-golang` node with raw handler data.
- `connect`: connects two steps. Branching steps require `on`.

## Engine-Golang Coverage

For advanced engine features, use `add_engine_node`:

```json
{
  "action": "add_engine_node",
  "key": "fetch_status",
  "node_type": "HTTPRequestNode",
  "data": {
    "method": "GET",
    "url": "https://api.example.com/status",
    "responseVariable": "status_response",
    "timeout": 10
  },
  "after": "start"
}
```

The catalog is generated from `engine-golang` executor support and exposed to the AI prompt in `agentv4.engine_catalog`. It includes trigger, media, message operation, flow-control, collection, HTTP, custom-code, scheduler, loop, state, and admin nodes.

Branching engine nodes use engine source handles:

- `IfConditionNode`: `true`, `false`
- `RandomNode`: `option_0`, `option_1`, ...
- `CheckMembershipNode`: `is-member`, `not-member`
- `ForLoopNode`: `loop-body`, `no-items`
- `ForLoopContinueNode`: `loop-continue`, `loop-done`

## Output

Successful responses include:

- `spec`: the controlled internal bot spec.
- `flow`: Botmother engine-compatible `{ "nodes": [], "edges": [] }`.
- `issues`: validation errors when the bot cannot be compiled.

## Production Notes

- Keep `OPENAI_API_KEY` only in runtime secrets; do not commit it.
- `/v1/build` and `/v1/compile` are deterministic and do not call OpenAI.
- `/v1/ai/build` has an OpenAI timeout and returns `504` if the provider call times out.
- Sessions are in-memory and intended for interactive editing. Store compiled `flow` in backend if it must survive restarts.
- Treat `CustomCodeNode`, `HTTPRequestNode`, and collection nodes as privileged features in your product UI. AgentV4 validates required fields and engine branch handles, but your backend should still enforce tenant/project permissions and allowed URLs/collections.

## AgentV3 Frontend Compatibility

AgentV4 exposes the endpoints used by the current dashboard agent UI:

- `POST /projects/{project_id}/runs`
- `GET /projects/{project_id}/runs/{request_id}`
- `GET /projects/{project_id}/runs/{request_id}/events`
- `POST /projects/{project_id}/runs/{request_id}/clarify`
- `POST /projects/{project_id}/runs/{request_id}/approve`
- `POST /projects/{project_id}/runs/{request_id}/cancel`
- `GET /usage`

These endpoints use the same bearer/internal-token auth and backend-backed AI token limit check as AgentV3.

Point the dashboard to AgentV4:

```bash
NEXT_PUBLIC_AGENT_URL=http://localhost:8094
```

The compatibility run API streams AgentV3-style SSE events: `run_started`, `tool_call`, `tool_result`, `assistant_message`, `run_completed`, `run_failed`, and `close`.

Current compatibility scope:

- Creates and compiles Botmother flow JSON via AgentV4.
- Enforces auth and monthly AI token limits through the Django backend.
- Returns the compiled flow in `tool_result.data.flow`.
- Keeps run state in memory.
- Does not yet persist flow to backend project releases automatically.
