from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any

from .models import BotSpec, ButtonMode, CompileResult, FlowDocument, FlowEdge, FlowNode, Step, ValidationIssue
from ..engine_catalog import (
    ENGINE_NODE_TYPES,
    engine_node_requires_branch,
    is_valid_engine_branch,
    validate_engine_node_data,
)


def compile_spec(spec: BotSpec) -> CompileResult:
    issues = validate_spec(spec)
    if issues:
        return CompileResult(ok=False, spec=spec, issues=issues)

    nodes: list[FlowNode] = []
    edges: list[FlowEdge] = []
    levels = layout_levels(spec)

    for key, step in spec.steps.items():
        x = levels.get(key, 0) * 560
        y = row_for_key(key, levels) * 260
        compiled_nodes, internal_edges = compile_step(step, x=x, y=y, spec=spec)
        nodes.extend(compiled_nodes)
        edges.extend(internal_edges)

    for edge in spec.edges:
        source_step = spec.steps[edge.source]
        target_step = spec.steps[edge.target]
        for source_handle in source_handles_for(source_step, edge.on):
            edges.append(
                FlowEdge(
                    id=edge_id(source_node_for_edge(source_step), target_node_for_edge(target_step), source_handle),
                    source=source_node_for_edge(source_step),
                    target=target_node_for_edge(target_step),
                    sourceHandle=source_handle,
                )
            )

    return CompileResult(ok=True, spec=spec, flow=FlowDocument(nodes=nodes, edges=edges))


def validate_spec(spec: BotSpec) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not spec.steps:
        issues.append(ValidationIssue(code="empty_bot", message="Bot has no steps.", field="steps"))
        return issues

    if spec.entry is None:
        issues.append(ValidationIssue(code="missing_entry", message="Bot entry step is missing.", field="entry"))
    elif spec.entry not in spec.steps:
        issues.append(
            ValidationIssue(code="unknown_entry", message=f"Entry step `{spec.entry}` does not exist.", field="entry")
        )

    step_keys = set(spec.steps)
    for key, step in spec.steps.items():
        if key != step.key:
            issues.append(ValidationIssue(code="step_key_mismatch", message=f"Step key mismatch for `{key}`."))
        if step.kind in {"text", "question", "buttons", "finish"} and not step.text:
            issues.append(ValidationIssue(code="missing_text", message=f"Step `{key}` requires text.", field=key))
        if step.kind == "question" and not step.variable:
            issues.append(ValidationIssue(code="missing_variable", message=f"Question `{key}` requires variable."))
        if step.kind == "buttons" and not step.buttons:
            issues.append(ValidationIssue(code="missing_buttons", message=f"Buttons step `{key}` requires buttons."))
        if step.kind == "condition" and not step.conditions:
            issues.append(ValidationIssue(code="missing_conditions", message=f"Condition `{key}` requires conditions."))
        if step.kind == "engine":
            if not step.node_type:
                issues.append(ValidationIssue(code="missing_node_type", message=f"Engine step `{key}` requires node_type."))
            elif step.node_type not in ENGINE_NODE_TYPES:
                issues.append(
                    ValidationIssue(
                        code="unknown_engine_node",
                        message=f"Engine node type `{step.node_type}` is not registered in engine-golang.",
                        field=key,
                    )
                )
            else:
                for message in validate_engine_node_data(step.node_type, step.data):
                    issues.append(ValidationIssue(code="invalid_engine_node_data", message=message, field=key))

    edge_counts = Counter((edge.source, edge.target, edge.on) for edge in spec.edges)
    for (source, target, on), count in edge_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    code="duplicate_edge",
                    message=f"Duplicate edge `{source}` -> `{target}` on `{on or 'default'}`.",
                )
            )

    for edge in spec.edges:
        if edge.source not in step_keys:
            issues.append(ValidationIssue(code="unknown_source", message=f"Edge source `{edge.source}` does not exist."))
            continue
        if edge.target not in step_keys:
            issues.append(ValidationIssue(code="unknown_target", message=f"Edge target `{edge.target}` does not exist."))
            continue
        source = spec.steps[edge.source]
        if source.kind == "condition" and not edge.on:
            issues.append(
                ValidationIssue(
                    code="missing_branch",
                    message=f"Edge from `{edge.source}` needs `on` because it is a branching step.",
                )
            )
        if source.kind == "buttons" and edge.on:
            allowed = {button.text for button in source.buttons}
            if edge.on not in allowed:
                issues.append(
                    ValidationIssue(
                        code="unknown_button",
                        message=f"Button branch `{edge.on}` is not defined on `{edge.source}`.",
                    )
                )
        if source.kind == "condition" and normalize_condition_branch(edge.on) not in {"true", "false"}:
            issues.append(
                ValidationIssue(
                    code="unknown_condition_branch",
                    message=f"Condition edge from `{edge.source}` must use `on=true` or `on=false`.",
                )
            )
        if source.kind == "engine" and source.node_type and engine_node_requires_branch(source.node_type) and edge.on is None:
            issues.append(
                ValidationIssue(
                    code="missing_branch",
                    message=f"Edge from `{edge.source}` needs `on` because `{source.node_type}` is branching.",
                )
            )
        elif source.kind == "engine" and source.node_type and not is_valid_engine_branch(source.node_type, edge.on):
            issues.append(
                ValidationIssue(
                    code="unknown_engine_branch",
                    message=f"Engine edge from `{edge.source}` uses unsupported branch `{edge.on}` for `{source.node_type}`.",
                )
            )

    issues.extend(reachability_issues(spec))
    return issues


def reachability_issues(spec: BotSpec) -> list[ValidationIssue]:
    if spec.entry is None or spec.entry not in spec.steps:
        return []

    graph: dict[str, list[str]] = defaultdict(list)
    for edge in spec.edges:
        if edge.source in spec.steps and edge.target in spec.steps:
            graph[edge.source].append(edge.target)

    seen = {spec.entry}
    queue = deque([spec.entry])
    while queue:
        current = queue.popleft()
        for target in graph[current]:
            if target not in seen:
                seen.add(target)
                queue.append(target)

    return [
        ValidationIssue(code="unreachable_step", message=f"Step `{key}` is not reachable from entry.")
        for key in sorted(set(spec.steps) - seen)
    ]


def compile_step(step: Step, *, x: int, y: int, spec: BotSpec) -> tuple[list[FlowNode], list[FlowEdge]]:
    position = {"x": float(x), "y": float(y)}
    if step.kind == "start":
        return (
            [
                FlowNode(
                    id=step.key,
                    type="CommandTriggerNode",
                    data={"command": "/start", "withArgs": False, "global": True},
                    position=position,
                    width=320,
                    height=220,
                )
            ],
            [],
        )
    if step.kind in {"text", "finish"}:
        return [message_node(step, position)], []
    if step.kind == "question":
        prompt_id = question_prompt_id(step.key)
        trigger_position = {"x": float(x + 360), "y": float(y)}
        return (
            [
                FlowNode(
                    id=prompt_id,
                    type="SendTextMessageNode",
                    data={"messageText": step.text, "parseMode": "HTML", "keyboard": {"active": "none"}},
                    position=position,
                    width=400,
                    height=300,
                ),
                FlowNode(
                    id=step.key,
                    type="MessageTriggerNode",
                    data={
                        "type": step.answer_type or "text",
                        "filter": "any",
                        "context": {"value": ""},
                        "state": {"key": step.variable, "type": step.answer_type or "text"},
                    },
                    position=trigger_position,
                    width=320,
                    height=420,
                ),
            ],
            [
                FlowEdge(
                    id=edge_id(prompt_id, step.key, "source-handler"),
                    source=prompt_id,
                    target=step.key,
                )
            ],
        )
    if step.kind == "buttons":
        return (
            [
                FlowNode(
                    id=step.key,
                    type="EditOrSendMessageNode",
                    data={
                        "messageText": step.text,
                        "parseMode": "HTML",
                        "keyboard": keyboard_data(step),
                    },
                    position=position,
                    width=400,
                    height=360,
                )
            ],
            [],
        )
    if step.kind == "condition":
        return (
            [
                FlowNode(
                    id=step.key,
                    type="IfConditionNode",
                    data={
                        "operator": "AND",
                        "conditions": [normalize_condition(condition.model_dump(), spec) for condition in step.conditions],
                    },
                    position=position,
                    width=420,
                    height=320,
                )
            ],
            [],
        )
    if step.kind == "engine":
        return (
            [
                FlowNode(
                    id=step.key,
                    type=step.node_type or "UnknownNode",
                    data=step.data,
                    position=position,
                    width=step.width,
                    height=step.height,
                )
            ],
            [],
        )
    raise ValueError(f"unsupported step kind: {step.kind}")


def message_node(step: Step, position: dict[str, float]) -> FlowNode:
    return FlowNode(
        id=step.key,
        type="SendTextMessageNode",
        data={"messageText": step.text, "parseMode": "HTML", "keyboard": {"active": "none"}},
        position=position,
        width=400,
        height=300,
    )


def keyboard_data(step: Step) -> dict[str, Any]:
    active = step.mode or ButtonMode.INLINE
    rows: list[list[dict[str, str]]] = []
    for button in step.buttons:
        if active == ButtonMode.INLINE:
            btn_type = "url" if button.url else "callback"
            rows.append([{"text": button.text, "type": btn_type, "value": button.url or button.value or button.text}])
        else:
            rows.append([{"text": button.text}])
    return {
        "active": active,
        "inline": rows if active == ButtonMode.INLINE else [],
        "reply": rows if active == ButtonMode.REPLY else [],
        "replyOptions": {
            "remove_after_use": True,
            "resize_keyboard": True,
            "one_time_keyboard": False,
            "is_persistent": False,
        },
    }


def source_handles_for(source: Step, branch: str | None) -> list[str]:
    if source.kind == "buttons":
        if branch:
            return [f"target-handler-{source.key}-{branch}"]
        return [f"target-handler-{source.key}-{button.text}" for button in source.buttons]
    if source.kind == "condition" and branch:
        return [normalize_condition_branch(branch) or branch]
    if source.kind == "engine":
        if source.node_type == "IfConditionNode" and branch:
            return [normalize_condition_branch(branch) or branch]
        if branch:
            return [branch]
    return ["source-handler"]


def normalize_condition_branch(branch: str | None) -> str | None:
    if branch is None:
        return None
    value = str(branch).strip().lower()
    true_aliases = {"true", "yes", "ha", "ok", "allowed", "allow", "adult", "valid", "ruxsat"}
    false_aliases = {"false", "no", "yoq", "yo'q", "denied", "deny", "minor", "invalid", "rad"}
    if value in true_aliases:
        return "true"
    if value in false_aliases:
        return "false"
    return branch


def question_prompt_id(step_key: str) -> str:
    return f"{step_key}_prompt"


def source_node_for_edge(step: Step) -> str:
    return step.key


def target_node_for_edge(step: Step) -> str:
    if step.kind == "question":
        return question_prompt_id(step.key)
    return step.key


def normalize_condition(condition: dict[str, Any], spec: BotSpec) -> dict[str, Any]:
    variable = condition.get("variable")
    if not isinstance(variable, str) or variable.startswith(("state_", "{{", "$")):
        return condition
    question_variables = {
        str(step.variable)
        for step in spec.steps.values()
        if step.kind == "question" and step.variable is not None
    }
    if variable in question_variables:
        condition = {**condition, "variable": f"state_{variable}"}
    return condition


def edge_id(source: str, target: str, source_handle: str) -> str:
    return f"reactflow__edge-{source}{source_handle}-{target}"


def layout_levels(spec: BotSpec) -> dict[str, int]:
    if spec.entry is None:
        return {key: idx for idx, key in enumerate(spec.steps)}
    graph: dict[str, list[str]] = defaultdict(list)
    for edge in spec.edges:
        graph[edge.source].append(edge.target)
    levels = {spec.entry: 0}
    queue = deque([spec.entry])
    while queue:
        current = queue.popleft()
        for target in graph[current]:
            next_level = levels[current] + 1
            if target not in levels or levels[target] < next_level:
                levels[target] = next_level
                queue.append(target)
    for key in spec.steps:
        levels.setdefault(key, 0)
    return levels


def row_for_key(key: str, levels: dict[str, int]) -> int:
    same_level = sorted(k for k, level in levels.items() if level == levels.get(key, 0))
    return same_level.index(key) if key in same_level else 0
