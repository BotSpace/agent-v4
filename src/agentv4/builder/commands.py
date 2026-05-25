from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import AnswerType, BotSpec, Button, ButtonMode, Condition, EdgeSpec, Step, StepKey


class CreateBotCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["create_bot"]
    name: str = Field(min_length=1, max_length=120)
    language: str = Field(default="uz", min_length=2, max_length=12)


class AddTextCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["add_text"]
    key: StepKey
    text: str = Field(min_length=1, max_length=4096)
    after: StepKey | None = None


class AddQuestionCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["add_question"]
    key: StepKey
    text: str = Field(min_length=1, max_length=4096)
    variable: StepKey
    answer_type: AnswerType = AnswerType.TEXT
    after: StepKey | None = None


class AddButtonsCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["add_buttons"]
    key: StepKey
    text: str = Field(min_length=1, max_length=4096)
    buttons: list[Button] = Field(min_length=1, max_length=20)
    mode: ButtonMode = ButtonMode.INLINE
    after: StepKey | None = None


class AddConditionCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["add_condition"]
    key: StepKey
    conditions: list[Condition] = Field(min_length=1, max_length=10)
    after: StepKey | None = None


class AddFinishCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["add_finish"]
    key: StepKey
    text: str = Field(min_length=1, max_length=4096)
    after: StepKey | None = None


class AddEngineNodeCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["add_engine_node"]
    key: StepKey
    node_type: str = Field(min_length=3, max_length=80)
    data: dict[str, Any] = Field(default_factory=dict)
    after: StepKey | None = None
    width: int = Field(default=400, ge=120, le=1200)
    height: int = Field(default=300, ge=120, le=1200)


class ConnectCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["connect"]
    source: StepKey
    target: StepKey
    on: str | None = Field(default=None, max_length=128)


BotCommand = Annotated[
    CreateBotCommand
    | AddTextCommand
    | AddQuestionCommand
    | AddButtonsCommand
    | AddConditionCommand
    | AddFinishCommand
    | AddEngineNodeCommand
    | ConnectCommand,
    Field(discriminator="action"),
]


def apply_command(spec: BotSpec, command: BotCommand) -> BotSpec:
    spec = spec.model_copy(deep=True)

    match command.action:
        case "create_bot":
            spec.name = command.name
            spec.language = command.language
            if "start" not in spec.steps:
                spec.steps["start"] = Step(key="start", kind="start")
            spec.entry = spec.entry or "start"
        case "add_text":
            add_step(spec, Step(key=command.key, kind="text", text=command.text), command.after)
        case "add_question":
            add_step(
                spec,
                Step(
                    key=command.key,
                    kind="question",
                    text=command.text,
                    variable=command.variable,
                    answer_type=command.answer_type,
                ),
                command.after,
            )
        case "add_buttons":
            add_step(
                spec,
                Step(
                    key=command.key,
                    kind="buttons",
                    text=command.text,
                    buttons=command.buttons,
                    mode=command.mode,
                ),
                command.after,
            )
        case "add_condition":
            add_step(
                spec, Step(key=command.key, kind="condition", conditions=command.conditions), command.after
            )
        case "add_finish":
            add_step(spec, Step(key=command.key, kind="finish", text=command.text), command.after)
        case "add_engine_node":
            add_step(
                spec,
                Step(
                    key=command.key,
                    kind="engine",
                    node_type=command.node_type,
                    data=command.data,
                    width=command.width,
                    height=command.height,
                ),
                command.after,
            )
        case "connect":
            spec.edges.append(EdgeSpec(source=command.source, target=command.target, on=command.on))

    return spec


def add_step(spec: BotSpec, step: Step, after: StepKey | None) -> None:
    existing_step_keys = set(spec.steps)
    spec.steps[step.key] = step
    if spec.entry is None:
        spec.entry = "start"
    if after is None and spec.entry == "start" and existing_step_keys == {"start"}:
        after = "start"
    if after is not None:
        after_step = spec.steps.get(after)
        if after_step is None:
            return
        if after_step.kind == "condition":
            return
        spec.edges.append(EdgeSpec(source=after, target=step.key))
