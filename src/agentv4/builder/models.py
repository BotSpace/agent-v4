from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

StepKey = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$")]


class AnswerType(StrEnum):
    TEXT = "text"
    PHONE = "phone"
    NUMBER = "number"
    LOCATION = "location"
    FILE = "file"


class ButtonMode(StrEnum):
    INLINE = "inline"
    REPLY = "reply"


class Button(BaseModel):
    text: str = Field(min_length=1, max_length=64)
    value: str | None = Field(default=None, max_length=128)
    url: str | None = Field(default=None, max_length=2048)

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        return value.strip()


class Condition(BaseModel):
    variable: str = Field(min_length=1, max_length=128)
    operator: Literal[
        "equals",
        "not_equals",
        "contains",
        "not_contains",
        "starts_with",
        "ends_with",
        "not_empty",
        "is_empty",
        "greater_than",
        "greater_than_or_equal",
        ">=",
        "less_than",
        "less_than_or_equal",
        "<=",
    ]
    value: str | int | float | bool | None = None


class BotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="Untitled bot", min_length=1, max_length=120)
    language: str = Field(default="uz", min_length=2, max_length=12)
    entry: StepKey | None = None
    steps: dict[str, Step] = Field(default_factory=dict)
    edges: list[EdgeSpec] = Field(default_factory=list)


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: StepKey
    kind: Literal["start", "text", "question", "buttons", "condition", "finish", "engine"]
    text: str | None = Field(default=None, max_length=4096)
    variable: StepKey | None = None
    answer_type: AnswerType | None = None
    mode: ButtonMode | None = None
    buttons: list[Button] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    node_type: str | None = Field(default=None, max_length=80)
    data: dict[str, Any] = Field(default_factory=dict)
    width: int = Field(default=400, ge=120, le=1200)
    height: int = Field(default=300, ge=120, le=1200)


class EdgeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: StepKey
    target: StepKey
    on: str | None = Field(default=None, max_length=128)


class FlowNode(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    data: dict[str, Any] = Field(default_factory=dict)
    position: dict[str, float] = Field(default_factory=dict)
    width: int = 400
    height: int = 240
    targetPosition: str = "left"
    sourcePosition: str = "right"


class FlowEdge(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    source: str
    target: str
    sourceHandle: str = "source-handler"
    targetHandle: str | None = "target-handler"
    type: str = "floating"
    markerEnd: dict[str, Any] = Field(
        default_factory=lambda: {"type": "arrow", "width": 30, "height": 30, "color": "#495057"}
    )
    style: dict[str, Any] = Field(default_factory=lambda: {"strokeWidth": 4, "stroke": "#495057"})
    animated: bool = False


class FlowDocument(BaseModel):
    nodes: list[FlowNode]
    edges: list[FlowEdge]


class ValidationIssue(BaseModel):
    code: str
    message: str
    field: str | None = None


class CompileResult(BaseModel):
    ok: bool
    spec: BotSpec
    flow: FlowDocument | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)
