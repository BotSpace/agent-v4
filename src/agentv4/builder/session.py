from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from uuid import uuid4

from .commands import BotCommand, apply_command
from .compiler import compile_spec
from .models import BotSpec, CompileResult


@dataclass
class BuilderSession:
    id: str
    spec: BotSpec = field(default_factory=BotSpec)
    commands: list[BotCommand] = field(default_factory=list)

    def apply(self, command: BotCommand) -> CompileResult:
        self.spec = apply_command(self.spec, command)
        self.commands.append(command)
        return compile_spec(self.spec)


class SessionStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[str, BuilderSession] = {}

    def create(self) -> BuilderSession:
        with self._lock:
            session = BuilderSession(id=str(uuid4()))
            self._sessions[session.id] = session
            return session

    def get(self, session_id: str) -> BuilderSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


def build_from_commands(commands: list[BotCommand]) -> CompileResult:
    spec = BotSpec()
    for command in commands:
        spec = apply_command(spec, command)
    return compile_spec(spec)
