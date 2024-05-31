import copy
import enum
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hanashi.types.utils import uuid


class Role(str, enum.Enum):
    System = "system"
    User = "user"
    Assistant = "assistant"


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=uuid)
    role: Role
    content: str
    metadata: dict = Field(default_factory=dict)

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)


class Conversation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=uuid)
    messages: list[Message] = Field(default_factory=list)

    def add(
        self,
        *,
        id: str | None = None,
        role: str | Role,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not id:
            id = uuid()

        if isinstance(role, str):
            role = Role(role)

        if not metadata:
            metadata = {}

        message = Message(id=id, role=role, content=content, metadata=metadata)
        self.messages.append(message)

    def get_system_message(self, *, which: Literal["first", "last"]) -> Message | None:
        messages = self.messages if which == "first" else reversed(self.messages)

        return next((x for x in messages if x.role == Role.System), None)

    def update_system_message(self, message: Message) -> None:
        messages = [message]
        for message in self.messages:
            if message.role != Role.System:
                messages += [message]

        self.messages = messages

    def insert(self, index: int, message: Message) -> None:
        self.messages.insert(index, message)

    def pop(self, index: int = -1) -> Message:
        return self.messages.pop(index)

    def last(self) -> Message:
        return self.messages[-1]

    def history(self) -> Iterable[Message]:
        yield from iter(self.messages[:-1])

    def clone(self) -> "Conversation":
        return copy.deepcopy(self)

    def new(
        self,
        *,
        with_system_message: Literal["first", "last"] | None = None,
    ) -> "Conversation":
        conversation = Conversation()

        if with_system_message and (
            system_message := self.get_system_message(which=with_system_message)
        ):
            conversation.update_system_message(system_message)

        return conversation

    def format(  # noqa: PLR0913
        self,
        *,
        include_system_message: bool = False,
        include_last: bool = True,
        message_template="[{role}]: {content} ",
        newline: str = "\n\n",
        limit: int | None = None,
    ) -> str:
        messages = self.messages
        if not include_system_message:
            messages = [x for x in messages if x.role != Role.System]

        if not include_last:
            messages = messages[:-1]

        if limit is not None:
            messages = messages[-limit:]

        return newline.join(
            message_template.format(role=x.role, content=x.content) for x in messages
        )
