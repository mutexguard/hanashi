from collections.abc import AsyncGenerator
from typing import cast

import structlog

from hanashi.core.llm import LLM as _LLM
from hanashi.types.conversation import Conversation, Message, Role

logger = structlog.get_logger()


class LLM:
    def __init__(
        self,
        *,
        llm: _LLM,
        system_prompt: str | None = None,
        stream: bool = True,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.stream = stream

    async def run(
        self,
        *,
        conversation: Conversation,
        prompt_template: str | None = None,
        **kwargs,
    ) -> str | AsyncGenerator:
        if not prompt_template:
            prompt_template = self.system_prompt

        if prompt_template:
            conversation = conversation.clone()
            conversation.update_system_message(
                Message(role=Role.System, content=prompt_template),
            )

        if self.stream:
            return cast(
                AsyncGenerator,
                await self.llm.streaming_response(conversation, **kwargs),
            )

        return cast(str, await self.llm.response(conversation, **kwargs))
