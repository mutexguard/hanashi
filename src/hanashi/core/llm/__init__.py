from collections.abc import AsyncGenerator

import structlog
from llm_taxi.conversation import Message, Role
from llm_taxi.factory import llm

from hanashi.types import Conversation
from hanashi.utils import log_time

logger = structlog.get_logger()


def to_llm_taxi_messages(conversation: Conversation) -> list[Message]:
    return [
        Message(
            role=Role(message.role),
            content=message.content,
        )
        for message in conversation.messages
    ]


class LLM:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        call_kwargs: dict | None = None,
        **client_kwargs,
    ) -> None:
        self.model = model
        self.client = llm(
            model=model,
            api_key=api_key,
            base_url=base_url,
            call_kwargs=call_kwargs,
            **client_kwargs,
        )

    @log_time(
        args=lambda args: {
            "model": args["self"].model,
        },
    )
    async def streaming_response(
        self,
        conversation: Conversation,
        **kwargs,
    ) -> AsyncGenerator:
        messages = to_llm_taxi_messages(conversation)
        logger.debug(
            "LLM streaming response",
            model=self.model,
            messages=messages,
            **kwargs,
        )

        return await self.client.streaming_response(messages, **kwargs)

    @log_time(
        args=lambda args: {
            "model": args["self"].model,
        },
    )
    async def response(self, conversation: Conversation, **kwargs) -> str:
        messages = to_llm_taxi_messages(conversation)
        logger.debug("LLM response", model=self.model, messages=messages, **kwargs)

        return await self.client.response(messages, **kwargs)
