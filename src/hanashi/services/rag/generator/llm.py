from collections.abc import AsyncGenerator
from typing import Generic

from pydantic import BaseModel

from hanashi.core.llm import LLM
from hanashi.services.rag.base import (
    BaseFormatter,
    BaseGenerator,
    T_FormatParams,
    T_Formattable,
)
from hanashi.types.conversation import Conversation, Role


class GenerateParams(BaseModel):
    stream: bool = True
    temperature: float = 0.1
    max_tokens: int = 4098


class LLMGenerator(
    BaseGenerator,
    Generic[T_Formattable, T_FormatParams],
):
    def __init__(
        self,
        llm: LLM,
        formatter: BaseFormatter[T_Formattable, T_FormatParams],
    ) -> None:
        self.llm = llm
        self.formatter = formatter

    def _format_prompt(
        self,
        conversation: Conversation,
        documents: T_Formattable,
        format_params: T_FormatParams | None = None,
    ) -> str:
        return self.formatter.format(
            conversation=conversation,
            documents=documents,
            params=format_params,
        )

    async def generate(
        self,
        *,
        conversation: Conversation,
        documents: T_Formattable,
        format_params: T_FormatParams | None = None,
        generate_params: GenerateParams | None = None,
    ) -> AsyncGenerator | str:
        content = self._format_prompt(conversation, documents)

        task_conversation = conversation.new()
        task_conversation.add(role=Role.User, content=content)

        call_kwargs = {}
        if generate_params:
            call_kwargs = {
                "temperature": generate_params.temperature,
                "max_tokens": generate_params.max_tokens,
            }
            stream = generate_params.stream
        else:
            stream = True

        if stream:
            return await self.llm.streaming_response(task_conversation, **call_kwargs)

        return await self.llm.response(task_conversation, **call_kwargs)
