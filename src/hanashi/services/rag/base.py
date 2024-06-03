import abc
from collections.abc import AsyncGenerator
from typing import Generic, TypeVar

from hanashi.types.conversation import Conversation

T_RetrieveParams = TypeVar("T_RetrieveParams")
T_Formattable = TypeVar("T_Formattable")
T_FormatParams = TypeVar("T_FormatParams")
T_GenerateParams = TypeVar("T_GenerateParams")


class BaseRetriever(
    Generic[
        T_RetrieveParams,
        T_Formattable,
    ],
    metaclass=abc.ABCMeta,
):
    @abc.abstractmethod
    async def retrieve(
        self,
        *,
        conversation: Conversation,
        params: T_RetrieveParams,
    ) -> T_Formattable:
        raise NotImplementedError


class BaseFormatter(
    Generic[
        T_Formattable,
        T_FormatParams,
    ],
    metaclass=abc.ABCMeta,
):
    @abc.abstractmethod
    def format(
        self,
        *,
        conversation: Conversation,
        documents: T_Formattable,
        params: T_FormatParams | None = None,
    ) -> str:
        raise NotImplementedError


class BaseGenerator(
    Generic[
        T_Formattable,
        T_FormatParams,
        T_GenerateParams,
    ],
    metaclass=abc.ABCMeta,
):
    @abc.abstractmethod
    async def generate(
        self,
        *,
        conversation: Conversation,
        documents: T_Formattable,
        format_params: T_FormatParams | None = None,
        generate_params: T_GenerateParams | None = None,
    ) -> AsyncGenerator | str:
        raise NotImplementedError


class RAG(
    Generic[
        T_RetrieveParams,
        T_FormatParams,
        T_GenerateParams,
    ],
):
    def __init__(
        self,
        *,
        retriever: BaseRetriever,
        generator: BaseGenerator,
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    async def run(
        self,
        *,
        conversation: Conversation,
        retrieve_params: T_RetrieveParams | None = None,
        format_params: T_FormatParams | None = None,
        generate_params: T_GenerateParams | None = None,
    ) -> AsyncGenerator | str:
        documents = await self.retriever.retrieve(
            conversation=conversation,
            params=retrieve_params,
        )

        return await self.generator.generate(
            conversation=conversation,
            documents=documents,
            format_params=format_params,
            generate_params=generate_params,
        )
