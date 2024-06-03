from collections.abc import Iterable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T_Document = TypeVar("T_Document", bound=BaseModel)


class Document(BaseModel):
    id: str
    source_id: str | None = None
    index: int | None = None
    content: str


class ScoredDocument(BaseModel, Generic[T_Document]):
    model_config = ConfigDict(extra="forbid")

    document: T_Document | Any
    score: float


class VectorSearch(Generic[T_Document]):
    async def retrieve_documents(
        self,
        query: str,
        limit: int = 10,
        filters: list[dict] | None = None,
        model: type[T_Document] | None = None,
        **kwargs,
    ) -> list[ScoredDocument[T_Document]]:
        raise NotImplementedError

    async def batch_retrieve_documents(
        self,
        queries: list[str],
        limit: int = 10,
        filters: list[dict] | list[list[dict]] | None = None,
        model: type[T_Document] | None = None,
        **kwargs,
    ) -> list[list[ScoredDocument[T_Document]]]:
        raise NotImplementedError

    async def list_documents(
        self,
        limit: int,
        filters: list[dict] | None = None,
        model: type[T_Document] | None = None,
        **kwargs,
    ) -> list[Any]:
        raise NotImplementedError


def filter_search_results(
    results: Iterable[ScoredDocument[T_Document]],
    *,
    score_threshold: float,
    require_score: bool = True,
    keep_score: bool = False,
) -> (
    Iterable[ScoredDocument[T_Document]]
    | Iterable[ScoredDocument[Any]]
    | Iterable[Document]
):
    for result in results:
        if (
            not require_score and (not result.score or result.score >= score_threshold)
        ) or (require_score and (result.score and result.score >= score_threshold)):
            if keep_score:
                yield result
            else:
                yield result.document
