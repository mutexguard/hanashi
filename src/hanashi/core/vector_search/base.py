from collections.abc import Iterable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T_Document = TypeVar("T_Document", bound=BaseModel)


class ScoredDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc: Any
    score: float | None = None


class VectorSearch(Generic[T_Document]):
    async def retrieve_documents(
        self,
        query: str,
        limit: int = 10,
        filters: list[dict] | None = None,
        model: type[T_Document] | None = None,
        **kwargs,
    ) -> list[ScoredDocument]:
        raise NotImplementedError

    async def batch_retrieve_documents(
        self,
        queries: list[str],
        limit: int = 10,
        filters: list[dict] | list[list[dict]] | None = None,
        model: type[T_Document] | None = None,
        **kwargs,
    ) -> list[list[ScoredDocument]]:
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
    results: Iterable[ScoredDocument],
    score_threshold: float,
    *,
    require_score: bool = True,
    keep_score: bool = False,
) -> Iterable:
    for result in results:
        if (
            not require_score and (not result.score or result.score >= score_threshold)
        ) or (require_score and (result.score and result.score >= score_threshold)):
            if keep_score:
                yield result
            else:
                yield result.doc
