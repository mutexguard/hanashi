import itertools
from collections.abc import Callable
from typing import Generic, cast

import structlog
from pydantic import BaseModel

from hanashi.core.vector_search import ScoredDocument, filter_search_results
from hanashi.core.vector_search.base import VectorSearch
from hanashi.core.vector_search.qdrant import T_Document
from hanashi.services.rag.base import BaseRetriever
from hanashi.types import Conversation
from hanashi.utils.text import count_approximate_tokens

logger = structlog.get_logger()


def merge_docuemnt_splits(docs: list[T_Document]) -> list[T_Document]:
    docs = sorted(docs, key=lambda x: (x.id is None, x.source_id))
    groups = itertools.groupby(docs, key=lambda x: x.source_id)

    merged_docs = []
    for _key, group in groups:
        group_docs: list[T_Document] = []
        for doc in group:
            if doc.index == 0:
                group_docs = [doc]
                break

            group_docs += [doc]

        merged_docs += group_docs

    return merged_docs


def post_process_documents(
    scored_docs: list[ScoredDocument[T_Document]],
    *,
    score_threshold: float,
    max_length_per_doc: int | None = None,
    merge_splits: bool = True,
    sort_by: Callable | None = None,
) -> list[T_Document]:
    docs = cast(
        list[T_Document],
        list(
            filter_search_results(
                scored_docs,
                score_threshold=score_threshold,
                require_score=False,
            ),
        ),
    )
    logger.info(
        "Filter documents by score",
        score_threshold=score_threshold,
        num_remaining_documents=len(docs),
    )

    docs = list({x.id: x for x in docs}.values())
    logger.info("Deduplicated documents", num_unique_documents=len(docs))

    if max_length_per_doc:
        docs = [
            x for x in docs if count_approximate_tokens(x.content) < max_length_per_doc
        ]
        logger.info(
            "Filter documents by max length",
            max_length_per_doc=max_length_per_doc,
            num_remaining_documents=len(docs),
        )

    if merge_splits:
        docs = merge_docuemnt_splits(docs)
        logger.info("Merge documents by", num_remaining_documents=len(docs))

    if sort_by:
        docs.sort(key=sort_by)

    return docs


class RetrieveParams(BaseModel, Generic[T_Document]):
    top_k: int
    score_threshold: float
    filters: list[dict] | None = None
    model: type[T_Document] | None = None


class VectorSearchRetriever(BaseRetriever, Generic[T_Document]):
    def __init__(
        self,
        vector_search: VectorSearch,
        filters: list[dict] | None = None,
        max_length_per_doc: int | None = None,
        merge_splits: bool = True,
        sort_by: Callable | None = None,
    ) -> None:
        self.vector_search = vector_search

        if not filters:
            filters = []
        self.filters = filters

        self.max_length_per_doc = max_length_per_doc
        self.merge_splits = merge_splits
        self.sort_by = sort_by

    async def retrieve(
        self,
        *,
        conversation: Conversation,
        params: RetrieveParams,
    ) -> list[T_Document]:
        filters = params.filters if params.filters else []
        filters += self.filters

        # Retrieve related documents
        query = conversation.last().content
        docs = await self.vector_search.retrieve_documents(
            query,
            params.top_k,
            filters=filters,
            model=params.model,
        )
        logger.info(
            "Retrieved documents",
            query=query,
            num_documents=len(docs),
        )

        # Post processing
        docs = post_process_documents(
            docs,
            score_threshold=params.score_threshold,
            max_length_per_doc=self.max_length_per_doc,
            merge_splits=self.merge_splits,
            sort_by=self.sort_by,
        )
        logger.info(
            "Post processing of documents, will only keep top k documents",
            num_remaining_documents=len(docs),
            top_k=params.top_k,
        )
        return docs[: params.top_k]
