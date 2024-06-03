import os
from typing import Any, TypeVar, cast

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from hanashi.core.embedding import Embedding
from hanashi.core.vector_search.base import Document, ScoredDocument, VectorSearch
from hanashi.utils.logging import log_time

logger = structlog.get_logger()


T_Document = TypeVar("T_Document", bound=Document)


def create_filters(filters) -> models.Filter | None:
    # FIXME Support more complex filters

    if not filters:
        return None

    def _create_field_condition(f):
        if "range" in f:
            return models.FieldCondition(
                key=f["key"],
                range=models.Range(**f["range"]),
            )

        if "match" in f:
            if "text" in f["match"]:
                return models.FieldCondition(
                    key=f["key"],
                    match=models.MatchText(**f["match"]),
                )

            if "any" in f["match"]:
                return models.FieldCondition(
                    key=f["key"],
                    match=models.MatchAny(**f["match"]),
                )

            return models.FieldCondition(
                key=f["key"],
                match=models.MatchValue(**f["match"]),
            )

        if "in" in f:
            return models.FieldCondition(
                key=f["key"],
                match=models.MatchAny(**f["in"]),
            )

        raise NotImplementedError

    must_filters = []
    must_not_filters = []
    should_filters = []
    for f in filters:
        type_ = f.get("type")
        if not type_:
            must_filters.append(f)
            continue

        f.pop("type")
        if type_ == "must_not":
            must_not_filters.append(f)
        elif type_ == "should":
            should_filters.append(f)

    return models.Filter(
        must=[_create_field_condition(f) for f in must_filters],
        must_not=[_create_field_condition(f) for f in must_not_filters],
        should=[_create_field_condition(f) for f in should_filters],
    )


class Qdrant(VectorSearch):
    def __init__(  # noqa: PLR0913
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        collection: str,
        vector_name: str | None = None,
        embedding: Embedding,
        timeout: int = 10,
    ) -> None:
        if not base_url:
            base_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        self.qdrant = AsyncQdrantClient(
            base_url,
            timeout=timeout,
            api_key=api_key,
        )
        self.collection = collection
        self.vector_name = vector_name
        self.embedding = embedding

    @log_time(
        args=[
            "query",
            "limit",
            "filters",
        ],
    )
    async def retrieve_documents(
        self,
        query: str,
        limit: int = 10,
        filters: list[dict] | None = None,
        model: type[T_Document] | None = None,
        **_kwargs,
    ) -> list[ScoredDocument]:
        query_vector = await self.embedding.embed_text(query)

        if not self.vector_name:
            query_vector_params: list[float] | tuple[str, list[float]] = query_vector
        else:
            query_vector_params = (self.vector_name, query_vector)

        query_filter = create_filters(filters)
        logger.info(
            "Parsed search filters",
            query_filter=query_filter.dict() if query_filter else None,
        )

        docs = await self.qdrant.search(
            collection_name=f"{self.collection}",
            query_vector=query_vector_params,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        logger.info(
            "Retrieved documents",
            query=query,
            num_documents=len(docs),
            ids=[
                {
                    "id": cast(models.Payload, x.payload)["id"],
                    "score": x.score,
                }
                for x in docs
            ],
        )

        return [
            ScoredDocument(
                doc=model.parse_obj(x.payload) if model else x.payload,
                score=x.score,
            )
            for x in docs
        ]

    @log_time(
        args=[
            "queries",
            "limit",
            "filters",
        ],
    )
    async def batch_retrieve_documents(
        self,
        queries: list[str],
        limit: int = 10,
        filters: list[dict] | list[list[dict]] | None = None,
        model: type[T_Document] | None = None,
        **_kwargs,
    ) -> list[list[ScoredDocument]]:
        if not filters:
            query_filters = [None] * len(queries)
        elif isinstance(filters[0], dict):
            query_filters_for_query = create_filters(filters)
            query_filters = [query_filters_for_query for _ in range(len(queries))]
        else:
            query_filters = list(map(create_filters, filters))

        logger.info(
            "Parsed batch search filters",
            query_filters=[x.dict() if x is not None else x for x in query_filters],
            collection_name=self.collection,
        )

        query_vectors = await self.embedding.embed_texts(queries)
        search_queries = [
            models.SearchRequest(
                vector=(
                    vector
                    if not self.vector_name
                    else models.NamedVector(name=self.vector_name, vector=vector)
                ),
                filter=filters,
                limit=limit,
                with_payload=True,
            )
            for vector, filters in zip(query_vectors, query_filters, strict=True)
        ]
        batch_docs = await self.qdrant.search_batch(
            collection_name=f"{self.collection}",
            requests=search_queries,
        )
        for query, docs in zip(queries, batch_docs, strict=True):
            logger.info(
                "Retrieved documents",
                query=query,
                num_documents=len(docs),
                ids=[
                    {
                        "id": cast(models.Payload, x.payload)["id"],
                        "score": x.score,
                    }
                    for x in docs
                ],
            )

        return [
            [
                ScoredDocument(
                    doc=model.parse_obj(x.payload) if model else x.payload,
                    score=x.score,
                )
                for x in docs
            ]
            for docs in batch_docs
        ]

    @log_time(
        task="qdrant_retriever.list_documents",
        args=[
            "limit",
            "filters",
        ],
    )
    async def list_documents(
        self,
        limit: int,
        filters: list[dict] | None = None,
        model: type[T_Document] | None = None,
        **kwargs,  # noqa: ARG002
    ) -> list[Any]:
        query_filter = create_filters(filters)
        logger.info(
            "Parsed listing filters",
            query_filter=query_filter.model_dump() if query_filter else None,
        )

        docs, _ = await self.qdrant.scroll(
            collection_name=self.collection,
            limit=limit,
            offset=0,
            with_payload=True,
            with_vectors=False,
            scroll_filter=query_filter,
        )

        return [model.parse_obj(x.payload) if model else x.payload for x in docs]
