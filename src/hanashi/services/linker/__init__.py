import asyncio
from collections.abc import Callable
from typing import Generic, TypeVar, cast

import structlog
from pydantic import BaseModel

from hanashi.core.llm import LLM
from hanashi.core.vector_search import filter_search_results
from hanashi.core.vector_search.base import ScoredDocument, VectorSearch
from hanashi.services.extractor import Entity
from hanashi.types import Conversation
from hanashi.types.conversation import Role

logger = structlog.get_logger()


class LinkedEntity(Entity):
    metadata: dict


class LinkerResponse(BaseModel):
    linked_entities: list[LinkedEntity]
    unlinked_entities: list[Entity]


T_LinkedEntity = TypeVar("T_LinkedEntity", bound=LinkedEntity)
CandidatePostProcessFunction = Callable[
    [Entity, list[T_LinkedEntity]],
    list[T_LinkedEntity],
]


class Linker(Generic[T_LinkedEntity]):
    def __init__(
        self,
        *,
        vector_search: VectorSearch[T_LinkedEntity],
        llm: LLM,
        prompt_template: str,
        cross_search_types: dict[str, list[str]] | None = None,
        candidate_format_function: Callable[[Entity], str] | None = None,
        candidate_postprocess_fns: (
            dict[str, CandidatePostProcessFunction] | None
        ) = None,
        skip_llm_check_confidence: float | None = None,
    ) -> None:
        self.vector_search = vector_search
        self.llm = llm
        self.prompt_template = prompt_template
        if not cross_search_types:
            cross_search_types = {}
        self.cross_search_types = cross_search_types
        self.candiate_format_function = candidate_format_function
        if not candidate_postprocess_fns:
            candidate_postprocess_fns = {}
        self.candidate_postprocess_fns = candidate_postprocess_fns
        self.skip_llm_check_confidence = skip_llm_check_confidence

    async def _retrieve_candidates(
        self,
        _query: str,
        entities: list[Entity],
        top_k: int,
        score_threshold: float,
    ) -> tuple[list[dict], list[Entity], list[LinkedEntity]]:
        # TODO: Integrate with _query information
        # TODO: Handle non-string entities correctly.
        unlinked_entities: list[Entity] = []
        to_link_entities: list[Entity] = []
        for entity in entities:
            if isinstance(entity.name, str):
                to_link_entities += [entity]
            else:
                unlinked_entities += [entity]

        if not to_link_entities:
            return [], unlinked_entities, []

        filters = [
            [  # Filter points using entity `type` information.
                {
                    "key": "type",
                    "in": {
                        "any": [
                            entity.type,
                            *self.cross_search_types.get(entity.type, []),
                        ],
                    },
                },
            ]
            for entity in to_link_entities
        ]
        batch_response = await self.vector_search.batch_retrieve_documents(
            [x.name for x in to_link_entities],
            limit=top_k,
            filters=filters,
        )
        batch_docs = [
            list(
                cast(
                    list[ScoredDocument[T_LinkedEntity]],
                    filter_search_results(
                        docs,
                        score_threshold=score_threshold,
                        require_score=True,
                        keep_score=True,
                    ),
                ),
            )
            for docs in batch_response
        ]

        linked_entities: list[LinkedEntity] = []
        entity_with_candidates: list[dict] = []
        for entity, docs_with_scores in zip(to_link_entities, batch_docs, strict=True):
            docs = [x.document for x in docs_with_scores]
            if process_fn := self.candidate_postprocess_fns.get(entity.type):
                processed_docs = process_fn(entity, docs)
                logger.info(
                    "Post processing linking candidates",
                    type=entity.type,
                    name=entity.name,
                    before=docs,
                    after=processed_docs,
                )
                docs = processed_docs

            # That means if a entity with no candidates is ignored here.
            if docs:
                if self.skip_llm_check_confidence:
                    skip_check_docs = [
                        x
                        for x in docs_with_scores
                        if x.score > self.skip_llm_check_confidence
                    ]
                    if len(skip_check_docs) == 1:
                        link_entity = LinkedEntity(
                            type=entity.type,
                            name=entity.name,
                            metadata=cast(dict, skip_check_docs[0].document),
                        )
                        linked_entities += [link_entity]
                        logger.info(
                            "Link entity and skip LLM check",
                            skip_check_confidence=self.skip_llm_check_confidence,
                            score=skip_check_docs[0].score,
                            entity=entity,
                            link_entity=link_entity,
                        )
                        continue

                entity_with_candidates += [
                    {
                        "entity": entity,
                        # NOTE: `name` key is required
                        "candidates": docs,
                    },
                ]

            else:
                logger.warning(
                    "No linking candidates found",
                    type=entity.type,
                    name=entity.name,
                )
                unlinked_entities += [entity]

        return entity_with_candidates, unlinked_entities, linked_entities

    async def _link_by_llm(
        self,
        query: str,
        entities_with_candidates: list[dict],
        **kwargs,
    ) -> tuple[list[LinkedEntity], list[Entity]]:
        # Make request for every entity
        futures = []
        for entity_with_candidates in entities_with_candidates:
            content = self.prompt_template.format(
                entity_type=entity_with_candidates["entity"].type,
                entity_name=entity_with_candidates["entity"].name,
                text=query,
                normalized_entities="\n".join(
                    f"{i + 1}. {self.candiate_format_function(candidate) if self.candiate_format_function else candidate['name']}"
                    for i, candidate in enumerate(entity_with_candidates["candidates"])
                ),
            )

            logger.debug("Formatted prompt", content=content)

            task = Conversation()
            task.add(role=Role.User, content=content)

            futures += [self.llm.response(task, **kwargs)]

        # Gather results concurrently
        responses = await asyncio.gather(*futures)

        # Verify generated results from LLM
        linked_entities: list[LinkedEntity] = []
        unlinked_entities: list[Entity] = []
        for request_index, (entity_with_candidates, response) in enumerate(
            zip(
                entities_with_candidates,
                responses,
                strict=True,
            ),
        ):
            entity = entity_with_candidates["entity"]

            logger.info(
                "LLM response for linking",
                candidates=entity_with_candidates["candidates"],
                response=response,
                request_index=request_index,
            )

            try:
                index = int(response) - 1
            except ValueError:
                logger.warning(
                    "LLM generated unparsable response",
                    entity_name=entity.name,
                    response=response,
                    request_index=request_index,
                )
                continue

            # LLM generates unexpected contents
            if index > len(entity_with_candidates["candidates"]):
                logger.warning(
                    "LLM generated unexpected candidate index",
                    index=index,
                    entity_name=entity.name,
                    request_index=request_index,
                )
                unlinked_entities += [entity]
                continue

            if index < 0:
                logger.info(
                    "No suitable candidate found for linking",
                    index=index,
                    entity_name=entity.name,
                    request_index=request_index,
                )
                unlinked_entities += [entity]
                continue

            normailzed_entity = entity_with_candidates["candidates"][index]
            linked_entities += [
                LinkedEntity(
                    type=entity.type,
                    name=entity.name,
                    metadata=normailzed_entity,
                ),
            ]
            logger.info(
                "LLM generate normalized entity",
                normailzed_entity=normailzed_entity,
                entity_name=entity.name,
                request_index=request_index,
            )

        return linked_entities, unlinked_entities

    async def run(
        self,
        *,
        conversation: Conversation,
        entities: list[Entity],
        top_k: int,
        score_threshold: float,
        **kwargs,
    ) -> LinkerResponse:
        if not entities:
            return LinkerResponse(linked_entities=[], unlinked_entities=[])

        question = conversation.last().content
        (
            entity_with_candidates,
            unlinked_entities,
            linked_entities,
        ) = await self._retrieve_candidates(
            question,
            entities,
            top_k,
            score_threshold,
        )
        if not entity_with_candidates:
            logger.info(
                "No entities to link",
                entities=entities,
                unlinked_entities=unlinked_entities,
                linked_entities=linked_entities,
            )
            return LinkerResponse(
                linked_entities=linked_entities,
                unlinked_entities=unlinked_entities,
            )

        # TODO Add chat history information
        linked_entities_by_llm, unlinked_entities_by_llm = await self._link_by_llm(
            question,
            entity_with_candidates,
            **kwargs,
        )
        logger.info(
            "Linked entities",
            linked_entities=linked_entities,
            linked_entities_by_llm=linked_entities_by_llm,
            unlinked_entities_by_llm=unlinked_entities_by_llm,
        )

        return LinkerResponse(
            linked_entities=linked_entities + linked_entities_by_llm,
            unlinked_entities=unlinked_entities + unlinked_entities_by_llm,
        )
