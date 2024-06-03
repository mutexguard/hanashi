import structlog
from pydantic import BaseModel

from hanashi.core.llm import LLM
from hanashi.core.llm.utils import extract_json
from hanashi.types.conversation import Conversation, Role

logger = structlog.get_logger()


class Entity(BaseModel):
    type: str
    name: str


class Extractor:
    def __init__(self, llm: LLM, prompt_template: str) -> None:
        self.llm = llm
        self.prompt_template = prompt_template

    def _format_prompt(self, conversation: Conversation) -> str:
        return self.prompt_template.format(text=conversation.last().content)

    async def _run_llm(self, conversation: Conversation, **kwargs) -> str:
        # Format prompt
        content = self._format_prompt(conversation)
        logger.debug("Formatted extraction prompt", content=content)

        # Response
        task = Conversation()
        task.add(role=Role.User, content=content)
        return await self.llm.response(task, **kwargs)

    def _post_process(self, response: str) -> list[Entity]:
        logger.debug("LLM response for extraction", response=response)
        json_data = extract_json(response)
        if not json_data:
            return []

        entities = []
        for entity_type, entity_names in json_data.items():
            if isinstance(entity_names, str):
                entity_names = [entity_names]

            for entity_name in entity_names:
                if entity_name := entity_name.strip():
                    entities += [Entity(type=entity_type, name=entity_name)]

        logger.info("LLM Extracted entities", entities=entities)

        return entities

    async def run(self, *, conversation: Conversation, **kwargs) -> list[Entity]:
        response = await self._run_llm(conversation, **kwargs)

        return self._post_process(response)
