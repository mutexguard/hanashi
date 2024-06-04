import structlog

from hanashi.core.llm import LLM
from hanashi.core.llm.utils import extract_json
from hanashi.types.conversation import Conversation, Message, Role

logger = structlog.get_logger()


class Rephraser:
    def __init__(self, *, llm: LLM, prompt_template: str) -> None:
        self.llm = llm
        self.prompt_template = prompt_template

    def _format_template(
        self,
        conversation: Conversation,
        num_questions: int = 1,
    ) -> str:
        return self.prompt_template.format(
            chat_history=conversation.format(include_last=False, limit=5),
            question=conversation.last().content,
            num_questions=num_questions,
        )

    async def run(
        self,
        *,
        conversation: Conversation,
        num_questions: int = 1,
        **kwargs,
    ) -> list[str]:
        content = self._format_template(conversation, num_questions=num_questions)
        logger.debug("Formatted rephrasing template", content=content)

        task = Conversation()
        task.add(Message(role=Role.User, content=content))

        response = await self.llm.response(task, **kwargs)
        logger.debug("LLM response for rephrasing", response=response)

        rephrased_questions = extract_json(response)
        logger.info("Rephrased questions", rephrased_questions=rephrased_questions)

        return rephrased_questions
