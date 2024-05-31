import structlog
from llm_taxi.factory import embedding

from hanashi.utils import log_time

logger = structlog.get_logger()


class Embedding:
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
        self.client = embedding(
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
    async def embed_text(self, text: str, **kwargs) -> list[float]:
        logger.debug("Embed text", text=text, model=self.model)

        return await self.client.embed_text(text=text, **kwargs)

    @log_time(
        args=lambda args: {
            "model": args["self"].model,
        },
    )
    async def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
        logger.debug("Embed texts", texts=texts, model=self.model)

        return await self.client.embed_texts(texts=texts, **kwargs)
