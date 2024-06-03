import asyncio
import re
from collections.abc import AsyncGenerator

import structlog

logger = structlog.get_logger()


async def oneshot_stream(
    s: str | list[str],
    delay: float | None = None,
) -> AsyncGenerator:
    if isinstance(s, list):
        for chunk in s:
            yield chunk

        if delay:
            await asyncio.sleep(delay)

    else:
        for chunk in re.split(r"(\W+)", s):
            yield chunk

        if delay:
            await asyncio.sleep(delay)


async def merge_streams(*streams, sep: str = "\n\n") -> AsyncGenerator:
    for stream in streams:
        async for chunk in stream:
            yield chunk

        yield sep
