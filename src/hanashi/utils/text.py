from typing import cast

import tiktoken


def count_approximate_tokens(
    text: str,
    encoding: str | None = "cl100k_base",
    model: str | None = None,
) -> int:
    if encoding is not None and model is not None:
        msg = "Only one of 'encoding' and 'model' should be provided."
        raise ValueError(msg)

    if encoding is None and model is None:
        msg = "One of 'encoding' and 'model' should be provided."
        raise ValueError(msg)

    if encoding is not None:
        enc = tiktoken.get_encoding(encoding)

    else:
        enc = tiktoken.encoding_for_model(cast(str, model))

    return len(enc.encode(text))
