import contextlib
import json
import re
from typing import Any

import structlog

logger = structlog.get_logger()


def extract_inline_json(s: str) -> Any:
    data = None

    if match := re.search(r"`(.+)`", s):
        s = match[1]

    with contextlib.suppress(json.decoder.JSONDecodeError):
        data = json.loads(s)

    return data


def extract_multi_line_json(s: str) -> Any:
    data = None

    if match := re.search(r"```(?:json)?(.+)```", s, re.S):
        s = match[1]

    with contextlib.suppress(json.decoder.JSONDecodeError):
        data = json.loads(s)

    return data


def extract_json_from_string(s: str) -> Any:
    data = None
    with contextlib.suppress(json.decoder.JSONDecodeError):
        data = json.loads(s)

    return data


def extract_json(s: str) -> Any:
    if data := extract_inline_json(s):
        return data

    if data := extract_multi_line_json(s):
        return data

    if data := extract_json_from_string(s):
        return data

    return None
