import functools
import inspect
import time
from collections.abc import Callable
from typing import Any

import structlog

logger = structlog.get_logger()


def log_time(
    *,
    task: str | None = None,
    args: list[str] | Callable[[dict], dict[str, Any]] | None = None,
):
    def _fn(fn):
        @functools.wraps(fn)
        async def _wrapper(*fn_args, **fn_kwargs):
            start_time = time.time()
            response = await fn(*fn_args, **fn_kwargs)
            end_time = time.time()

            nonlocal args, task
            if not task:
                task = fn.__name__
            if not args:
                args = []

            arg_names = inspect.getfullargspec(fn).args
            fn_args = dict(zip(arg_names, fn_args, strict=False))
            default_fn_kwargs = {
                k: v.default
                for k, v in inspect.signature(fn).parameters.items()
                if v.default is not v.empty
            }
            all_log_args = fn_args | default_fn_kwargs | fn_kwargs
            if callable(args):
                log_kwargs = args(all_log_args)
            else:
                log_kwargs = {x: all_log_args[x] for x in args}

            logger.info(
                "Execute time",
                task=task,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                **log_kwargs,
            )

            return response

        return _wrapper

    return _fn
