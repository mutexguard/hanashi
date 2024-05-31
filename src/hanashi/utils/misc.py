import itertools
from collections.abc import Iterable


def chunk(iterator: Iterable, size: int) -> Iterable[tuple]:
    it = iter(iterator)

    return iter(lambda: tuple(itertools.islice(it, size)), ())
