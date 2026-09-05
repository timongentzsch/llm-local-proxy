"""Explicit iterator cleanup across stream wrappers and worker threads."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")


@contextmanager
def closing_iterator(events: Iterator[T]) -> Iterator[Iterator[T]]:
    """Close generators on early exit, while also accepting plain iterators."""
    try:
        yield events
    finally:
        close = getattr(events, "close", None)
        if close is not None:
            close()
