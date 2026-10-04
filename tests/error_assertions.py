"""Helpers for asserting stable diagnostics on broadly caught exceptions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast


class _StructuredError(Protocol):
    def errors(self) -> Sequence[Mapping[str, object]]: ...


def error_code(error: BaseException) -> str:
    """Return the first structured error code from an exception."""

    code = cast(_StructuredError, error).errors()[0]["type"]
    assert isinstance(code, str)
    return code
