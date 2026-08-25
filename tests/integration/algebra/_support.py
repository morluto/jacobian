"""Structured validation assertions for algebra integration contracts."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pydantic import ValidationError


@contextmanager
def algebra_validation_error() -> Iterator[None]:
    with pytest.raises(ValidationError) as caught:
        yield
    assert caught.value.errors()[0]["type"] not in {"value_error", "assertion_error"}
