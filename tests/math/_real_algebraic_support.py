"""Structured validation assertions for real-algebraic values."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pydantic import ValidationError


@contextmanager
def real_algebraic_validation_error() -> Iterator[None]:
    with pytest.raises(ValidationError) as caught:
        yield
    assert caught.value.errors()[0]["type"].startswith("real_algebraic.")
