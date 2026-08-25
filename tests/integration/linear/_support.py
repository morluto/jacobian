"""Shared assertions for typed linear-programming contracts."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pydantic import ValidationError


@contextmanager
def linear_validation_error() -> Iterator[None]:
    """Require a stable Pydantic error type rather than rendered prose."""

    with pytest.raises(ValidationError) as caught:
        yield
    assert caught.value.errors()[0]["type"] not in {"value_error", "assertion_error"}
