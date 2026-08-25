"""Shared assertions for polynomial validation contracts."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pydantic import ValidationError


@contextmanager
def polynomial_validation_error() -> Iterator[None]:
    """Assert that a polynomial model rejects input by a structured reason."""

    with pytest.raises(ValidationError) as caught:
        yield
    assert caught.value.errors()[0]["type"] not in {"value_error", "assertion_error"}
