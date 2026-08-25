"""Shared assertions for polynomial validation contracts."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pydantic import ValidationError


@contextmanager
def polynomial_validation_error() -> Iterator[None]:
    """Assert that a polynomial model rejects invalid input."""

    with pytest.raises(ValidationError):
        yield
