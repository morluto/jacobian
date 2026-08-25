"""Shared assertions for typed linear-programming contracts."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pydantic import ValidationError


@contextmanager
def linear_validation_error() -> Iterator[None]:
    """Require a Pydantic validation boundary for invalid linear data."""

    with pytest.raises(ValidationError):
        yield
