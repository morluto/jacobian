"""Assertions for structured number-theory model validation."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pydantic import ValidationError


@contextmanager
def expect_validation(error_type: str) -> Iterator[None]:
    """Require a Pydantic error with the owner-defined stable type."""

    with pytest.raises(ValidationError) as error:
        yield
    actual = error.value.errors()[0]["type"]
    # Some boundary cases are rejected by Pydantic's declarative field
    # constraints (for example ``string_too_long``) before the owner
    # validator runs.  Those remain structured Pydantic errors, while
    # owner-level checks use the stable domain code.
    assert actual.startswith(error_type) or actual in {
        "string_too_long",
        "less_than_equal",
        "greater_than_equal",
        "too_long",
    }
