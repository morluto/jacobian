"""Shared assertions for numerical-semigroup validation contracts."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pydantic import ValidationError


@contextmanager
def numerical_semigroup_error() -> Iterator[None]:
    """Assert that a model rejects input with its owner-local error code."""

    with pytest.raises(ValidationError) as caught:
        yield
    error_type = caught.value.errors()[0]["type"]
    assert error_type.startswith("numerical_semigroup.")
    assert error_type != "numerical_semigroup.invariant"
