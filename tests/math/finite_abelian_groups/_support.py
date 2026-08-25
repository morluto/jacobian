"""Shared assertions for finite-Abelian validation contracts."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pydantic import ValidationError


@contextmanager
def finite_abelian_validation_error() -> Iterator[None]:
    """Reject message-coupled model validation without masking raw errors."""

    with pytest.raises(ValidationError) as caught:
        yield
    assert caught.value.errors()[0]["type"] not in {"value_error", "assertion_error"}
