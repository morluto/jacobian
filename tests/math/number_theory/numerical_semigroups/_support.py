"""Shared assertions for numerical-semigroup validation contracts."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from jacobian.catalog.models import OperationDomainValidationError


@contextmanager
def operation_domain_error() -> Iterator[None]:
    """Assert that a native operation rejects an admitted-domain input."""

    with pytest.raises(OperationDomainValidationError) as caught:
        yield
    error_type = caught.value.errors()[0]["type"]
    assert error_type.startswith("numerical_semigroup.")
