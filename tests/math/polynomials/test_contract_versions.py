"""Contract-version regression tests for polynomial result schema changes."""

from __future__ import annotations

import pytest

from jacobian.math.polynomials._invariants import POLYNOMIAL_INVARIANT_OPERATIONS


@pytest.mark.parametrize(
    ("operation_id", "version"),
    [
        ("polynomial.compute.square_free_decomposition", "3"),
        ("polynomial.factor.compute", "3"),
    ],
)
def test_result_contract_version_tracks_the_source_binding_schema_change(
    operation_id: str,
    version: str,
) -> None:
    """Adding the required source ``polynomial`` field changed the published
    result schema of both operations, so each must declare its own bumped
    version instead of inheriting the shared v2 default."""
    operation = next(
        item
        for item in POLYNOMIAL_INVARIANT_OPERATIONS
        if item.operation_id == operation_id
    )
    assert operation.version == version
