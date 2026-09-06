from __future__ import annotations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationMatchRequest


@pytest.mark.parametrize(
    ("query", "operation_id"),
    (
        ("return every positive divisor of an integer", "integer.compute.divisors"),
        ("return the sum of all positive divisors", "integer.compute.divisor_sum"),
    ),
)
def test_positive_divisor_language_keeps_existing_first_rank(
    query: str,
    operation_id: str,
) -> None:
    result = Catalog.open().match(OperationMatchRequest(need=query, limit=8))

    assert result.matches[0].operation_id == operation_id


def test_complete_divisor_declarations_do_not_claim_proper_result_postconditions() -> (
    None
):
    catalog = Catalog.open()
    divisors = catalog.operation("integer.compute.divisors")
    divisor_sum = catalog.operation("integer.compute.divisor_sum")

    assert divisors is not None
    assert "every positive divisor" in divisors.description
    assert "proper-divisor" not in divisors.description
    assert "aliquot" not in divisors.description
    assert divisor_sum is not None
    assert "every positive divisor" in divisor_sum.description
    assert "proper-divisor" not in divisor_sum.description
    assert "aliquot" not in divisor_sum.description


def test_prime_field_system_discovery_surfaces_applicable_primitives() -> None:
    matches = (
        Catalog.open()
        .match(
            OperationMatchRequest(
                need="Solve Ax=b over GF(2), returning a particular solution and nullspace basis, or a left-nullspace inconsistency certificate.",
                limit=5,
            )
        )
        .matches
    )
    ids = [match.operation_id for match in matches]
    assert set(ids[:2]) == {
        "prime_field.matrix.nullspace.compute",
        "prime_field.matrix.rref.compute",
    }
    assert "prime_field.matrix.rref.compute" in ids[:3]
