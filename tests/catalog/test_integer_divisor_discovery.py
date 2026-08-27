from __future__ import annotations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDiscoveryRequest


@pytest.mark.parametrize(
    ("query", "operation_id"),
    (
        ("positive divisors", "integer.compute.divisors"),
        ("sum positive divisors", "integer.compute.divisor_sum"),
    ),
)
def test_positive_divisor_language_keeps_existing_first_rank(
    query: str,
    operation_id: str,
) -> None:
    result = Catalog.open().search(OperationDiscoveryRequest(query=query, limit=8))

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
