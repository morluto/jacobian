"""Refuse excessive polymorphism domains before touching supplied tables."""

from collections.abc import Sequence

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    _admission,
    check_polymorphism,
)


@pytest.mark.parametrize("table", [(), [0] * 16_385, range(32_768)])
def test_native_oversized_domain_is_resource_refused(table: Sequence[int]) -> None:
    source = FiniteRelationalStructure(
        carrier_size=32, signature=(), relation_tables=()
    )
    with pytest.raises(OperationResourceAdmissionError) as raised:
        check_polymorphism(source, 3, table)
    assert raised.value.errors()[0]["type"] == "relational.polymorphism.table_bound"


def test_oversized_domain_does_not_touch_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_snapshot(*args: object, **kwargs: object) -> None:
        pytest.fail("table snapshot ran before domain-size admission")

    monkeypatch.setattr(_admission, "_bounded_sequence_snapshot", unexpected_snapshot)
    source = FiniteRelationalStructure(
        carrier_size=32, signature=(), relation_tables=()
    )
    with pytest.raises(OperationResourceAdmissionError) as raised:
        _admission.admit_polymorphism_check(source, 3, range(32_768))
    assert raised.value.errors()[0]["type"] == "relational.polymorphism.table_bound"


def test_small_total_table_still_produces_a_polymorphism() -> None:
    source = FiniteRelationalStructure(carrier_size=2, signature=(), relation_tables=())
    result = check_polymorphism(source, 2, (0, 0, 1, 1))
    assert result.polymorphism is not None
    assert result.polymorphism.operation_table == (0, 0, 1, 1)
