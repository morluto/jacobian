"""Local-structure slice (#3718)."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.finite_semigroups._models import FiniteSemigroup
from jacobian.math.finite_semigroups.operations import (
    ideal_enumeration,
    local_structure,
)


def _z3() -> FiniteSemigroup:
    return FiniteSemigroup(
        elements=("0", "1", "2"),
        multiplication=(("0", "1", "2"), ("1", "2", "0"), ("2", "0", "1")),
    )


def _left_zero_2() -> FiniteSemigroup:
    # x*y = x; no identity, both idempotent, minimal ideal is whole set.
    return FiniteSemigroup(
        elements=("a", "b"),
        multiplication=(("a", "a"), ("b", "b")),
    )


def test_z3_units_and_monoids() -> None:
    result = local_structure(_z3())
    assert result.identity == "0"
    assert result.units == ("0", "1", "2")
    assert len(result.local_monoids) == 1
    assert result.local_monoids[0].idempotent == "0"
    assert result.local_monoids[0].carrier == ("0", "1", "2")
    assert result.local_monoids[0].maximal_subgroup == ("0", "1", "2")
    assert result.minimal_ideal == ("0", "1", "2")


def test_left_zero_has_no_identity() -> None:
    result = local_structure(_left_zero_2())
    assert result.identity is None
    assert result.units == ()
    assert {m.idempotent for m in result.local_monoids} == {"a", "b"}
    for monoid in result.local_monoids:
        assert monoid.carrier == (monoid.idempotent,)
        assert monoid.maximal_subgroup == (monoid.idempotent,)


def test_enumeration_z3() -> None:
    result = ideal_enumeration(_z3())
    assert result.ideals == (("0", "1", "2"),)
    assert ("0",) in result.subsemigroups
    assert ("0", "1", "2") in result.subsemigroups


def test_enumeration_bound() -> None:
    elements = tuple(f"s{i}" for i in range(13))
    table = tuple(tuple("s0" for _ in elements) for _ in elements)
    semigroup = FiniteSemigroup(elements=elements, multiplication=table)
    with pytest.raises(OperationResourceAdmissionError, match="at most 12"):
        ideal_enumeration(semigroup)
