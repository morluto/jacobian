"""Tests for finite semigroup operations."""

import pytest

from jacobian.math.finite_semigroups_ops._models import (
    ElementPowerRequest,
    GeneratedSubsemigroupRequest,
    IdempotentsRequest,
    PrincipalIdealsRequest,
)
from jacobian.math.finite_semigroups_ops._operations import (
    compute_element_power,
    compute_generated_subsemigroup,
    compute_idempotents,
    compute_principal_ideals,
)
from jacobian.math.finite_semigroups_ops._tools import TOOLS

Z3_TABLE = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "semigroup.element.power.compute",
        "semigroup.idempotents.compute",
        "semigroup.generated_subsemigroup.compute",
        "semigroup.principal_ideals.compute",
    }


def test_element_power_in_z3() -> None:
    request = ElementPowerRequest(multiplication_table=Z3_TABLE, element=1, exponent=2)
    result = compute_element_power(request)
    assert result.result == 2


def test_element_power_identity() -> None:
    request = ElementPowerRequest(multiplication_table=Z3_TABLE, element=1, exponent=1)
    result = compute_element_power(request)
    assert result.result == 1


def test_idempotents_of_band() -> None:
    request = IdempotentsRequest(multiplication_table=((0, 0), (0, 1)))
    result = compute_idempotents(request)
    assert result.idempotents == (0, 1)


def test_generated_subsemigroup_generates_all() -> None:
    request = GeneratedSubsemigroupRequest(
        multiplication_table=Z3_TABLE, generators=(1,)
    )
    result = compute_generated_subsemigroup(request)
    assert set(result.elements) == {0, 1, 2}
    assert result.size == 3


def test_principal_ideals_in_z3() -> None:
    request = PrincipalIdealsRequest(multiplication_table=Z3_TABLE, elements=(1,))
    result = compute_principal_ideals(request)
    assert len(result.ideals) == 1
    assert set(result.ideals[0]) == {0, 1, 2}


def test_element_power_rejects_zero_exponent() -> None:
    request = ElementPowerRequest(multiplication_table=Z3_TABLE, element=1, exponent=0)
    with pytest.raises(ValueError, match="identity"):
        compute_element_power(request)
