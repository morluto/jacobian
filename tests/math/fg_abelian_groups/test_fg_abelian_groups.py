"""Tests for finitely generated abelian group operations."""

from jacobian.math.finite_abelian_groups_v2._models import (
    ElementEqualRequest,
    ElementOrderRequest,
    ElementReduceRequest,
    QuotientRequest,
    SubgroupGeneratedRequest,
)
from jacobian.math.finite_abelian_groups_v2._operations import (
    compute_element_equal,
    compute_element_order,
    compute_element_reduce,
    compute_presentation_normalize,
    compute_quotient,
    compute_subgroup_generated,
)
from jacobian.math.finite_abelian_groups_v2._tools import TOOLS


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "abelian_group.element.equal.decide",
        "abelian_group.element.order.compute",
        "abelian_group.element.reduce",
        "abelian_group.presentation.normalize",
        "abelian_group.quotient.compute",
        "abelian_group.subgroup.generated.compute",
    }


def test_element_reduce_modular() -> None:
    request = ElementReduceRequest(invariant_factors=(6,), coordinates=(7,))
    result = compute_element_reduce(request)
    assert result.reduced == (1,)


def test_element_equal_same() -> None:
    request = ElementEqualRequest(
        invariant_factors=(6,), coordinates_a=(1,), coordinates_b=(7,)
    )
    result = compute_element_equal(request)
    assert result.equal is True


def test_element_equal_different() -> None:
    request = ElementEqualRequest(
        invariant_factors=(6,), coordinates_a=(1,), coordinates_b=(2,)
    )
    result = compute_element_equal(request)
    assert result.equal is False


def test_element_order_in_z6() -> None:
    request = ElementOrderRequest(invariant_factors=(6,), coordinates=(2,))
    result = compute_element_order(request)
    assert result.order == 3


def test_element_order_identity() -> None:
    request = ElementOrderRequest(invariant_factors=(6,), coordinates=(0,))
    result = compute_element_order(request)
    assert result.order == 1


def test_subgroup_generated_index() -> None:
    request = SubgroupGeneratedRequest(invariant_factors=(6,), generators=((2,),))
    result = compute_subgroup_generated(request)
    assert result.index == 2


def test_quotient_z6_by_2z() -> None:
    request = QuotientRequest(invariant_factors=(6,), subgroup_generators=((2,),))
    result = compute_quotient(request)
    assert result.quotient_order == 2


def test_presentation_normalize_z6_z4() -> None:
    result = compute_presentation_normalize((6, 4))
    assert result.order == 24
    assert len(result.invariant_factors) == 2
