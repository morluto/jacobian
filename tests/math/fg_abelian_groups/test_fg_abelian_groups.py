"""Tests for finitely generated abelian group operations."""

import pytest

from jacobian.math.finite_abelian_groups_v2._models import (
    AbelianPresentation,
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


def test_invariant_factor_divisibility_canonical_order_accepted() -> None:
    """d_1 | d_2: (2, 4) is the canonical presentation of Z/2 x Z/4."""
    p = AbelianPresentation(invariant_factors=(2, 4))
    assert p.invariant_factors == (2, 4)


def test_invariant_factor_divisibility_reversed_order_rejected() -> None:
    """(4, 2) violates d_1 | d_2 since 4 does not divide 2."""
    with pytest.raises(ValueError, match="must satisfy d_i"):
        AbelianPresentation(invariant_factors=(4, 2))


def test_invariant_factor_rejects_zero_free_summand() -> None:
    """Zero (free) summands are not admitted by the finite-group contract."""
    with pytest.raises(ValueError, match="integers >= 2"):
        AbelianPresentation(invariant_factors=(0, 6))


def test_invariant_factor_rejects_trivial_one() -> None:
    """Trivial factors of 1 must be omitted."""
    with pytest.raises(ValueError, match="integers >= 2"):
        AbelianPresentation(invariant_factors=(1, 6))


def test_subgroup_generated_rejects_unbounded_group() -> None:
    """A single large cyclic factor that exceeds the bound must be rejected."""
    with pytest.raises(ValueError, match="bound"):
        SubgroupGeneratedRequest(invariant_factors=(4_097,), generators=((1,),))


def test_quotient_rejects_unbounded_group() -> None:
    with pytest.raises(ValueError, match="bound"):
        QuotientRequest(invariant_factors=(4_097,), subgroup_generators=((1,),))


def test_subgroup_generated_z2_x_z4() -> None:
    """Subgroup <(1,0)> in Z/2 x Z/4 has index 4."""
    request = SubgroupGeneratedRequest(invariant_factors=(2, 4), generators=((1, 0),))
    result = compute_subgroup_generated(request)
    assert result.index == 4


def test_presentation_normalize_method_name() -> None:
    """Normalization method should not have a typo."""
    result = compute_presentation_normalize((2, 4))
    assert result.method == "SmithNormalForm"
