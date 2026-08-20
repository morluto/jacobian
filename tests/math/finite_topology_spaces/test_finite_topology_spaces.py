"""Tests for finite topological space operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.finite_topology_spaces import (
    FiniteTopologicalMap,
    FiniteTopologicalSpace,
)
from jacobian.math.finite_topology_spaces._models import (
    ContinuousCheckRequest,
    KolmogorovQuotientRequest,
    SubsetRequest,
)
from jacobian.math.finite_topology_spaces._operations import (
    compute_boundary,
    compute_closure,
    compute_continuous_check,
    compute_interior,
    compute_kolmogorov_quotient,
)
from jacobian.math.finite_topology_spaces._tools import TOOLS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sierpinski() -> FiniteTopologicalSpace:
    """Sierpinski space: points {a, b}, a <= b (open sets: {}, {a}, {a,b})."""
    return FiniteTopologicalSpace(
        points=("a", "b"),
        preorder=((0,), (0, 1)),
    )


def _discrete_2() -> FiniteTopologicalSpace:
    """Discrete 2-point space: every point is isolated."""
    return FiniteTopologicalSpace(
        points=("x", "y"),
        preorder=((0,), (1,)),
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_catalog_contains_only_audited_agent_outcomes() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "topology.finite.interior.compute",
        "topology.finite.closure.compute",
        "topology.finite.boundary.compute",
        "topology.finite.kolmogorov_quotient.compute",
        "topology.finite.continuity_check.compute",
    }


# ---------------------------------------------------------------------------
# Interior
# ---------------------------------------------------------------------------


class TestInterior:
    def test_sierpinski_interior_a(self) -> None:
        # Interior of {a}: {a} is open, so interior = {a}.
        result = compute_interior(SubsetRequest(space=_sierpinski(), subset=(0,)))
        assert result.interior == (0,)

    def test_sierpinski_interior_b(self) -> None:
        # Interior of {b}: {b} is not open (its minimal nbhd is {a,b}), so interior = {}.
        result = compute_interior(SubsetRequest(space=_sierpinski(), subset=(1,)))
        assert result.interior == ()

    def test_sierpinski_interior_ab(self) -> None:
        result = compute_interior(SubsetRequest(space=_sierpinski(), subset=(0, 1)))
        assert result.interior == (0, 1)


# ---------------------------------------------------------------------------
# Closure
# ---------------------------------------------------------------------------


class TestClosure:
    def test_sierpinski_closure_a(self) -> None:
        # Closure of {a}: up-set of a = {b} (since b >= a in specialization).
        result = compute_closure(SubsetRequest(space=_sierpinski(), subset=(0,)))
        assert result.closure == (0, 1)

    def test_sierpinski_closure_b(self) -> None:
        result = compute_closure(SubsetRequest(space=_sierpinski(), subset=(1,)))
        assert result.closure == (1,)


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_sierpinski_boundary_a(self) -> None:
        # Closure({a}) = {a,b}, Interior({a}) = {a}. Boundary = {b}.
        result = compute_boundary(SubsetRequest(space=_sierpinski(), subset=(0,)))
        assert result.boundary == (1,)


# ---------------------------------------------------------------------------
# Kolmogorov quotient
# ---------------------------------------------------------------------------


class TestKolmogorovQuotient:
    def test_sierpinski_is_t0(self) -> None:
        result = compute_kolmogorov_quotient(
            KolmogorovQuotientRequest(space=_sierpinski())
        )
        # Sierpinski space is T0, so the quotient has 2 points.
        assert len(result.quotient_points) == 2

    def test_discrete_is_t0(self) -> None:
        result = compute_kolmogorov_quotient(
            KolmogorovQuotientRequest(space=_discrete_2())
        )
        assert len(result.quotient_points) == 2


# ---------------------------------------------------------------------------
# Continuity check
# ---------------------------------------------------------------------------


class TestContinuityCheck:
    def test_identity_is_continuous(self) -> None:
        space = _sierpinski()
        m = FiniteTopologicalMap(source=space, target=space, point_map=(0, 1))
        result = compute_continuous_check(ContinuousCheckRequest(point_map=m))
        assert result.is_continuous is True

    def test_swap_not_continuous(self) -> None:
        space = _sierpinski()
        m = FiniteTopologicalMap(source=space, target=space, point_map=(1, 0))
        result = compute_continuous_check(ContinuousCheckRequest(point_map=m))
        assert result.is_continuous is False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_non_reflexive_preorder_rejected(self) -> None:
        with pytest.raises(ValidationError, match="reflexive"):
            FiniteTopologicalSpace(
                points=("a", "b"),
                preorder=((1,), (0, 1)),
            )

    def test_out_of_range_preorder_rejected(self) -> None:
        with pytest.raises(ValidationError, match="out of range"):
            FiniteTopologicalSpace(
                points=("a", "b"),
                preorder=((0, 5), (0, 1)),
            )
