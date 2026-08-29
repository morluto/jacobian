"""Tests for finite topological space operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.finite.spaces import (
    FiniteTopologicalMap,
    FiniteTopologicalSpace,
)
from jacobian.math.topology.finite.spaces._models import (
    ContinuousCheckRequest,
    KolmogorovQuotientRequest,
    SubsetRequest,
)
from jacobian.math.topology.finite.spaces._tools import (
    TOOLS,
    _boundary,
    _closure,
    _continuous_check,
    _interior,
    _kolmogorov_quotient,
)

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
        result = _interior(SubsetRequest(space=_sierpinski(), subset=(0,)))
        assert result.interior == (0,)

    def test_sierpinski_interior_b(self) -> None:
        # Interior of {b}: {b} is not open (its minimal nbhd is {a,b}), so interior = {}.
        result = _interior(SubsetRequest(space=_sierpinski(), subset=(1,)))
        assert result.interior == ()

    def test_sierpinski_interior_ab(self) -> None:
        result = _interior(SubsetRequest(space=_sierpinski(), subset=(0, 1)))
        assert result.interior == (0, 1)


# ---------------------------------------------------------------------------
# Closure
# ---------------------------------------------------------------------------


class TestClosure:
    def test_sierpinski_closure_a(self) -> None:
        # Closure of {a}: up-set of a = {b} (since b >= a in specialization).
        result = _closure(SubsetRequest(space=_sierpinski(), subset=(0,)))
        assert result.closure == (0, 1)

    def test_sierpinski_closure_b(self) -> None:
        result = _closure(SubsetRequest(space=_sierpinski(), subset=(1,)))
        assert result.closure == (1,)


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_sierpinski_boundary_a(self) -> None:
        # Closure({a}) = {a,b}, Interior({a}) = {a}. Boundary = {b}.
        result = _boundary(SubsetRequest(space=_sierpinski(), subset=(0,)))
        assert result.boundary == (1,)


# ---------------------------------------------------------------------------
# Kolmogorov quotient
# ---------------------------------------------------------------------------


class TestKolmogorovQuotient:
    def test_sierpinski_is_t0(self) -> None:
        result = _kolmogorov_quotient(KolmogorovQuotientRequest(space=_sierpinski()))
        # Sierpinski space is T0, so the quotient has 2 points.
        assert len(result.quotient_points) == 2

    def test_discrete_is_t0(self) -> None:
        result = _kolmogorov_quotient(KolmogorovQuotientRequest(space=_discrete_2()))
        assert len(result.quotient_points) == 2

    def test_equivalence_classes_retain_long_source_labels(self) -> None:
        left = "a" * 64
        right = "b" * 64
        space = FiniteTopologicalSpace(
            points=(left, right),
            preorder=((0, 1), (0, 1)),
        )

        result = _kolmogorov_quotient(KolmogorovQuotientRequest(space=space))

        assert result.quotient_points == ((left, right),)
        assert result.quotient_preorder == ((0,),)
        assert result.class_map == (0, 0)


# ---------------------------------------------------------------------------
# Continuity check
# ---------------------------------------------------------------------------


class TestContinuityCheck:
    def test_identity_is_continuous(self) -> None:
        space = _sierpinski()
        m = FiniteTopologicalMap(source=space, target=space, point_map=(0, 1))
        result = _continuous_check(ContinuousCheckRequest(point_map=m))
        assert result.is_continuous is True

    def test_swap_not_continuous(self) -> None:
        space = _sierpinski()
        m = FiniteTopologicalMap(source=space, target=space, point_map=(1, 0))
        result = _continuous_check(ContinuousCheckRequest(point_map=m))
        assert result.is_continuous is False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize(
        "operation",
        (_interior, _closure, _boundary),
    )
    def test_subset_admission_rejects_out_of_range_indices(self, operation) -> None:
        request = SubsetRequest(space=_sierpinski(), subset=(2,))

        with pytest.raises(OperationDomainValidationError) as error:
            operation(request)

        assert error.value.errors()[0]["loc"] == ("subset",)
        assert (
            error.value.errors()[0]["type"]
            == "finite_topology_space.subset_index_out_of_range"
        )

    def test_duplicate_point_labels_rejected(self) -> None:
        with pytest.raises(ValidationError) as error:
            FiniteTopologicalSpace(
                points=("a", "a"),
                preorder=((0,), (1,)),
            )
        assert (
            error.value.errors()[0]["type"]
            == "finite_topology_space.point_labels_not_distinct"
        )

    def test_non_reflexive_preorder_rejected(self) -> None:
        with pytest.raises(ValidationError) as error:
            FiniteTopologicalSpace(
                points=("a", "b"),
                preorder=((1,), (0, 1)),
            )
        assert (
            error.value.errors()[0]["type"]
            == "finite_topology_space.preorder_not_reflexive"
        )

    def test_out_of_range_preorder_rejected(self) -> None:
        with pytest.raises(ValidationError) as error:
            FiniteTopologicalSpace(
                points=("a", "b"),
                preorder=((0, 5), (0, 1)),
            )
        assert (
            error.value.errors()[0]["type"]
            == "finite_topology_space.preorder_index_out_of_range"
        )
