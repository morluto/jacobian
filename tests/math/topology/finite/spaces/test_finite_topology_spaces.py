"""Tests for finite topological space operations."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.finite.spaces import (
    FiniteTopologicalMap,
    FiniteTopologicalSpace,
    verify_boundary,
    verify_closure,
    verify_continuity,
    verify_interior,
)
from jacobian.math.topology.finite.spaces._models import (
    BoundaryResult,
    ClosureResult,
    ContinuousCheckRequest,
    ContinuousCheckResult,
    InteriorResult,
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
    """Sierpinski space: points {a, b}, a <= b (open sets: {}, {b}, {a,b})."""
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
        # Interior of {a}: {a} is not open (minimal nbhd of a is {a,b}), so interior = {}.
        space = _sierpinski()
        result = _interior(SubsetRequest(space=space, subset=(0,)))
        assert result.interior.indices == ()
        assert result.interior.space == space
        assert result.subset.indices == (0,)
        assert result.space == space

    def test_sierpinski_interior_b(self) -> None:
        # Interior of {b}: {b} is open (minimal nbhd of b is {b}), so interior = {b}.
        result = _interior(SubsetRequest(space=_sierpinski(), subset=(1,)))
        assert result.interior.indices == (1,)

    def test_sierpinski_interior_ab(self) -> None:
        result = _interior(SubsetRequest(space=_sierpinski(), subset=(0, 1)))
        assert result.interior.indices == (0, 1)


# ---------------------------------------------------------------------------
# Closure
# ---------------------------------------------------------------------------


class TestClosure:
    def test_sierpinski_closure_a(self) -> None:
        # Closure of {a}: down-set of a = {a} (preorder row).
        space = _sierpinski()
        result = _closure(SubsetRequest(space=space, subset=(0,)))
        assert result.closure.indices == (0,)
        assert result.closure.space == space
        assert result.subset.indices == (0,)

    def test_sierpinski_closure_b(self) -> None:
        result = _closure(SubsetRequest(space=_sierpinski(), subset=(1,)))
        assert result.closure.indices == (0, 1)


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_sierpinski_boundary_a(self) -> None:
        # Closure({a}) = {a}, Interior({a}) = {}. Boundary = {a}.
        space = _sierpinski()
        result = _boundary(SubsetRequest(space=space, subset=(0,)))
        assert result.boundary.indices == (0,)
        assert result.boundary.space == space

    def test_empty_subset_stays_bound(self) -> None:
        space = _sierpinski()
        result = _boundary(SubsetRequest(space=space, subset=()))
        assert result.boundary.indices == ()
        assert result.subset.indices == ()


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
        assert result.point_map == m
        assert verify_continuity(
            ContinuousCheckResult.model_validate_json(result.model_dump_json())
        )

    def test_swap_not_continuous(self) -> None:
        space = _sierpinski()
        m = FiniteTopologicalMap(source=space, target=space, point_map=(1, 0))
        result = _continuous_check(ContinuousCheckRequest(point_map=m))
        assert result.is_continuous is False
        assert verify_continuity(
            ContinuousCheckResult.model_validate_json(result.model_dump_json())
        )

    def test_verifier_rejects_flipped_claim(self) -> None:
        space = _sierpinski()
        m = FiniteTopologicalMap(source=space, target=space, point_map=(0, 1))
        payload = ContinuousCheckResult(point_map=m, is_continuous=False).model_dump(
            mode="json"
        )
        assert not verify_continuity(ContinuousCheckResult.model_validate(payload))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize(
        "operation",
        (_interior, _closure, _boundary),
    )
    def test_subset_admission_rejects_out_of_range_indices(
        self, operation: Callable[[SubsetRequest], object]
    ) -> None:
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
        from jacobian.catalog.models import OperationDomainValidationError
        from jacobian.math.topology.finite.spaces import from_preorder

        with pytest.raises(OperationDomainValidationError) as error:
            from_preorder(
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


class TestSerializedSubsetClaims:
    def test_interior_round_trip_retains_source(self) -> None:
        space = _sierpinski()
        result = _interior(SubsetRequest(space=space, subset=(1,)))
        decoded = InteriorResult.model_validate_json(result.model_dump_json())
        assert decoded.space == space
        assert decoded.subset.indices == (1,)
        assert decoded.interior.indices == (1,)
        assert verify_interior(decoded)
        forged = decoded.model_copy(
            update={"interior": decoded.interior.model_copy(update={"indices": ()})}
        )
        assert not verify_interior(forged)
        other_space = _discrete_2()
        assert not verify_interior(
            decoded.model_copy(
                update={
                    "interior": decoded.interior.model_copy(
                        update={"space": other_space}
                    )
                }
            )
        )

    def test_closure_and_boundary_verifiers_reject_forgery(self) -> None:
        space = _sierpinski()
        closure_result = _closure(SubsetRequest(space=space, subset=(1,)))
        boundary_result = _boundary(SubsetRequest(space=space, subset=(1,)))
        closure_decoded = ClosureResult.model_validate_json(
            closure_result.model_dump_json()
        )
        boundary_decoded = BoundaryResult.model_validate_json(
            boundary_result.model_dump_json()
        )
        assert verify_closure(closure_decoded)
        assert verify_boundary(boundary_decoded)
        assert not verify_closure(
            closure_decoded.model_copy(
                update={
                    "closure": closure_decoded.closure.model_copy(
                        update={"indices": ()}
                    )
                }
            )
        )

    def test_subset_indices_are_canonical(self) -> None:
        space = _sierpinski()
        with pytest.raises(ValidationError) as error:
            InteriorResult(
                space=space,
                subset={"space": space, "indices": (1, 0)},
                interior={"space": space, "indices": ()},
            )
        assert (
            error.value.errors()[0]["type"]
            == "finite_topology_space.subset_indices_not_canonical"
        )

    def test_quotient_verifier_checks_relation_directly(self) -> None:
        from jacobian.math.topology.finite.spaces import (
            kolmogorov_quotient,
            verify_kolmogorov_quotient,
        )

        space = FiniteTopologicalSpace(
            points=("a", "b", "c"), preorder=((0, 1), (0, 1), (0, 1, 2))
        )
        result = kolmogorov_quotient(space)
        assert verify_kolmogorov_quotient(
            type(result).model_validate_json(result.model_dump_json())
        )
        payload = result.model_dump(mode="json")
        payload["quotient_map"]["target"]["preorder"] = [[0], [1]]
        assert not verify_kolmogorov_quotient(type(result).model_validate(payload))
