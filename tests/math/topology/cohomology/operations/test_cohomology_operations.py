"""Tests for cohomology operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.topology.cohomology.operations._models import (
    BocksteinRequest,
    BocksteinResult,
    SteenrodSquareRequest,
)
from jacobian.math.topology.cohomology.operations._operations import (
    compute_bockstein,
    compute_steenrod_square,
)


def _assert_error_code(
    excinfo: pytest.ExceptionInfo[ValidationError], code: str
) -> None:
    assert any(error["type"] == code for error in excinfo.value.errors())


class TestSteenrodSquare:
    """Test Steenrod square computation."""

    def test_sq0_identity(self) -> None:
        """Sq^0 is the identity."""
        # Minimal ambient where a single edge is vacuously a cocycle (no triangles).
        ambient = ((0,), (1,), (0, 1))
        result = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=0,
                ambient_simplices=ambient,
            )
        )
        assert not result.is_zero
        assert result.result_degree == 1

    def test_sq_above_degree_is_zero(self) -> None:
        """Sq^k(x) = 0 for k > deg(x) (instability)."""
        ambient = ((0,), (1,), (0, 1))
        result = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=2,
                ambient_simplices=ambient,
            )
        )
        assert result.is_zero

    def test_sq_above_degree_zero(self) -> None:
        """Sq^3(x) = 0 for a degree-1 cocycle."""
        ambient = ((0,), (1,), (0, 1))
        result = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=3,
                ambient_simplices=ambient,
            )
        )
        assert result.is_zero

    def test_sq_n_cup_n(self) -> None:
        """Sq^n(x) = x cup x for a degree-n cocycle."""
        # For degree 1, x supported on edges (0,1) and (1,2) cups to triangle
        # (0,1,2) only when that 2-simplex lies in the ambient complex.
        ambient = ((0,), (1,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2))
        result = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1), (1, 2)),
                simplex_coefficients=(1, 1),
                square_degree=1,
                ambient_simplices=ambient,
            )
        )
        assert result.result_degree == 2
        assert result.result_simplex_values == ((0, 1, 2),)

    def test_top_square_requires_target_in_ambient(self) -> None:
        """Edges [0,1],[1,2] of a graph emit no absent triangle [0,1,2]."""
        import pytest

        with pytest.raises(ValidationError) as excinfo:
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1), (1, 2)),
                simplex_coefficients=(1, 1),
                square_degree=1,
            )
        _assert_error_code(excinfo, "cohomology_operation.ambient_required_for_nonzero")
        # With the triangle absent from the complex, the square is zero.
        result = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1), (1, 2)),
                simplex_coefficients=(1, 1),
                square_degree=1,
                ambient_simplices=((0,), (1,), (2,), (0, 1), (1, 2)),
            )
        )
        assert result.is_zero
        assert result.result_simplex_values == ()

    def test_support_must_lie_in_ambient_complex(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((5, 6),),
                simplex_coefficients=(1,),
                square_degree=1,
                ambient_simplices=((0,), (1,), (0, 1)),
            )
        _assert_error_code(excinfo, "cohomology_operation.support_outside_ambient")

    # Cups that don't share the middle vertex should not contribute
    def test_disjoint_edges_do_not_contribute(self) -> None:
        result2 = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1), (0, 2)),
                simplex_coefficients=(1, 1),
                square_degree=1,
                ambient_simplices=((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)),
            )
        )
        assert result2.is_zero

    def test_sq0_duplicate_support_sums_modulo_two(self) -> None:
        """Repeated simplex keys denote one cochain: coefficients sum in GF(2)."""
        result = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=0,
                simplex_values=((0,), (0,)),
                simplex_coefficients=(1, 1),
                square_degree=0,
            )
        )
        assert result.is_zero
        assert result.result_simplex_values == ()
        assert result.result_simplex_coefficients == ()

    def test_sq0_duplicate_support_survivor(self) -> None:
        """Only keys whose summed coefficient survives mod 2 remain."""
        result = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=0,
                simplex_values=((1,), (0,), (0,), (1,)),
                simplex_coefficients=(1, 1, 1, 1),
                square_degree=0,
            )
        )
        assert result.is_zero

        # Non-zero 0-cochain requires an ambient where it is a cocycle;
        # isolated vertices have no edges, so every 0-cochain is a cocycle.
        result2 = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=0,
                simplex_values=((1,), (0,), (0,)),
                simplex_coefficients=(1, 1, 1),
                square_degree=0,
                ambient_simplices=((0,), (1,)),
            )
        )
        assert not result2.is_zero
        assert result2.result_simplex_values == ((1,),)
        assert result2.result_simplex_coefficients == (1,)

    def test_cup_product_with_duplicate_support_is_linear(self) -> None:
        """Sq^deg of a zero cochain represented by duplicated keys is zero."""
        result = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1), (0, 1)),
                simplex_coefficients=(1, 1),
                square_degree=1,
                ambient_simplices=((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)),
            )
        )
        assert result.is_zero


class TestBockstein:
    """Test Bockstein homomorphism."""

    def test_zero_cocycle(self) -> None:
        """Bockstein of a zero cocycle is zero."""
        result = compute_bockstein(
            BocksteinRequest(
                prime=2,
                cochain_degree=1,
                simplex_values=(),
                simplex_coefficients=(),
            )
        )
        assert result.is_zero

    def test_cancelling_duplicate_support_is_zero(self) -> None:
        """Sparse supports that cancel modulo p are the zero cocycle."""
        result = compute_bockstein(
            BocksteinRequest(
                prime=3,
                cochain_degree=2,
                simplex_values=((0, 1, 2), (0, 1, 2)),
                simplex_coefficients=(1, 2),
            )
        )
        assert result.is_zero
        assert result.result_degree == 3
        assert result.result_simplex_values == ()

    def test_zero_cocycle_with_ambient_complex(self) -> None:
        """The optional ambient stays validated and the result is unchanged."""
        ambient = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
        result = compute_bockstein(
            BocksteinRequest(
                prime=5,
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(10,),
                ambient_simplices=ambient,
            )
        )
        assert result.is_zero
        assert result.result_degree == 2

    def test_every_admissible_input_returns_predetermined_zero(self) -> None:
        """Admission reduces every request to the zero cocycle first, so
        execution can only return the empty degree-(n+1) cochain."""
        admissible = [
            (2, 0, (), ()),
            (2, 1, (), ()),
            (3, 2, ((0, 1, 2), (0, 1, 2)), (4, 2)),
            (7, 0, ((5,),), (-14,)),
            (9973, 16, (tuple(range(17)),), (9973 * 3,)),
        ]
        for prime, degree, values, coeffs in admissible:
            request = BocksteinRequest(
                prime=prime,
                cochain_degree=degree,
                simplex_values=values,
                simplex_coefficients=coeffs,
            )
            result = compute_bockstein(request)
            assert (
                result.result_degree,
                result.result_simplex_values,
                result.result_simplex_coefficients,
                result.is_zero,
            ) == (degree + 1, (), (), True)

    def test_result_shape_is_structurally_validated(self) -> None:
        request = BocksteinRequest(
            prime=2,
            cochain_degree=1,
            simplex_values=(),
            simplex_coefficients=(),
        )
        payload = compute_bockstein(request).model_dump()
        payload["result_degree"] = 3
        with pytest.raises(ValidationError) as excinfo:
            BocksteinResult.model_validate(payload)
        _assert_error_code(excinfo, "cohomology_operation.result_shape")

    def test_nonzero_cocycle(self) -> None:
        """Bockstein of a non-zero cocycle is unsupported without the complex."""
        with pytest.raises(ValidationError) as excinfo:
            BocksteinRequest(
                prime=2,
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
            )
        _assert_error_code(
            excinfo, "cohomology_operation.nonzero_bockstein_unsupported"
        )

    def test_prime_5(self) -> None:
        """Bockstein with a different prime still requires zero cocycle."""
        import pytest

        with pytest.raises(ValidationError) as excinfo:
            BocksteinRequest(
                prime=5,
                cochain_degree=2,
                simplex_values=((0, 1, 2),),
                simplex_coefficients=(3,),
            )
        _assert_error_code(
            excinfo, "cohomology_operation.nonzero_bockstein_unsupported"
        )


class TestCocycleAdmission:
    def test_non_cocycle_rejected(self) -> None:
        """A degree-1 cochain on one edge of the triangle is not a cocycle."""
        with pytest.raises(ValidationError) as excinfo:
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=1,
                ambient_simplices=((0,), (1,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2)),
            )
        _assert_error_code(excinfo, "cohomology_operation.not_cocycle")

    def test_genuine_cocycle_admitted(self) -> None:
        result = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1), (0, 2)),
                simplex_coefficients=(1, 1),
                square_degree=1,
                ambient_simplices=((0,), (1,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2)),
            )
        )
        assert result.is_zero


# The full boundary-plus-interior data of the 3-simplex (0,1,2,3): a
# downward-closed complex on four vertices.
_TETRAHEDRON = (
    (0,),
    (1,),
    (2,),
    (3,),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
    (0, 1, 2, 3),
)


class TestAmbientComplexClosure:
    """The supplied complex must carry every face its simplices imply."""

    def test_non_closed_complex_rejected(self) -> None:
        """A tetrahedron entry implies faces that are not listed here."""
        with pytest.raises(ValidationError) as excinfo:
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=1,
                ambient_simplices=((0, 1), (0, 1, 2, 3)),
            )
        _assert_error_code(excinfo, "cohomology_operation.ambient_not_downward_closed")

    def test_implied_triangle_enforces_cocycle(self) -> None:
        """On the closed tetrahedron the triangle (0,1,2) exists, so an edge
        (0,1)-supported cochain has nonzero coboundary and cannot pass."""
        with pytest.raises(ValidationError) as excinfo:
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=1,
                ambient_simplices=_TETRAHEDRON,
            )
        _assert_error_code(excinfo, "cohomology_operation.not_cocycle")

    def test_coboundary_of_vertex_on_closed_tetrahedron_admitted(self) -> None:
        """d(vertex 0) is a genuine cocycle of the closed tetrahedron."""
        result = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1), (0, 2), (0, 3)),
                simplex_coefficients=(1, 1, 1),
                square_degree=1,
                ambient_simplices=_TETRAHEDRON,
            )
        )
        assert result.is_zero
        assert result.result_simplex_values == ()

    def test_ambient_simplex_vertex_cap(self) -> None:
        """The advertised per-simplex cap fails at schema parse time."""
        with pytest.raises(ValidationError) as excinfo:
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=1,
                ambient_simplices=(tuple(range(65)),),
            )
        _assert_error_code(excinfo, "too_long")

    def test_huge_single_ambient_simplex_rejected_before_traversal(self) -> None:
        """One extremely large inner array dies at the schema length bound.

        The rejection must come from the ``BoundedAmbientSimplex`` schema
        constraint, not from validator-side traversal, hashing, or sorting
        of the simplex (which would report the 64-vertex value error).
        """
        with pytest.raises(ValidationError) as excinfo:
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=1,
                ambient_simplices=(tuple(range(500_000)),),
            )
        error_types = [error["type"] for error in excinfo.value.errors()]
        assert "too_long" in error_types
        assert "cohomology_operation.ambient_simplex_bound" not in error_types

    def test_bockstein_huge_single_ambient_simplex_rejected_before_traversal(
        self,
    ) -> None:
        """Bockstein shares the schema-bounded inner simplex type."""
        with pytest.raises(ValidationError) as excinfo:
            BocksteinRequest(
                prime=2,
                cochain_degree=1,
                simplex_values=(),
                simplex_coefficients=(),
                ambient_simplices=(tuple(range(500_000)),),
            )
        error_types = [error["type"] for error in excinfo.value.errors()]
        assert "too_long" in error_types
        assert "cohomology_operation.ambient_simplex_bound" not in error_types

    def test_ambient_vertex_label_magnitude_is_schema_bounded(self) -> None:
        """A 7-digit label is rejected by the element constraint itself."""
        with pytest.raises(ValidationError) as excinfo:
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=1,
                ambient_simplices=((10**6, 1),),
            )
        _assert_error_code(excinfo, "less_than_equal")


class TestInstabilityDegreeAdmission:
    """Sq^k = 0 for k > deg(x) is admitted output-sensitively."""

    def test_high_degree_trivial_square_returns_tiny_exact_zero(self) -> None:
        """cochain_degree=16 with square_degree=17 exceeds the old ceiling."""
        result = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=16,
                simplex_values=(),
                simplex_coefficients=(),
                square_degree=17,
            )
        )
        assert result.is_zero
        assert result.result_degree == 33
        assert result.result_simplex_values == ()
        assert result.result_simplex_coefficients == ()

    def test_result_degree_budget_boundary(self) -> None:
        """Requests are admitted exactly up to the returned-degree budget."""
        edge = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=16,
                simplex_values=(),
                simplex_coefficients=(),
                square_degree=112,
            )
        )
        assert edge.is_zero
        assert edge.result_degree == 128

        with pytest.raises(ValidationError) as excinfo:
            SteenrodSquareRequest(
                cochain_degree=16,
                simplex_values=(),
                simplex_coefficients=(),
                square_degree=113,
            )
        _assert_error_code(excinfo, "cohomology_operation.result_degree_bound")

    def test_nonzero_cocycle_instability_square_above_old_ceiling(self) -> None:
        """A genuine cocycle admits large trivial squares too."""
        ambient = (
            (0,),
            (1,),
            (2,),
            (3,),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
            (0, 1, 2, 3),
        )
        result = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1), (0, 2), (0, 3)),
                simplex_coefficients=(1, 1, 1),
                square_degree=40,
                ambient_simplices=ambient,
            )
        )
        assert result.is_zero
        assert result.result_degree == 41

    def test_top_and_intermediate_squares_keep_prior_boundaries(self) -> None:
        """k <= n envelopes are unchanged by the output-sensitive bound."""
        # Top square still requires ambient targets.
        with pytest.raises(ValidationError) as excinfo:
            SteenrodSquareRequest(
                cochain_degree=16,
                simplex_values=(),
                simplex_coefficients=(),
                square_degree=16,
            )
        _assert_error_code(
            excinfo, "cohomology_operation.ambient_required_for_top_square"
        )
        # Intermediate squares stay unsupported.
        with pytest.raises(ValidationError) as excinfo:
            SteenrodSquareRequest(
                cochain_degree=2,
                simplex_values=((0, 1, 2),),
                simplex_coefficients=(1,),
                square_degree=1,
            )
        _assert_error_code(
            excinfo, "cohomology_operation.intermediate_square_unsupported"
        )


class TestCatalogAdmission:
    """Owner-local admission expectations for the cohomology domain."""

    def test_bockstein_is_native_only(self) -> None:
        from jacobian.catalog.admission import AdmissionDecision
        from jacobian.math.topology.cohomology.operations._admission import ADMISSIONS

        record = next(
            entry
            for entry in ADMISSIONS
            if entry.operation_id == "cohomology.bockstein.compute"
        )
        assert record.decision is AdmissionDecision.NATIVE_ONLY

    def test_published_catalog_keeps_only_the_steenrod_square(self) -> None:
        from jacobian.catalog.admission import curate_public_tools
        from jacobian.math.topology.cohomology.operations._admission import ADMISSIONS
        from jacobian.math.topology.cohomology.operations._tools import TOOLS

        published = tuple(
            tool.operation_id for tool in curate_public_tools(TOOLS, ADMISSIONS)
        )
        assert published == ("cohomology.steenrod_square.compute",)

    def test_bockstein_native_symbol_is_supported(self) -> None:
        import jacobian.math.topology.cohomology.operations as public_module

        assert "compute_bockstein" in public_module.__all__
        assert callable(public_module.compute_bockstein)
