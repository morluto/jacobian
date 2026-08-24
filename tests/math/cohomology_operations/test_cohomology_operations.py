"""Tests for cohomology operations."""

from __future__ import annotations

import pytest

from jacobian.math.cohomology_operations._models import (
    BocksteinRequest,
    SteenrodSquareRequest,
    SteenrodSquareResult,
)
from jacobian.math.cohomology_operations._operations import (
    compute_bockstein,
    compute_steenrod_square,
)


class TestSteenrodSquare:
    """Test Steenrod square computation."""

    def test_sq0_identity(self):
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

    def test_sq_above_degree_is_zero(self):
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

    def test_sq_above_degree_zero(self):
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

    def test_sq_n_cup_n(self):
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

    def test_top_square_requires_target_in_ambient(self):
        """Edges [0,1],[1,2] of a graph emit no absent triangle [0,1,2]."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="ambient"):
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1), (1, 2)),
                simplex_coefficients=(1, 1),
                square_degree=1,
            )
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

    def test_support_must_lie_in_ambient_complex(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="inside the ambient"):
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((5, 6),),
                simplex_coefficients=(1,),
                square_degree=1,
                ambient_simplices=((0,), (1,), (0, 1)),
            )

    # Cups that don't share the middle vertex should not contribute
    def test_disjoint_edges_do_not_contribute(self):
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

    def test_sq0_duplicate_support_sums_modulo_two(self):
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

    def test_sq0_duplicate_support_survivor(self):
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

    def test_cup_product_with_duplicate_support_is_linear(self):
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

    def test_forged_result_rejected(self):
        """An authored payload cannot detach from the retained cochain."""
        import pytest
        from pydantic import ValidationError

        request = SteenrodSquareRequest(
            cochain_degree=1,
            simplex_values=((0, 1), (1, 2)),
            simplex_coefficients=(1, 1),
            square_degree=1,
            ambient_simplices=((0,), (1,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2)),
        )
        genuine = compute_steenrod_square(request)
        assert genuine.result_simplex_values == ((0, 1, 2),)
        payload = genuine.model_dump()
        payload["result_simplex_values"] = [[2, 3, 4]]
        payload["result_simplex_coefficients"] = [1]
        payload["is_zero"] = False
        with pytest.raises(ValidationError, match="replay"):
            SteenrodSquareResult.model_validate(payload)


class TestBockstein:
    """Test Bockstein homomorphism."""

    def test_zero_cocycle(self):
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

    def test_cancelling_duplicate_support_is_zero(self):
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

    def test_zero_cocycle_with_ambient_complex(self):
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

    def test_every_admissible_input_returns_predetermined_zero(self):
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

    def test_nonzero_cocycle(self):
        """Bockstein of a non-zero cocycle is unsupported without the complex."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BocksteinRequest(
                prime=2,
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
            )

    def test_prime_5(self):
        """Bockstein with a different prime still requires zero cocycle."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BocksteinRequest(
                prime=5,
                cochain_degree=2,
                simplex_values=((0, 1, 2),),
                simplex_coefficients=(3,),
            )


class TestCocycleAdmission:
    def test_non_cocycle_rejected(self):
        from pydantic import ValidationError

        """A degree-1 cochain on one edge of the triangle is not a cocycle."""
        with pytest.raises(ValidationError, match="not a cocycle"):
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=1,
                ambient_simplices=((0,), (1,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2)),
            )

    def test_genuine_cocycle_admitted(self):
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

    def test_non_closed_complex_rejected(self):
        """A tetrahedron entry implies faces that are not listed here."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="downward closed"):
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=1,
                ambient_simplices=((0, 1), (0, 1, 2, 3)),
            )

    def test_implied_triangle_enforces_cocycle(self):
        """On the closed tetrahedron the triangle (0,1,2) exists, so an edge
        (0,1)-supported cochain has nonzero coboundary and cannot pass."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="not a cocycle"):
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=1,
                ambient_simplices=_TETRAHEDRON,
            )

    def test_coboundary_of_vertex_on_closed_tetrahedron_admitted(self):
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

    def test_ambient_simplex_vertex_cap(self):
        """The advertised per-simplex cap fails at schema parse time."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="at most 64 items"):
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=1,
                ambient_simplices=(tuple(range(65)),),
            )

    def test_huge_single_ambient_simplex_rejected_before_traversal(self):
        """One extremely large inner array dies at the schema length bound.

        The rejection must come from the ``BoundedAmbientSimplex`` schema
        constraint, not from validator-side traversal, hashing, or sorting
        of the simplex (which would report the 64-vertex value error).
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as excinfo:
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=1,
                ambient_simplices=(tuple(range(500_000)),),
            )
        messages = [error["msg"] for error in excinfo.value.errors()]
        assert any("at most 64 items" in message for message in messages)
        assert not any("64 vertices" in message for message in messages)

    def test_bockstein_huge_single_ambient_simplex_rejected_before_traversal(self):
        """Bockstein shares the schema-bounded inner simplex type."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as excinfo:
            BocksteinRequest(
                prime=2,
                cochain_degree=1,
                simplex_values=(),
                simplex_coefficients=(),
                ambient_simplices=(tuple(range(500_000)),),
            )
        messages = [error["msg"] for error in excinfo.value.errors()]
        assert any("at most 64 items" in message for message in messages)
        assert not any("64 vertices" in message for message in messages)

    def test_ambient_vertex_label_magnitude_is_schema_bounded(self):
        """A 7-digit label is rejected by the element constraint itself."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="999999"):
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=1,
                ambient_simplices=((10**6, 1),),
            )


class TestInstabilityDegreeAdmission:
    """Sq^k = 0 for k > deg(x) is admitted output-sensitively."""

    def test_high_degree_trivial_square_returns_tiny_exact_zero(self):
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

    def test_result_degree_budget_boundary(self):
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

        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="exact-result budget"):
            SteenrodSquareRequest(
                cochain_degree=16,
                simplex_values=(),
                simplex_coefficients=(),
                square_degree=113,
            )

    def test_nonzero_cocycle_instability_square_above_old_ceiling(self):
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

    def test_top_and_intermediate_squares_keep_prior_boundaries(self):
        """k <= n envelopes are unchanged by the output-sensitive bound."""
        from pydantic import ValidationError

        # Top square still requires ambient targets.
        with pytest.raises(ValidationError, match="ambient"):
            SteenrodSquareRequest(
                cochain_degree=16,
                simplex_values=(),
                simplex_coefficients=(),
                square_degree=16,
            )
        # Intermediate squares stay unsupported.
        with pytest.raises(ValidationError, match="intermediate"):
            SteenrodSquareRequest(
                cochain_degree=2,
                simplex_values=((0, 1, 2),),
                simplex_coefficients=(1,),
                square_degree=1,
            )


class TestCatalogAdmission:
    """Owner-local admission expectations for the cohomology domain."""

    def test_bockstein_is_native_only(self):
        from jacobian.catalog.admission import AdmissionDecision
        from jacobian.math.cohomology_operations._admission import ADMISSIONS

        record = next(
            entry
            for entry in ADMISSIONS
            if entry.operation_id == "cohomology.bockstein.compute"
        )
        assert record.decision is AdmissionDecision.NATIVE_ONLY

    def test_published_catalog_keeps_only_the_steenrod_square(self):
        from jacobian.catalog.admission import curate_public_tools
        from jacobian.math.cohomology_operations._admission import ADMISSIONS
        from jacobian.math.cohomology_operations._tools import TOOLS

        published = tuple(
            tool.operation_id for tool in curate_public_tools(TOOLS, ADMISSIONS)
        )
        assert published == ("cohomology.steenrod_square.compute",)

    def test_bockstein_native_symbol_is_supported(self):
        import jacobian.math.cohomology_operations as public_module

        assert "compute_bockstein" in public_module.__all__
        assert callable(public_module.compute_bockstein)
