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
        result = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=0,
            )
        )
        assert not result.is_zero
        assert result.result_degree == 1

    def test_sq_above_degree_is_zero(self):
        """Sq^k(x) = 0 for k > deg(x) (instability)."""
        result = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=2,
            )
        )
        assert result.is_zero

    def test_sq_above_degree_zero(self):
        """Sq^3(x) = 0 for a degree-1 cocycle."""
        result = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=3,
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

        result2 = compute_steenrod_square(
            SteenrodSquareRequest(
                cochain_degree=0,
                simplex_values=((1,), (0,), (0,)),
                simplex_coefficients=(1, 1, 1),
                square_degree=0,
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
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="64 vertices"):
            SteenrodSquareRequest(
                cochain_degree=1,
                simplex_values=((0, 1),),
                simplex_coefficients=(1,),
                square_degree=1,
                ambient_simplices=(tuple(range(65)),),
            )
