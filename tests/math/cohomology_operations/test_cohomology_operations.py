"""Tests for cohomology operations."""

from __future__ import annotations

from jacobian.math.cohomology_operations._models import (
    BocksteinRequest,
    SteenrodSquareRequest,
)
from jacobian.math.cohomology_operations._operations import (
    compute_bockstein,
    compute_steenrod_square,
)


class TestSteenrodSquare:
    """Test Steenrod square computation."""

    def test_sq0_identity(self):
        """Sq^0 is the identity."""
        result = compute_steenrod_square(SteenrodSquareRequest(
            cochain_degree=1,
            simplex_values=((0, 1),),
            simplex_coefficients=(1,),
            square_degree=0,
        ))
        assert not result.is_zero
        assert result.result_degree == 1

    def test_sq_above_degree_is_zero(self):
        """Sq^k(x) = 0 for k > deg(x) (instability)."""
        result = compute_steenrod_square(SteenrodSquareRequest(
            cochain_degree=1,
            simplex_values=((0, 1),),
            simplex_coefficients=(1,),
            square_degree=2,
        ))
        assert result.is_zero

    def test_sq_above_degree_zero(self):
        """Sq^3(x) = 0 for a degree-1 cocycle."""
        result = compute_steenrod_square(SteenrodSquareRequest(
            cochain_degree=1,
            simplex_values=((0, 1),),
            simplex_coefficients=(1,),
            square_degree=3,
        ))
        assert result.is_zero

    def test_sq_n_cup_n(self):
        """Sq^n(x) = x cup x for a degree-n cocycle."""
        # For degree 1, x supported on edges (0,1) and (1,2) cups to triangle (0,1,2)
        result = compute_steenrod_square(SteenrodSquareRequest(
            cochain_degree=1,
            simplex_values=((0, 1), (1, 2)),
            simplex_coefficients=(1, 1),
            square_degree=1,
        ))
        assert result.result_degree == 2
        # Cups that don't share the middle vertex should not contribute
        result2 = compute_steenrod_square(SteenrodSquareRequest(
            cochain_degree=1,
            simplex_values=((0, 1), (0, 2)),
            simplex_coefficients=(1, 1),
            square_degree=1,
        ))
        assert result2.is_zero


class TestBockstein:
    """Test Bockstein homomorphism."""

    def test_zero_cocycle(self):
        """Bockstein of a zero cocycle is zero."""
        result = compute_bockstein(BocksteinRequest(
            prime=2,
            cochain_degree=1,
            simplex_values=(),
            simplex_coefficients=(),
        ))
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
