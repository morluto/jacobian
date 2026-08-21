"""Tests for moment-functional and orthogonal-polynomial operations (#1900)."""

import pytest
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.moments_orthogonal._models import (
    ChristoffelDarbouxRequest,
    HankelRequest,
    JacobiMatrixRequest,
    OrthogonalPolynomialRequest,
    RecurrenceRequest,
    ShiftedHankelRequest,
)
from jacobian.math.moments_orthogonal.operations import (
    compute_christoffel_darboux,
    compute_hankel_matrix,
    compute_jacobi_matrix,
    compute_orthogonal_polynomials,
    compute_recurrence,
    compute_shifted_hankel,
)
from jacobian.math.moments_orthogonal.values import OrthogonalPolynomialFamily


def _moments_uniform(n: int) -> tuple[CanonicalRational, ...]:
    """Uniform measure on [-1,1]: mu_k = 2/(k+1) for even k, 0 for odd k."""
    return tuple(
        CanonicalRational(num="2", den=str(k + 1)) if k % 2 == 0 else CanonicalRational(num="0", den="1")
        for k in range(n)
    )


class TestHankel:
    def test_hankel_order_0(self) -> None:
        result = compute_hankel_matrix(
            HankelRequest(moments=_moments_uniform(3), order=0, variable="x")
        )
        assert result.order == 0
        assert len(result.entries) == 1
        assert len(result.entries[0]) == 1
        assert int(result.entries[0][0].num) == 2
        assert int(result.entries[0][0].den) == 1
        assert int(result.determinant.num) == 2
        assert result.rank == 1

    def test_hankel_order_2(self) -> None:
        result = compute_hankel_matrix(
            HankelRequest(moments=_moments_uniform(5), order=2, variable="x")
        )
        assert result.order == 2
        assert result.rank == 3
        # det should be positive (positive definite)
        assert int(result.determinant.num) * int(result.determinant.den) > 0
        assert int(result.determinant.den) > 0

    def test_insufficient_moments(self) -> None:
        with pytest.raises(Exception, match="moment"):
            HankelRequest(moments=_moments_uniform(3), order=2, variable="x")


class TestShiftedHankel:
    def test_shifted_hankel(self) -> None:
        result = compute_shifted_hankel(
            ShiftedHankelRequest(moments=_moments_uniform(6), order=2, variable="x")
        )
        assert result.order == 2


class TestOrthogonalPolynomials:
    def test_uniform_gives_legendre(self) -> None:
        result = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                moments=_moments_uniform(7), max_degree=3, variable="x"
            )
        )
        assert result.is_quasi_definite
        assert result.is_positive_definite

        # p_0 = 1
        p0 = result.polynomials[0]
        assert len(p0.coefficients) == 1
        assert int(p0.coefficients[0].num) == 1

        # p_1 = x (alpha_0 = 0 for symmetric measure)
        p1 = result.polynomials[1]
        assert int(p1.coefficients[0].num) == 0  # constant term is 0
        assert int(p1.coefficients[1].num) == 1  # leading coefficient is 1 (monic)

        # p_2 = x^2 - 1/3
        p2 = result.polynomials[2]
        assert int(p2.coefficients[2].num) == 1  # monic
        assert int(p2.coefficients[0].num) == -1  # constant term -1/3
        assert int(p2.coefficients[0].den) == 3

        # Norms: h_0=2, h_1=2/3, h_2=8/45
        assert int(result.polynomials[0].squared_norm.num) == 2
        assert int(result.polynomials[0].squared_norm.den) == 1
        assert int(result.polynomials[1].squared_norm.num) == 2
        assert int(result.polynomials[1].squared_norm.den) == 3

    def test_insufficient_moments(self) -> None:
        with pytest.raises(Exception, match="moment"):
            OrthogonalPolynomialRequest(
                moments=_moments_uniform(3), max_degree=2, variable="x"
            )


class TestRecurrence:
    def test_recurrence_from_legendre(self) -> None:
        family = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                moments=_moments_uniform(7), max_degree=3, variable="x"
            )
        )
        rec = compute_recurrence(RecurrenceRequest(family=family))

        # All alphas should be 0 for symmetric measure
        for a in rec.alpha:
            assert int(a.num) == 0

        # beta_1 = 1/3, beta_2 = 4/15, beta_3 = 9/35
        assert int(rec.beta[1].num) == 1
        assert int(rec.beta[1].den) == 3


class TestChristoffelDarboux:
    def test_cd_kernel_degree_0(self) -> None:
        family = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                moments=_moments_uniform(7), max_degree=3, variable="x"
            )
        )
        result = compute_christoffel_darboux(
            ChristoffelDarbouxRequest(family=family, degree=0)
        )
        # K_0 = p_0(x)^2 / h_0 = 1/2
        assert len(result.numerator_x_coefficients) == 1
        assert int(result.numerator_x_coefficients[0].num) == 1
        assert int(result.numerator_x_coefficients[0].den) == 2


class TestJacobiMatrix:
    def test_jacobi_matrix(self) -> None:
        family = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                moments=_moments_uniform(7), max_degree=3, variable="x"
            )
        )
        result = compute_jacobi_matrix(JacobiMatrixRequest(family=family))
        # 3x3 matrix (n-1 = 3-1 = 2... wait, n=4 polynomials, so 3x3)
        # Actually we have polynomials 0..3 (4 total), so matrix is 3x3
        # Wait: the matrix is (n-1) x (n-1) = 3x3
        assert len(result.matrix) == 3
        assert len(result.alphas) == 3
        # Diagonal should all be 0 for symmetric measure
        for i in range(3):
            assert int(result.matrix[i][i].num) == 0
