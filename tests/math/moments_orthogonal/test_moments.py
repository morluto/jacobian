"""Tests for moment-functional and orthogonal-polynomial operations (#1900)."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.moments_orthogonal._models import (
    ChristoffelDarbouxRequest,
    GaussianQuadratureRequest,
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


def _prefix(moments):
    from jacobian.math.moments_orthogonal.values import MomentFunctionalPrefix

    return MomentFunctionalPrefix(moments=moments, variable="x")


def _moments_uniform(n: int) -> tuple[CanonicalRational, ...]:
    """Uniform measure on [-1,1]: mu_k = 2/(k+1) for even k, 0 for odd k."""
    return tuple(
        CanonicalRational(num="2", den=str(k + 1))
        if k % 2 == 0
        else CanonicalRational(num="0", den="1")
        for k in range(n)
    )


class TestHankel:
    def test_hankel_order_0(self) -> None:
        result = compute_hankel_matrix(
            HankelRequest(prefix=_prefix(_moments_uniform(3)), order=0)
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
            HankelRequest(prefix=_prefix(_moments_uniform(5)), order=2)
        )
        assert result.order == 2
        assert result.rank == 3
        # det should be positive (positive definite)
        assert int(result.determinant.num) * int(result.determinant.den) > 0
        assert int(result.determinant.den) > 0

    def test_insufficient_moments(self) -> None:
        with pytest.raises(Exception, match="moment"):
            HankelRequest(prefix=_prefix(_moments_uniform(3)), order=2)


class TestShiftedHankel:
    def test_shifted_hankel(self) -> None:
        result = compute_shifted_hankel(
            ShiftedHankelRequest(prefix=_prefix(_moments_uniform(6)), order=2)
        )
        assert result.order == 2


class TestOrthogonalPolynomials:
    def test_uniform_gives_legendre(self) -> None:
        result = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                prefix=_prefix(_moments_uniform(7)), max_degree=3
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
                prefix=_prefix(_moments_uniform(3)), max_degree=2
            )

    def test_zero_norm_prefix_rejected(self) -> None:
        """moments [1,0,0,...]: p_1 has zero norm; no orthogonal family
        through higher degrees exists and the boundary must reject."""
        moments = (
            CanonicalRational(num="1", den="1"),
            CanonicalRational(num="0", den="1"),
            CanonicalRational(num="0", den="1"),
            CanonicalRational(num="0", den="1"),
            CanonicalRational(num="0", den="1"),
        )
        with pytest.raises(ValueError, match="quasi-definite"):
            compute_orthogonal_polynomials(
                OrthogonalPolynomialRequest(prefix=_prefix(moments), max_degree=2)
            )
        # The same prefix is rejected at request admission.
        with pytest.raises(ValueError, match="quasi-definite"):
            OrthogonalPolynomialRequest(prefix=_prefix(moments), max_degree=2)

    def test_quasi_definite_is_not_positive_definite(self) -> None:
        """Moments (1, 0, -1): norms are 1 and -1 - quasi-definite but not
        positive-definite."""
        moments = (
            CanonicalRational(num="1", den="1"),
            CanonicalRational(num="0", den="1"),
            CanonicalRational(num="-1", den="1"),
        )
        result = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(prefix=_prefix(moments), max_degree=1)
        )
        assert result.is_quasi_definite is True
        assert result.is_positive_definite is False


class TestRecurrence:
    def test_recurrence_from_legendre(self) -> None:
        family = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                prefix=_prefix(_moments_uniform(7)), max_degree=3
            )
        )
        rec = compute_recurrence(RecurrenceRequest(family=family))

        # alpha_0..alpha_2 are determined by adjacent polynomials; the
        # terminal alpha_3 is undetermined without p_4 and must be omitted.
        assert len(rec.alpha) == 3
        for a in rec.alpha:
            assert int(a.num) == 0

        # beta[0] is a placeholder; beta_k = h_k / h_{k-1}.
        # beta_1 = 1/3, beta_2 = 4/15, beta_3 = 9/35
        assert len(rec.beta) == 4
        assert int(rec.beta[1].num) == 1
        assert int(rec.beta[1].den) == 3

    def test_singleton_family_has_no_determined_alpha(self) -> None:
        """A family containing only p_0=1 determines no recurrence
        coefficient: mu_1/mu_0 is arbitrary and must not be reported as 0."""
        family = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                prefix=_prefix(
                    (
                        CanonicalRational(num="2", den="1"),
                        CanonicalRational(num="5", den="1"),
                        CanonicalRational(num="9", den="1"),
                    )
                ),
                max_degree=0,
            )
        )
        rec = compute_recurrence(RecurrenceRequest(family=family))
        assert len(rec.alpha) == 0


class TestChristoffelDarboux:
    def test_cd_kernel_degree_0(self) -> None:
        family = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                prefix=_prefix(_moments_uniform(7)), max_degree=3
            )
        )
        result = compute_christoffel_darboux(
            ChristoffelDarbouxRequest(family=family, degree=0)
        )
        # K_0(x,y) = p_0(x) p_0(y) / h_0 = 1/2
        assert result.coefficients == ((CanonicalRational(num="1", den="2"),),)

    def test_cd_kernel_is_bivariate(self) -> None:
        """Degree-1 Legendre-like kernel is 1/2 + (3/2) x y, not the
        diagonal specialization 1/2 + (3/2) x^2."""
        family = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                prefix=_prefix(_moments_uniform(7)), max_degree=1
            )
        )
        result = compute_christoffel_darboux(
            ChristoffelDarbouxRequest(family=family, degree=1)
        )
        half = CanonicalRational(num="1", den="2")
        zero = CanonicalRational(num="0", den="1")
        three_halves = CanonicalRational(num="3", den="2")
        assert result.coefficients == (
            (half, zero),
            (zero, three_halves),
        )
        # Off-diagonal evaluation K_1(2, 3) = 1/2 + (3/2)*6 = 37/4 replays.

        def evaluate(matrix, xv, yv):
            return sum(
                Fraction(int(entry.num), int(entry.den)) * xv**i * yv**j
                for i, row in enumerate(matrix)
                for j, entry in enumerate(row)
            )

        assert evaluate(result.coefficients, Fraction(2), Fraction(3)) == Fraction(
            19, 2
        )


class TestJacobiMatrix:
    def test_jacobi_matrix(self) -> None:
        family = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                prefix=_prefix(_moments_uniform(7)), max_degree=3
            )
        )
        result = compute_jacobi_matrix(JacobiMatrixRequest(family=family))
        # 3x3 matrix: n=4 polynomials, so the multiplication operator acts
        # on p_0..p_2.
        assert len(result.matrix) == 3
        assert len(result.alphas) == 3
        # Diagonal should all be 0 for symmetric measure
        for i in range(3):
            assert int(result.matrix[i][i].num) == 0
        # Monic basis: subdiagonal carries 1, superdiagonal carries beta.
        # beta_1 = 1/3 and beta_2 = 4/15 for Legendre moments; betas keeps
        # the recurrence convention with an unused placeholder first.
        one = CanonicalRational(num="1", den="1")
        third = CanonicalRational(num="1", den="3")
        four_fifteenths = CanonicalRational(num="4", den="15")
        assert result.betas == (
            CanonicalRational(num="0", den="1"),
            third,
            four_fifteenths,
        )
        assert result.matrix[1][0] == one
        assert result.matrix[2][1] == one
        assert result.matrix[0][1] == third
        assert result.matrix[1][2] == four_fifteenths


class TestGaussianQuadrature:
    def _moments_rational_nodes(self) -> tuple:
        """Measure with weight 7 at +-1 and 5 at +-2: mu_(2j) = 14 + 10*4^j,
        odd moments 0. mu_2/mu_0 = 54/24 = 9/4, so p_2 = x^2 - 9/4 with
        rational nodes +-3/2."""
        values = ("24", "0", "54", "0", "174")
        return tuple(
            CanonicalRational(num=v, den="1")
            if v != "0"
            else CanonicalRational(num="0", den="1")
            for v in values
        )

    def test_exact_rule_for_rational_nodes(self) -> None:
        from jacobian.math.moments_orthogonal.operations import (
            compute_gaussian_quadrature,
        )

        result = compute_gaussian_quadrature(
            GaussianQuadratureRequest(
                prefix=_prefix(self._moments_rational_nodes()), order=2
            )
        )
        assert result.order == 2
        assert len(result.nodes) == 2
        nodes = {(int(n.node.num), int(n.node.den)) for n in result.nodes}
        assert nodes == {(-3, 2), (3, 2)}
        for node in result.nodes:
            assert (int(node.weight.num), int(node.weight.den)) == (12, 1)
        assert result.exactness_degree == 3

    def test_requires_two_n_plus_one_moments(self) -> None:
        """The nested orthogonal-polynomial request needs h_n, so the public
        boundary requires 2n+1 moments, not the published minimum of 2n."""

        with pytest.raises(ValueError, match="need at least 5"):
            GaussianQuadratureRequest(
                prefix=_prefix(self._moments_rational_nodes()[:4]), order=2
            )

    def test_algebraic_nodes_rejected_at_admission(self) -> None:
        """Uniform [0,1] moments give p_2 = x^2 - x + 1/6 with irrational
        roots; the canonical rational node contract cannot carry them, so
        admission rejects instead of crashing on .p/.q access."""

        uniform = tuple(CanonicalRational(num="1", den=str(k + 1)) for k in range(5))
        with pytest.raises(ValueError, match="rational"):
            GaussianQuadratureRequest(prefix=_prefix(uniform), order=2)
