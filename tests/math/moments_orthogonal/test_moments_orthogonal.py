"""Known-answer and adversarial tests for moments and orthogonal polynomials."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.moments_orthogonal import (
    christoffel_darboux,
    gaussian_quadrature,
    hankel_matrix,
    jacobi_matrix,
    recurrence_coefficients,
)
from jacobian.math.moments_orthogonal._models import (
    ChristoffelDarbouxRequest,
    ChristoffelDarbouxResult,
    GaussianQuadratureRequest,
    GaussianQuadratureResult,
    HankelMatrixRequest,
    HankelMatrixResult,
    JacobiMatrixRequest,
    JacobiMatrixResult,
    RecurrenceCoefficientsRequest,
    RecurrenceCoefficientsResult,
)
from jacobian.math.moments_orthogonal._operations import (
    compute_christoffel_darboux,
    compute_gaussian_quadrature,
    compute_hankel_matrix,
    compute_jacobi_matrix,
    compute_recurrence_coefficients,
)
from jacobian.math.moments_orthogonal._tools import TOOLS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frac(num: int, den: int) -> Fraction:
    return Fraction(num, den)


def _cr(num: int, den: int) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(num, den)


_HARMONIC_MOMENTS = tuple(_cr(1, k) for k in range(1, 8))


# ---------------------------------------------------------------------------
# Hankel matrix
# ---------------------------------------------------------------------------


class TestHankelMatrix:
    def test_known_values(self) -> None:
        moments = (_frac(1, 1), _frac(1, 2), _frac(1, 3), _frac(1, 4), _frac(1, 5))
        result = hankel_matrix(moments)
        assert result.matrix[0] == (_frac(1, 1), _frac(1, 2), _frac(1, 3))
        assert result.matrix[1] == (_frac(1, 2), _frac(1, 3), _frac(1, 4))
        assert result.matrix[2] == (_frac(1, 3), _frac(1, 4), _frac(1, 5))

    def test_dimension(self) -> None:
        moments = tuple(_frac(1, k) for k in range(1, 8))
        result = hankel_matrix(moments)
        assert len(result.matrix) == 4
        assert len(result.matrix[0]) == 4

    def test_single_moment(self) -> None:
        result = hankel_matrix((_frac(1, 1),))
        assert result.matrix == ((_frac(1, 1),),)

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="between 1 and 64"):
            hankel_matrix(())

    def test_too_many_moments_rejected(self) -> None:
        moments = tuple(_frac(1, k) for k in range(1, 200))
        with pytest.raises(ValueError, match="between 1 and 64"):
            hankel_matrix(moments)

    def test_non_fraction_rejected(self) -> None:
        with pytest.raises(TypeError, match="Fractions"):
            hankel_matrix([1, 2, 3])


# ---------------------------------------------------------------------------
# Recurrence coefficients
# ---------------------------------------------------------------------------


class TestRecurrenceCoefficients:
    def test_uniform_measure(self) -> None:
        """Uniform measure on [0,1] has moments 1/(k+1), giving Legendre-like recurrence."""
        moments = tuple(_frac(1, k) for k in range(1, 8))
        result = recurrence_coefficients(moments)
        # For Legendre polynomials on [0,1], alpha_k = 1/2 for all k
        assert all(a == _frac(1, 2) for a in result.alpha)

    def test_beta_zero_is_mu0(self) -> None:
        moments = (_frac(2, 1), _frac(1, 1), _frac(2, 3), _frac(1, 2))
        result = recurrence_coefficients(moments)
        assert result.beta[0] == _frac(2, 1)

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="between 1 and 64"):
            recurrence_coefficients(())

    def test_zeroth_moment_nonzero(self) -> None:
        moments = (_frac(0, 1), _frac(1, 1), _frac(1, 2))
        with pytest.raises(ValueError, match="positive"):
            recurrence_coefficients(moments)

    def test_negative_zeroth_moment_rejected_before_short_return(self) -> None:
        """A one-term sequence must not publish a negative beta_0."""
        with pytest.raises(ValueError, match="positive"):
            recurrence_coefficients((_frac(-1, 1),))

    def test_even_length_sequence_fully_consumed(self) -> None:
        """The first four standard-normal moments determine two coefficient pairs."""
        moments = (_frac(1, 1), _frac(0, 1), _frac(1, 1), _frac(0, 1))
        result = recurrence_coefficients(moments)
        assert result.alpha == (_frac(0, 1), _frac(0, 1))
        assert result.beta == (_frac(1, 1), _frac(1, 1))

    def test_odd_length_sequence_unchanged(self) -> None:
        """Five standard-normal moments still give the same two coefficient pairs."""
        moments = (
            _frac(1, 1),
            _frac(0, 1),
            _frac(1, 1),
            _frac(0, 1),
            _frac(3, 1),
        )
        result = recurrence_coefficients(moments)
        assert result.alpha == (_frac(0, 1), _frac(0, 1))
        assert result.beta == (_frac(1, 1), _frac(1, 1))

    def test_two_moments_determine_one_coefficient_pair(self) -> None:
        """Two moments fully determine alpha_0 = mu_1/mu_0 and beta_0 = mu_0."""
        moments = (_frac(1, 1), _frac(1, 2))
        result = recurrence_coefficients(moments)
        assert result.alpha == (_frac(1, 2),)
        assert result.beta == (_frac(1, 1),)


# ---------------------------------------------------------------------------
# Jacobi matrix
# ---------------------------------------------------------------------------


class TestJacobiMatrix:
    def test_assembly(self) -> None:
        alpha = (_frac(1, 2), _frac(1, 2))
        beta = (_frac(1, 1), _frac(1, 12), _frac(1, 15))
        result = jacobi_matrix(alpha, beta)
        assert result.diagonal == (_frac(1, 2), _frac(1, 2))
        # beta_0 is the zeroth moment; the subdiagonal carries beta_1 only.
        assert result.off_diagonal == (_frac(1, 12),)

    def test_empty_alpha(self) -> None:
        beta = (_frac(1, 1),)
        result = jacobi_matrix((), beta)
        assert result.diagonal == ()
        assert result.off_diagonal == ()

    def test_zero_beta_rejected(self) -> None:
        with pytest.raises(ValueError, match="nonzero"):
            jacobi_matrix((_frac(0, 1),), (_frac(0, 1), _frac(1, 1)))


# ---------------------------------------------------------------------------
# Christoffel-Darboux kernel
# ---------------------------------------------------------------------------


class TestChristoffelDarboux:
    def test_diagonal_kernel(self) -> None:
        """K_n(x,x) = sum_{k=0}^{n-1} p_k(x)^2 / h_k is positive."""
        alpha = (_frac(1, 2), _frac(1, 2))
        beta = (_frac(1, 1), _frac(1, 12), _frac(1, 15))
        result = christoffel_darboux(alpha, beta, _frac(1, 2), _frac(1, 2))
        assert result.kernel > 0

    def test_zero_when_x_neq_y_symmetric(self) -> None:
        """K_n(x,y) is a reproducing kernel: K_n(x,y) = K_n(y,x)."""
        alpha = (_frac(0, 1), _frac(1, 3))
        beta = (_frac(2, 1), _frac(1, 9), _frac(4, 45))
        result_xy = christoffel_darboux(alpha, beta, _frac(1, 4), _frac(3, 4))
        result_yx = christoffel_darboux(alpha, beta, _frac(3, 4), _frac(1, 4))
        assert result_xy.kernel == result_yx.kernel

    def test_first_polynomial_is_one(self) -> None:
        alpha = (_frac(0, 1), _frac(1, 3))
        beta = (_frac(2, 1), _frac(1, 9), _frac(4, 45))
        result = christoffel_darboux(alpha, beta, _frac(1, 1), _frac(1, 1))
        assert result.polynomials_evaluated[0] == _frac(1, 1)

    def test_empty_alpha(self) -> None:
        beta = (_frac(1, 1),)
        result = christoffel_darboux((), beta, _frac(1, 1), _frac(1, 1))
        assert result.kernel == _frac(0, 1)
        assert result.polynomials_evaluated == (_frac(1, 1),)

    def test_huge_coefficients_charged_at_every_recurrence_step(self) -> None:
        """x=y=0 with 4,096-digit alphas must overflow the admission bound."""
        big = "1" + "0" * 4095  # exactly 4,096 digits: admissible per coefficient
        with pytest.raises((ValueError, ValidationError), match="growth exceeds"):
            ChristoffelDarbouxRequest(
                alpha=tuple(_cr(int(big), 1) for _ in range(15)),
                beta=tuple(_cr(1, 1) for _ in range(16)),
                x=_cr(0, 1),
                y=_cr(0, 1),
            )

    def test_large_y_charged_at_every_recurrence_step(self) -> None:
        """A huge y must be charged at every step, not only the first."""
        alpha = tuple(_frac(0, 1) for _ in range(15))
        beta = tuple(_frac(1, 1) for _ in range(16))
        y = Fraction(10) ** 3000
        with pytest.raises((ValueError, ValidationError), match="growth exceeds"):
            ChristoffelDarbouxRequest(
                alpha=tuple(_cr(a.numerator, a.denominator) for a in alpha),
                beta=tuple(_cr(b.numerator, b.denominator) for b in beta),
                x=_cr(2, 1),
                y=_cr(y.numerator, y.denominator),
            )


# ---------------------------------------------------------------------------
# Gaussian quadrature
# ---------------------------------------------------------------------------


class TestGaussianQuadrature:
    def test_weights_sum_to_mu0(self) -> None:
        """Sum of weights equals mu_0 (the zeroth moment)."""
        alpha = (_frac(1, 2), _frac(1, 2), _frac(1, 2))
        beta = (_frac(1, 1), _frac(1, 12), _frac(1, 15), _frac(4, 45))
        result = gaussian_quadrature(alpha, beta)
        mu0 = float(beta[0])
        assert abs(sum(result.weights) - mu0) < 1e-10

    def test_nodes_count(self) -> None:
        alpha = (_frac(1, 2), _frac(1, 2), _frac(1, 2))
        beta = (_frac(1, 1), _frac(1, 12), _frac(1, 15), _frac(4, 45))
        result = gaussian_quadrature(alpha, beta)
        assert len(result.nodes) == 3
        assert len(result.weights) == 3

    def test_single_point(self) -> None:
        alpha = (_frac(0, 1),)
        beta = (_frac(1, 1), _frac(1, 3))
        result = gaussian_quadrature(alpha, beta)
        assert len(result.nodes) == 1
        assert abs(result.nodes[0] - 0.0) < 1e-10
        assert abs(result.weights[0] - 1.0) < 1e-10

    def test_alpha_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="between 1 and 16"):
            gaussian_quadrature((), (_frac(1, 1),))


# ---------------------------------------------------------------------------
# Wire adapter tests
# ---------------------------------------------------------------------------


class TestWireAdapters:
    def test_hankel_wire(self) -> None:
        request = HankelMatrixRequest(
            moments=tuple(_cr(1, k) for k in range(1, 8))
        )
        result = compute_hankel_matrix(request)
        assert result.dimension == 4
        assert isinstance(result, HankelMatrixResult)

    def test_recurrence_wire(self) -> None:
        request = RecurrenceCoefficientsRequest(
            moments=tuple(_cr(1, k) for k in range(1, 8))
        )
        result = compute_recurrence_coefficients(request)
        assert isinstance(result, RecurrenceCoefficientsResult)
        assert len(result.alpha) == 3

    def test_jacobi_wire(self) -> None:
        request = JacobiMatrixRequest(
            alpha=(_cr(1, 2), _cr(1, 2)),
            beta=(_cr(1, 1), _cr(1, 12), _cr(1, 15)),
        )
        result = compute_jacobi_matrix(request)
        assert isinstance(result, JacobiMatrixResult)
    def test_christoffel_darboux_wire(self) -> None:
        request = ChristoffelDarbouxRequest(
            alpha=(_cr(1, 2), _cr(1, 2)),
            beta=(_cr(1, 1), _cr(1, 12), _cr(1, 15)),
            x=_cr(1, 1),
            y=_cr(1, 1),
        )
        result = compute_christoffel_darboux(request)
        assert isinstance(result, ChristoffelDarbouxResult)

    def test_gaussian_quadrature_wire(self) -> None:
        request = GaussianQuadratureRequest(
            alpha=(_cr(1, 2), _cr(1, 2), _cr(1, 2)),
            beta=(_cr(1, 1), _cr(1, 12), _cr(1, 15), _cr(4, 45)),
        )
        result = compute_gaussian_quadrature(request)
        assert isinstance(result, GaussianQuadratureResult)
        assert len(result.nodes) == 3

    def test_hankel_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            HankelMatrixRequest(moments=())


# ---------------------------------------------------------------------------
# Tools and examples
# ---------------------------------------------------------------------------


class TestToolsAndExamples:
    def test_five_tools(self) -> None:
        assert len(TOOLS) == 5

    @pytest.mark.parametrize(
        "tool",
        TOOLS,
        ids=[t.operation_id for t in TOOLS],
    )
    def test_example_runs(self, tool) -> None:
        for ex in tool.examples:
            request = tool.request_type.model_validate(ex.input)
            result = tool.run(request)
            assert result is not None

    def test_all_examples_have_unique_names(self) -> None:
        for tool in TOOLS:
            names = [ex.name for ex in tool.examples]
            assert len(names) == len(set(names))
