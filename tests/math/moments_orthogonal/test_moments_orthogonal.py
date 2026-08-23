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
        with pytest.raises(ValueError, match="between 1 and 32"):
            recurrence_coefficients(())

    def test_zeroth_moment_nonzero(self) -> None:
        moments = (_frac(0, 1), _frac(1, 1), _frac(1, 2))
        with pytest.raises(ValueError, match="nonzero"):
            recurrence_coefficients(moments)

    def test_two_moments_determine_one_pair(self) -> None:
        """Two moments fully determine alpha_0 and beta_0."""
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

    def test_nonpositive_beta_rejected(self) -> None:
        """beta_0 is the zeroth moment of a positive functional."""
        with pytest.raises(ValueError, match=r"zeroth moment.*positive"):
            jacobi_matrix((_frac(0, 1),), (_frac(0, 1), _frac(1, 1)))
        with pytest.raises(ValueError, match=r"zeroth moment.*positive"):
            jacobi_matrix((_frac(0, 1),), (_frac(-2, 1), _frac(1, 1)))


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

    def test_nonpositive_beta_zero_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="beta_0"):
            christoffel_darboux((), (_frac(-2, 1),), _frac(2, 1), _frac(2, 1))
        with pytest.raises(ValueError, match="beta_0"):
            christoffel_darboux((), (_frac(0, 1),), _frac(2, 1), _frac(2, 1))

    def test_nonpositive_subdiagonal_beta_is_rejected(self) -> None:
        """A kernel from a positive-definite functional needs positive h_k."""
        with pytest.raises(ValueError, match="squared-norm ratios must be positive"):
            christoffel_darboux(
                (_frac(0, 1), _frac(1, 3)),
                (_frac(2, 1), _frac(-1, 9)),
                _frac(1, 4),
                _frac(1, 4),
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

    def test_negative_mass_rejected(self) -> None:
        """beta_0 is the zeroth moment; a negative mass admits no quadrature rule."""
        with pytest.raises(ValueError, match=r"zeroth moment.*positive"):
            gaussian_quadrature((Fraction(0),), (Fraction(-1),))

    def test_zero_mass_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"zeroth moment.*positive"):
            gaussian_quadrature((Fraction(0),), (Fraction(0),))

    def test_oversized_exact_coefficients_are_rejected_before_float_conversion(
        self,
    ) -> None:
        """Exact inputs outside the finite-double domain are a validation error."""
        with pytest.raises(ValueError, match="finite-float magnitude bound"):
            gaussian_quadrature((Fraction(10**1000),), (Fraction(1),))
        with pytest.raises(ValueError, match="finite-float magnitude bound"):
            gaussian_quadrature((Fraction(0),), (Fraction(10**500), Fraction(1)))

    def test_tiny_subdiagonal_is_rejected_by_the_underflow_bound(self) -> None:
        alpha = (Fraction(0), Fraction(0))
        beta = (Fraction(1), Fraction(1, 10**400))
        with pytest.raises(ValueError, match="underflow bound"):
            gaussian_quadrature(alpha, beta)

    def test_interior_zero_subdiagonal_is_rejected(self) -> None:
        """The wire contract requires every used squared-norm ratio to be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            gaussian_quadrature((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)))

    def test_bound_magnitude_boundary_coefficient_is_admitted(self) -> None:
        """A coefficient at the admitted magnitude converts and returns nodes."""
        result = gaussian_quadrature((Fraction(10**299),), (Fraction(1),))
        assert len(result.nodes) == 1
        assert len(result.weights) == 1


# ---------------------------------------------------------------------------
# Wire adapter tests
# ---------------------------------------------------------------------------


class TestWireAdapters:
    def test_hankel_wire(self) -> None:
        request = HankelMatrixRequest(moments=tuple(_cr(1, k) for k in range(1, 8)))
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


class TestRecurrenceDerivedBudget:
    @pytest.mark.exhaustive
    def test_request_whose_derived_coefficients_overflow_is_rejected(self) -> None:
        """Admission rejects positive sequences whose Gram-Schmidt output
        cannot be carried by the canonical rational result type."""
        import random

        denominator = 10**1250 + 12345
        rng = random.Random(1)
        weights = [
            Fraction(rng.randrange(10**1149, 10**1150), denominator) for _ in range(16)
        ]
        moments = tuple(
            sum(w * Fraction(n**k) for n, w in enumerate(weights)) for k in range(31)
        )
        request = {
            "moments": [
                CanonicalRational.from_fraction(m).model_dump() for m in moments
            ]
        }
        with pytest.raises(ValidationError, match="canonical"):
            RecurrenceCoefficientsRequest.model_validate(request)


class TestChristoffelDarbouxDerivedBudget:
    def _request(self, exponent: int) -> dict:
        alpha = [{"num": "0", "den": "1"} for _ in range(9)]
        beta = [{"num": "1", "den": "1"} for _ in range(10)]
        huge = str(10**exponent)
        return {
            "alpha": alpha,
            "beta": beta,
            "x": {"num": huge, "den": "1"},
            "y": {"num": huge, "den": "1"},
        }

    def test_request_whose_derived_kernel_overflows_is_rejected(self) -> None:
        """Recurrence amplification past the canonical bound fails at admission.

        Nine zero alphas and unit betas give p_k(x) ~ x^k, so K_9(x, x) has a
        leading term near 10^(4095*16) ~ 65,521 digits: each input fits the
        4,096-digit input bound but the kernel exceeds the 32,768-digit result
        bound, so the request must never reach execution.
        """
        with pytest.raises(ValidationError, match="canonical"):
            ChristoffelDarbouxRequest.model_validate(self._request(4095))

    def test_request_whose_derived_kernel_fits_is_admitted_and_returns(self) -> None:
        """The same family one magnitude smaller returns its typed kernel."""
        request = ChristoffelDarbouxRequest.model_validate(self._request(1000))
        result = compute_christoffel_darboux(request)
        assert result.kernel.num.startswith("-") is False
        assert len(result.polynomials_evaluated) == 9


class TestRecurrenceOrderAdmissionBound:
    def test_sequence_beyond_the_computed_recurrence_order_is_rejected(self) -> None:
        """35 harmonic moments determine order 17 but the cap computes 16.

        Admission must reject sequences longer than the 2 * MAX + 1 moments
        the maximum supported recurrence order consumes instead of returning
        complete=True while silently ignoring trailing moments.
        """
        request = {
            "moments": [
                CanonicalRational.from_fraction(Fraction(1, k + 1)).model_dump()
                for k in range(35)
            ]
        }
        with pytest.raises(ValidationError, match="maximum supported recurrence order"):
            RecurrenceCoefficientsRequest.model_validate(request)

    def test_boundary_sequence_of_thirty_two_moments_is_fully_consumed(self) -> None:
        request = RecurrenceCoefficientsRequest(
            moments=tuple(_cr(1, k + 1) for k in range(32))
        )
        result = compute_recurrence_coefficients(request)
        assert len(result.alpha) == 16
        assert len(result.beta) == 16

    def test_sequence_longer_than_the_maximum_consumed_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="consumed by the maximum"):
            RecurrenceCoefficientsRequest(
                moments=tuple(_cr(1, k + 1) for k in range(33))
            )


class TestRecurrenceFullConsumption:
    """Every retained moment must determine the returned coefficients."""

    def test_even_length_fourth_moment_determines_alpha_one(self) -> None:
        """(1,0,1,0) and (1,0,1,10) share the first three moments but their
        fourth moment determines alpha_1; neither may be ignored."""
        first = recurrence_coefficients(
            (_frac(1, 1), _frac(0, 1), _frac(1, 1), _frac(0, 1))
        )
        second = recurrence_coefficients(
            (_frac(1, 1), _frac(0, 1), _frac(1, 1), _frac(10, 1))
        )
        assert first.alpha == (_frac(0, 1), _frac(0, 1))
        assert second.alpha == (_frac(0, 1), _frac(10, 1))
        assert first.beta == (_frac(1, 1), _frac(1, 1))
        assert second.beta == (_frac(1, 1), _frac(1, 1))

    def test_odd_length_trailing_moment_determines_final_beta(self) -> None:
        """mu_2 distinguishes (1,0,1) from (1,0,2) through beta_1."""
        first = recurrence_coefficients((_frac(1, 1), _frac(0, 1), _frac(1, 1)))
        second = recurrence_coefficients((_frac(1, 1), _frac(0, 1), _frac(2, 1)))
        assert first.alpha == (_frac(0, 1),)
        assert second.alpha == (_frac(0, 1),)
        assert first.beta == (_frac(1, 1), _frac(1, 1))
        assert second.beta == (_frac(1, 1), _frac(2, 1))

    def test_odd_length_indefinite_trailing_hankel_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive-definite"):
            recurrence_coefficients((_frac(1, 1), _frac(0, 1), _frac(-1, 1)))


class TestQuadratureExactCoefficients:
    """The native quadrature kernel requires exact Fraction inputs."""

    def test_float_coefficients_rejected(self) -> None:
        with pytest.raises(TypeError, match="exact Fractions"):
            gaussian_quadrature((0.0,), (1.0,))

    def test_non_finite_floats_rejected(self) -> None:
        import math

        with pytest.raises(TypeError, match="exact Fractions"):
            gaussian_quadrature((math.inf,), (1.0,))
