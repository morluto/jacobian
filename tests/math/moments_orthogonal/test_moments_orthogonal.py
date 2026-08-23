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


def _cr_num(digits: int) -> CanonicalRational:
    """A maximal nines-only integer with the given digit count."""
    return CanonicalRational.from_integer_ratio(10**digits - 1, 1)


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
        with pytest.raises(ValueError, match="must be positive"):
            recurrence_coefficients(moments)

    def test_negative_short_sequence_rejected(self) -> None:
        """A one-moment sequence skips Gram-Schmidt but still needs mu_0 > 0."""
        with pytest.raises(ValueError, match="must be positive"):
            recurrence_coefficients((_frac(-1, 1),))
        with pytest.raises(ValueError, match="must be positive"):
            recurrence_coefficients((_frac(-1, 1), _frac(-1, 2)))

    def test_insufficient_moments_for_recurrence(self) -> None:
        """With only 2 moments we can't produce any recurrence coefficient."""
        moments = (_frac(1, 1), _frac(1, 2))
        result = recurrence_coefficients(moments)
        assert result.alpha == ()
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
        with pytest.raises(ValueError, match="must be positive"):
            jacobi_matrix((_frac(0, 1),), (_frac(0, 1), _frac(1, 1)))

    def test_negative_subdiagonal_rejected(self) -> None:
        """beta_1.. are squared subdiagonal entries; negatives cannot rebuild
        the claimed symmetric real Jacobi matrix."""
        with pytest.raises(ValueError, match="positive squared-norm ratios"):
            jacobi_matrix((_frac(0, 1), _frac(0, 1)), (_frac(1, 1), _frac(-1, 1)))

    def test_zero_subdiagonal_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive squared-norm ratios"):
            jacobi_matrix((_frac(0, 1), _frac(0, 1)), (_frac(1, 1), _frac(0, 1)))

    def test_unused_trailing_beta_not_required_positive(self) -> None:
        """alpha length bounds the used subdiagonal; unused tail is inert."""
        result = jacobi_matrix((_frac(0, 1),), (_frac(2, 1), _frac(-1, 1)))
        assert result.off_diagonal == ()


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

    def test_negative_beta0_rejected(self) -> None:
        """h_0 = beta_0 is a squared norm; the diagonal kernel cannot be -1."""
        with pytest.raises(ValueError, match="must be positive"):
            christoffel_darboux(
                (_frac(0, 1),), (_frac(-1, 1),), _frac(0, 1), _frac(0, 1)
            )

    def test_negative_subdiagonal_rejected(self) -> None:
        """Used beta_1.. advance the squared norms; negatives make them
        indefinite and cannot define an orthogonal family."""
        with pytest.raises(ValueError, match="positive squared-norm ratios"):
            christoffel_darboux(
                (_frac(0, 1), _frac(0, 1)),
                (_frac(1, 1), _frac(-1, 1)),
                _frac(0, 1),
                _frac(0, 1),
            )

    def test_zero_subdiagonal_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive squared-norm ratios"):
            christoffel_darboux(
                (_frac(0, 1), _frac(0, 1)),
                (_frac(1, 1), _frac(0, 1)),
                _frac(0, 1),
                _frac(0, 1),
            )

    def test_unused_trailing_beta_not_required_positive(self) -> None:
        """alpha length bounds the used subdiagonal; unused tail is inert."""
        result = christoffel_darboux(
            (_frac(0, 1),), (_frac(2, 1), _frac(-1, 1)), _frac(1, 2), _frac(1, 2)
        )
        assert result.polynomials_evaluated == (_frac(1, 1),)


class TestChristoffelDarbouxJointBound:
    """The exact kernel and every reported polynomial must stay canonical."""

    def test_reviewer_counterexample_rejected_at_request(self) -> None:
        """Order 16 with 4096-digit x and y would overflow the canonical limit."""
        alpha = tuple(_cr(0, 1) for _ in range(16))
        beta = tuple(_cr(1, 1) for _ in range(16))
        x = CanonicalRational.from_integer_ratio(10**4095, 1)
        with pytest.raises(ValidationError, match="32768-digit"):
            ChristoffelDarbouxRequest(alpha=alpha, beta=beta, x=x, y=x)

    def test_growth_bound_rejects_larger_orders_and_heights(self) -> None:
        from fractions import Fraction

        from jacobian.math.moments_orthogonal._models import (
            _require_bounded_kernel_growth,
        )

        def probe(order: int, digits: int) -> bool:
            alpha = tuple(Fraction(10**digits - 1) for _ in range(order))
            beta = tuple(Fraction(10 ** (digits - 1) + 7, 3) for _ in range(order))
            point = Fraction(10**digits - 1)
            try:
                _require_bounded_kernel_growth(alpha, beta, point, point)
            except ValueError:
                return False
            return True

        assert probe(16, 72)
        assert not probe(16, 73)
        assert not probe(16, 4095)

    def test_maximal_admitted_boundary_executes(self) -> None:
        """Order 16 at the admitted digit height executes and round-trips."""
        alpha = tuple(_cr_num(16) for _ in range(16))
        beta = tuple(
            CanonicalRational.from_integer_ratio(10**15 + 7, 3) for _ in range(16)
        )
        x = _cr_num(16)
        request = ChristoffelDarbouxRequest(alpha=alpha, beta=beta, x=x, y=x)
        result = compute_christoffel_darboux(request)
        assert len(result.polynomials_evaluated) == 16
        replayed = ChristoffelDarbouxResult.model_validate(result.model_dump())
        assert replayed == result

    def test_low_order_admits_taller_inputs(self) -> None:
        """A single recurrence step tolerates far taller evaluation points."""
        alpha = (_cr_num(600),)
        beta = (CanonicalRational.from_integer_ratio(10**599 + 3, 7),)
        request = ChristoffelDarbouxRequest(
            alpha=alpha, beta=beta, x=_cr_num(600), y=_cr_num(600)
        )
        result = compute_christoffel_darboux(request)
        assert isinstance(result, ChristoffelDarbouxResult)


# ---------------------------------------------------------------------------
# Gaussian quadrature
# ---------------------------------------------------------------------------


class TestGaussianQuadrature:
    def test_weights_sum_to_mu0(self) -> None:
        """Sum of approximate weights equals mu_0 (the zeroth moment)."""
        alpha = (_frac(1, 2), _frac(1, 2), _frac(1, 2))
        beta = (_frac(1, 1), _frac(1, 12), _frac(1, 15), _frac(4, 45))
        result = gaussian_quadrature(alpha, beta)
        mu0 = float(beta[0])
        assert abs(sum(result.approximate_weights) - mu0) < 1e-10

    def test_nodes_count(self) -> None:
        alpha = (_frac(1, 2), _frac(1, 2), _frac(1, 2))
        beta = (_frac(1, 1), _frac(1, 12), _frac(1, 15), _frac(4, 45))
        result = gaussian_quadrature(alpha, beta)
        assert len(result.approximate_nodes) == 3
        assert len(result.approximate_weights) == 3

    def test_single_point(self) -> None:
        alpha = (_frac(0, 1),)
        beta = (_frac(1, 1), _frac(1, 3))
        result = gaussian_quadrature(alpha, beta)
        assert len(result.approximate_nodes) == 1
        assert abs(result.approximate_nodes[0] - 0.0) < 1e-10
        assert abs(result.approximate_weights[0] - 1.0) < 1e-10

    def test_alpha_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="between 1 and 16"):
            gaussian_quadrature((), (_frac(1, 1),))

    def test_approximate_irrational_nodes(self) -> None:
        """Nodes ±sqrt(2) are irrational and returned as IEEE-double approximations."""
        # alpha=(0,0) and beta=(1,2) has exact nodes ±sqrt(2)
        alpha2 = (_frac(0, 1), _frac(0, 1))
        beta2 = (_frac(1, 1), _frac(2, 1))
        result = gaussian_quadrature(alpha2, beta2)
        # approximate nodes must be close to ±sqrt(2) within double precision
        import math

        nodes = sorted(float(v) for v in result.approximate_nodes)
        assert abs(nodes[0] + math.sqrt(2)) < 1e-12
        assert abs(nodes[1] - math.sqrt(2)) < 1e-12
        # They are declared approximate, not exact algebraic numbers
        assert result.approximate_nodes != ()


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
        assert len(result.approximate_nodes) == 3
        assert result.approximation == "IEEE_DOUBLE"
        assert result.method == "GOLUB_WELSCH_APPROXIMATE"

    def test_hankel_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            HankelMatrixRequest(moments=())

    def test_recurrence_wire_rejects_negative_short_sequence(self) -> None:
        """mu_0 < 0 with fewer than three moments must fail at this boundary."""
        with pytest.raises(ValidationError, match="must be positive"):
            RecurrenceCoefficientsRequest(moments=(_cr(-1, 1),))
        with pytest.raises(ValidationError, match="must be positive"):
            RecurrenceCoefficientsRequest(moments=(_cr(-1, 1), _cr(-1, 2)))

    def test_output_beyond_canonical_limit_rejected_at_admission(self) -> None:
        """Positive-definite moments whose Gram-Schmidt alpha overflows.

        mu_k = 1/(k+1) + 1/(10^300+2k+1) for k = 0..16 is positive definite
        with every component near the admitted input height, but the final
        exact alpha coefficient has ~34,000 digits. Admission must reject it
        instead of letting construction of RecurrenceCoefficientsResult fail
        after the request was accepted.
        """
        big = 10**300
        moments = tuple(
            _cr((big + 2 * k + 1) + (k + 1), (k + 1) * (big + 2 * k + 1))
            for k in range(17)
        )
        with pytest.raises(ValidationError, match="canonical"):
            RecurrenceCoefficientsRequest(moments=moments)

    def test_moment_heights_at_admitted_bound_succeed(self) -> None:
        """A small positive-definite sequence inside every bound computes."""
        from jacobian.math.moments_orthogonal._operations import (
            compute_recurrence_coefficients,
        )

        request = RecurrenceCoefficientsRequest(
            moments=(_cr(1, 1), _cr(1, 2), _cr(1, 3), _cr(1, 4))
        )
        result = compute_recurrence_coefficients(request)
        assert len(result.alpha) == 1


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
