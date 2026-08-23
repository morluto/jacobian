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
    RecurrenceCoefficientsValue,
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
        with pytest.raises(ValueError, match="nonzero"):
            recurrence_coefficients(moments)

    def test_two_moments_determine_first_coefficient(self) -> None:
        """mu=(1, 1/2) already determines alpha_0 = mu_1/mu_0 = 1/2; the
        subdiagonal stays at beta_0 = mu_0 because h_1 is not determined."""
        moments = (_frac(1, 1), _frac(1, 2))
        result = recurrence_coefficients(moments)
        assert result.alpha == (_frac(1, 2),)
        assert result.beta == (_frac(1, 1),)

    def test_even_length_sequence_yields_trailing_alpha(self) -> None:
        """mu=(1,1,2,6) determines alpha=(1,3) and beta=(1,1): the final
        shift coefficient consumes only the last available moment."""
        moments = (_frac(1, 1), _frac(1, 1), _frac(2, 1), _frac(6, 1))
        result = recurrence_coefficients(moments)
        assert result.alpha == (_frac(1, 1), _frac(3, 1))
        assert result.beta == (_frac(1, 1), _frac(1, 1))


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

    def test_negative_subdiagonal_beta_rejected(self) -> None:
        """A negative used beta would need sqrt(-1) on the Jacobi
        subdiagonal; the native boundary rejects it like the wire request."""
        with pytest.raises(ValueError, match="subdiagonal beta entries"):
            jacobi_matrix(
                (_frac(0, 1), _frac(0, 1)),
                (_frac(1, 1), _frac(-1, 1)),
            )

    def test_zero_subdiagonal_beta_rejected(self) -> None:
        with pytest.raises(ValueError, match="subdiagonal beta entries"):
            jacobi_matrix(
                (_frac(0, 1), _frac(0, 1)),
                (_frac(1, 1), _frac(0, 1)),
            )

    def test_unused_negative_beta_allowed(self) -> None:
        """beta entries beyond the assembled subdiagonal are not used and so
        are not constrained."""
        result = jacobi_matrix((_frac(0, 1),), (_frac(1, 1), _frac(-1, 1)))
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
        assert len(result.coefficients.alpha) == 3

    def test_jacobi_wire(self) -> None:
        request = JacobiMatrixRequest(
            coefficients=RecurrenceCoefficientsValue(
                alpha=(_cr(1, 2), _cr(1, 2)),
                beta=(_cr(1, 1), _cr(1, 12), _cr(1, 15)),
            )
        )
        result = compute_jacobi_matrix(request)
        assert isinstance(result, JacobiMatrixResult)

    def test_christoffel_darboux_wire(self) -> None:
        request = ChristoffelDarbouxRequest(
            coefficients=RecurrenceCoefficientsValue(
                alpha=(_cr(1, 2), _cr(1, 2)),
                beta=(_cr(1, 1), _cr(1, 12), _cr(1, 15)),
            ),
            x=_cr(1, 1),
            y=_cr(1, 1),
        )
        result = compute_christoffel_darboux(request)
        assert isinstance(result, ChristoffelDarbouxResult)

    def test_christoffel_darboux_height_growth_rejected(self) -> None:
        """Sixteen unrelated 501-digit coefficient denominators compound in
        the exact kernel past the canonical result limit; the request must
        be rejected at admission instead of raising during conversion."""

        def tall(i: int) -> CanonicalRational:
            return CanonicalRational(num="1", den=str(10**500 + 2 * i + 1))

        with pytest.raises(ValidationError):
            ChristoffelDarbouxRequest(
                coefficients=RecurrenceCoefficientsValue(
                    alpha=tuple(tall(i) for i in range(16)),
                    beta=tuple(tall(16 + i) for i in range(16)),
                ),
                x=tall(32),
                y=tall(33),
            )

    def test_christoffel_darboux_representable_growth_accepted(self) -> None:
        """A request whose exact kernel stays representable is admitted even
        when individual inputs carry hundreds of digits."""

        def small(i: int) -> CanonicalRational:
            return CanonicalRational(num="1", den=str(10**40 + 2 * i + 1))

        request = ChristoffelDarbouxRequest(
            coefficients=RecurrenceCoefficientsValue(
                alpha=tuple(small(i) for i in range(16)),
                beta=tuple(small(16 + i) for i in range(16)),
            ),
            x=small(32),
            y=small(33),
        )
        result = compute_christoffel_darboux(request)
        assert isinstance(result, ChristoffelDarbouxResult)

    def test_recurrence_coefficient_height_rejected(self) -> None:
        """Nine positive-definite moments with 3601-digit denominators grow
        ~97,000-digit recurrence coefficients in the exact Gram-Schmidt
        kernel; the request must be rejected at admission instead of raising
        during result conversion."""

        def moment(k: int) -> CanonicalRational:
            q = 10**3600 + 2 * k + 1
            value = Fraction(1, k + 1) + Fraction(1, q)
            return CanonicalRational.from_fraction(value)

        with pytest.raises(ValidationError):
            RecurrenceCoefficientsRequest(moments=tuple(moment(k) for k in range(9)))

    def test_recurrence_representable_moments_accepted(self) -> None:
        """A positive-definite sequence whose exact coefficients stay inside
        the canonical limit is admitted and computes end-to-end."""
        request = RecurrenceCoefficientsRequest(
            moments=tuple(_cr(1, k) for k in range(1, 8))
        )
        result = compute_recurrence_coefficients(request)
        assert isinstance(result, RecurrenceCoefficientsResult)
        assert len(result.coefficients.alpha) == 3

    def test_gaussian_quadrature_wire(self) -> None:
        request = GaussianQuadratureRequest(
            coefficients=RecurrenceCoefficientsValue(
                alpha=(_cr(1, 2), _cr(1, 2), _cr(1, 2)),
                beta=(_cr(1, 1), _cr(1, 12), _cr(1, 15), _cr(4, 45)),
            )
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


class TestOddPrefixTrailingCoefficient:
    """Every retained moment must determine the returned coefficients."""

    def test_odd_length_trailing_moment_determines_final_beta(self) -> None:
        """mu_2 distinguishes (1,0,1) from (1,0,2) through beta_1."""
        first = recurrence_coefficients((_frac(1, 1), _frac(0, 1), _frac(1, 1)))
        second = recurrence_coefficients((_frac(1, 1), _frac(0, 1), _frac(2, 1)))
        assert first.alpha == (_frac(0, 1),)
        assert second.alpha == (_frac(0, 1),)
        assert first.beta == (_frac(1, 1), _frac(1, 1))
        assert second.beta == (_frac(1, 1), _frac(2, 1))

    def test_odd_length_indefinite_trailing_hankel_rejected(self) -> None:
        """(1, 0, -1) has an indefinite trailing Hankel minor."""
        with pytest.raises(ValueError, match="positive-definite"):
            recurrence_coefficients((_frac(1, 1), _frac(0, 1), _frac(-1, 1)))


class TestCanonicalCoefficientsComposition:
    """The serialized recurrence value feeds every consumer unchanged."""

    def test_result_coefficients_feed_all_consumers(self) -> None:
        request = RecurrenceCoefficientsRequest(
            moments=tuple(_cr(2, k) for k in range(1, 8))
        )
        produced = compute_recurrence_coefficients(request)
        value = produced.coefficients

        jacobi = compute_jacobi_matrix(JacobiMatrixRequest(coefficients=value))
        assert isinstance(jacobi, JacobiMatrixResult)

        cd = compute_christoffel_darboux(
            ChristoffelDarbouxRequest(
                coefficients=value,
                x=_cr(1, 1),
                y=_cr(1, 1),
            )
        )
        assert isinstance(cd, ChristoffelDarbouxResult)

        quadrature = compute_gaussian_quadrature(
            GaussianQuadratureRequest(coefficients=value)
        )
        assert isinstance(quadrature, GaussianQuadratureResult)
