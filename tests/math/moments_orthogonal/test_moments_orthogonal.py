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
    RecurrenceCoefficients,
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

    def test_negative_zeroth_moment_rejected(self) -> None:
        """A negative mu_0 is not a positive functional and seeds beta_0 < 0."""
        moments = (_frac(-1, 1), _frac(0, 1), _frac(1, 1))
        with pytest.raises(ValueError, match="positive"):
            recurrence_coefficients(moments)
        with pytest.raises(ValidationError):
            RecurrenceCoefficientsRequest(
                moments=(
                    CanonicalRational(num="-1", den="1"),
                    CanonicalRational(num="0", den="1"),
                    CanonicalRational(num="1", den="1"),
                )
            )

    @staticmethod
    def _overflowing_moments(count: int):
        """Positive functional whose exact coefficients overflow canonically."""
        moments = []
        for k in range(count):
            if k % 2 == 0:
                num = sum(x**k for x in range(-8, 9))
                moments.append(CanonicalRational(num=str(num), den="1"))
            else:
                moments.append(CanonicalRational(num="1", den=str(10**135 + 2 * k + 1)))
        return tuple(moments)

    def test_large_but_admissible_coefficients_pass(self) -> None:
        """Large-but-bounded moments stay admissible under the result limit."""
        moments = []
        for k in range(21):
            if k % 2 == 0:
                num = sum(x**k for x in range(-8, 9))
                moments.append(CanonicalRational(num=str(num), den="1"))
            else:
                moments.append(CanonicalRational(num="1", den=str(10**20 + 2 * k + 1)))
        request = RecurrenceCoefficientsRequest(moments=tuple(moments))
        assert len(request.moments) == 21

    @pytest.mark.exhaustive
    def test_result_growth_rejected_before_execution(self) -> None:
        """Positivity alone does not admit overflowing coefficients."""
        # Every input component stays small, but one exact recurrence
        # coefficient exceeds the 4,096-digit limit the result model applies.
        with pytest.raises((ValueError, ValidationError), match="result limit"):
            RecurrenceCoefficientsRequest(moments=self._overflowing_moments(33))

    def test_admission_matches_the_result_digit_contract(self) -> None:
        """Admission applies the same digit limit the result model enforces.

        These 33 moments pass the positivity replay and every derived
        coefficient fits the 32,768-digit canonical bound, yet the largest
        has ~6,233 digits and the result validator rejects anything beyond
        4,096; an admitted request therefore used to fail while constructing
        its typed result.
        """
        moments = []
        for k in range(33):
            if k % 2 == 0:
                num = sum(x**k for x in range(-8, 9))
                moments.append(CanonicalRational(num=str(num), den="1"))
            else:
                moments.append(CanonicalRational(num="1", den=str(10**25 + 2 * k + 1)))
        with pytest.raises((ValueError, ValidationError), match=r"4,?096"):
            RecurrenceCoefficientsRequest(moments=tuple(moments))

    def test_native_rejects_unverified_moment_tail(self) -> None:
        """Direct callers cannot smuggle an unverified tail past the cap."""
        moments = (*(_frac(1, k + 1) for k in range(33)), _frac(0, 1), _frac(-1, 1))
        with pytest.raises(ValueError, match="consumed by"):
            recurrence_coefficients(moments)
        # 33 moments remain admissible: exactly the consumed prefix.
        result = recurrence_coefficients(tuple(_frac(1, k + 1) for k in range(33)))
        assert len(result.alpha) == 16

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
        with pytest.raises(ValueError, match="positive"):
            jacobi_matrix((_frac(0, 1),), (_frac(0, 1), _frac(1, 1)))

    def test_negative_beta_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
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
        # alpha=(0,0), beta=(1,2) gives a Jacobi matrix with exact ±sqrt(2)
        alpha = (_frac(0, 1), _frac(0, 1))
        beta = (_frac(1, 1), _frac(2, 1))
        result = gaussian_quadrature(alpha, beta)
        # approximate nodes must be close to ±sqrt(2) within double precision
        import math

        nodes = sorted(float(v) for v in result.approximate_nodes)
        assert abs(nodes[0] + math.sqrt(2)) < 1e-12
        assert abs(nodes[1] - math.sqrt(2)) < 1e-12
        # They are declared approximate, not exact algebraic numbers
        assert result.approximate_nodes != ()

    def test_native_kernel_rejects_underflowed_beta_zero(self) -> None:
        """beta_0 below the underflow bound cannot silently lose the mass."""
        with pytest.raises(ValueError, match="underflow bound"):
            gaussian_quadrature((_frac(0, 1),), (_frac(1, 10**400),))

    def test_native_kernel_rejects_underflowed_subdiagonal(self) -> None:
        # A two-point Jacobi matrix uses beta[1] as its subdiagonal; below the
        # underflow bound its square root would not survive double conversion.
        with pytest.raises(ValueError, match="underflow bound"):
            gaussian_quadrature(
                (_frac(0, 1), _frac(0, 1)),
                (_frac(1, 1), _frac(1, 10**400)),
            )

    def test_native_kernel_rejects_magnitude_overflow(self) -> None:
        with pytest.raises(ValueError, match="finite-float magnitude"):
            gaussian_quadrature((_frac(0, 1),), (_frac(10**400, 1),))

    def test_native_kernel_matches_request_model_float_domain(self) -> None:
        """The wire request rejects the same payloads as the native kernel."""
        coefficients = RecurrenceCoefficients(
            alpha=(_cr(0, 1),),
            beta=(_cr(1, 1), _cr(1, 10**400)),
        )
        with pytest.raises(ValidationError):
            GaussianQuadratureRequest(coefficients=coefficients)
        # With one node the unused trailing entry still cannot underflow to
        # zero silently; kernel and model reject it through the same bound.
        with pytest.raises(ValueError, match="underflow"):
            gaussian_quadrature((_frac(0, 1),), (_frac(1, 1), _frac(1, 10**400)))


# ---------------------------------------------------------------------------
# Wire adapter tests
# ---------------------------------------------------------------------------


class TestWireAdapters:
    def test_hankel_wire(self) -> None:
        request = HankelMatrixRequest(moments=tuple(_cr(1, k) for k in range(1, 8)))
        result = compute_hankel_matrix(request)
        assert result.dimension == 4
        assert isinstance(result, HankelMatrixResult)

    def test_hankel_matrix_composes_with_canonical_matrix_consumers(self) -> None:
        """The producer field must feed canonical matrix consumers unchanged."""
        from jacobian.math.matrices._operation_models import (
            MatrixRankRequest,
            RationalMatrixRequest,
        )

        request = HankelMatrixRequest(moments=tuple(_cr(1, k) for k in range(1, 8)))
        result = compute_hankel_matrix(request)
        rank_request = MatrixRankRequest(matrix=result.matrix)
        assert rank_request.matrix.entries == result.matrix.entries
        rref_request = RationalMatrixRequest(matrix=result.matrix)
        assert rref_request.matrix.domain == "QQ"

    def test_recurrence_wire(self) -> None:
        request = RecurrenceCoefficientsRequest(
            moments=tuple(_cr(1, k) for k in range(1, 8))
        )
        result = compute_recurrence_coefficients(request)
        assert isinstance(result, RecurrenceCoefficientsResult)
        assert len(result.coefficients.alpha) == 3

    def test_jacobi_wire(self) -> None:
        request = JacobiMatrixRequest(
            coefficients=RecurrenceCoefficients(
                alpha=(_cr(1, 2), _cr(1, 2)),
                beta=(_cr(1, 1), _cr(1, 12), _cr(1, 15)),
            ),
        )
        result = compute_jacobi_matrix(request)
        assert isinstance(result, JacobiMatrixResult)

    def test_christoffel_darboux_wire(self) -> None:
        request = ChristoffelDarbouxRequest(
            coefficients=RecurrenceCoefficients(
                alpha=(_cr(1, 2), _cr(1, 2)),
                beta=(_cr(1, 1), _cr(1, 12), _cr(1, 15)),
            ),
            x=_cr(1, 1),
            y=_cr(1, 1),
        )
        result = compute_christoffel_darboux(request)
        assert isinstance(result, ChristoffelDarbouxResult)

    def test_gaussian_quadrature_wire(self) -> None:
        request = GaussianQuadratureRequest(
            coefficients=RecurrenceCoefficients(
                alpha=(_cr(1, 2), _cr(1, 2), _cr(1, 2)),
                beta=(_cr(1, 1), _cr(1, 12), _cr(1, 15), _cr(4, 45)),
            ),
        )
        result = compute_gaussian_quadrature(request)
        assert isinstance(result, GaussianQuadratureResult)
        assert len(result.approximate_nodes) == 3
        assert result.approximation == "IEEE_DOUBLE"
        assert result.method == "GOLUB_WELSCH_APPROXIMATE"

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


class TestCanonicalRecurrenceValueComposition:
    """The native producer's value composes into every consumer unchanged."""

    def test_producer_value_feeds_jacobi_matrix(self) -> None:
        coeffs = recurrence_coefficients(tuple(_frac(1, k + 1) for k in range(9)))
        result = jacobi_matrix(coeffs)
        assert result.diagonal == tuple(_frac(1, 2) for _ in result.diagonal)
        assert result.off_diagonal[0] > 0

    def test_producer_value_feeds_christoffel_darboux(self) -> None:
        coeffs = recurrence_coefficients(tuple(_frac(1, k + 1) for k in range(9)))
        separate = christoffel_darboux(
            coeffs.alpha, coeffs.beta, _frac(1, 3), _frac(1, 3)
        )
        canonical = christoffel_darboux(coeffs, _frac(1, 3), _frac(1, 3))
        assert canonical == separate

    def test_producer_value_feeds_gaussian_quadrature(self) -> None:
        coeffs = recurrence_coefficients(tuple(_frac(1, k + 1) for k in range(9)))
        separate = gaussian_quadrature(coeffs.alpha, coeffs.beta)
        assert gaussian_quadrature(coeffs).approximate_weights == (
            separate.approximate_weights
        )

    def test_mixed_forms_rejected(self) -> None:
        coeffs = recurrence_coefficients(tuple(_frac(1, k + 1) for k in range(5)))
        with pytest.raises(TypeError):
            jacobi_matrix(coeffs, coeffs.beta)
        with pytest.raises(TypeError):
            gaussian_quadrature(coeffs, coeffs.beta)
        with pytest.raises(TypeError):
            christoffel_darboux(
                coeffs,
                coeffs.beta,
                _frac(1, 3),
                _frac(1, 3),
            )
