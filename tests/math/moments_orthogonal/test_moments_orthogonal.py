"""Known-answer and adversarial tests for moments and orthogonal polynomials."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.values import RationalMatrix
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
from jacobian.math.moments_orthogonal.values import RecurrenceCoefficients

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frac(num: int, den: int) -> Fraction:
    return Fraction(num, den)


def _cr(num: int, den: int) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(num, den)


_HARMONIC_MOMENTS = tuple(_cr(1, k) for k in range(1, 8))

_LEGENDRE_ALPHA = (_cr(1, 2), _cr(1, 2))
_LEGENDRE_BETA = (_cr(1, 1), _cr(1, 12), _cr(1, 15))


def _legendre_coefficients() -> RecurrenceCoefficients:
    return RecurrenceCoefficients(alpha=_LEGENDRE_ALPHA, beta=_LEGENDRE_BETA)


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
        assert all(a.as_fraction() == _frac(1, 2) for a in result.alpha)

    def test_beta_zero_is_mu0(self) -> None:
        moments = (_frac(2, 1), _frac(1, 1), _frac(2, 3))
        result = recurrence_coefficients(moments)
        assert result.beta[0].as_fraction() == _frac(2, 1)

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="between 1 and 64"):
            recurrence_coefficients(())

    def test_zeroth_moment_nonzero(self) -> None:
        moments = (_frac(0, 1), _frac(1, 1), _frac(1, 2))
        with pytest.raises(ValueError, match="positive"):
            recurrence_coefficients(moments)

    def test_negative_singleton_moments_rejected(self) -> None:
        """A non-positive zeroth moment cannot seed a positive functional."""
        with pytest.raises(ValueError, match="positive"):
            recurrence_coefficients((_frac(-1, 1),))

    def test_insufficient_moments_for_recurrence(self) -> None:
        """A single moment admits no recurrence coefficient; an even prefix
        is rejected because its final moment would determine alpha while
        the matching norm ratio stays undetermined."""
        result = recurrence_coefficients((_frac(1, 1),))
        assert result.alpha == ()
        assert [b.as_fraction() for b in result.beta] == [_frac(1, 1)]
        with pytest.raises(ValueError, match="odd length"):
            recurrence_coefficients((_frac(1, 1), _frac(1, 2)))

    def test_even_prefix_final_moment_is_never_ignored(self) -> None:
        """Every shorter even prefix is rejected rather than returning the
        same result as its odd truncation: (1, 0) and (1, 1) determine
        different alpha_0 values (0 vs 1), so neither may be reported as
        complete=True with alpha=(), beta=(1,)."""
        for moments in (
            (_frac(1, 1), _frac(0, 1)),
            (_frac(1, 1), _frac(1, 1)),
            (_frac(2, 1), _frac(1, 1), _frac(2, 3), _frac(1, 2)),
        ):
            with pytest.raises(ValueError, match="odd length"):
                recurrence_coefficients(moments)

    def test_wire_request_rejects_even_length_moments(self) -> None:
        """The wire request replays the native kernel's odd-length rule so
        an accepted request cannot fail during execution."""
        with pytest.raises(ValidationError, match="odd length"):
            RecurrenceCoefficientsRequest(moments=(_cr(1, 1), _cr(0, 1)))


# ---------------------------------------------------------------------------
# Jacobi matrix
# ---------------------------------------------------------------------------


class TestJacobiMatrix:
    def test_assembly(self) -> None:
        result = jacobi_matrix(_legendre_coefficients())
        assert result.diagonal == (_frac(1, 2), _frac(1, 2))
        # beta_0 is the zeroth moment; the subdiagonal carries beta_1 only.
        assert result.off_diagonal == (_frac(1, 12),)

    def test_empty_alpha(self) -> None:
        coefficients = RecurrenceCoefficients(alpha=(), beta=(_cr(1, 1),))
        result = jacobi_matrix(coefficients)
        assert result.diagonal == ()
        assert result.off_diagonal == ()

    def test_zero_beta0_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            RecurrenceCoefficients(alpha=(_cr(0, 1),), beta=(_cr(0, 1), _cr(1, 1)))

    def test_negative_beta0_rejected(self) -> None:
        """The canonical value matches the wire contract: mu_0 must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            RecurrenceCoefficients(alpha=(_cr(0, 1),), beta=(_cr(-1, 1),))


# ---------------------------------------------------------------------------
# Christoffel-Darboux kernel
# ---------------------------------------------------------------------------


class TestChristoffelDarboux:
    def test_diagonal_kernel(self) -> None:
        """K_n(x,x) = sum_{k=0}^{n-1} p_k(x)^2 / h_k is positive."""
        result = christoffel_darboux(_legendre_coefficients(), _frac(1, 2), _frac(1, 2))
        assert result.kernel > 0

    def test_zero_when_x_neq_y_symmetric(self) -> None:
        """K_n(x,y) is a reproducing kernel: K_n(x,y) = K_n(y,x)."""
        coefficients = RecurrenceCoefficients(
            alpha=(_cr(0, 1), _cr(1, 3)),
            beta=(_cr(2, 1), _cr(1, 9), _cr(4, 45)),
        )
        result_xy = christoffel_darboux(coefficients, _frac(1, 4), _frac(3, 4))
        result_yx = christoffel_darboux(coefficients, _frac(3, 4), _frac(1, 4))
        assert result_xy.kernel == result_yx.kernel

    def test_first_polynomial_is_one(self) -> None:
        coefficients = RecurrenceCoefficients(
            alpha=(_cr(0, 1), _cr(1, 3)),
            beta=(_cr(2, 1), _cr(1, 9), _cr(4, 45)),
        )
        result = christoffel_darboux(coefficients, _frac(1, 1), _frac(1, 1))
        assert result.polynomials_evaluated[0] == _frac(1, 1)

    def test_empty_alpha(self) -> None:
        coefficients = RecurrenceCoefficients(alpha=(), beta=(_cr(1, 1),))
        result = christoffel_darboux(coefficients, _frac(1, 1), _frac(1, 1))
        assert result.kernel == _frac(0, 1)
        assert result.polynomials_evaluated == (_frac(1, 1),)

    def test_nonpositive_beta_rejected_at_the_value_boundary(self) -> None:
        """The canonical value enforces the wire contract: mu_0 and every
        norm ratio must be positive before any consumer runs."""
        with pytest.raises(ValidationError, match="must be positive"):
            RecurrenceCoefficients(
                alpha=(_cr(0, 1), _cr(0, 1)), beta=(_cr(-1, 1), _cr(1, 2))
            )
        with pytest.raises(ValidationError, match="squared-norm"):
            RecurrenceCoefficients(
                alpha=(_cr(0, 1), _cr(0, 1)), beta=(_cr(1, 1), _cr(-1, 2))
            )


# ---------------------------------------------------------------------------
# Gaussian quadrature
# ---------------------------------------------------------------------------


class TestGaussianQuadrature:
    def test_weights_sum_to_mu0(self) -> None:
        """Sum of weights equals mu_0 (the zeroth moment)."""
        coefficients = RecurrenceCoefficients(
            alpha=(_cr(1, 2), _cr(1, 2), _cr(1, 2)),
            beta=(_cr(1, 1), _cr(1, 12), _cr(1, 15), _cr(4, 45)),
        )
        result = gaussian_quadrature(coefficients)
        mu0 = float(coefficients.beta[0].as_fraction())
        assert abs(sum(result.weights) - mu0) < 1e-10

    def test_nodes_count(self) -> None:
        coefficients = RecurrenceCoefficients(
            alpha=(_cr(1, 2), _cr(1, 2), _cr(1, 2)),
            beta=(_cr(1, 1), _cr(1, 12), _cr(1, 15), _cr(4, 45)),
        )
        result = gaussian_quadrature(coefficients)
        assert len(result.nodes) == 3
        assert len(result.weights) == 3

    def test_single_point(self) -> None:
        coefficients = RecurrenceCoefficients(
            alpha=(_cr(0, 1),), beta=(_cr(1, 1), _cr(1, 3))
        )
        result = gaussian_quadrature(coefficients)
        assert len(result.nodes) == 1
        assert abs(result.nodes[0] - 0.0) < 1e-10
        assert abs(result.weights[0] - 1.0) < 1e-10

    def test_alpha_empty_rejected(self) -> None:
        coefficients = RecurrenceCoefficients(alpha=(), beta=(_cr(1, 1),))
        with pytest.raises(ValueError, match="between 1 and 16"):
            gaussian_quadrature(coefficients)

    def test_zero_subdiagonal_rejected_at_the_value_boundary(self) -> None:
        """The canonical value rejects zero subdiagonals like the wire contract."""
        with pytest.raises(ValidationError, match="squared-norm"):
            RecurrenceCoefficients(
                alpha=(_cr(0, 1), _cr(1, 1)), beta=(_cr(1, 1), _cr(0, 1))
            )


# ---------------------------------------------------------------------------
# Wire adapter tests
# ---------------------------------------------------------------------------


class TestWireAdapters:
    def test_recurrence_trailing_beta_must_be_positive(self) -> None:
        """A partial recurrence (len(alpha) == len(beta) - 1) still validates
        every subdiagonal entry, including the trailing beta."""
        with pytest.raises(ValidationError, match="squared-norm"):
            RecurrenceCoefficients(alpha=(_cr(0, 1),), beta=(_cr(1, 1), _cr(-1, 1)))
        with pytest.raises(ValidationError, match="squared-norm"):
            RecurrenceCoefficients(
                alpha=(_cr(0, 1), _cr(0, 1)), beta=(_cr(1, 1), _cr(-1, 1))
            )
        accepted = RecurrenceCoefficients(
            alpha=(_cr(0, 1),), beta=(_cr(1, 1), _cr(1, 2))
        )
        assert accepted.beta[-1] == _cr(1, 2)

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
        request = JacobiMatrixRequest(coefficients=_legendre_coefficients())
        result = compute_jacobi_matrix(request)
        assert isinstance(result, JacobiMatrixResult)

    def test_christoffel_darboux_wire(self) -> None:
        request = ChristoffelDarbouxRequest(
            coefficients=_legendre_coefficients(),
            x=_cr(1, 1),
            y=_cr(1, 1),
        )
        result = compute_christoffel_darboux(request)
        assert isinstance(result, ChristoffelDarbouxResult)

    def test_kernel_growth_bound_rejects_unrepresentable_output(self) -> None:
        """Sixteen recurrence steps at a huge evaluation point would drive the
        exact kernel past the canonical rational digit limit; admission must
        reject the request instead of failing during execution."""
        alpha = tuple(_cr(0, 1) for _ in range(16))
        beta = tuple(_cr(1, 1) for _ in range(16))
        huge = CanonicalRational.from_fraction(Fraction(1, 10**4095))
        with pytest.raises(ValidationError, match="order-and-height"):
            ChristoffelDarbouxRequest(
                coefficients=RecurrenceCoefficients(alpha=alpha, beta=beta),
                x=huge,
                y=huge,
            )

    def test_kernel_growth_bound_admits_moderate_inputs(self) -> None:
        alpha = tuple(_cr(0, 1) for _ in range(4))
        beta = tuple(_cr(1, k + 1) for k in range(5))
        x = CanonicalRational.from_fraction(Fraction(10**80 + 1, 10**79 - 3))
        request = ChristoffelDarbouxRequest(
            coefficients=RecurrenceCoefficients(alpha=alpha, beta=beta),
            x=x,
            y=x,
        )
        assert compute_christoffel_darboux(request).kernel

    def test_gaussian_quadrature_wire(self) -> None:
        request = GaussianQuadratureRequest(
            coefficients=RecurrenceCoefficients(
                alpha=(_cr(1, 2), _cr(1, 2), _cr(1, 2)),
                beta=(_cr(1, 1), _cr(1, 12), _cr(1, 15), _cr(4, 45)),
            ),
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


class TestRecurrenceTruncation:
    def test_overlong_sequence_rejected(self) -> None:
        """35 usable moments would be silently truncated to 16 coefficients;
        the kernel rejects them instead of reporting complete=True."""
        moments = tuple(Fraction(1, k + 1) for k in range(35))
        with pytest.raises(ValueError, match="at most 33"):
            recurrence_coefficients(moments)


class TestRecurrenceTruncationBoundary:
    def test_34th_moment_rejected(self) -> None:
        """(34-1)//2 == 16 still fits the cap, but the 34th moment would be
        silently ignored; it must be rejected like any overlong prefix."""
        moments = tuple(Fraction(1, k + 1) for k in range(34))
        with pytest.raises(ValueError, match="at most 33"):
            recurrence_coefficients(moments)


class TestNativeRecurrenceValueComposition:
    """The exported producer's canonical value feeds consumers unchanged."""

    def test_producer_value_feeds_all_consumers_unchanged(self) -> None:
        coefficients = recurrence_coefficients(
            tuple(Fraction(1, k) for k in range(1, 8))
        )
        assert isinstance(coefficients, RecurrenceCoefficients)

        matrix = jacobi_matrix(coefficients)
        assert matrix.diagonal == tuple(
            value.as_fraction() for value in coefficients.alpha
        )
        assert (
            matrix.off_diagonal
            == (tuple(value.as_fraction() for value in coefficients.beta))[
                1 : len(coefficients.alpha)
            ]
        )

        kernel = christoffel_darboux(coefficients, Fraction(1), Fraction(1))
        assert kernel.kernel > 0

        rule = gaussian_quadrature(coefficients)
        assert len(rule.nodes) == len(coefficients.alpha)
        assert all(weight > 0 for weight in rule.weights)

    def test_forged_value_cannot_bypass_consumer_admission(self) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            jacobi_matrix(
                RecurrenceCoefficients(alpha=(_cr(0, 1),), beta=(_cr(-1, 1),))
            )

    def test_producer_and_wire_result_values_agree(self) -> None:
        native = recurrence_coefficients(tuple(Fraction(1, k) for k in range(1, 8)))
        result = compute_recurrence_coefficients(
            RecurrenceCoefficientsRequest(moments=_HARMONIC_MOMENTS)
        )
        assert native == result.coefficients


class TestCanonicalRecurrenceCoefficientsValue:
    """The producer's coefficients value composes into consumers unchanged."""

    def test_serialized_coefficients_feed_consumers_unchanged(self) -> None:
        request = RecurrenceCoefficientsRequest(moments=_HARMONIC_MOMENTS)
        result = compute_recurrence_coefficients(request)
        payload = result.model_dump(mode="json")

        jacobi = JacobiMatrixRequest.model_validate(
            {"coefficients": payload["coefficients"]}
        )
        assert len(compute_jacobi_matrix(jacobi).diagonal) == 3

        kernel = ChristoffelDarbouxRequest.model_validate(
            {
                "coefficients": payload["coefficients"],
                "x": {"num": "1", "den": "1"},
                "y": {"num": "1", "den": "1"},
            }
        )
        assert compute_christoffel_darboux(kernel).kernel

        quadrature = GaussianQuadratureRequest.model_validate(
            {"coefficients": payload["coefficients"]}
        )
        assert len(compute_gaussian_quadrature(quadrature).nodes) == 3

    def test_consumer_rejects_nonpositive_beta0(self) -> None:
        with pytest.raises(ValidationError, match="positive"):
            JacobiMatrixRequest.model_validate(
                {
                    "coefficients": {
                        "alpha": [],
                        "beta": [{"num": "-1", "den": "1"}],
                    }
                }
            )

    def test_result_revalidates_from_serialized_payload(self) -> None:
        result = compute_recurrence_coefficients(
            RecurrenceCoefficientsRequest(moments=_HARMONIC_MOMENTS)
        )
        assert (
            RecurrenceCoefficientsResult.model_validate(result.model_dump(mode="json"))
            == result
        )


class TestHankelCanonicalMatrixValue:
    def test_hankel_result_is_canonical_rational_matrix(self) -> None:
        request = HankelMatrixRequest(moments=_HARMONIC_MOMENTS)
        result = compute_hankel_matrix(request)
        assert isinstance(result.matrix, RationalMatrix)
        assert result.dimension == 4

    def test_hankel_matrix_feeds_rational_matrix_consumers(self) -> None:
        from jacobian.math.matrices._operation_models import RationalMatrixRequest

        result = compute_hankel_matrix(HankelMatrixRequest(moments=_HARMONIC_MOMENTS))
        consumer = RationalMatrixRequest(matrix=result.matrix)
        assert len(consumer.matrix.entries) == 4

    def test_hankel_result_revalidates_and_rejects_forgery(self) -> None:
        result = compute_hankel_matrix(HankelMatrixRequest(moments=_HARMONIC_MOMENTS))
        payload = result.model_dump(mode="json")
        assert HankelMatrixResult.model_validate(payload) == result

        payload["matrix"]["entries"][0][0] = {"num": "7", "den": "1"}
        with pytest.raises(ValidationError, match="exact Hankel matrix"):
            HankelMatrixResult.model_validate(payload)


class TestConditionedQuadratureAdmission:
    def test_combined_extremes_rejected_at_admission(self) -> None:
        """alpha=(0, 10^300), beta=(1, 10^-300) passes every individual
        finite-float bound but numerically disconnects the Jacobi matrix:
        the large node's eigenvector first component underflows and its mass
        would be zero. Admission replays Golub-Welsch and rejects it."""
        coefficients = RecurrenceCoefficients(
            alpha=(_cr(0, 1), CanonicalRational.from_fraction(Fraction(10**300))),
            beta=(
                _cr(1, 1),
                CanonicalRational.from_fraction(Fraction(1, 10**300)),
            ),
        )
        with pytest.raises(ValidationError, match="positive n-point rule"):
            GaussianQuadratureRequest(coefficients=coefficients)

    def test_moderate_conditioning_still_admitted(self) -> None:
        request = GaussianQuadratureRequest(coefficients=_legendre_coefficients())
        assert len(request.coefficients.alpha) == 2
