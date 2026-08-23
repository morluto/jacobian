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
from jacobian.math.moments_orthogonal.values import MAX_RECURRENCE_ORDER

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
        with pytest.raises(ValueError, match="positive"):
            recurrence_coefficients(moments)

    def test_two_moments_determine_one_pair(self) -> None:
        """Two moments determine exactly alpha_0 and beta_0."""
        moments = (_frac(1, 1), _frac(1, 2))
        result = recurrence_coefficients(moments)
        assert result.alpha == (_frac(1, 2),)
        assert result.beta == (_frac(1, 1),)

    def test_even_length_sequence_fully_consumed(self) -> None:
        """Four uniform moments determine both coefficient pairs."""
        result = recurrence_coefficients(tuple(_frac(1, k) for k in range(1, 5)))
        assert result.alpha == (_frac(1, 2), _frac(1, 2))
        assert result.beta == (_frac(1, 1), _frac(1, 12))


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


class TestChristoffelDarbouxBoundedness:
    """Admission must bound coefficient growth as a function of order."""

    def test_large_alpha_coefficient_growth_rejected(self) -> None:
        coefficient = _cr(10**4090 + 123, 10**4088 + 7)
        unit = _cr(1, 1)
        zero = _cr(0, 1)
        with pytest.raises(ValidationError, match="canonical"):
            ChristoffelDarbouxRequest(
                coefficients=RecurrenceCoefficientsValue(
                    alpha=tuple(coefficient for _ in range(16)),
                    beta=tuple(unit for _ in range(16)),
                ),
                x=zero,
                y=zero,
            )

    def test_large_x_growth_rejected(self) -> None:
        big_x = _cr(10**2001 + 5, 10**2000 + 3)
        small_alpha = tuple(_cr(1, k + 2) for k in range(16))
        unit = _cr(1, 1)
        with pytest.raises(ValidationError, match="canonical"):
            ChristoffelDarbouxRequest(
                coefficients=RecurrenceCoefficientsValue(
                    alpha=small_alpha,
                    beta=tuple(unit for _ in range(16)),
                ),
                x=big_x,
                y=big_x,
            )

    def test_fitting_coefficients_at_order_succeed(self) -> None:
        alpha = tuple(_cr(10**399 + k, 10**400 + k) for k in range(16))
        beta = tuple(_cr(10**400 - k, 10**399 + k) for k in range(16))
        result = compute_christoffel_darboux(
            ChristoffelDarbouxRequest(
                coefficients=RecurrenceCoefficientsValue(alpha=alpha, beta=beta),
                x=_cr(0, 1),
                y=_cr(0, 1),
            )
        )
        assert len(result.polynomials_evaluated) == 16

    def test_low_order_large_coefficients_succeed(self) -> None:
        big = _cr(10**4090 + 123, 10**4088 + 7)
        result = compute_christoffel_darboux(
            ChristoffelDarbouxRequest(
                coefficients=RecurrenceCoefficientsValue(
                    alpha=(big, big), beta=(_cr(1, 1), big)
                ),
                x=big,
                y=big,
            )
        )
        assert isinstance(result, ChristoffelDarbouxResult)


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
        RecurrenceCoefficientsResult.model_validate(result.model_dump())

    def test_jacobi_wire(self) -> None:
        request = JacobiMatrixRequest(
            coefficients=RecurrenceCoefficientsValue(
                alpha=(_cr(1, 2), _cr(1, 2)),
                beta=(_cr(1, 1), _cr(1, 12), _cr(1, 15)),
            ),
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

    def test_gaussian_quadrature_wire(self) -> None:
        request = GaussianQuadratureRequest(
            coefficients=RecurrenceCoefficientsValue(
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


class TestRecurrencePositiveFunctional:
    def test_negative_zeroth_moment_rejected_before_kernel(self) -> None:
        """A one-moment request with mu_0 < 0 cannot yield beta_0 = mu_0."""
        with pytest.raises((ValueError, ValidationError), match="positive"):
            RecurrenceCoefficientsRequest(
                moments=(CanonicalRational(num="-1", den="1"),)
            )


class TestRecurrenceValueComposition:
    """The serialized recurrence value must feed every consumer unchanged."""

    def test_result_coefficients_feed_all_consumers_unchanged(self) -> None:
        request = RecurrenceCoefficientsRequest(
            moments=tuple(_cr(1, k) for k in range(1, 8))
        )
        result = compute_recurrence_coefficients(request)
        payload = result.model_dump(mode="json")
        coefficients = payload["coefficients"]

        jacobi = JacobiMatrixRequest.model_validate({"coefficients": coefficients})
        assert len(jacobi.coefficients.alpha) == 3

        christoffel = ChristoffelDarbouxRequest.model_validate(
            {
                "coefficients": coefficients,
                "x": {"num": "1", "den": "1"},
                "y": {"num": "2", "den": "1"},
            }
        )
        assert compute_christoffel_darboux(christoffel).kernel is not None

        quadrature = GaussianQuadratureRequest.model_validate(
            {"coefficients": coefficients}
        )
        assert compute_gaussian_quadrature(quadrature).nodes is not None

    def test_tampered_coefficients_rejected_by_replay(self) -> None:
        request = RecurrenceCoefficientsRequest(
            moments=tuple(_cr(1, k) for k in range(1, 8))
        )
        result = compute_recurrence_coefficients(request)
        payload = result.model_dump()
        forged = dict(payload["coefficients"])
        forged["alpha"] = [
            {"num": "1", "den": str(int(entry["den"]) + 2)} for entry in forged["alpha"]
        ]
        payload["coefficients"] = forged
        with pytest.raises(ValidationError, match="exact recurrence"):
            RecurrenceCoefficientsResult.model_validate(payload)


class TestRecurrenceAdmissionBoundaries:
    def test_thirty_three_moment_boundary_rejected(self):
        """33 harmonic moments would derive a 17th beta entry outside the
        canonical coefficient value; admission rejects at the boundary."""
        moments = tuple(_cr(1, k) for k in range(1, 34))
        with pytest.raises(ValidationError, match="32 moments"):
            RecurrenceCoefficientsRequest(moments=moments)

    def test_thirty_two_moment_sequence_roundtrips(self):
        moments = tuple(_cr(1, k) for k in range(1, 33))
        request = RecurrenceCoefficientsRequest(moments=moments)
        result = compute_recurrence_coefficients(request)
        assert len(result.coefficients.beta) <= MAX_RECURRENCE_ORDER
        assert (
            RecurrenceCoefficientsResult.model_validate(result.model_dump()) == result
        )

    def test_derived_coefficient_overflow_rejected_at_admission(self):
        """17 concentrated positive-definite moments whose exact recurrence
        coefficients leave the canonical rational domain cannot be accepted."""
        denominator_scale = Fraction(10) ** 299
        moments = tuple(
            CanonicalRational.from_fraction(
                Fraction(1, k + 1) + Fraction(1, denominator_scale + 2 * k + 1)
            )
            for k in range(17)
        )
        with pytest.raises(ValidationError, match="canonical"):
            RecurrenceCoefficientsRequest(moments=moments)

    def test_odd_prefix_final_norm_validated(self):
        """An odd-length prefix must validate the norm its last moment fixes.

        moments=(1, 0, -1) has Hankel determinant -1, so it defines no
        positive functional; the final even-moment norm check must reject it
        instead of returning coefficients with complete=True.
        """
        with pytest.raises(ValueError, match="positive-definite"):
            recurrence_coefficients((Fraction(1), Fraction(0), Fraction(-1)))
        # The same prefix extended to a positive sequence stays admitted and
        # keeps its known answer alpha=(0) plus the final beta ratio the
        # retained third moment determines.
        result = recurrence_coefficients((Fraction(1), Fraction(0), Fraction(1)))
        assert result.alpha == (Fraction(0),)
        assert result.beta == (Fraction(1), Fraction(1))

    def test_odd_prefix_wire_rejects_negative_hankel_minor(self):
        with pytest.raises((ValidationError, ValueError)):
            RecurrenceCoefficientsRequest(
                moments=(
                    _cr(1, 1),
                    _cr(0, 1),
                    _cr(-1, 1),
                )
            )

    def test_three_harmonic_moments_return_final_beta(self):
        """The reviewer's counterexample: three harmonic moments.

        p_1 = x - 1/2 has squared norm h_1 = mu_2 - mu_1 + mu_0/4 = 1/12,
        so the odd prefix fully determines beta_1 = h_1/h_0 = 1/12; omitting
        it would return an incomplete coefficient set with complete=True.
        """
        result = recurrence_coefficients((Fraction(1), Fraction(1, 2), Fraction(1, 3)))
        assert result.alpha == (Fraction(1, 2),)
        assert result.beta == (Fraction(1), Fraction(1, 12))

    def test_seven_harmonic_moments_return_final_beta(self):
        """Seven uniform moments determine beta_3 through their last entry."""
        result = recurrence_coefficients(tuple(_frac(1, k) for k in range(1, 8)))
        assert len(result.alpha) == 3
        assert len(result.beta) == 4

    def test_even_length_stops_at_determined_coefficients(self):
        """An even prefix leaves the final norm undetermined: no extra beta."""
        result = recurrence_coefficients(tuple(_frac(1, k) for k in range(1, 5)))
        assert result.alpha == (_frac(1, 2), _frac(1, 2))
        assert result.beta == (_frac(1, 1), _frac(1, 12))

    def test_wire_result_carries_final_beta_and_composes(self):
        """The wire result replays the final beta and feeds every consumer."""
        request = RecurrenceCoefficientsRequest(
            moments=(_cr(1, 1), _cr(1, 2), _cr(1, 3))
        )
        result = compute_recurrence_coefficients(request)
        assert result.coefficients.beta[-1] == _cr(1, 12)
        payload = result.model_dump(mode="json")["coefficients"]
        jacobi = compute_jacobi_matrix(
            JacobiMatrixRequest.model_validate({"coefficients": payload})
        )
        assert jacobi.off_diagonal == ()
        quadrature = GaussianQuadratureRequest.model_validate({"coefficients": payload})
        computed = compute_gaussian_quadrature(quadrature)
        assert len(computed.nodes) == 1
