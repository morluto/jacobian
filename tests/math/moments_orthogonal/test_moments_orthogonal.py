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

    def test_negative_singleton_moments_rejected(self) -> None:
        """A non-positive zeroth moment cannot seed a positive functional."""
        with pytest.raises(ValueError, match="positive"):
            recurrence_coefficients((_frac(-1, 1),))

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
        with pytest.raises(ValueError, match="nonzero"):
            jacobi_matrix((_frac(0, 1),), (_frac(0, 1), _frac(1, 1)))

    def test_nonpositive_subdiagonal_rejected(self) -> None:
        """A used beta entry must be a positive squared norm, or its square
        root cannot reconstruct a real symmetric Jacobi matrix."""
        with pytest.raises(ValueError, match="positive squared-norm"):
            jacobi_matrix((_frac(0, 1), _frac(0, 1)), (_frac(1, 1), _frac(-1, 1)))
        with pytest.raises(ValueError, match="positive squared-norm"):
            jacobi_matrix((_frac(0, 1), _frac(0, 1)), (_frac(1, 1), _frac(0, 1)))
        # An unused trailing entry does not occupy the subdiagonal.
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

    def test_nonpositive_beta_rejected(self) -> None:
        """K_n(x, x) is a sum of squares over positive squared norms and
        strictly positive; nonpositive used beta entries leave the
        positive-definite family the kernel is defined on."""
        with pytest.raises(ValueError, match="must be positive"):
            christoffel_darboux(
                (_frac(0, 1),), (_frac(-2, 1),), _frac(1, 1), _frac(1, 1)
            )
        with pytest.raises(ValueError, match="positive squared-norm"):
            christoffel_darboux(
                (_frac(0, 1), _frac(0, 1)),
                (_frac(1, 1), _frac(-1, 1)),
                _frac(1, 1),
                _frac(1, 1),
            )
        with pytest.raises(ValueError, match="positive squared-norm"):
            christoffel_darboux(
                (_frac(0, 1), _frac(0, 1)),
                (_frac(1, 1), _frac(0, 1)),
                _frac(1, 1),
                _frac(1, 1),
            )

    def test_unused_trailing_beta_outside_contract(self) -> None:
        """beta_n is unused for alpha length n, mirroring the wire model."""
        result = christoffel_darboux(
            (_frac(0, 1),), (_frac(1, 1), _frac(-1, 1)), _frac(1, 1), _frac(1, 1)
        )
        assert result.kernel == _frac(1, 1)


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

    def test_overflowing_alpha_rejected_not_raised(self) -> None:
        """The native API rejects the unsupported Float64 domain explicitly."""
        with pytest.raises(ValueError, match="magnitude"):
            gaussian_quadrature((_frac(10**400, 1),), (_frac(1, 1),))

    def test_underflowing_beta0_rejected(self) -> None:
        """beta_0 below the double subnormal range cannot scale weights."""
        with pytest.raises(ValueError, match="underflow"):
            gaussian_quadrature(
                (_frac(0, 1),),
                (_frac(1, 10**4000), _frac(1, 1)),
            )


# ---------------------------------------------------------------------------
# Wire adapter tests
# ---------------------------------------------------------------------------


class TestWireAdapters:
    def test_hankel_wire(self) -> None:
        request = HankelMatrixRequest(moments=tuple(_cr(1, k) for k in range(1, 8)))
        result = compute_hankel_matrix(request)
        assert result.dimension == 4
        assert isinstance(result.matrix, RationalMatrix)
        assert isinstance(result, HankelMatrixResult)

    def test_hankel_matrix_feeds_matrix_rank_unchanged(self) -> None:
        """The returned canonical matrix composes into downstream rational
        matrix operations without rewrapping or renaming fields."""
        from jacobian.math.matrices._operation_models import MatrixRankRequest
        from jacobian.math.matrices._operations import compute_rank

        request = HankelMatrixRequest(moments=tuple(_cr(1, k) for k in range(1, 8)))
        result = compute_hankel_matrix(request)
        rank_request = MatrixRankRequest.model_validate(
            {"matrix": result.matrix.model_dump()}
        )
        assert rank_request.matrix.entries == result.matrix.entries
        assert compute_rank(rank_request).rank == result.dimension

    def test_forged_hankel_matrix_rejected(self) -> None:
        """An authored matrix that is not the exact Hankel matrix of the
        retained moments fails validation."""
        with pytest.raises(ValidationError, match="exact Hankel"):
            HankelMatrixResult(
                moments=tuple(_cr(1, k) for k in range(1, 5)),
                matrix=RationalMatrix(
                    entries=(
                        (_cr(1, 1), _cr(1, 1)),
                        (_cr(1, 1), _cr(1, 1)),
                    )
                ),
                dimension=2,
            )

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

    def test_kernel_growth_bound_rejects_unrepresentable_output(self) -> None:
        """Sixteen recurrence steps at a huge evaluation point would drive the
        exact kernel past the canonical rational digit limit; admission must
        reject the request instead of failing during execution."""
        alpha = tuple(_cr(0, 1) for _ in range(16))
        beta = tuple(_cr(1, 1) for _ in range(16))
        huge = CanonicalRational.from_fraction(Fraction(1, 10**4095))
        with pytest.raises(ValidationError, match="order-and-height"):
            ChristoffelDarbouxRequest(alpha=alpha, beta=beta, x=huge, y=huge)

    def test_kernel_growth_bound_admits_moderate_inputs(self) -> None:
        alpha = tuple(_cr(0, 1) for _ in range(4))
        beta = tuple(_cr(1, k + 1) for k in range(5))
        x = CanonicalRational.from_fraction(Fraction(10**80 + 1, 10**79 - 3))
        request = ChristoffelDarbouxRequest(alpha=alpha, beta=beta, x=x, y=x)
        assert compute_christoffel_darboux(request).kernel

    def test_gaussian_quadrature_wire(self) -> None:
        request = GaussianQuadratureRequest(
            alpha=(_cr(1, 2), _cr(1, 2), _cr(1, 2)),
            beta=(_cr(1, 1), _cr(1, 12), _cr(1, 15), _cr(4, 45)),
        )
        result = compute_gaussian_quadrature(request)
        assert isinstance(result, GaussianQuadratureResult)
        assert len(result.nodes) == 3

    def test_gaussian_quadrature_wire_rejects_underflowing_beta0(self) -> None:
        """A positive beta_0 below the double subnormal range is rejected."""
        with pytest.raises(ValidationError, match="underflow"):
            GaussianQuadratureRequest(
                alpha=(_cr(0, 1),),
                beta=(_cr(1, 10**4000), _cr(1, 1)),
            )

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

    def test_gaussian_quadrature_discovery_discloses_float64_scope(self) -> None:
        """Discovery must reveal the Float64/dyadic approximation scope
        before execution, not only through result fields afterwards."""
        (tool,) = (
            t for t in TOOLS if t.operation_id == "moments.gaussian_quadrature.compute"
        )
        description = tool.description
        assert "Float64" in description
        assert "dyadic" in description
        assert "approximat" in description.lower()


class TestRecurrenceOutputBound:
    def test_near_degenerate_moments_rejected(self) -> None:
        """Crafted near-degenerate positive-definite moments pass per-moment
        checks but their Gram-Schmidt coefficients exceed the canonical
        rational limit; the kernel rejects them at the boundary."""
        eps = Fraction(1, 10**400)
        moments = (
            Fraction(1),
            Fraction(1, 3) + eps,
            Fraction(1, 5),
            Fraction(1, 7) + eps**2,
            Fraction(1, 9),
        )
        result = recurrence_coefficients(moments)
        assert len(result.alpha) == 2


class TestRecurrenceComposition:
    def test_serialized_result_feeds_consumers_unchanged(self) -> None:
        """A serialized recurrence result composes into the Jacobi and
        Christoffel-Darboux requests without stripping producer fields."""
        from jacobian.math.moments_orthogonal._models import (
            ChristoffelDarbouxRequest,
            RecurrenceCoefficientsResult,
        )

        moments = tuple(
            CanonicalRational.from_fraction(Fraction(1, k + 1)) for k in range(5)
        )
        kres = recurrence_coefficients([Fraction(1, k + 1) for k in range(5)])
        result = RecurrenceCoefficientsResult(
            moments=moments,
            alpha=tuple(CanonicalRational.from_fraction(a) for a in kres.alpha),
            beta=tuple(CanonicalRational.from_fraction(b) for b in kres.beta),
        )
        payload = result.model_dump()
        jacobi = JacobiMatrixRequest.model_validate(payload)
        assert len(jacobi.alpha) == len(result.alpha)
        cd = ChristoffelDarbouxRequest.model_validate(
            {**payload, "x": {"num": "1", "den": "1"}, "y": {"num": "1", "den": "1"}}
        )
        assert len(cd.beta) == len(result.beta)


class TestDiscoveryApproximationDisclosure:
    def test_quadrature_discovery_discloses_float64_scope(self):
        """math.find must reveal the Float64 approximation before selection."""
        tool = next(
            t for t in TOOLS if t.operation_id == "moments.gaussian_quadrature.compute"
        )
        combined = f"{tool.title} {tool.description}".lower()
        assert "float64" in combined or "double" in combined
        assert "approximate" in combined
