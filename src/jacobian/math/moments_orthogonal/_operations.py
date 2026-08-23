"""Wire adapters for exact moments and orthogonal polynomials."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
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


def _to_fractions(
    values: tuple[CanonicalRational, ...],
) -> tuple[Fraction, ...]:
    return tuple(v.as_fraction() for v in values)


def _from_fractions(
    values: tuple[Fraction, ...],
) -> tuple[CanonicalRational, ...]:
    return tuple(CanonicalRational.from_fraction(v) for v in values)


def compute_hankel_matrix(request: HankelMatrixRequest) -> HankelMatrixResult:
    from jacobian.math.matrices.values import RationalMatrix
    from jacobian.math.moments_orthogonal.operations import hankel_matrix

    result = hankel_matrix(_to_fractions(request.moments))
    return HankelMatrixResult(
        moments=request.moments,
        matrix=RationalMatrix(
            entries=tuple(
                tuple(CanonicalRational.from_fraction(v) for v in row)
                for row in result.matrix
            ),
        ),
        dimension=len(result.matrix),
    )


def compute_recurrence_coefficients(
    request: RecurrenceCoefficientsRequest,
) -> RecurrenceCoefficientsResult:
    from jacobian.math.moments_orthogonal.operations import (
        recurrence_coefficients,
    )

    result = recurrence_coefficients(_to_fractions(request.moments))
    return RecurrenceCoefficientsResult(
        moments=request.moments,
        coefficients=RecurrenceCoefficients(
            alpha=_from_fractions(result.alpha),
            beta=_from_fractions(result.beta),
        ),
    )


def compute_jacobi_matrix(request: JacobiMatrixRequest) -> JacobiMatrixResult:
    from jacobian.math.moments_orthogonal.operations import jacobi_matrix

    result = jacobi_matrix(
        _to_fractions(request.coefficients.alpha),
        _to_fractions(request.coefficients.beta),
    )
    return JacobiMatrixResult(
        coefficients=request.coefficients,
        diagonal=_from_fractions(result.diagonal),
        off_diagonal=_from_fractions(result.off_diagonal),
    )


def compute_christoffel_darboux(
    request: ChristoffelDarbouxRequest,
) -> ChristoffelDarbouxResult:
    from jacobian.math.moments_orthogonal.operations import christoffel_darboux

    result = christoffel_darboux(
        _to_fractions(request.coefficients.alpha),
        _to_fractions(request.coefficients.beta),
        request.x.as_fraction(),
        request.y.as_fraction(),
    )
    return ChristoffelDarbouxResult(
        coefficients=request.coefficients,
        x=request.x,
        y=request.y,
        kernel=CanonicalRational.from_fraction(result.kernel),
        polynomials_evaluated=_from_fractions(result.polynomials_evaluated),
    )


def compute_gaussian_quadrature(
    request: GaussianQuadratureRequest,
) -> GaussianQuadratureResult:
    from jacobian.math.moments_orthogonal.operations import gaussian_quadrature

    result = gaussian_quadrature(
        _to_fractions(request.coefficients.alpha),
        _to_fractions(request.coefficients.beta),
    )
    return GaussianQuadratureResult(
        coefficients=request.coefficients,
        approximate_nodes=_from_fractions(result.approximate_nodes),
        approximate_weights=_from_fractions(result.approximate_weights),
    )


__all__ = [
    "compute_christoffel_darboux",
    "compute_gaussian_quadrature",
    "compute_hankel_matrix",
    "compute_jacobi_matrix",
    "compute_recurrence_coefficients",
]
