"""Wire adapters for exact moments and orthogonal polynomials."""

from collections.abc import Iterable
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.values import RationalMatrix
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


def _to_fractions(values: tuple[CanonicalRational, ...]) -> tuple[Fraction, ...]:
    return tuple(v.as_fraction() for v in values)


def _from_fractions(values: Iterable[Fraction]) -> tuple[CanonicalRational, ...]:
    return tuple(CanonicalRational.from_fraction(v) for v in values)


def compute_hankel_matrix(request: HankelMatrixRequest) -> HankelMatrixResult:
    from jacobian.math.moments_orthogonal.operations import hankel_matrix

    result = hankel_matrix(_to_fractions(request.moments))
    matrix = RationalMatrix(
        entries=tuple(
            tuple(CanonicalRational.from_fraction(v) for v in row)
            for row in result.matrix
        )
    )
    return HankelMatrixResult(
        moments=request.moments,
        matrix=matrix,
        dimension=len(matrix.entries),
    )


def compute_recurrence_coefficients(
    request: RecurrenceCoefficientsRequest,
) -> RecurrenceCoefficientsResult:
    from jacobian.math.moments_orthogonal.operations import (
        recurrence_coefficients,
    )

    return RecurrenceCoefficientsResult(
        moments=request.moments,
        coefficients=recurrence_coefficients(_to_fractions(request.moments)),
    )


def compute_jacobi_matrix(request: JacobiMatrixRequest) -> JacobiMatrixResult:
    from jacobian.math.moments_orthogonal.operations import jacobi_matrix

    result = jacobi_matrix(request.coefficients)
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
        request.coefficients,
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

    result = gaussian_quadrature(request.coefficients)
    return GaussianQuadratureResult(
        coefficients=request.coefficients,
        nodes=_from_fractions(result.nodes),
        weights=_from_fractions(result.weights),
        is_approximate=True,
        precision="FLOAT64",
        exactness="APPROXIMATE_DYADIC",
    )


__all__ = [
    "compute_christoffel_darboux",
    "compute_gaussian_quadrature",
    "compute_hankel_matrix",
    "compute_jacobi_matrix",
    "compute_recurrence_coefficients",
]
