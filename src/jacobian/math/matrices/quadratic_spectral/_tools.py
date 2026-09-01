"""Exact real-quadratic matrix operation declarations."""

from collections.abc import Callable
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.matrices.quadratic_spectral import operations as native
from jacobian.math.matrices.quadratic_spectral._models import (
    RealQuadraticInertiaRequest,
    RealQuadraticSingularSpectrumRequest,
    RealQuadraticSymmetricSpectrumRequest,
)
from jacobian.math.matrices.quadratic_spectral.values import (
    RealQuadraticInertia,
    RealQuadraticSpectrum,
)


def _run[ResultT](operation: Callable[[], ResultT]) -> ResultT:
    try:
        return operation()
    except (PydanticCustomError, ValueError) as exc:
        code = (
            exc.type
            if isinstance(exc, PydanticCustomError)
            else "matrix.domain_invalid"
        )
        message = exc.message() if isinstance(exc, PydanticCustomError) else str(exc)
        raise OperationDomainValidationError(
            location=("matrix",), code=code, message=message
        ) from exc


def compute_symmetric_spectrum(
    request: RealQuadraticSymmetricSpectrumRequest,
) -> RealQuadraticSpectrum:
    return _run(lambda: native.symmetric_spectrum(request.matrix))


def compute_singular_spectrum(
    request: RealQuadraticSingularSpectrumRequest,
) -> RealQuadraticSpectrum:
    return _run(lambda: native.singular_spectrum(request.matrix))


def compute_inertia(request: RealQuadraticInertiaRequest) -> RealQuadraticInertia:
    return _run(lambda: native.inertia(request.matrix))


def _quadratic(
    rational_numerator: int,
    radical_numerator: int,
    radicand: int,
    *,
    rational_denominator: int = 1,
    radical_denominator: int = 1,
) -> dict[str, object]:
    return {
        "rational_part": {
            "num": str(rational_numerator),
            "den": str(rational_denominator),
        },
        "radical_coefficient": {
            "num": str(radical_numerator),
            "den": str(radical_denominator),
        },
        "radicand": radicand,
    }


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="matrix.real_quadratic.symmetric_spectrum.compute",
        title="Compute an exact symmetric 2 by 2 spectrum over Q(sqrt(d))",
        description="Return the complete descending eigenvalue spectrum of an exact "
        "symmetric 2 by 2 matrix over one real quadratic field. Values use "
        "canonical minimal polynomials and increasing real-root indices. "
        "The exact annihilating polynomial is limited to 996 decimal digits "
        "per coefficient.",
        request_type=RealQuadraticSymmetricSpectrumRequest,
        result_type=RealQuadraticSpectrum,
        run=compute_symmetric_spectrum,
        tags=("matrix", "eigenvalue", "spectrum", "quadratic-field", "exact"),
        examples=(
            OperationExample(
                name="pang_weighted_sum_spectrum",
                description="Compute the exact eigenvalues 1/2 +/- sqrt(3)/20 of Pang's "
                "weighted projection sum.",
                input={
                    "matrix": {
                        "entries": [
                            [
                                _quadratic(1, 0, 3, rational_denominator=2),
                                _quadratic(0, 1, 3, radical_denominator=20),
                            ],
                            [
                                _quadratic(0, 1, 3, radical_denominator=20),
                                _quadratic(1, 0, 3, rational_denominator=2),
                            ],
                        ]
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.real_quadratic.singular_spectrum.compute",
        title="Compute an exact 2 by 2 singular spectrum over Q(sqrt(d))",
        description="Return the complete descending singular-value spectrum of an exact "
        "2 by 2 matrix over one real quadratic field. Values use canonical "
        "minimal polynomials and increasing real-root indices. The exact "
        "annihilating polynomial is limited to 996 decimal digits per "
        "coefficient.",
        request_type=RealQuadraticSingularSpectrumRequest,
        result_type=RealQuadraticSpectrum,
        run=compute_singular_spectrum,
        tags=("matrix", "singular-value", "spectrum", "quadratic-field", "exact"),
        examples=(
            OperationExample(
                name="pang_projection_product_spectrum",
                description="Compute the singular values 3*sqrt(3)/8 and 0 of Pang's "
                "four-projection product.",
                input={
                    "matrix": {
                        "entries": [
                            [
                                _quadratic(0, 0, 3),
                                _quadratic(0, 3, 3, radical_denominator=8),
                            ],
                            [_quadratic(0, 0, 3), _quadratic(0, 0, 3)],
                        ]
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.real_quadratic.inertia.compute",
        title="Compute exact inertia over Q(sqrt(d))",
        description="Return Sylvester inertia and definiteness for an exact symmetric "
        "matrix of dimension at most four over one real quadratic field. "
        "Signs and congruence pivots are computed exactly; no floating "
        "tolerance is used.",
        request_type=RealQuadraticInertiaRequest,
        result_type=RealQuadraticInertia,
        run=compute_inertia,
        tags=("matrix", "inertia", "definiteness", "quadratic-field", "exact"),
        examples=(
            OperationExample(
                name="maxwell_hessian_inertia",
                description="Compute the exact (+--) inertia of the Maxwell critical-point "
                "Hessian at (1/3, 0, sqrt(39)/12).",
                input={
                    "matrix": {
                        "entries": [
                            [
                                _quadratic(19, 0, 39, rational_denominator=8),
                                _quadratic(0, 0, 39),
                                _quadratic(0, -1, 39, radical_denominator=2),
                            ],
                            [
                                _quadratic(0, 0, 39),
                                _quadratic(-45, 0, 39, rational_denominator=8),
                                _quadratic(0, 0, 39),
                            ],
                            [
                                _quadratic(0, -1, 39, radical_denominator=2),
                                _quadratic(0, 0, 39),
                                _quadratic(13, 0, 39, rational_denominator=4),
                            ],
                        ]
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
