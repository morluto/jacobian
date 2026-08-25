"""Exact real-quadratic matrix operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.matrices.quadratic_spectral._models import (
    RealQuadraticInertiaRequest,
    RealQuadraticSingularSpectrumRequest,
    RealQuadraticSymmetricSpectrumRequest,
)
from jacobian.math.matrices.quadratic_spectral._operations import (
    compute_inertia,
    compute_singular_spectrum,
    compute_symmetric_spectrum,
)
from jacobian.math.matrices.quadratic_spectral.values import (
    RealQuadraticInertia,
    RealQuadraticSpectrum,
)


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


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
    _op(
        "matrix.real_quadratic.symmetric_spectrum.compute",
        "Compute an exact symmetric 2 by 2 spectrum over Q(sqrt(d))",
        "Return the complete descending eigenvalue spectrum of an exact "
        "symmetric 2 by 2 matrix over one real quadratic field. Values use "
        "canonical minimal polynomials and increasing real-root indices. "
        "The exact annihilating polynomial is limited to 996 decimal digits "
        "per coefficient.",
        RealQuadraticSymmetricSpectrumRequest,
        RealQuadraticSpectrum,
        compute_symmetric_spectrum,
        "matrix",
        "eigenvalue",
        "spectrum",
        "quadratic-field",
        "exact",
        examples=(
            example(
                "pang_weighted_sum_spectrum",
                "Compute the exact eigenvalues 1/2 +/- sqrt(3)/20 of Pang's "
                "weighted projection sum.",
                {
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
    _op(
        "matrix.real_quadratic.singular_spectrum.compute",
        "Compute an exact 2 by 2 singular spectrum over Q(sqrt(d))",
        "Return the complete descending singular-value spectrum of an exact "
        "2 by 2 matrix over one real quadratic field. Values use canonical "
        "minimal polynomials and increasing real-root indices. The exact "
        "annihilating polynomial is limited to 996 decimal digits per "
        "coefficient.",
        RealQuadraticSingularSpectrumRequest,
        RealQuadraticSpectrum,
        compute_singular_spectrum,
        "matrix",
        "singular-value",
        "spectrum",
        "quadratic-field",
        "exact",
        examples=(
            example(
                "pang_projection_product_spectrum",
                "Compute the singular values 3*sqrt(3)/8 and 0 of Pang's "
                "four-projection product.",
                {
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
    _op(
        "matrix.real_quadratic.inertia.compute",
        "Compute exact inertia over Q(sqrt(d))",
        "Return Sylvester inertia and definiteness for an exact symmetric "
        "matrix of dimension at most four over one real quadratic field. "
        "Signs and congruence pivots are computed exactly; no floating "
        "tolerance is used.",
        RealQuadraticInertiaRequest,
        RealQuadraticInertia,
        compute_inertia,
        "matrix",
        "inertia",
        "definiteness",
        "quadratic-field",
        "exact",
        examples=(
            example(
                "maxwell_hessian_inertia",
                "Compute the exact (+--) inertia of the Maxwell critical-point "
                "Hessian at (1/3, 0, sqrt(39)/12).",
                {
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
