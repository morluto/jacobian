"""Catalog declaration for exact free-algebra polynomial scalar multiplication."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.free_algebras._models import FreeAlgebraPolynomial
from jacobian.math.free_algebras.scalar_multiply._models import (
    FreeAlgebraPolynomialScalarMultiplyRequest,
)
from jacobian.math.free_algebras.scalar_multiply.operations import scalar_multiply


def _run_scalar_multiply(
    request: FreeAlgebraPolynomialScalarMultiplyRequest,
) -> FreeAlgebraPolynomial:
    return scalar_multiply(request.polynomial, request.scalar)


TOOLS = (
    MathTool(
        operation_id="free_algebra.polynomial.scalar_multiply.compute",
        title="Scale a free-algebra polynomial exactly",
        description=(
            "Multiply every coefficient of a sparse noncommutative QQ-polynomial "
            "by one exact rational. The ordered alphabet and word support are preserved."
        ),
        request_type=FreeAlgebraPolynomialScalarMultiplyRequest,
        result_type=FreeAlgebraPolynomial,
        run=_run_scalar_multiply,
        tags=("algebra", "free-algebra", "polynomial", "scalar", "exact"),
        discovery_terms=(
            "scale noncommutative polynomial",
            "free algebra rational scalar action",
            "multiply polynomial coefficients by a rational",
        ),
        examples=(
            OperationExample(
                name="rational_scaling",
                description=(
                    "Scale 3xy-2x by 2/3; polynomial terms must use canonical "
                    "descending degree-lexicographic order."
                ),
                input={
                    "polynomial": {
                        "alphabet": ["x", "y"],
                        "terms": [
                            {
                                "coefficient": {"num": "3", "den": "1"},
                                "word": ["x", "y"],
                            },
                            {
                                "coefficient": {"num": "-2", "den": "1"},
                                "word": ["x"],
                            },
                        ],
                    },
                    "scalar": {"num": "2", "den": "3"},
                },
            ),
        ),
    ),
)
