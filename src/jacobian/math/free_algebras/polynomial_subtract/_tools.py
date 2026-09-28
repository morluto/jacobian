"""Catalog declaration for exact free-algebra polynomial subtraction."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.free_algebras._models import FreeAlgebraPolynomial
from jacobian.math.free_algebras.polynomial_subtract._models import (
    FreeAlgebraPolynomialSubtractionRequest,
)
from jacobian.math.free_algebras.polynomial_subtract.operations import subtract


def _run(
    request: FreeAlgebraPolynomialSubtractionRequest,
) -> FreeAlgebraPolynomial:
    return subtract(request.left, request.right)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="free_algebra.polynomial.subtract.compute",
        title="Subtract sparse noncommutative polynomials exactly",
        description=(
            "Subtract two sparse QQ-linear polynomials over the identical "
            "ordered generator alphabet. Equal words have their exact rational "
            "coefficients differenced, zero results are omitted, and the result "
            "is returned in canonical degree-lexicographic order. Admission "
            "bounds operand size, coefficient growth, output allocation, and "
            "exact work before coefficient aggregation."
        ),
        request_type=FreeAlgebraPolynomialSubtractionRequest,
        result_type=FreeAlgebraPolynomial,
        run=_run,
        tags=("algebra", "free-algebra", "noncommutative", "polynomial", "exact"),
        discovery_terms=(
            "free associative algebra subtraction",
            "noncommutative polynomial difference",
            "sparse nc polynomial subtraction",
        ),
        examples=(
            OperationExample(
                name="collect_and_cancel_like_words",
                description=(
                    "Subtract x+y from 2x+3/2 xy. The x term remains; xy and y "
                    "remain distinct words over the same ordered alphabet."
                ),
                input={
                    "left": {
                        "alphabet": ["x", "y"],
                        "terms": [
                            {
                                "coefficient": {"num": "3", "den": "2"},
                                "word": ["x", "y"],
                            },
                            {
                                "coefficient": {"num": "2", "den": "1"},
                                "word": ["x"],
                            },
                        ],
                    },
                    "right": {
                        "alphabet": ["x", "y"],
                        "terms": [
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "word": ["y"],
                            },
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "word": ["x"],
                            },
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
