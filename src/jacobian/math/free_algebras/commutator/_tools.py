"""Catalog operation for exact free-algebra commutators."""

from __future__ import annotations

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.free_algebras.commutator._models import (
    FreeAlgebraCommutatorRequest,
    FreeAlgebraCommutatorResult,
)
from jacobian.math.free_algebras.commutator.operations import commutator


def _run(request: FreeAlgebraCommutatorRequest) -> FreeAlgebraCommutatorResult:
    return commutator(request.left, request.right)


_LEFT = {
    "alphabet": ["x", "y"],
    "terms": [
        {"coefficient": {"num": "1", "den": "1"}, "word": ["y"]},
        {"coefficient": {"num": "1", "den": "1"}, "word": ["x"]},
    ],
}
_RIGHT = {
    "alphabet": ["x", "y"],
    "terms": [
        {"coefficient": {"num": "-1", "den": "1"}, "word": ["y"]},
        {"coefficient": {"num": "1", "den": "1"}, "word": ["x"]},
    ],
}

TOOLS = (
    MathTool(
        operation_id="free_algebra.polynomial.commutator.compute",
        title="Compute a free-algebra polynomial commutator",
        description=(
            "Compute [f,g] = fg - gf for sparse exact QQ-polynomials in a "
            "free associative algebra. It preserves the ordered generator "
            "alphabet and canonical degree-lexicographic term order. Before "
            "convolution, admission bounds each operand to 64 terms and "
            "32-letter words, the candidate output to 4,096 terms and "
            "262,144 word cells, coefficient growth to 64 digits, and "
            "estimated exact work to 20,000,000 units."
        ),
        request_type=FreeAlgebraCommutatorRequest,
        result_type=FreeAlgebraCommutatorResult,
        run=_run,
        tags=("free-algebra", "noncommutative", "commutator", "exact"),
        discovery_terms=(
            "free algebra commutator",
            "noncommutative polynomial bracket",
            "associative polynomial bracket fg minus gf",
        ),
        examples=(
            OperationExample(
                name="linear_polynomial_bracket",
                description=(
                    "Compute the exact bracket of x+y and x-y; both polynomials "
                    "must use the same ordered generator alphabet."
                ),
                input={"left": _LEFT, "right": _RIGHT},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
