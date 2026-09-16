"""Public declaration for exact free associative algebra multiplication."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.free_algebras._models import (
    FreeAlgebraPolynomialProductRequest,
    FreeAlgebraPolynomialProductResult,
)
from jacobian.math.free_algebras.operations import multiply


def _run_multiply(
    request: FreeAlgebraPolynomialProductRequest,
) -> FreeAlgebraPolynomialProductResult:
    return multiply(request.left, request.right)


_EXAMPLE_ALPHABET = ["x", "y"]

# (y + x)(x - y) = yx - yy + xx - xy, with xy and yx distinct.
_EXAMPLE_LEFT = {
    "alphabet": _EXAMPLE_ALPHABET,
    "terms": [
        {"coefficient": {"num": "1", "den": "1"}, "word": ["y"]},
        {"coefficient": {"num": "1", "den": "1"}, "word": ["x"]},
    ],
}
_EXAMPLE_RIGHT = {
    "alphabet": _EXAMPLE_ALPHABET,
    "terms": [
        {"coefficient": {"num": "-1", "den": "1"}, "word": ["y"]},
        {"coefficient": {"num": "1", "den": "1"}, "word": ["x"]},
    ],
}

TOOLS = (
    MathTool(
        operation_id="free_algebra.polynomial.multiply.compute",
        title="Multiply sparse noncommutative polynomials exactly",
        description=(
            "Compute the exact distributive concatenation product of two "
            "sparse QQ-linear polynomials over one shared finite ordered "
            "generator alphabet: (sum a_u u)(sum b_v v) = sum a_u b_v (uv). "
            "Like product words are collected and zero sums dropped, so the "
            "result is a canonical sparse free-algebra polynomial in "
            "descending degree-lexicographic order. The free algebra is "
            "noncommutative, so x*y and y*x stay distinct for distinct "
            "generators. Admission bounds each alphabet to 26 distinct "
            "letters, operand words to 32 letters, operand terms to 64, "
            "product term pairs and result terms to 4096, and predicted "
            "coefficient growth to 64 digits before expansion; the bounded "
            "term-pair and collected-like-word ledger is returned. Ideals, "
            "Grobner-Shirshov bases, and derivations are not implemented."
        ),
        request_type=FreeAlgebraPolynomialProductRequest,
        result_type=FreeAlgebraPolynomialProductResult,
        run=_run_multiply,
        tags=("algebra", "free-algebra", "noncommutative", "polynomial", "exact"),
        discovery_terms=(
            "free associative algebra",
            "noncommutative polynomial",
            "word multiplication",
            "distributive product",
            "sparse nc polynomial",
        ),
        examples=(
            OperationExample(
                name="xy_yx_distinct",
                description=(
                    "Multiply (y + x) by (x - y) over the alphabet [x, y] and "
                    "return the canonical product with its term-pair ledger; "
                    "the distinct words x*y and y*x must not be collected."
                ),
                input={"left": _EXAMPLE_LEFT, "right": _EXAMPLE_RIGHT},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
