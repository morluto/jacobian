"""Public declaration for exact free associative algebra multiplication."""

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.free_algebras._models import (
    FreeAlgebraIdealPrefixRequest,
    FreeAlgebraIdealPrefixResult,
    FreeAlgebraPolynomialProductRequest,
    FreeAlgebraPolynomialProductResult,
    GroebnerShirshovRequest,
    GroebnerShirshovResult,
)
from jacobian.math.free_algebras.operations import (
    groebner_shirshov_through_degree,
    ideal_generated_prefix,
    multiply,
)


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


def _run_left_prefix(
    request: FreeAlgebraIdealPrefixRequest,
) -> FreeAlgebraIdealPrefixResult:
    if request.ideal.side != "left":
        raise OperationDomainValidationError(
            location=("ideal", "side"),
            code="free_algebra.left_side",
            message="the left-ideal operation requires side='left'",
        )
    return ideal_generated_prefix(request.ideal, request.degree)


def _run_right_prefix(
    request: FreeAlgebraIdealPrefixRequest,
) -> FreeAlgebraIdealPrefixResult:
    if request.ideal.side != "right":
        raise OperationDomainValidationError(
            location=("ideal", "side"),
            code="free_algebra.right_side",
            message="the right-ideal operation requires side='right'",
        )
    return ideal_generated_prefix(request.ideal, request.degree)


def _run_two_sided_prefix(
    request: FreeAlgebraIdealPrefixRequest,
) -> FreeAlgebraIdealPrefixResult:
    if request.ideal.side != "two-sided":
        raise OperationDomainValidationError(
            location=("ideal", "side"),
            code="free_algebra.two_sided_side",
            message="the two-sided ideal operation requires side='two-sided'",
        )
    return ideal_generated_prefix(request.ideal, request.degree)


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
            "term-pair and collected-like-word ledger is returned."
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
    MathTool(
        operation_id="free_algebra.left_ideal.generated_prefix.compute",
        title="Generate a bounded left free-algebra ideal prefix",
        description="Generate exact left multiples through one finite degree; the ideal is bound to one ordered free alphabet and must have side left.",
        request_type=FreeAlgebraIdealPrefixRequest,
        result_type=FreeAlgebraIdealPrefixResult,
        run=_run_left_prefix,
        tags=("free-algebra", "left-ideal", "noncommutative", "exact"),
        examples=(
            OperationExample(
                name="left_prefix",
                description="Generate degree-two left multiples of x; the ideal side must be left.",
                input={
                    "ideal": {
                        "alphabet": ["x", "y"],
                        "generators": [
                            {
                                "alphabet": ["x", "y"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["x"],
                                    }
                                ],
                            }
                        ],
                        "side": "left",
                    },
                    "degree": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_algebra.right_ideal.generated_prefix.compute",
        title="Generate a bounded right free-algebra ideal prefix",
        description="Generate exact right multiples through one finite degree; the ideal is bound to one ordered free alphabet and must have side right.",
        request_type=FreeAlgebraIdealPrefixRequest,
        result_type=FreeAlgebraIdealPrefixResult,
        run=_run_right_prefix,
        tags=("free-algebra", "right-ideal", "noncommutative", "exact"),
        examples=(
            OperationExample(
                name="right_prefix",
                description="Generate degree-two right multiples of x; the ideal side must be right.",
                input={
                    "ideal": {
                        "alphabet": ["x", "y"],
                        "generators": [
                            {
                                "alphabet": ["x", "y"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["x"],
                                    }
                                ],
                            }
                        ],
                        "side": "right",
                    },
                    "degree": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_algebra.two_sided_ideal.generated_prefix.compute",
        title="Generate a bounded two-sided free-algebra ideal prefix",
        description="Generate exact two-sided multiples through one finite degree; the ideal is bound to one ordered free alphabet and must have side two-sided.",
        request_type=FreeAlgebraIdealPrefixRequest,
        result_type=FreeAlgebraIdealPrefixResult,
        run=_run_two_sided_prefix,
        tags=("free-algebra", "two-sided-ideal", "noncommutative", "exact"),
        examples=(
            OperationExample(
                name="two_sided_prefix",
                description="Generate degree-two two-sided multiples of x; the ideal side must be two-sided.",
                input={
                    "ideal": {
                        "alphabet": ["x", "y"],
                        "generators": [
                            {
                                "alphabet": ["x", "y"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["x"],
                                    }
                                ],
                            }
                        ],
                        "side": "two-sided",
                    },
                    "degree": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_algebra.ideal.generated_prefix.compute",
        title="Generate a bounded sided free-algebra ideal prefix",
        description="Generate the exact homogeneous degree prefix of a left, right, or two-sided ideal over one ordered free alphabet; sidedness remains part of the ideal value.",
        request_type=FreeAlgebraIdealPrefixRequest,
        result_type=FreeAlgebraIdealPrefixResult,
        run=lambda request: ideal_generated_prefix(request.ideal, request.degree),
        tags=("free-algebra", "ideal", "noncommutative", "exact"),
        examples=(
            OperationExample(
                name="left_right_distinction",
                description="Generate degree-two multiples of the generator x on the left; sidedness and the ordered alphabet are explicit.",
                input={
                    "ideal": {
                        "alphabet": ["x", "y"],
                        "generators": [
                            {
                                "alphabet": ["x", "y"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["x"],
                                    }
                                ],
                            }
                        ],
                        "side": "left",
                    },
                    "degree": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_algebra.two_sided_ideal.groebner_shirshov_through_degree.compute",
        title="Complete two-sided Groebner-Shirshov compositions through a degree",
        description=(
            "Reduce every ordered overlap and inclusion composition through the "
            "requested degree to a fixed point and return a basis complete "
            "through that degree; this bounded state is not a global completion "
            "claim."
        ),
        request_type=GroebnerShirshovRequest,
        result_type=GroebnerShirshovResult,
        run=lambda request: groebner_shirshov_through_degree(
            request.ideal, request.degree
        ),
        tags=("free-algebra", "groebner-shirshov", "two-sided", "exact"),
        examples=(
            OperationExample(
                name="commutator_generator",
                description="Complete the one-generator two-sided ideal through degree three; every overlap composition through the bound must reduce to zero.",
                input={
                    "ideal": {
                        "alphabet": ["x", "y"],
                        "generators": [
                            {
                                "alphabet": ["x", "y"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "-1", "den": "1"},
                                        "word": ["y", "x"],
                                    },
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["x", "y"],
                                    },
                                ],
                            }
                        ],
                        "side": "two-sided",
                    },
                    "degree": 3,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
