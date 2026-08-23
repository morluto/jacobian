"""Exact multivariate polynomial operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.multivariate._models import (
    MultivariateDivisionRequest,
    MultivariateDivisionResult,
    MultivariateGcdRequest,
    MultivariateGcdResult,
    MultivariateResultantRequest,
    MultivariateResultantResult,
    MultivariateSubresultantSequenceRequest,
    MultivariateSubresultantSequenceResult,
)
from jacobian.math.polynomials.multivariate._operations import (
    compute_multivariate_division,
    compute_multivariate_gcd,
    compute_multivariate_resultant,
    compute_multivariate_subresultant_sequence,
)


def multivariate_polynomial_operation[
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
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


_GCD_EXAMPLE = example(
    "gcd_xy_minus_one_coprime",
    "Compute the GCD of two coprime multivariate polynomials.",
    {
        "left": {
            "polynomial_schema_version": "1",
            "domain": "QQ",
            "variables": ["x", "y"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [1, 1],
                    },
                    {
                        "coefficient": {"num": "-1", "den": "1"},
                        "exponents": [0, 0],
                    },
                ]
            },
        },
        "right": {
            "polynomial_schema_version": "1",
            "domain": "QQ",
            "variables": ["x", "y"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [2, 0],
                    },
                    {
                        "coefficient": {"num": "-1", "den": "1"},
                        "exponents": [0, 0],
                    },
                ]
            },
        },
    },
)

_DIVISION_EXAMPLE = example(
    "division_xy_minus_one",
    "Divide x^2*y + x by x*y - 1 under lex order.",
    {
        "left": {
            "polynomial_schema_version": "1",
            "domain": "QQ",
            "variables": ["x", "y"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [2, 1],
                    },
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [1, 0],
                    },
                ]
            },
        },
        "right": {
            "polynomial_schema_version": "1",
            "domain": "QQ",
            "variables": ["x", "y"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [1, 1],
                    },
                    {
                        "coefficient": {"num": "-1", "den": "1"},
                        "exponents": [0, 0],
                    },
                ]
            },
        },
        "monomial_order": "lex",
    },
)

_RESULTANT_EXAMPLE = example(
    "resultant_xy_minus_one_x_squared_minus_one",
    "Compute the resultant of x*y-1 and x^2-1 w.r.t. x.",
    {
        "left": {
            "polynomial_schema_version": "1",
            "domain": "QQ",
            "variables": ["x", "y"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [1, 1],
                    },
                    {
                        "coefficient": {"num": "-1", "den": "1"},
                        "exponents": [0, 0],
                    },
                ]
            },
        },
        "right": {
            "polynomial_schema_version": "1",
            "domain": "QQ",
            "variables": ["x", "y"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [2, 0],
                    },
                    {
                        "coefficient": {"num": "-1", "den": "1"},
                        "exponents": [0, 0],
                    },
                ]
            },
        },
        "elimination_variable": "x",
    },
)

_SUBRESULTANT_EXAMPLE = example(
    "subresultants_x_squared_minus_y_x_minus_y",
    (
        "Compute the exact nonzero subresultant PRS of x^2-y and x-y in x; "
        "both polynomials must share one ordered multivariate QQ ring and "
        "have positive degree in the declared main variable."
    ),
    {
        "left": {
            "polynomial_schema_version": "1",
            "domain": "QQ",
            "variables": ["x", "y"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [2, 0],
                    },
                    {
                        "coefficient": {"num": "-1", "den": "1"},
                        "exponents": [0, 1],
                    },
                ]
            },
        },
        "right": {
            "polynomial_schema_version": "1",
            "domain": "QQ",
            "variables": ["x", "y"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [1, 0],
                    },
                    {
                        "coefficient": {"num": "-1", "den": "1"},
                        "exponents": [0, 1],
                    },
                ]
            },
        },
        "main_variable": "x",
    },
)


MULTIVARIATE_POLYNOMIAL_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    multivariate_polynomial_operation(
        "polynomial.multivariate.gcd.compute",
        "Compute a multivariate polynomial GCD over QQ",
        (
            "Compute the monic GCD of two bounded multivariate polynomials over "
            "QQ[x_1, ..., x_n].  Backed by SymPy's multivariate polynomial GCD."
        ),
        MultivariateGcdRequest,
        MultivariateGcdResult,
        compute_multivariate_gcd,
        "polynomial",
        "gcd",
        "multivariate",
        "rational",
        examples=(_GCD_EXAMPLE,),
    ),
    multivariate_polynomial_operation(
        "polynomial.multivariate.divide.compute",
        "Divide multivariate polynomials with remainder",
        (
            "Compute the quotient and remainder of one multivariate polynomial "
            "divided by another over QQ[x_1, ..., x_n] under a declared monomial "
            "order.  Backed by SymPy's multivariate polynomial division."
        ),
        MultivariateDivisionRequest,
        MultivariateDivisionResult,
        compute_multivariate_division,
        "polynomial",
        "division",
        "multivariate",
        "rational",
        examples=(_DIVISION_EXAMPLE,),
    ),
    multivariate_polynomial_operation(
        "polynomial.multivariate.resultant.compute",
        "Compute a multivariate polynomial resultant over QQ",
        (
            "Compute the exact resultant of two multivariate polynomials with "
            "respect to one declared variable over QQ[x_1, ..., x_n].  The "
            "resultant lives in the ring over the remaining variables.  Backed "
            "by SymPy's resultant."
        ),
        MultivariateResultantRequest,
        MultivariateResultantResult,
        compute_multivariate_resultant,
        "polynomial",
        "resultant",
        "multivariate",
        "rational",
        "elimination",
        examples=(_RESULTANT_EXAMPLE,),
    ),
    multivariate_polynomial_operation(
        "polynomial.multivariate.subresultant_sequence.compute",
        "Compute an exact multivariate-coefficient subresultant sequence",
        (
            "Compute the complete nonzero Brown subresultant PRS of two "
            "bounded polynomials in one declared main variable over the exact "
            "QQ polynomial ring in all remaining variables. Return every "
            "source-bound member, skipped member degrees, every principal "
            "subresultant coefficient including zeros, the original-orientation "
            "Sylvester resultant, and the final fraction-field GCD relation. "
            "A pinned SymPy Brown PRS kernel stays private; Jacobian fixes "
            "ordering, signs, bounds, and exact replay."
        ),
        MultivariateSubresultantSequenceRequest,
        MultivariateSubresultantSequenceResult,
        compute_multivariate_subresultant_sequence,
        "polynomial",
        "subresultant",
        "polynomial remainder sequence",
        "multivariate",
        "rational",
        "projection",
        "lifting",
        examples=(_SUBRESULTANT_EXAMPLE,),
    ),
)


TOOLS = MULTIVARIATE_POLYNOMIAL_OPERATIONS

__all__ = ["TOOLS"]
