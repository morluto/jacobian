"""Exact multivariate polynomial operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.multivariate_polynomial import (
    MultivariateDivisionRequest,
    MultivariateDivisionResult,
    MultivariateGcdRequest,
    MultivariateGcdResult,
    MultivariateResultantRequest,
    MultivariateResultantResult,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.multivariate_polynomial.operations import (
    compute_multivariate_division,
    compute_multivariate_gcd,
    compute_multivariate_resultant,
)
from jacobian.math_tools import MathTool


def multivariate_polynomial_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
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
)


__all__ = ["MULTIVARIATE_POLYNOMIAL_OPERATIONS"]
