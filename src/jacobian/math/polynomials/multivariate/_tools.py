"""Exact multivariate polynomial operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.multivariate._division import (
    MultivariateDivisionRequest,
    MultivariateDivisionResult,
)
from jacobian.math.polynomials.multivariate._factor_models import (
    MultivariateFactorRequest,
    MultivariateFactorResult,
)
from jacobian.math.polynomials.multivariate._gcd import (
    MultivariateGcdRequest,
    MultivariateGcdResult,
)
from jacobian.math.polynomials.multivariate._resultant import (
    MultivariateResultantRequest,
    MultivariateResultantResult,
)
from jacobian.math.polynomials.multivariate._subresultants import (
    MultivariateSubresultantSequenceRequest,
    MultivariateSubresultantSequenceResult,
)
from jacobian.math.polynomials.multivariate.operations import (
    multivariate_division,
    multivariate_factor,
    multivariate_gcd,
    multivariate_resultant,
    multivariate_subresultant_sequence,
)


def _compute_gcd(request: MultivariateGcdRequest) -> MultivariateGcdResult:
    return multivariate_gcd(request.left, request.right)


def _compute_division(
    request: MultivariateDivisionRequest,
) -> MultivariateDivisionResult:
    return multivariate_division(request.left, request.right, request.monomial_order)


def _compute_resultant(
    request: MultivariateResultantRequest,
) -> MultivariateResultantResult:
    return multivariate_resultant(
        request.left, request.right, request.elimination_variable
    )


def _compute_subresultants(
    request: MultivariateSubresultantSequenceRequest,
) -> MultivariateSubresultantSequenceResult:
    return multivariate_subresultant_sequence(
        request.left, request.right, request.main_variable
    )


def _compute_factor(request: MultivariateFactorRequest) -> MultivariateFactorResult:
    return multivariate_factor(request.polynomial)


_GCD_EXAMPLE = OperationExample(
    name="gcd_xy_minus_one_coprime",
    description="Compute the GCD of two coprime multivariate polynomials.",
    input={
        "left": {
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

_DIVISION_EXAMPLE = OperationExample(
    name="division_xy_minus_one",
    description="Divide x^2*y + x by x*y - 1 under lex order.",
    input={
        "left": {
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

_RESULTANT_EXAMPLE = OperationExample(
    name="resultant_xy_minus_one_x_squared_minus_one",
    description="Compute the resultant of x*y-1 and x^2-1 w.r.t. x.",
    input={
        "left": {
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

_SUBRESULTANT_EXAMPLE = OperationExample(
    name="subresultants_x_squared_minus_y_x_minus_y",
    description=(
        "Compute the exact nonzero subresultant PRS of x^2-y and x-y in x; "
        "both polynomials must share one ordered multivariate QQ ring and "
        "have positive degree in the declared main variable."
    ),
    input={
        "left": {
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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial.multivariate.gcd.compute",
        title="Compute a multivariate polynomial GCD over QQ",
        description=(
            "Compute the monic GCD of two bounded multivariate polynomials over "
            "QQ[x_1, ..., x_n].  Backed by SymPy's multivariate polynomial GCD."
        ),
        request_type=MultivariateGcdRequest,
        result_type=MultivariateGcdResult,
        run=_compute_gcd,
        tags=("polynomial", "gcd", "multivariate", "rational"),
        examples=(_GCD_EXAMPLE,),
    ),
    MathTool(
        operation_id="polynomial.multivariate.divide.compute",
        title="Divide multivariate polynomials with remainder",
        description=(
            "Compute the quotient and remainder of one multivariate polynomial "
            "divided by another over QQ[x_1, ..., x_n] under a declared monomial "
            "order.  Backed by SymPy's multivariate polynomial division."
        ),
        request_type=MultivariateDivisionRequest,
        result_type=MultivariateDivisionResult,
        run=_compute_division,
        tags=("polynomial", "division", "multivariate", "rational"),
        examples=(_DIVISION_EXAMPLE,),
    ),
    MathTool(
        operation_id="polynomial.multivariate.resultant.compute",
        title="Compute a multivariate polynomial resultant over QQ",
        description=(
            "Compute the exact resultant of two multivariate polynomials with "
            "respect to one declared variable over QQ[x_1, ..., x_n].  The "
            "resultant lives in the ring over the remaining variables.  "
            "Nonzero inputs that are constant in the eliminated variable "
            "follow the Sylvester power rule Res_x(f, c) = c^deg_x(f) "
            "(symmetrically for the right input), two such constants give "
            "the empty-determinant value 1, and a zero input gives 0.  Backed "
            "by SymPy's resultant."
        ),
        request_type=MultivariateResultantRequest,
        result_type=MultivariateResultantResult,
        run=_compute_resultant,
        tags=("polynomial", "resultant", "multivariate", "rational", "elimination"),
        examples=(_RESULTANT_EXAMPLE,),
    ),
    MathTool(
        operation_id="polynomial.multivariate.subresultant_sequence.compute",
        title="Compute an exact multivariate-coefficient subresultant sequence",
        description=(
            "Compute the complete nonzero Brown subresultant PRS of two "
            "bounded polynomials in one declared main variable over the exact "
            "QQ polynomial ring in all remaining variables. Return every "
            "source-bound member, skipped member degrees, every principal "
            "subresultant coefficient including zeros, the original-orientation "
            "Sylvester resultant, and the final fraction-field GCD relation. "
            "A pinned SymPy Brown PRS kernel stays private; Jacobian fixes "
            "ordering, signs, bounds, and exact replay."
        ),
        request_type=MultivariateSubresultantSequenceRequest,
        result_type=MultivariateSubresultantSequenceResult,
        run=_compute_subresultants,
        tags=(
            "polynomial",
            "subresultant",
            "polynomial remainder sequence",
            "multivariate",
            "rational",
            "projection",
            "lifting",
        ),
        examples=(_SUBRESULTANT_EXAMPLE,),
    ),
    MathTool(
        operation_id="polynomial.multivariate.factor.compute",
        title="Factor a multivariate polynomial over QQ",
        description=(
            "Exact content and complete irreducible factorization with "
            "multiplicities for one bounded nonzero multivariate polynomial over "
            "QQ in >=2 variables (univariate inputs use polynomial.factor.compute). "
            "Worker interruption, failure, or a decomposition outside the admitted "
            "factor representation is reported as an operational error. Backed by "
            "SymPy factor_list."
        ),
        request_type=MultivariateFactorRequest,
        result_type=MultivariateFactorResult,
        run=_compute_factor,
        tags=("polynomial", "factorization", "multivariate", "rational"),
        examples=(
            OperationExample(
                name="factor_xy_squared_minus_x",
                description="Factor the nonzero multivariate polynomial x^2*y - x in Q[x,y]; "
                "the polynomial must use the same canonical ordered QQ ring and "
                "contain at least two variables.",
                input={
                    "polynomial": {
                        "domain": "QQ",
                        "variables": ["x", "y"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2, 1],
                                },
                                {
                                    "coefficient": {"num": "-1", "den": "1"},
                                    "exponents": [1, 0],
                                },
                            ]
                        },
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
