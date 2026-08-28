"""Typed declarations for sum-of-squares operations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.sum_of_squares._models import (
    GramCertificateRequest,
    GramCertificateResult,
    SOSDecompositionCheckRequest,
    SOSDecompositionCheckResult,
)
from jacobian.math.polynomials.sum_of_squares.operations import (
    check_gram_certificate,
    check_sos_decomposition,
)


def _check_sos(request: SOSDecompositionCheckRequest) -> SOSDecompositionCheckResult:
    return check_sos_decomposition(request.polynomial, request.summands)


def _check_gram(request: GramCertificateRequest) -> GramCertificateResult:
    return check_gram_certificate(
        request.polynomial, request.monomial_basis, request.gram_matrix
    )


def sos_operation[
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


_SOS_CHECK_EXAMPLE: dict[str, Any] = {
    "polynomial": {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [2]},
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
            ]
        },
    },
    "summands": [
        {
            "domain": "QQ",
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": "1", "den": "1"}, "exponents": [1]},
                ]
            },
        },
        {
            "domain": "QQ",
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
                ]
            },
        },
    ],
}

_GRAM_CHECK_EXAMPLE: dict[str, Any] = {
    "polynomial": {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [2]},
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
            ]
        },
    },
    "monomial_basis": [
        {
            "domain": "QQ",
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
                ]
            },
        },
        {
            "domain": "QQ",
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": "1", "den": "1"}, "exponents": [1]},
                ]
            },
        },
    ],
    "gram_matrix": {
        "domain": "QQ",
        "entries": [
            [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
            [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
        ],
    },
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    sos_operation(
        "polynomial.sos.decomposition.check",
        "Check a rational sum-of-squares decomposition by exact coefficient identity",
        "Given a polynomial p and zero or more summands q_1, ..., q_r, check that "
        "p = q_1^2 + ... + q_r^2 by exact coefficient identity over QQ "
        "(the empty sum is zero). "
        "This is the trust-critical boundary: a floating comparison or "
        "solver status is not a mathematical certificate.",
        SOSDecompositionCheckRequest,
        SOSDecompositionCheckResult,
        _check_sos,
        "sum-of-squares",
        "decomposition",
        "exact",
        "certificate",
        examples=(
            example(
                "x_squared_plus_one",
                "Check that x^2 + 1 = x^2 + 1^2.",
                _SOS_CHECK_EXAMPLE,
            ),
        ),
    ),
    sos_operation(
        "polynomial.sos.gram.check",
        "Check a rational Gram certificate for sum-of-squares",
        "Given a polynomial p, monomial basis z, and symmetric rational "
        "matrix Q, check that p = z^T Q z with Q symmetric and PSD by exact "
        "rational arithmetic. This proves nonnegativity without floating "
        "eigenvalue computation.",
        GramCertificateRequest,
        GramCertificateResult,
        _check_gram,
        "sum-of-squares",
        "gram",
        "exact",
        "certificate",
        examples=(
            example(
                "x_squared_plus_one_gram",
                "Check that x^2+1 = [1, x]^T I [1, x] with identity Gram matrix.",
                _GRAM_CHECK_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
