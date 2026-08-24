"""Typed declarations for sum-of-squares operations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.sum_of_squares._models import (
    GramCertificateRequest,
    GramCertificateResult,
    SOSDecompositionCheckRequest,
    SOSDecompositionCheckResult,
)
from jacobian.math.sum_of_squares._operations import (
    check_gram_certificate,
    check_sos_decomposition,
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


_SOS_CHECK_EXAMPLE: dict[str, Any] = {
    "polynomial": {
        "polynomial_schema_version": "1",
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
            "polynomial_schema_version": "1",
            "domain": "QQ",
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": "1", "den": "1"}, "exponents": [1]},
                ]
            },
        },
        {
            "polynomial_schema_version": "1",
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
        "polynomial_schema_version": "1",
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
            "polynomial_schema_version": "1",
            "domain": "QQ",
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
                ]
            },
        },
        {
            "polynomial_schema_version": "1",
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
        "matrix_schema_version": "1",
        "domain": "QQ",
        "entries": [
            [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
            [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
        ],
    },
}


SOS_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    sos_operation(
        "polynomial.sos.decomposition.check",
        "Check a rational sum-of-squares decomposition by exact coefficient identity",
        "Given a polynomial p and summands q_1, ..., q_r, check that "
        "p = q_1^2 + ... + q_r^2 by exact coefficient identity over QQ. "
        "This is the trust-critical boundary: a floating comparison or "
        "solver status is not a mathematical certificate.",
        SOSDecompositionCheckRequest,
        SOSDecompositionCheckResult,
        check_sos_decomposition,
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
        check_gram_certificate,
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

TOOLS = SOS_OPERATIONS

__all__ = ["TOOLS"]
