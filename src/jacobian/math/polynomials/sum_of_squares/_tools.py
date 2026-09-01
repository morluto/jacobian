"""Typed declarations for sum-of-squares operations."""

from typing import Any

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
    MathTool(
        operation_id="polynomial.sos.decomposition.check",
        title="Check a rational sum-of-squares decomposition by exact coefficient identity",
        description="Given a polynomial p and zero or more summands q_1, ..., q_r, check that "
        "p = q_1^2 + ... + q_r^2 by exact coefficient identity over QQ "
        "(the empty sum is zero). "
        "This is the trust-critical boundary: a floating comparison or "
        "solver status is not a mathematical certificate.",
        request_type=SOSDecompositionCheckRequest,
        result_type=SOSDecompositionCheckResult,
        run=_check_sos,
        tags=("sum-of-squares", "decomposition", "exact", "certificate"),
        examples=(
            OperationExample(
                name="x_squared_plus_one",
                description="Check that x^2 + 1 = x^2 + 1^2.",
                input=_SOS_CHECK_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.sos.gram.check",
        title="Check a rational Gram certificate for sum-of-squares",
        description="Given a polynomial p, monomial basis z, and symmetric rational "
        "matrix Q, check that p = z^T Q z with Q symmetric and PSD by exact "
        "rational arithmetic. This proves nonnegativity without floating "
        "eigenvalue computation.",
        request_type=GramCertificateRequest,
        result_type=GramCertificateResult,
        run=_check_gram,
        tags=("sum-of-squares", "gram", "exact", "certificate"),
        examples=(
            OperationExample(
                name="x_squared_plus_one_gram",
                description="Check that x^2+1 = [1, x]^T I [1, x] with identity Gram matrix.",
                input=_GRAM_CHECK_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
