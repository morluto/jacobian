"""Matrix analysis operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.matrices.analysis._models import (
    FarkasCertificateRequest,
    FarkasCertificateResult,
    InertiaResult,
    RationalSpectrumClaimRequest,
    RationalSpectrumClaimResult,
    SymmetricMatrixRequest,
)
from jacobian.math.matrices.analysis._operations import (
    check_farkas_certificate,
    check_rational_spectrum_claim,
    compute_inertia,
)


def _op[
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


MATRIX_ANALYSIS_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "matrix.symmetric.rational_spectrum_claim.check",
        "Check a complete rational spectrum claim for a symmetric matrix",
        "For a bounded symmetric matrix over QQ and pairwise-distinct rational "
        "eigenvalue claims, return the retained source and exact shifted-nullity "
        "ledger, and decide whether the claimed multiplicities give the complete "
        "spectrum. A claim is complete exactly when every multiplicity equals its "
        "nullity and the multiplicities sum to the matrix order.",
        RationalSpectrumClaimRequest,
        RationalSpectrumClaimResult,
        check_rational_spectrum_claim,
        "matrix",
        "symmetric",
        "spectrum",
        "eigenvalue",
        "multiplicity",
        "exact",
        "check",
        examples=(
            example(
                "repeated_diagonal_rational_spectrum",
                "Check that diag(2, 2, -1) has complete spectrum 2^2, (-1)^1; "
                "the matrix must be symmetric and claimed eigenvalues distinct.",
                {
                    "matrix": {
                        "entries": [
                            [
                                {"num": "2", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "2", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "-1", "den": "1"},
                            ],
                        ]
                    },
                    "claimed_profile": [
                        {
                            "eigenvalue": {"num": "2", "den": "1"},
                            "multiplicity": 2,
                        },
                        {
                            "eigenvalue": {"num": "-1", "den": "1"},
                            "multiplicity": 1,
                        },
                    ],
                },
            ),
        ),
    ),
    _op(
        "matrix.inertia.compute",
        "Compute Sylvester inertia of a symmetric rational matrix",
        "Given a symmetric rational matrix, compute its Sylvester inertia "
        "(n_positive, n_negative, n_zero) and definiteness classification "
        "using exact rational LDL decomposition.",
        SymmetricMatrixRequest,
        InertiaResult,
        compute_inertia,
        "matrix",
        "inertia",
        "definiteness",
        "exact",
        examples=(
            example(
                "identity_inertia",
                "3x3 identity matrix has inertia (3, 0, 0).",
                {
                    "dimension": 3,
                    "entries": [
                        {"row": 0, "col": 0, "value": {"num": "1", "den": "1"}},
                        {"row": 1, "col": 1, "value": {"num": "1", "den": "1"}},
                        {"row": 2, "col": 2, "value": {"num": "1", "den": "1"}},
                    ],
                },
            ),
        ),
    ),
    _op(
        "matrix.farkas_certificate.check",
        "Check a rational Farkas infeasibility certificate",
        "Given system Ax <= b and non-negative multiplier y, verify "
        "y^T A = 0 and y^T b < 0.",
        FarkasCertificateRequest,
        FarkasCertificateResult,
        check_farkas_certificate,
        "matrix",
        "farkas",
        "infeasibility",
        "exact",
        examples=(
            example(
                "simple_farkas",
                "Simple Farkas certificate check.",
                {
                    "constraint_matrix": [
                        [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                        [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                    ],
                    "rhs_vector": [
                        {"num": "-1", "den": "1"},
                        {"num": "-1", "den": "1"},
                    ],
                    "multipliers": [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}],
                },
            ),
        ),
    ),
)


TOOLS = MATRIX_ANALYSIS_OPERATIONS

__all__ = ["TOOLS"]
