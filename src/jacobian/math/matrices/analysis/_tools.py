"""Matrix analysis operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.matrices.analysis._models import (
    FarkasCertificateRequest,
    FarkasCertificateResult,
    InertiaResult,
    RationalSpectrumClaimRequest,
    RationalSpectrumClaimResult,
    SymmetricMatrixRequest,
)
from jacobian.math.matrices.analysis.operations import (
    check_farkas_certificate as _check_farkas_certificate_native,
)
from jacobian.math.matrices.analysis.operations import (
    check_rational_spectrum_claim as _check_rational_spectrum_claim_native,
)
from jacobian.math.matrices.analysis.operations import (
    compute_inertia as _compute_inertia_native,
)


def check_rational_spectrum_claim(
    request: RationalSpectrumClaimRequest,
) -> RationalSpectrumClaimResult:
    """Unpack a wire claim for the canonical spectrum checker."""

    return _check_rational_spectrum_claim_native(
        request.matrix, request.claimed_profile
    )


def compute_inertia(request: SymmetricMatrixRequest) -> InertiaResult:
    """Compute inertia of the request's canonical exact-real matrix."""

    return _compute_inertia_native(request.matrix)


def check_farkas_certificate(
    request: FarkasCertificateRequest,
) -> FarkasCertificateResult:
    """Unpack a wire certificate for the canonical Farkas checker."""

    return _check_farkas_certificate_native(
        request.constraint_matrix,
        request.rhs_vector,
        request.multipliers,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="matrix.symmetric.rational_spectrum_claim.check",
        title="Check a complete rational spectrum claim for a symmetric matrix",
        description="For a bounded symmetric matrix over QQ and pairwise-distinct rational "
        "eigenvalue claims, return the retained source and exact shifted-nullity "
        "ledger, and decide whether the claimed multiplicities give the complete "
        "spectrum. A claim is complete exactly when every multiplicity equals its "
        "nullity and the multiplicities sum to the matrix order.",
        request_type=RationalSpectrumClaimRequest,
        result_type=RationalSpectrumClaimResult,
        run=check_rational_spectrum_claim,
        tags=(
            "matrix",
            "symmetric",
            "spectrum",
            "eigenvalue",
            "multiplicity",
            "exact",
            "check",
        ),
        examples=(
            OperationExample(
                name="repeated_diagonal_rational_spectrum",
                description="Check that diag(2, 2, -1) has complete spectrum 2^2, (-1)^1; "
                "the matrix must be symmetric and claimed eigenvalues distinct.",
                input={
                    "matrix": {
                        "domain": "QQ",
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
                        ],
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
    MathTool(
        operation_id="matrix.inertia.compute",
        title="Compute Sylvester inertia of an exact real symmetric matrix",
        description="Given a canonical rational or common-embedding real simple-number-field "
        "symmetric matrix, compute its exact Sylvester inertia "
        "(n_positive, n_negative, n_zero) and definiteness classification by "
        "congruence reduction in the retained scalar domain.",
        request_type=SymmetricMatrixRequest,
        result_type=InertiaResult,
        run=compute_inertia,
        tags=("matrix", "inertia", "definiteness", "exact"),
        examples=(
            OperationExample(
                name="identity_inertia",
                description="3x3 identity matrix has inertia (3, 0, 0).",
                input={
                    "matrix": {
                        "domain": "QQ",
                        "entries": [
                            [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                        ],
                    }
                },
            ),
            OperationExample(
                name="positive_quadratic_embedding",
                description="The positive embedding of sqrt(2) has inertia (1, 0, 0).",
                input={
                    "matrix": {
                        "domain": "EMBEDDED_REAL_SIMPLE_NUMBER_FIELD",
                        "embedding": {
                            "kind": "REAL",
                            "presentation": {
                                "domain": "QQ",
                                "coefficients_descending": ["1", "0", "-2"],
                            },
                            "root": {
                                "polynomial": ["1", "0", "-2"],
                                "real_root_index": 1,
                            },
                        },
                        "entries": [
                            [
                                {
                                    "presentation": {
                                        "domain": "QQ",
                                        "coefficients_descending": ["1", "0", "-2"],
                                    },
                                    "coefficients_ascending": [
                                        {"num": "0", "den": "1"},
                                        {"num": "1", "den": "1"},
                                    ],
                                }
                            ]
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.farkas_certificate.check",
        title="Check a rational Farkas infeasibility certificate",
        description="Given system Ax <= b and non-negative multiplier y, verify "
        "y^T A = 0 and y^T b < 0.",
        request_type=FarkasCertificateRequest,
        result_type=FarkasCertificateResult,
        run=check_farkas_certificate,
        tags=("matrix", "farkas", "infeasibility", "exact"),
        examples=(
            OperationExample(
                name="simple_farkas",
                description="Simple Farkas certificate check.",
                input={
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


__all__ = ["TOOLS"]
