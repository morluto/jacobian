"""Combinatorial-matrix operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.matrices.combinatorial._models import (
    DeterminantProfileRequest,
    DeterminantProfileResult,
    GramProfileRequest,
    GramProfileResult,
    NormalizeRequest,
    NormalizeResult,
    SignProfileRequest,
    SignProfileResult,
    SylvesterRequest,
    SylvesterResult,
)
from jacobian.math.matrices.combinatorial.operations import (
    determinant_profile,
    gram_profile,
    normalize,
    sign_profile,
    sylvester,
)


def compute_sign_profile(request: SignProfileRequest) -> SignProfileResult:
    return sign_profile(request.matrix)


def compute_gram_profile(request: GramProfileRequest) -> GramProfileResult:
    return gram_profile(request.matrix)


def compute_normalize(request: NormalizeRequest) -> NormalizeResult:
    return normalize(request.matrix)


def compute_determinant_profile(
    request: DeterminantProfileRequest,
) -> DeterminantProfileResult:
    return determinant_profile(request.matrix)


def compute_sylvester(request: SylvesterRequest) -> SylvesterResult:
    return sylvester(request.k)


# The order-2 Hadamard matrix.
_H2 = [[1, 1], [1, -1]]


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="matrix.sign.profile.compute",
        title="Compute the sign profile of a sign matrix",
        description="Return dimensions, entry counts, row/column sums, and square-ness "
        "for a general {-1, +1} sign matrix.",
        request_type=SignProfileRequest,
        result_type=SignProfileResult,
        run=compute_sign_profile,
        tags=("combinatorial-matrix", "sign-profile", "exact"),
        examples=(
            OperationExample(
                name="order_2_sign_profile",
                description="Sign profile of the order-2 Hadamard matrix.",
                input={"matrix": {"rows": _H2}},
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.hadamard.gram_profile.compute",
        title="Compute the Gram profile of a sign matrix",
        description="Return order, exact H H^T, diagonal residuals from n, all nonzero "
        "off-diagonal inner products, and is_hadamard. Row and column counts "
        "are admitted by Gram multiply-add work and exact-result size.",
        request_type=GramProfileRequest,
        result_type=GramProfileResult,
        run=compute_gram_profile,
        tags=("combinatorial-matrix", "gram-profile", "exact"),
        examples=(
            OperationExample(
                name="order_2_gram_profile",
                description="Gram profile of the order-2 Hadamard matrix.",
                input={"matrix": {"rows": _H2}},
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.hadamard.normalize.compute",
        title="Normalize a sign matrix so first row/column are all +1",
        description="Return a deterministically normalized sign matrix whose first row "
        "and first column are all +1, plus the exact row/column sign switches "
        "used. Normalization preserves the full matrix and is idempotent.",
        request_type=NormalizeRequest,
        result_type=NormalizeResult,
        run=compute_normalize,
        tags=("combinatorial-matrix", "normalize", "exact"),
        examples=(
            OperationExample(
                name="order_2_normalize",
                description="Normalize the order-2 Hadamard matrix.",
                input={"matrix": {"rows": _H2}},
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.hadamard.determinant_profile.compute",
        title="Compute the determinant profile of a Hadamard matrix",
        description="For a square sign matrix of order n, return |det H| = n^(n/2), "
        "the Gram determinant = n^n, and the identity det(H)^2 = "
        "det(H H^T) when H H^T = n I_n exactly. Determinant magnitude is "
        "not inferred from a matrix that fails exact orthogonality.",
        request_type=DeterminantProfileRequest,
        result_type=DeterminantProfileResult,
        run=compute_determinant_profile,
        tags=("combinatorial-matrix", "determinant-profile", "exact"),
        examples=(
            OperationExample(
                name="order_2_determinant",
                description="Determinant profile of the order-2 Hadamard matrix.",
                input={"matrix": {"rows": _H2}},
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.hadamard.sylvester.compute",
        title="Construct the Sylvester Hadamard matrix of order 2^k",
        description="For bounded k, return the recursively defined order 2^k Hadamard "
        "matrix with construction ledger. A finite constructor, not an "
        "existence search.",
        request_type=SylvesterRequest,
        result_type=SylvesterResult,
        run=compute_sylvester,
        tags=("combinatorial-matrix", "sylvester", "exact"),
        examples=(
            OperationExample(
                name="sylvester_k1",
                description="Sylvester construction for k=1 (order 2).",
                input={"k": 1},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
