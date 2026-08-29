"""Combinatorial-matrix operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


# The order-2 Hadamard matrix.
_H2 = [[1, 1], [1, -1]]


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "matrix.sign.profile.compute",
        "Compute the sign profile of a sign matrix",
        "Return dimensions, entry counts, row/column sums, and square-ness "
        "for a general {-1, +1} sign matrix.",
        SignProfileRequest,
        SignProfileResult,
        compute_sign_profile,
        "combinatorial-matrix",
        "sign-profile",
        "exact",
        examples=(
            example(
                "order_2_sign_profile",
                "Sign profile of the order-2 Hadamard matrix.",
                {"matrix": {"rows": _H2}},
            ),
        ),
    ),
    _op(
        "matrix.hadamard.gram_profile.compute",
        "Compute the Gram profile of a sign matrix",
        "Return order, exact H H^T, diagonal residuals from n, all nonzero "
        "off-diagonal inner products, and is_hadamard. Row and column counts "
        "are admitted by Gram multiply-add work and exact-result size.",
        GramProfileRequest,
        GramProfileResult,
        compute_gram_profile,
        "combinatorial-matrix",
        "gram-profile",
        "exact",
        examples=(
            example(
                "order_2_gram_profile",
                "Gram profile of the order-2 Hadamard matrix.",
                {"matrix": {"rows": _H2}},
            ),
        ),
    ),
    _op(
        "matrix.hadamard.normalize.compute",
        "Normalize a sign matrix so first row/column are all +1",
        "Return a deterministically normalized sign matrix whose first row "
        "and first column are all +1, plus the exact row/column sign switches "
        "used. Normalization preserves the full matrix and is idempotent.",
        NormalizeRequest,
        NormalizeResult,
        compute_normalize,
        "combinatorial-matrix",
        "normalize",
        "exact",
        examples=(
            example(
                "order_2_normalize",
                "Normalize the order-2 Hadamard matrix.",
                {"matrix": {"rows": _H2}},
            ),
        ),
    ),
    _op(
        "matrix.hadamard.determinant_profile.compute",
        "Compute the determinant profile of a Hadamard matrix",
        "For a square sign matrix of order n, return |det H| = n^(n/2), "
        "the Gram determinant = n^n, and the identity det(H)^2 = "
        "det(H H^T) when H H^T = n I_n exactly. Determinant magnitude is "
        "not inferred from a matrix that fails exact orthogonality.",
        DeterminantProfileRequest,
        DeterminantProfileResult,
        compute_determinant_profile,
        "combinatorial-matrix",
        "determinant-profile",
        "exact",
        examples=(
            example(
                "order_2_determinant",
                "Determinant profile of the order-2 Hadamard matrix.",
                {"matrix": {"rows": _H2}},
            ),
        ),
    ),
    _op(
        "matrix.hadamard.sylvester.compute",
        "Construct the Sylvester Hadamard matrix of order 2^k",
        "For bounded k, return the recursively defined order 2^k Hadamard "
        "matrix with construction ledger. A finite constructor, not an "
        "existence search.",
        SylvesterRequest,
        SylvesterResult,
        compute_sylvester,
        "combinatorial-matrix",
        "sylvester",
        "exact",
        examples=(
            example(
                "sylvester_k1",
                "Sylvester construction for k=1 (order 2).",
                {"k": 1},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
