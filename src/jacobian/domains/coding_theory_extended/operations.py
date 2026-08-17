"""Domain adapter for extended coding theory operations."""

from __future__ import annotations

from jacobian.contracts.coding_theory_extended import (
    DualCodeRequest,
    ParityCheckResult,
    PunctureRequest,
    PunctureResult,
    ShortenRequest,
    ShortenResult,
)
from jacobian.math.coding_theory_extended import (
    matrix_rank,
    parity_check_matrix,
    puncture_code,
    shorten_code,
)


def compute_dual_code(request: DualCodeRequest) -> ParityCheckResult:
    code = request.code
    generator = [list(row) for row in code.generator_matrix]
    h = parity_check_matrix(generator, code.field_order)
    n_cols = len(generator[0])
    code_dimension = matrix_rank(generator, code.field_order)
    return ParityCheckResult(
        parity_check_matrix=tuple(tuple(row) for row in h),
        field_order=code.field_order,
        code_length=n_cols,
        code_dimension=code_dimension,
    )


def compute_puncture(request: PunctureRequest) -> PunctureResult:
    code = request.code
    result = puncture_code(
        [list(row) for row in code.generator_matrix],
        code.field_order,
        request.position,
    )
    return PunctureResult(
        generator_matrix=tuple(tuple(row) for row in result),
        field_order=code.field_order,
        code_length=len(result[0]) if result else 0,
    )


def compute_shorten(request: ShortenRequest) -> ShortenResult:
    code = request.code
    generator = [list(row) for row in code.generator_matrix]
    result = shorten_code(generator, code.field_order, request.position)
    # Shortening deletes exactly one coordinate, so the ambient length is the
    # original code length minus one, even when the shortened basis is empty
    # (the zero subcode).
    code_length = len(generator[0]) - 1
    return ShortenResult(
        generator_matrix=tuple(tuple(row) for row in result),
        field_order=code.field_order,
        code_length=code_length,
    )
