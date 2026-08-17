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
    parity_check_matrix,
    puncture_code,
    shorten_code,
)


def compute_dual_code(request: DualCodeRequest) -> ParityCheckResult:
    code = request.code
    h = parity_check_matrix(
        [list(row) for row in code.generator_matrix], code.field_order
    )
    return ParityCheckResult(
        parity_check_matrix=tuple(tuple(row) for row in h),
        field_order=code.field_order,
        code_length=len(code.generator_matrix[0]),
        code_dimension=len(code.generator_matrix),
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
    result = shorten_code(
        [list(row) for row in code.generator_matrix],
        code.field_order,
        request.position,
        request.value,
    )
    return ShortenResult(
        generator_matrix=tuple(tuple(row) for row in result),
        field_order=code.field_order,
        code_length=len(result[0]) if result else 0,
    )
