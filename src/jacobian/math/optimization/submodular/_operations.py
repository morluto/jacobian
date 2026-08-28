"""Domain-owned submodular optimization operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import format_canonical_rational, require_bounded_rational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.optimization.submodular._models import (
    MAX_SUBMODULAR_SCAN_VALUE_DIGITS,
    MonotonicityCheckRequest,
    MonotonicityCheckResult,
    SetFunction,
    SetFunctionEvalRequest,
    SetFunctionEvalResult,
    SubmodularityCheckRequest,
    SubmodularityCheckResult,
)


def _subset_label(mask: int, size: int) -> tuple[int, ...]:
    return tuple(index for index in range(size) if mask & (1 << index))


def _table_by_mask(function: SetFunction) -> dict[int, Fraction]:
    table: dict[int, Fraction] = {}
    for entry in function.entries:
        mask = 0
        for element in entry.subset:
            mask |= 1 << element
        table[mask] = entry.value.as_fraction()
    return table


def _admit_scan(function: SetFunction) -> None:
    for entry in function.entries:
        try:
            require_bounded_rational(
                entry.value,
                max_digits=MAX_SUBMODULAR_SCAN_VALUE_DIGITS,
                label="set-function scan value",
            )
        except ValueError as error:
            raise OperationDomainValidationError(
                location=("function", "entries"),
                code="submodular_opt.scan_value_height_exceeded",
                message=str(error),
            ) from error


def evaluate_set_function(
    request: SetFunctionEvalRequest,
) -> SetFunctionEvalResult:
    """Evaluate f(S) by table lookup."""
    val = _lookup(request.function, request.subset)
    if val is not None:
        return SetFunctionEvalResult(value=format_canonical_rational(val), found=True)
    return SetFunctionEvalResult(value="0", found=False)


def _lookup(
    function: SetFunction,
    subset: tuple[int, ...],
) -> Fraction | None:
    """Look up f(S) in the table; return None if not found."""
    key = tuple(sorted(subset))
    for entry in function.entries:
        if tuple(sorted(entry.subset)) == key:
            return entry.value.as_fraction()
    return None


def check_monotonicity(
    request: MonotonicityCheckRequest,
) -> MonotonicityCheckResult:
    """Check if a set function is monotone non-decreasing.

    f is monotone iff every covering relation preserves order: for each S
    and each i not in S, f(S) <= f(S | {i}).  This is O(n * 2^n) covering
    checks; violating any one covering relation violates some comparable
    pair, so the local scan is exact.
    """
    _admit_scan(request.function)
    size = request.function.ground_set_size
    table = _table_by_mask(request.function)

    for mask in range(1 << size):
        value_mask = table[mask]
        for index in range(size):
            bit = 1 << index
            if mask & bit:
                continue
            supersets_value = table[mask | bit]
            if value_mask > supersets_value:
                return MonotonicityCheckResult(
                    is_monotone=False,
                    violation=(
                        f"f({_subset_label(mask, size)}) > "
                        f"f({_subset_label(mask | bit, size)})"
                    ),
                )
    return MonotonicityCheckResult(is_monotone=True, violation="")


def check_submodularity(
    request: SubmodularityCheckRequest,
) -> SubmodularityCheckResult:
    """Check if a set function is submodular.

    Exact local characterization: f is submodular iff for every S and every
    two distinct i, j outside S,

        f(S | {i}) + f(S | {j}) >= f(S) + f(S | {i, j}).

    This needs C(n,2) checks per subset instead of the O(4^n) all-pairs
    scan, and it is complete: any violated inequality anywhere in 2^N has a
    violated local instance (take S minimal inside the differing part).
    """
    _admit_scan(request.function)
    size = request.function.ground_set_size
    table = _table_by_mask(request.function)

    full_mask = (1 << size) - 1
    complement_pairs = [
        (1 << i, 1 << j) for i in range(size) for j in range(i + 1, size)
    ]
    for mask in range(1 << size):
        base_value = table[mask]
        remaining = full_mask & ~mask
        for bit_i, bit_j in complement_pairs:
            if (bit_i | bit_j) & ~remaining:
                continue
            lhs = table[mask | bit_i] + table[mask | bit_j]
            rhs = base_value + table[mask | bit_i | bit_j]
            if lhs < rhs:
                return SubmodularityCheckResult(
                    is_submodular=False,
                    violation=(
                        f"f({_subset_label(mask | bit_i, size)}) + "
                        f"f({_subset_label(mask | bit_j, size)}) < "
                        f"f({_subset_label(mask, size)}) + "
                        f"f({_subset_label(mask | bit_i | bit_j, size)})"
                    ),
                )
    return SubmodularityCheckResult(is_submodular=True, violation="")


__all__ = [
    "check_monotonicity",
    "check_submodularity",
    "evaluate_set_function",
]
