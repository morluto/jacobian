"""Domain-owned submodular optimization operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian.canonical import format_canonical_integer
from jacobian.math.submodular_opt._models import (
    MonotonicityCheckRequest,
    MonotonicityCheckResult,
    SetFunction,
    SetFunctionEvalRequest,
    SetFunctionEvalResult,
    SubmodularityCheckRequest,
    SubmodularityCheckResult,
)


def _format_rational(value: Fraction) -> str:
    if value.denominator == 1:
        return format_canonical_integer(value.numerator)
    return (
        f"{format_canonical_integer(value.numerator)}/"
        f"{format_canonical_integer(value.denominator)}"
    )


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


def evaluate_set_function(
    request: SetFunctionEvalRequest,
) -> SetFunctionEvalResult:
    """Evaluate f(S) by table lookup."""
    val = _lookup(request.function, request.subset)
    if val is not None:
        return SetFunctionEvalResult(value=_format_rational(val), found=True)
    return SetFunctionEvalResult(value="0", found=False)


def check_monotonicity(
    request: MonotonicityCheckRequest,
) -> MonotonicityCheckResult:
    """Check if a set function is monotone non-decreasing.

    f is monotone iff for all S subset T: f(S) <= f(T).
    We check all pairs S, T with S subset T and |T| = |S| + 1.
    """
    entries = request.function.entries
    # Build dict
    table: dict[tuple[int, ...], Fraction] = {}
    for entry in entries:
        key = tuple(sorted(entry.subset))
        table[key] = entry.value.as_fraction()

    for subset_str, val_s in table.items():
        subset = set(subset_str)
        for entry in entries:
            other = set(entry.subset)
            if (
                subset < other
                and len(other) == len(subset) + 1
                and val_s > entry.value.as_fraction()
            ):
                return MonotonicityCheckResult(
                    is_monotone=False,
                    violation=f"f({subset_str}) > f({tuple(sorted(other))})",
                )
    return MonotonicityCheckResult(is_monotone=True, violation="")


def check_submodularity(
    request: SubmodularityCheckRequest,
) -> SubmodularityCheckResult:
    """Check if a set function is submodular.

    f is submodular iff for all S, T: f(S) + f(T) >= f(S union T) + f(S intersection T).
    """
    entries = request.function.entries
    table: dict[tuple[int, ...], Fraction] = {}
    for entry in entries:
        key = tuple(sorted(entry.subset))
        table[key] = entry.value.as_fraction()

    keys = list(table.keys())
    for i, s_key in enumerate(keys):
        s_set = set(s_key)
        for j in range(i + 1, len(keys)):
            t_key = keys[j]
            t_set = set(t_key)
            union_key = tuple(sorted(s_set | t_set))
            inter_key = tuple(sorted(s_set & t_set))
            if union_key in table and inter_key in table:
                lhs = table[s_key] + table[t_key]
                rhs = table[union_key] + table[inter_key]
                if lhs < rhs:
                    return SubmodularityCheckResult(
                        is_submodular=False,
                        violation=f"f({s_key}) + f({t_key}) < f({union_key}) + f({inter_key})",
                    )
    return SubmodularityCheckResult(is_submodular=True, violation="")


__all__ = [
    "check_monotonicity",
    "check_submodularity",
    "evaluate_set_function",
]
