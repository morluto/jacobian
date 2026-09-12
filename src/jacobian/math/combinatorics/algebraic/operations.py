"""Exact algebraic combinatorics kernels over Young diagrams.

The partition kernels consume the canonical ``IntegerPartition`` value. They
use only exact integer arithmetic and are private implementation details of
the public operations.
"""

from __future__ import annotations

from fractions import Fraction
from math import factorial, prod
from typing import cast

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.algebraic._models import DominanceRelation, RSKResult
from jacobian.math.combinatorics.algebraic._rsk import (
    _row_insert,
)
from jacobian.math.combinatorics.algebraic._rsk import (
    inverse_row_insertion_rsk as _inverse_row_insertion_rsk,
)
from jacobian.math.combinatorics.algebraic._rsk import (
    row_insertion_rsk as _row_insertion_rsk,
)
from jacobian.math.combinatorics.algebraic.values import RSKTableauPair
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
    StandardYoungTableau,
    require_semistandard,
    require_standard,
)
from jacobian.math.logic.languages.words.values import FiniteWord

__all__ = [
    "check_semistandard_tableau",
    "check_standard_tableau",
    "conjugate_partition",
    "hook_content_count",
    "hook_lengths",
    "inverse_row_insertion_rsk",
    "partition_dominance",
    "row_insertion_rsk",
    "standard_young_tableaux_count",
    "verify_rsk",
]


# Hook-content evaluation performs exact products of one factor per cell.  The
# first bound protects the serialized integer fields; the second bounds the
# bigint arithmetic.  CPython multiplies limb vectors with schoolbook
# products below 70 limbs and Karatsuba-style recursion above, so each
# multiplication of limb sizes (a, b) is charged by splitting the larger
# operand into ceil(a/b) blocks of the smaller size: schoolbook block cost
# below the threshold and the Karatsuba recurrence
# T(m) <= 3T(ceil(m/2)) + 32m above it, unrolled to schoolbook leaves.  All
# quantities derive from canonical source dimensions and the supplied
# alphabet before expansion.
MAX_HOOK_CONTENT_WORK = 8_000_000

_KARATSUBA_LIMB_THRESHOLD = 70
_KARATSUBA_SCHOOLBOOK_LEAF_COST = _KARATSUBA_LIMB_THRESHOLD * _KARATSUBA_LIMB_THRESHOLD


def _limbs_for_digits(digits: int) -> int:
    """Return a limb upper bound for a decimal-digit count (30-bit limbs)."""

    return (digits * 1108) // 10_000 + 1


def _balanced_multiplication_work(limbs: int) -> int:
    """Bound one balanced multiplication of the given limb size."""

    if limbs <= _KARATSUBA_LIMB_THRESHOLD:
        return limbs * limbs + 2 * limbs
    ratio = (limbs + _KARATSUBA_LIMB_THRESHOLD - 1) // _KARATSUBA_LIMB_THRESHOLD
    levels = (ratio - 1).bit_length()
    return (3**levels) * (_KARATSUBA_SCHOOLBOOK_LEAF_COST + 32 * limbs)


def _multiplication_work(a_limbs: int, b_limbs: int) -> int:
    """Bound one multiplication of the given limb sizes via blocking."""

    if a_limbs < b_limbs:
        a_limbs, b_limbs = b_limbs, a_limbs
    if b_limbs <= 0:
        return a_limbs
    blocks = (a_limbs + b_limbs - 1) // b_limbs
    return blocks * _balanced_multiplication_work(b_limbs)


def _upper_decimal_digits(value: int) -> int:
    """Return the exact decimal-digit count without converting to text."""
    estimate = (value.bit_length() * 30103) // 100000 + 1
    if value < 10 ** (estimate - 1):
        return estimate - 1
    return estimate


def _admit_hook_content(partition: IntegerPartition, alphabet_size: int) -> None:
    """Admit hook-content arithmetic before constructing any factors."""
    if type(alphabet_size) is not int or alphabet_size < 1:
        raise ValueError("alphabet_size must be a positive integer")

    cell_count = sum(partition.parts)
    alphabet_digits = _upper_decimal_digits(alphabet_size)
    if cell_count:
        # The positive numerator maximum is attained at the top-right cell;
        # the greatest negative magnitude is attained at the bottom-left
        # cell.  Use those actual offsets instead of charging alphabet + N,
        # which rejects a one-cell result when the alphabet is at the exact
        # scalar digit boundary.
        largest_positive = alphabet_size + partition.parts[0] - 1
        largest_negative = max(0, len(partition.parts) - 1 - alphabet_size)
        factor_digits = _upper_decimal_digits(max(largest_positive, largest_negative))
    else:
        factor_digits = 1
    # The alphabet is retained in the result even for the empty shape, so its
    # own serialized size is part of the output bound.
    hook_digits = _upper_decimal_digits(max(1, cell_count))
    output_digits = max(
        alphabet_digits,
        factor_digits,
        cell_count * factor_digits,
        cell_count * hook_digits,
    )
    if output_digits > MAX_CANONICAL_INTEGER_DIGITS:
        raise OperationResourceAdmissionError(
            location=("alphabet_size",),
            code="algebraic_combinatorics.hook_content_result_digits",
            message="hook-content factors exceed the exact output digit bound",
        )

    # The kernel accumulates each product left to right, so the i-th numerator
    # multiplication joins an accumulator of at most i factor widths with one
    # new factor; hooks accumulate likewise.  Charge every multiplication at
    # its own admitted size (not at the final size, and not as a fixed-size
    # digit-square product, which excludes cheap large-scalar requests), plus
    # the final exact division charged by the divisor width.
    factor_limbs = _limbs_for_digits(factor_digits)
    hook_limbs = _limbs_for_digits(hook_digits)
    hook_product_digits = cell_count * hook_digits
    work = 0
    for step in range(1, cell_count):
        accumulator_limbs = _limbs_for_digits(step * factor_digits)
        work += _multiplication_work(accumulator_limbs, factor_limbs)
        accumulator_hook_limbs = _limbs_for_digits(step * hook_digits)
        work += _multiplication_work(accumulator_hook_limbs, hook_limbs)
        if work > MAX_HOOK_CONTENT_WORK:
            raise OperationResourceAdmissionError(
                location=("alphabet_size",),
                code="algebraic_combinatorics.hook_content_work",
                message="hook-content arithmetic exceeds the admitted work bound",
            )
    work += _multiplication_work(
        _limbs_for_digits(output_digits), _limbs_for_digits(hook_product_digits)
    )
    if work > MAX_HOOK_CONTENT_WORK:
        raise OperationResourceAdmissionError(
            location=("alphabet_size",),
            code="algebraic_combinatorics.hook_content_work",
            message="hook-content arithmetic exceeds the admitted work bound",
        )


def row_insertion_rsk(word: FiniteWord) -> RSKTableauPair:
    """Compute ordinary row-insertion RSK for a word of at most 500 letters."""
    return _row_insertion_rsk(word)


def inverse_row_insertion_rsk(pair: RSKTableauPair) -> FiniteWord:
    """Reconstruct the unique word represented by a pair of at most 500 cells."""
    return _inverse_row_insertion_rsk(pair)


def _rsk_permutation(
    permutation: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Validate and insert one strict permutation through the native kernel."""
    if sorted(permutation) != list(range(1, len(permutation) + 1)):
        raise ValueError("permutation must be a permutation of 1..n")
    return _row_insert(permutation)


def verify_rsk(claim: RSKResult) -> bool:
    """Check bounded permutation admission, RSK correspondence, and LIS/LDS.

    The retained source has at most 500 entries. Row insertion has the same
    quadratic search-count bound as the producing operation.
    """
    try:
        require_semistandard(claim.p_tableau)
        require_standard(claim.q_tableau)
        insertion, recording = _rsk_permutation(claim.permutation)
    except (TypeError, ValueError):
        return False
    return (
        insertion == claim.p_tableau.rows
        and recording == claim.q_tableau.rows
        and claim.lis_length == (len(insertion[0]) if insertion else 0)
        and claim.lds_length == len(insertion)
    )


def conjugate_partition(partition: IntegerPartition) -> IntegerPartition:
    """Return the conjugate of one canonical partition.

    The conjugate ``lambda'`` has ``lambda'_j`` equal to the number of parts of
    ``lambda`` that are at least ``j``, i.e. the column heights of the Ferrers
    diagram.
    """
    parts = partition.parts
    if not parts:
        return IntegerPartition(parts=())
    max_column = parts[0]
    return IntegerPartition(
        parts=tuple(
            sum(1 for part in parts if part >= column)
            for column in range(1, max_column + 1)
        )
    )


def hook_lengths(partition: IntegerPartition) -> tuple[tuple[int, ...], ...]:
    """Return the hook length of every cell of a canonical partition.

    The hook length of cell ``(i, j)`` (0-indexed) is
    ``lambda_i - j + lambda'_j - i - 1``: one arm step plus the cell itself
    plus the number of cells below it in its column.
    """
    parts = partition.parts
    conjugate = conjugate_partition(partition).parts
    hooks: list[list[int]] = []
    for row, length in enumerate(parts):
        row_hooks: list[int] = []
        for column in range(length):
            right = length - column - 1
            below = conjugate[column] - row - 1
            row_hooks.append(right + below + 1)
        hooks.append(row_hooks)
    return tuple(tuple(row) for row in hooks)


def _hook_length_product(hooks: tuple[tuple[int, ...], ...]) -> int:
    """Return the exact product of one hook-length array."""
    return prod(hook for row in hooks for hook in row)


def standard_young_tableaux_count(partition: IntegerPartition) -> int:
    """Count standard Young tableaux of a canonical partition.

    The number of standard Young tableaux of shape ``lambda`` is
    ``n! / prod_{(i,j) in lambda} h(i,j)`` where ``n = |lambda|`` and
    ``h(i,j)`` is the cell's hook length.
    """
    hooks = hook_lengths(partition)
    n = sum(partition.parts)
    return factorial(n) // _hook_length_product(hooks)


def hook_content_count(partition: IntegerPartition, alphabet_size: int) -> int:
    """Count SSYTs of ``partition`` over ``1..alphabet_size`` exactly.

    The hook-content factors are private kernel intermediates. The public
    result carries only the exact count and its source shape and alphabet.
    """
    _admit_hook_content(partition, alphabet_size)
    hooks = hook_lengths(partition)
    numerators = tuple(
        alphabet_size + column - row
        for row, length in enumerate(partition.parts)
        for column in range(length)
    )
    numerator_product = 1
    for index, factor in enumerate(numerators):
        if index and index % 32 == 0:
            request_checkpoint("during hook-content numerator accumulation")
        numerator_product *= factor
    hook_product = 1
    for index, row in enumerate(hooks):
        for position, hook in enumerate(row):
            if (index + position) and (index + position) % 32 == 0:
                request_checkpoint("during hook-content hook accumulation")
            hook_product *= hook
    quotient = Fraction(numerator_product, hook_product)
    if quotient.denominator != 1:
        raise ValueError("hook-content formula did not produce an integer")
    return quotient.numerator


def partition_dominance(
    left: IntegerPartition, right: IntegerPartition
) -> tuple[DominanceRelation, tuple[int, ...], tuple[int, ...]]:
    """Compare partitions using all leading-row prefix sums."""
    if sum(left.parts) != sum(right.parts):
        return "NOT_COMPARABLE_DIFFERENT_SIZE", (), ()
    length = max(len(left.parts), len(right.parts))
    left_sums: list[int] = []
    right_sums: list[int] = []
    left_total = right_total = 0
    for index in range(length):
        left_total += left.parts[index] if index < len(left.parts) else 0
        right_total += right.parts[index] if index < len(right.parts) else 0
        left_sums.append(left_total)
        right_sums.append(right_total)
    left_ge = all(a >= b for a, b in zip(left_sums, right_sums, strict=True))
    right_ge = all(a <= b for a, b in zip(left_sums, right_sums, strict=True))
    relation = (
        "EQUAL"
        if left_ge and right_ge
        else "LEFT_DOMINATES"
        if left_ge
        else "RIGHT_DOMINATES"
        if right_ge
        else "INCOMPARABLE"
    )
    return cast(DominanceRelation, relation), tuple(left_sums), tuple(right_sums)


def check_standard_tableau(tableau: StandardYoungTableau) -> StandardYoungTableau:
    """Return a candidate after replaying standard-tableau membership."""
    require_standard(tableau)
    return tableau


def check_semistandard_tableau(
    tableau: SemistandardYoungTableau,
) -> SemistandardYoungTableau:
    """Return a candidate after replaying semistandard membership."""
    require_semistandard(tableau)
    return tableau
