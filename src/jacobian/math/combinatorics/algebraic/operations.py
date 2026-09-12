"""Exact algebraic combinatorics kernels over Young diagrams.

The partition kernels consume the canonical ``IntegerPartition`` value. They
use only exact integer arithmetic and are private implementation details of
the public operations.
"""

from __future__ import annotations

from fractions import Fraction
from math import factorial, prod

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.algebraic._models import (
    DominanceRelation,
    RSKResult,
    SemistandardTableauCheckResult,
    StandardTableauCheckResult,
)
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
    "hook_lengths",
    "inverse_row_insertion_rsk",
    "partition_dominance",
    "row_insertion_rsk",
    "semistandard_young_tableaux_count",
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
# T(m) <= 3T(ceil(m/2)) + 32m above it, unrolled one level at a time so each
# level contributes its own shrunken linear term, down to schoolbook leaves
# at their actual shrunken size.  All quantities derive from canonical source
# dimensions and the supplied alphabet before expansion.
MAX_HOOK_CONTENT_WORK = 8_000_000
_MAX_SSYT_COUNT_DIGITS = MAX_CANONICAL_INTEGER_DIGITS

_KARATSUBA_LIMB_THRESHOLD = 70


def _limbs_for_digits(digits: int) -> int:
    """Return a limb upper bound for a decimal-digit count (30-bit limbs)."""

    return (digits * 1108) // 10_000 + 1


def _balanced_multiplication_work(limbs: int) -> int:
    """Bound one balanced multiplication of the given limb size."""

    if limbs <= _KARATSUBA_LIMB_THRESHOLD:
        return limbs * limbs + 2 * limbs
    work = 0
    size = limbs
    count = 1
    while size > _KARATSUBA_LIMB_THRESHOLD:
        work += count * 32 * size
        size = (size + 1) // 2
        count *= 3
    return work + count * (size * size + 2 * size)


def _product_tree_work(factor_count: int, factor_digits: int) -> int:
    """Bound a balanced product tree of equal-width factors."""

    if factor_count <= 1:
        return 0
    work = 0
    widths = [factor_digits] * factor_count
    while len(widths) > 1:
        nxt: list[int] = []
        for index in range(0, len(widths), 2):
            if index + 1 == len(widths):
                nxt.append(widths[index])
                continue
            left = widths[index]
            right = widths[index + 1]
            work += _multiplication_work(
                _limbs_for_digits(left), _limbs_for_digits(right)
            )
            if work > MAX_HOOK_CONTENT_WORK:
                return work
            nxt.append(left + right)
        widths = nxt
    return work


def _balanced_product(values: tuple[int, ...]) -> int:
    """Multiply factors in a balanced tree matching the admission cost model."""

    if not values:
        return 1
    pending = list(values)
    while len(pending) > 1:
        nxt: list[int] = []
        for index in range(0, len(pending), 2):
            if index + 1 == len(pending):
                nxt.append(pending[index])
            else:
                nxt.append(pending[index] * pending[index + 1])
        pending = nxt
        request_checkpoint("during hook-content product tree")
    return pending[0]


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


# log10(2) is in (0.301029995, 0.301029996). 1/ln(10) is in
# (0.434294481, 0.434294482). These 1e-9 units keep the per-factor bound
# sound while removing the bit_length * log10(2) slack on powers of two.
_LOG10_SCALE = 1_000_000_000
_LOG10_2_UPPER_UNITS = 301_029_996
_LOG10_2_LOWER_UNITS = 301_029_995
_INV_LN10_UPPER_UNITS = 434_294_482


def _log10_upper_units(value: int) -> int:
    """Return U such that log10(value) < U / _LOG10_SCALE for value >= 1."""

    bit_length = value.bit_length()
    leading = 1 << (bit_length - 1)
    remainder = value - leading
    units = (bit_length - 1) * _LOG10_2_UPPER_UNITS
    if remainder == 0:
        return units
    return units + (remainder * _INV_LN10_UPPER_UNITS + leading - 1) // leading


def _log10_lower_units(value: int) -> int:
    """Return L such that log10(value) >= L / _LOG10_SCALE for value >= 1."""

    return max(value.bit_length() - 1, 0) * _LOG10_2_LOWER_UNITS


def _digits_upper_from_log10_units(units: int) -> int:
    """Return a digit upper bound from a strict log10 upper bound.

    Decimal width is ``floor(log10 n) + 1``. A floor of an approximate log can
    undershoot that width, so this uses ``ceil(U) + 1`` for
    U = units/_LOG10_SCALE > log10 n.
    """

    if units <= 0:
        return 1
    return (units + _LOG10_SCALE - 1) // _LOG10_SCALE + 1


def _ssyt_count_digit_bound(
    partition: IntegerPartition, alphabet_size: int, alphabet_digits: int
) -> int:
    """Upper-bound the exact SSYT count's decimal width after hook cancellation."""

    cell_count = sum(partition.parts)
    if cell_count == 0:
        return max(1, alphabet_digits)
    conjugate = conjugate_partition(partition).parts
    numerator_log_units = 0
    hook_product = 1
    for row, length in enumerate(partition.parts):
        for column in range(length):
            content = alphabet_size + column - row
            if content <= 0:
                return max(1, alphabet_digits)
            numerator_log_units += _log10_upper_units(content)
            hook_product *= length - column + conjugate[column] - row - 1
    if cell_count == 1:
        return max(1, alphabet_digits, _upper_decimal_digits(alphabet_size))
    cancelled_units = numerator_log_units - _log10_lower_units(hook_product)
    if cancelled_units < 0:
        return max(1, alphabet_digits)
    return max(1, alphabet_digits, _digits_upper_from_log10_units(cancelled_units))


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
        _ssyt_count_digit_bound(partition, alphabet_size, alphabet_digits),
    )
    if output_digits > _MAX_SSYT_COUNT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("alphabet_size",),
            code="algebraic_combinatorics.hook_content_result_digits",
            message="hook-content factors exceed the exact output digit bound",
        )

    # Charge a balanced product tree of the admitted factor widths, then the
    # exact division by the hook product.  Left-to-right accumulation of many
    # similar-width factors overcharges cheap multi-cell counts near the
    # exact-output digit boundary.
    hook_product_digits = cell_count * hook_digits
    work = _product_tree_work(cell_count, factor_digits)
    work += _product_tree_work(cell_count, hook_digits)
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


def semistandard_young_tableaux_count(
    partition: IntegerPartition, alphabet_size: int
) -> int:
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
    numerator_product = _balanced_product(numerators)
    hook_product = _balanced_product(tuple(hook for row in hooks for hook in row))
    quotient = Fraction(numerator_product, hook_product)
    if quotient.denominator != 1:
        raise ValueError("hook-content formula did not produce an integer")
    return quotient.numerator


def partition_dominance(
    left: IntegerPartition, right: IntegerPartition
) -> DominanceRelation:
    """Compare partitions using leading-row prefix sums privately."""
    if sum(left.parts) != sum(right.parts):
        return "NOT_COMPARABLE_DIFFERENT_SIZE"
    length = max(len(left.parts), len(right.parts))
    left_total = right_total = 0
    left_ge = right_ge = True
    for index in range(length):
        left_total += left.parts[index] if index < len(left.parts) else 0
        right_total += right.parts[index] if index < len(right.parts) else 0
        left_ge = left_ge and left_total >= right_total
        right_ge = right_ge and left_total <= right_total
    if left_ge and right_ge:
        return "EQUAL"
    if left_ge:
        return "LEFT_DOMINATES"
    if right_ge:
        return "RIGHT_DOMINATES"
    return "INCOMPARABLE"


_MEMBERSHIP_SHAPE_ERRORS = frozenset(
    {
        "symmetric_function.partition_not_weakly_decreasing",
        "symmetric_function.partition_parts_not_positive",
    }
)


def _is_shape_rejection(error: ValidationError) -> bool:
    """Decide whether a wrapped failure is only a diagram-shape rejection."""

    types = [item["type"] for item in error.errors()]
    return bool(types) and all(item in _MEMBERSHIP_SHAPE_ERRORS for item in types)


def check_standard_tableau(tableau: StandardYoungTableau) -> StandardTableauCheckResult:
    """Return a source-bound membership decision for one standard candidate."""

    try:
        require_standard(tableau)
    except PydanticCustomError:
        return StandardTableauCheckResult(tableau=tableau, is_member=False)
    except ValidationError as error:
        if _is_shape_rejection(error):
            return StandardTableauCheckResult(tableau=tableau, is_member=False)
        raise
    return StandardTableauCheckResult(tableau=tableau, is_member=True)


def check_semistandard_tableau(
    tableau: SemistandardYoungTableau,
) -> SemistandardTableauCheckResult:
    """Return a source-bound membership decision for one semistandard candidate."""

    try:
        require_semistandard(tableau)
    except PydanticCustomError:
        return SemistandardTableauCheckResult(tableau=tableau, is_member=False)
    except ValidationError as error:
        if _is_shape_rejection(error):
            return SemistandardTableauCheckResult(tableau=tableau, is_member=False)
        raise
    return SemistandardTableauCheckResult(tableau=tableau, is_member=True)
