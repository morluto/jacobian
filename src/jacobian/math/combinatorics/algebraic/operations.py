"""Exact algebraic combinatorics kernels over Young diagrams.

The partition kernels consume the canonical ``IntegerPartition`` value. They
use only exact integer arithmetic and are private implementation details of
the public operations.
"""

from __future__ import annotations

from fractions import Fraction
from math import factorial, prod
from typing import cast

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS
from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.algebraic._models import (
    DominanceRelation,
    KnuthMovesResult,
    KnuthNeighbor,
    PartitionDominanceResult,
    PlacticEquivalenceResult,
    PlacticNormalFormResult,
    RSKResult,
    RSKWordTraceResult,
    SemistandardTableauCheckResult,
    SemistandardYoungTableauCountResult,
    SkewLittlewoodRichardsonCheckResult,
    SkewReadingConvention,
    StandardTableauCheckResult,
)
from jacobian.math.combinatorics.algebraic._rsk import (
    _row_insert,
    word_payload_scalars,
)
from jacobian.math.combinatorics.algebraic._rsk import (
    inverse_row_insertion_rsk as _inverse_row_insertion_rsk,
)
from jacobian.math.combinatorics.algebraic._rsk import (
    row_insertion_rsk as _row_insertion_rsk,
)
from jacobian.math.combinatorics.algebraic._rsk import (
    row_insertion_rsk_trace as _row_insertion_rsk_trace,
)
from jacobian.math.combinatorics.algebraic.values import (
    MAX_RSK_ALPHABET_RANK_DIGITS,
    MAX_RSK_WORD_LENGTH,
    MAX_RSK_WORD_PAYLOAD_SCALARS,
    RSKTableauPair,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    MAX_PARTITION_PARTS,
    MAX_PARTITION_SIZE,
    IntegerPartition,
    SemistandardYoungTableau,
    StandardYoungTableau,
    TableauCandidate,
    require_semistandard,
    require_standard,
)
from jacobian.math.logic.languages.words.values import (
    MAX_SYMBOL_LENGTH,
    FiniteWord,
)

MAX_PLACTIC_EQUIVALENCE_WORK = (
    2 * MAX_RSK_WORD_LENGTH * MAX_RSK_WORD_LENGTH * MAX_RSK_WORD_LENGTH.bit_length()
)
# Each side of the combined result retains its source word (alphabet plus
# positioned letters), emits a row-reading word (the alphabet again plus at
# most one MAX_SYMBOL_LENGTH scalar payload per source letter), and carries
# one insertion-tableau rank of at most MAX_RSK_ALPHABET_RANK_DIGITS decimal
# digits per cell. Every quantity is a cardinality or digit bound.
MAX_PLACTIC_EQUIVALENCE_OUTPUT_SCALARS = 2 * (
    2 * MAX_RSK_WORD_PAYLOAD_SCALARS
    + MAX_RSK_WORD_LENGTH * (MAX_SYMBOL_LENGTH + MAX_RSK_ALPHABET_RANK_DIGITS)
)

__all__ = [
    "check_semistandard_tableau",
    "check_skew_littlewood_richardson",
    "check_standard_tableau",
    "conjugate_partition",
    "hook_lengths",
    "inverse_row_insertion_rsk",
    "knuth_moves",
    "partition_dominance",
    "plactic_equivalence",
    "plactic_normal_form",
    "row_insertion_rsk",
    "row_insertion_rsk_trace",
    "semistandard_young_tableaux_count",
    "standard_young_tableaux_count",
    "tableau_row_reading_word",
    "verify_knuth_moves",
    "verify_rsk",
    "verify_skew_littlewood_richardson",
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
    # Materializing 10**(estimate - 1) costs work linear in the decimal width,
    # so only resolve the exact boundary near the canonical envelope; far
    # beyond it the estimate is already a sound upper bound.
    if estimate - 1 > _EXACT_BOUNDARY_MAX_OPERAND_DIGITS:
        return estimate
    if value < 10 ** (estimate - 1):
        return estimate - 1
    return estimate


# log10(2) is in (0.301029995, 0.301029996). 1/ln(10) is in
# (0.434294481, 0.434294482). These 1e-9 units keep the per-factor bound
# sound while removing the bit_length * log10(2) slack on powers of two.
_LOG10_SCALE = 1_000_000_000
# Digit-operations allowed between SSYT hook-content cancellation checks. This
# bounds one uninterrupted batch of ``_log10_upper_units`` probes regardless of
# the alphabet's integer width.
_SSYT_CHECKPOINT_DIGIT_WORK = 256_000
# Largest numerator the exact digit-boundary resolution will materialize; the
# canonical count envelope is 32,768 digits, so this only covers operands whose
# quotient can still be near the boundary.
_EXACT_BOUNDARY_MAX_OPERAND_DIGITS = 1_000_000
# Largest logarithmic-bound overshoot (in digits) that exact boundary
# resolution can still overturn. Each ``_log10_upper_units`` probe overshoots by
# less than ``(ln2 - 1/2 + 1/3)/ln10`` (the cubic series remainder), about
# 0.109 digits, and the hook lower bound undershoots by under 0.145 digits, so
# over the at most 500 cells of a canonical partition the total slack is below
# 55 digits; 64 leaves margin. A bound further past the canonical envelope
# cannot be admissible, and resolving it exactly would only materialize a huge
# numerator to refuse it.
_EXACT_BOUNDARY_SLACK_DIGITS = 64
_LOG10_2_UPPER_UNITS = 301_029_996
_LOG10_2_LOWER_UNITS = 301_029_995
_INV_LN10_UPPER_UNITS = 434_294_482
_INV_LN10_LOWER_UNITS = 434_294_481


def _log10_upper_units(value: int) -> int:
    """Return U such that log10(value) < U / _LOG10_SCALE for value >= 1."""

    bit_length = value.bit_length()
    leading = 1 << (bit_length - 1)
    remainder = value - leading
    units = (bit_length - 1) * _LOG10_2_UPPER_UNITS
    if remainder == 0:
        return units
    # ln(1+x) < x - x^2/2 + x^3/3 for x = remainder/leading in (0, 1).
    cubic = (
        6 * remainder * leading * leading
        - 3 * remainder * remainder * leading
        + 2 * remainder * remainder * remainder
    )
    denominator = 6 * leading * leading * leading
    return units + (cubic * _INV_LN10_UPPER_UNITS + denominator - 1) // denominator


def _log10_lower_units(value: int) -> int:
    """Return L such that log10(value) >= L / _LOG10_SCALE for value >= 1.

    Use the leading bit and a truncated series for the fractional part instead
    of discarding it; the discarded fraction is up to ``log10(2)``, which is
    enough to reject an exactly representable result at the digit boundary.
    """

    bit_length = value.bit_length()
    if bit_length == 0:
        return 0
    leading = 1 << (bit_length - 1)
    remainder = value - leading
    units = (bit_length - 1) * _LOG10_2_LOWER_UNITS
    if remainder == 0:
        return units
    # ln(1+x) > x - x^2/2 for x = remainder/leading in (0, 1).
    quadratic = 2 * remainder * leading - remainder * remainder
    denominator = 2 * leading * leading
    return units + (quadratic * _INV_LN10_LOWER_UNITS) // denominator


def _digits_upper_from_log10_units(units: int) -> int:
    """Return a digit upper bound from a strict log10 upper bound.

    Decimal width is ``floor(log10 n) + 1``. Because ``U`` is a strict upper
    bound, ``floor(U) + 1`` never undershoots that width.
    """

    if units <= 0:
        return 1
    return (units - 1) // _LOG10_SCALE + 1


def _ssyt_count_digit_bound(
    partition: IntegerPartition,
    alphabet_size: int,
    alphabet_digits: int,
    *,
    exact_out: list[int] | None = None,
) -> int:
    """Upper-bound the exact SSYT count's decimal width after hook cancellation.

    When ``exact_out`` is supplied and the canonical boundary is resolved
    exactly, the resolved quotient is appended so the kernel can reuse it
    instead of repeating the product and division.
    """

    cell_count = sum(partition.parts)
    if cell_count == 0:
        return max(1, alphabet_digits)
    # Cheap overflow test before the per-cell logarithmic scan.  For a
    # partition of ``n`` cells in ``r`` rows, every hook-content numerator
    # factor is at least ``alphabet_size - r + 1`` and the hook product is at
    # most ``n!``, so the exact count is at least
    # ``(alphabet_size - r + 1)**n / n!``.  If that lower bound already exceeds
    # the canonical envelope, return a cheap sound upper bound without walking
    # the whole alphabet for every cell (which costs seconds at 32,767 digits).
    smallest_factor = alphabet_size - len(partition.parts) + 1
    if smallest_factor >= 1:
        lower_units = cell_count * (
            _log10_lower_units(smallest_factor) - _log10_upper_units(cell_count)
        )
        if lower_units >= _MAX_SSYT_COUNT_DIGITS * _LOG10_SCALE:
            return max(
                1,
                alphabet_digits,
                cell_count * _upper_decimal_digits(alphabet_size + cell_count),
            )
    conjugate = conjugate_partition(partition).parts
    numerator_log_units = 0
    hook_product = 1
    probes = 0
    # ``_log10_upper_units`` walks the whole integer, so its cost grows with
    # ``alphabet_digits``.  A fixed probe batch would leave a schema-valid
    # 32767-digit alphabet uncancellable for seconds, so the batch is derived
    # from a bounded amount of per-probe digit work instead.
    probe_interval = min(
        256,
        max(1, _SSYT_CHECKPOINT_DIGIT_WORK // max(1, alphabet_digits)),
    )
    for row, length in enumerate(partition.parts):
        request_checkpoint("during SSYT digit admission")
        for column in range(length):
            probes += 1
            if probes % probe_interval == 0:
                request_checkpoint("during SSYT hook-content admission")
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
    bound = _digits_upper_from_log10_units(cancelled_units)
    if bound <= _MAX_SSYT_COUNT_DIGITS:
        return max(1, alphabet_digits, bound)
    # The logarithmic bound carries a small per-factor slack, which is enough to
    # overshoot by one digit exactly at the canonical output boundary: for
    # ``partition=(500)`` and ``alphabet_size=154850 * 2**208`` the exact count
    # has 32,768 digits while the estimate reports 32,769.  Resolve the boundary
    # exactly instead of refusing a cheaply executable count, and keep the
    # logarithmic estimate as the answer whenever it already fits.
    #
    # Only a genuinely near-boundary estimate can hide an admissible count: the
    # per-factor log bounds overshoot by fractions of a unit, at most about
    # 0.109 digits per cell plus a fraction for the hook lower bound, so over
    # the at most 500 cells of a canonical partition the total overshoot stays
    # below ``_EXACT_BOUNDARY_SLACK_DIGITS``.  A more distant estimate is a
    # sound refusal, and materializing its numerator could only confirm it.
    if bound > _MAX_SSYT_COUNT_DIGITS + _EXACT_BOUNDARY_SLACK_DIGITS:
        return max(1, alphabet_digits, bound)
    if (
        _digits_upper_from_log10_units(numerator_log_units)
        > _EXACT_BOUNDARY_MAX_OPERAND_DIGITS
    ):
        # A far-overflowing numerator must stay a refusal: materializing it to
        # resolve the boundary exactly would cost more than the result envelope.
        return max(1, alphabet_digits, bound)
    exact = _exact_count(partition, alphabet_size, hook_product)
    if exact is None:
        return max(1, alphabet_digits, bound)
    if exact_out is not None:
        exact_out.append(exact)
    return max(1, alphabet_digits, _decimal_width(exact))


def _exact_count(
    partition: IntegerPartition, alphabet_size: int, hook_product: int
) -> int | None:
    """Return the exact hook-content count, or ``None`` if it does not divide.

    Only reached at the canonical digit boundary, where the logarithmic
    estimate cannot decide.  The numerator product and the exact division are
    bounded by the same factor count the caller already charges.
    """

    numerator_factors: list[int] = []
    for row, length in enumerate(partition.parts):
        if row % 64 == 0:
            request_checkpoint("during exact SSYT boundary resolution")
        for column in range(length):
            numerator_factors.append(alphabet_size + column - row)
    # Match the admitted balanced-product cost model instead of a sequential
    # left-to-right accumulation, so the resolver does not outrun admission.
    numerator_product = _balanced_product(tuple(numerator_factors))
    quotient, remainder = divmod(numerator_product, hook_product)
    if remainder:
        # The hook-content numerator is only guaranteed to divide for an
        # admitted partition; anything else keeps the sound estimate.
        return None
    return quotient


def _decimal_width(value: int) -> int:
    """Return the decimal digit count of a positive integer without ``str``.

    ``str`` refuses integers above Python's active conversion limit, which is
    exactly the magnitude this boundary path handles, so the width is found by
    exact comparison against powers of ten.
    """

    if value <= 0:
        return 1
    # log10(2) < 30103 / 100000, so this is a tight upper estimate: the decimal
    # width is at most ``bit_length * log10(2) + 1`` and differs by at most one.
    estimate = value.bit_length() * 30103 // 100000 + 1
    if estimate > 1 and 10 ** (estimate - 1) > value:
        estimate -= 1
    if 10**estimate <= value:
        estimate += 1
    return estimate


def _admit_hook_content(partition: IntegerPartition, alphabet_size: int) -> int | None:
    """Admit hook-content arithmetic before constructing any factors.

    Returns the exact count when admission resolved the digit boundary exactly,
    so the kernel can reuse it rather than repeat the product and division.
    """
    if type(alphabet_size) is not int or alphabet_size < 1:
        raise OperationDomainValidationError(
            location=("alphabet_size",),
            code="algebraic_combinatorics.hook_content_alphabet",
            message="alphabet_size must be a positive integer",
        )
    if _upper_decimal_digits(alphabet_size) > MAX_CANONICAL_INTEGER_DIGITS:
        # Match the catalog request's ExactInteger carrier so a native call and
        # a wire call classify an oversized alphabet the same way, before any
        # big-integer admission arithmetic runs.
        raise OperationDomainValidationError(
            location=("alphabet_size",),
            code="algebraic_combinatorics.hook_content_alphabet",
            message=(
                "alphabet_size must have at most "
                f"{MAX_CANONICAL_INTEGER_DIGITS} decimal digits"
            ),
        )
    partition = _require_canonical_partition(partition)

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
    resolved: list[int] = []
    output_digits = max(
        alphabet_digits,
        _ssyt_count_digit_bound(
            partition, alphabet_size, alphabet_digits, exact_out=resolved
        ),
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
    return resolved[0] if resolved else None


def row_insertion_rsk(word: FiniteWord) -> RSKTableauPair:
    """Compute ordinary row-insertion RSK for a word of at most 500 letters."""
    return _row_insertion_rsk(word)


def inverse_row_insertion_rsk(pair: RSKTableauPair) -> FiniteWord:
    """Reconstruct the unique word represented by a pair of at most 500 cells."""
    return _inverse_row_insertion_rsk(pair)


def row_insertion_rsk_trace(word: FiniteWord) -> RSKWordTraceResult:
    """Return word RSK with its complete ordinary insertion bump path."""
    return _row_insertion_rsk_trace(word)


def _rsk_permutation(
    permutation: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Validate and insert one strict permutation through the native kernel."""
    if sorted(permutation) != list(range(1, len(permutation) + 1)):
        raise ValueError("permutation must be a permutation of 1..n")
    return _row_insert(permutation)


def knuth_moves(word: FiniteWord) -> tuple[KnuthNeighbor, ...]:
    """Return every one-step Knuth neighbor of a bounded ordered word.

    Ranks come from the word's explicit alphabet order under
    ROW_INSERTION_RSK_V1: ``xzy <-> zxy`` when ``x <= y < z`` (K1, swapping
    the first two window letters) and ``yxz <-> yzx`` when ``x < y <= z``
    (K2, swapping the last two). At most one relation applies per window, so
    the neighborhood holds at most ``len(letters) - 2`` rows.
    """

    try:
        canonical = FiniteWord(alphabet=word.alphabet, letters=word.letters)
    except ValidationError as exc:
        raise OperationDomainValidationError(
            location=("word",),
            code="algebraic_combinatorics.word_not_over_alphabet",
            message="knuth word letters must belong to the declared alphabet",
        ) from exc
    rank = {symbol: index for index, symbol in enumerate(canonical.alphabet)}
    letters = canonical.letters
    neighbors: list[KnuthNeighbor] = []
    for position in range(len(letters) - 2):
        first, second, third = (
            rank[letters[position]],
            rank[letters[position + 1]],
            rank[letters[position + 2]],
        )
        relation: str | None = None
        swapped: tuple[str, ...] | None = None
        if first <= third < second or second <= third < first:
            relation = "K1"
            swapped = (
                *letters[:position],
                letters[position + 1],
                letters[position],
                letters[position + 2],
                *letters[position + 3 :],
            )
        elif second < first <= third or third < first <= second:
            relation = "K2"
            swapped = (
                *letters[:position],
                letters[position],
                letters[position + 2],
                letters[position + 1],
                *letters[position + 3 :],
            )
        if relation is None or swapped is None:
            continue
        neighbors.append(
            KnuthNeighbor.model_construct(
                position=position,
                relation=relation,
                neighbor=FiniteWord.model_construct(
                    alphabet=canonical.alphabet, letters=swapped
                ),
            )
        )
    return tuple(neighbors)


def plactic_normal_form(word: FiniteWord) -> PlacticNormalFormResult:
    """Return the canonical bottom-to-top row-reading word of the RSK tableau."""

    pair = _row_insertion_rsk(word)
    normal_letters = tuple(
        pair.alphabet[entry - 1]
        for row in reversed(pair.insertion_tableau.rows)
        for entry in row
    )
    return PlacticNormalFormResult(
        source_word=word,
        insertion_tableau=pair.insertion_tableau,
        normal_form=FiniteWord(alphabet=pair.alphabet, letters=normal_letters),
    )


def plactic_equivalence(
    left: FiniteWord, right: FiniteWord
) -> PlacticEquivalenceResult:
    """Return canonical forms and equality in the row-insertion plactic monoid.

    Both finite canonical words are admitted together before either RSK
    insertion. An n-letter insertion performs at most n rows of
    ceil(log2(n+1)) comparisons for each of its n letters; the combined
    output is bounded from both source cardinalities before either normal
    form is constructed.
    """
    try:
        canonical_left = FiniteWord.model_validate(left.model_dump(mode="python"))
        canonical_right = FiniteWord.model_validate(right.model_dump(mode="python"))
    except (AttributeError, TypeError, ValidationError) as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="algebraic_combinatorics.plactic_equivalence_request",
            message="plactic equivalence requires two canonical words over one ordered alphabet",
        ) from exc
    if canonical_left.alphabet != canonical_right.alphabet:
        raise OperationDomainValidationError(
            location=("request",),
            code="algebraic_combinatorics.plactic_equivalence_request",
            message="plactic equivalence requires two canonical words over one ordered alphabet",
        )

    words = (canonical_left, canonical_right)
    source_payloads = tuple(word_payload_scalars(word) for word in words)
    if any(
        len(word.letters) > MAX_RSK_WORD_LENGTH
        or payload > MAX_RSK_WORD_PAYLOAD_SCALARS
        for word, payload in zip(words, source_payloads, strict=True)
    ):
        raise OperationResourceAdmissionError(
            location=("request",),
            code="algebraic_combinatorics.plactic_equivalence_source",
            message="each source word must fit the ordinary RSK size and payload bounds",
        )
    work = sum(
        len(word.letters) ** 2 * max(1, len(word.letters).bit_length())
        for word in words
    )
    if work > MAX_PLACTIC_EQUIVALENCE_WORK:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="algebraic_combinatorics.plactic_equivalence_work",
            message="combined RSK insertion work exceeds its admitted bound",
        )
    # Every resulting row-reading word contains no more letters than its
    # source, each emitted letter is one retained alphabet symbol of at most
    # MAX_SYMBOL_LENGTH scalar values, and each tableau cell stores one rank
    # of at most MAX_RSK_ALPHABET_RANK_DIGITS decimal digits.
    output_bound = sum(
        2 * payload
        + len(word.letters) * (MAX_SYMBOL_LENGTH + MAX_RSK_ALPHABET_RANK_DIGITS)
        for word, payload in zip(words, source_payloads, strict=True)
    )
    if output_bound > MAX_PLACTIC_EQUIVALENCE_OUTPUT_SCALARS:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="algebraic_combinatorics.plactic_equivalence_output",
            message="combined canonical-form output exceeds its admitted bound",
        )
    left_form = plactic_normal_form(canonical_left)
    right_form = plactic_normal_form(canonical_right)
    return PlacticEquivalenceResult(
        left=left_form,
        right=right_form,
        equivalent=left_form.insertion_tableau == right_form.insertion_tableau,
    )


def tableau_row_reading_word(pair: RSKTableauPair) -> FiniteWord:
    """Read P bottom-to-top, each row left-to-right, retaining its alphabet."""

    if type(pair) is not RSKTableauPair:
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_pair_required",
            message="row reading requires an alphabet-bound RSK tableau pair",
        )
    try:
        canonical = RSKTableauPair.model_validate(pair.model_dump())
    except (TypeError, ValueError, ValidationError) as exc:
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.invalid_rsk_pair",
            message="row reading requires valid semistandard and standard tableaux",
        ) from exc
    cell_count = sum(canonical.shape.parts)
    if cell_count > MAX_RSK_WORD_LENGTH:
        raise OperationResourceAdmissionError(
            location=("pair", "shape"),
            code="algebraic_combinatorics.rsk_tableau_size",
            message="tableau exceeds the admitted row-reading cell bound",
        )
    try:
        require_semistandard(canonical.insertion_tableau)
        require_standard(canonical.recording_tableau)
    except (TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.invalid_rsk_pair",
            message="row reading requires valid semistandard and standard tableaux",
        ) from exc
    if any(
        entry > len(canonical.alphabet)
        for row in canonical.insertion_tableau.rows
        for entry in row
    ):
        raise OperationDomainValidationError(
            location=("pair", "insertion_tableau"),
            code="algebraic_combinatorics.rsk_entry_outside_alphabet",
            message="insertion-tableau ranks must index the retained alphabet",
        )
    letters = tuple(
        canonical.alphabet[entry - 1]
        for row in reversed(canonical.insertion_tableau.rows)
        for entry in row
    )
    # The output retains the alphabet once and emits exactly one referenced
    # alphabet symbol per tableau cell, so its payload is the sum of the
    # alphabet scalars and the scalars of the emitted letters.
    output_scalars = sum(map(len, canonical.alphabet)) + sum(map(len, letters))
    if output_scalars > MAX_RSK_WORD_PAYLOAD_SCALARS:
        raise OperationResourceAdmissionError(
            location=("pair",),
            code="algebraic_combinatorics.row_reading_output_size",
            message="row-reading output exceeds its admitted payload bound",
        )
    return FiniteWord.model_construct(alphabet=canonical.alphabet, letters=letters)


def verify_knuth_moves(claim: KnuthMovesResult) -> bool:
    """Replay a Knuth neighborhood against its retained source word."""

    try:
        return knuth_moves(claim.word) == claim.neighbors and (
            claim.neighbor_count == len(claim.neighbors)
        )
    except OperationDomainValidationError:
        return False


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


def _require_canonical_partition(partition: object) -> IntegerPartition:
    if type(partition) is not IntegerPartition:
        raise OperationDomainValidationError(
            location=("partition",),
            code="algebraic_combinatorics.partition_carrier",
            message="partition operations require an IntegerPartition value",
        )
    parts = getattr(partition, "parts", None)
    if not isinstance(parts, tuple):
        raise OperationDomainValidationError(
            location=("partition", "parts"),
            code="algebraic_combinatorics.partition_carrier",
            message="a canonical partition has a tuple of parts",
        )
    # The part-count envelope is checked first: ``len`` is constant time while
    # every subsequent scan is linear in the tuple length, so an already
    # oversized forged carrier must be refused before it is traversed.
    if len(parts) > MAX_PARTITION_PARTS:
        raise OperationResourceAdmissionError(
            location=("partition", "parts"),
            code="algebraic_combinatorics.partition_size",
            message="partition size exceeds the admitted Ferrers envelope",
        )
    # Domain invariants are checked for every part before the resource
    # envelope, so a forged carrier is classified by what it actually violates
    # instead of by whichever field happens to be inspected first.
    if any(type(part) is not int or part <= 0 for part in parts):
        raise OperationDomainValidationError(
            location=("partition", "parts"),
            code="algebraic_combinatorics.partition_carrier",
            message="partition parts must be positive integers",
        )
    if any(parts[index] < parts[index + 1] for index in range(len(parts) - 1)):
        raise OperationDomainValidationError(
            location=("partition", "parts"),
            code="algebraic_combinatorics.partition_carrier",
            message="partition parts must be weakly decreasing",
        )
    if len(parts) > MAX_PARTITION_PARTS or sum(parts) > MAX_PARTITION_SIZE:
        raise OperationResourceAdmissionError(
            location=("partition", "parts"),
            code="algebraic_combinatorics.partition_size",
            message="partition size exceeds the admitted Ferrers envelope",
        )
    try:
        # A constructed instance is returned unchanged by the default Pydantic
        # configuration, so validate a fresh payload to rerun the canonical
        # weak-decreasing, positive-part invariants.
        return IntegerPartition.model_validate({"parts": tuple(parts)})
    except (ValidationError, PydanticCustomError) as error:
        raise OperationDomainValidationError(
            location=("partition",),
            code="algebraic_combinatorics.partition_carrier",
            message="partition operations require a canonical IntegerPartition",
        ) from error


def conjugate_partition(partition: IntegerPartition) -> IntegerPartition:
    """Return the conjugate of one canonical partition.

    The conjugate ``lambda'`` has ``lambda'_j`` equal to the number of parts of
    ``lambda`` that are at least ``j``, i.e. the column heights of the Ferrers
    diagram.
    """
    partition = _require_canonical_partition(partition)
    parts = partition.parts
    return IntegerPartition(parts=_conjugate_parts(parts))


def _conjugate_parts(parts: tuple[int, ...]) -> tuple[int, ...]:
    """Conjugate parts already admitted by ``_require_canonical_partition``."""
    if not parts:
        return ()
    max_column = parts[0]
    return tuple(
        sum(1 for part in parts if part >= column)
        for column in range(1, max_column + 1)
    )


def _hook_lengths_canonical(parts: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Compute hooks from parts admitted by ``_require_canonical_partition``."""
    conjugate = _conjugate_parts(parts)
    hooks: list[list[int]] = []
    for row, length in enumerate(parts):
        row_hooks: list[int] = []
        for column in range(length):
            right = length - column - 1
            below = conjugate[column] - row - 1
            row_hooks.append(right + below + 1)
        hooks.append(row_hooks)
    return tuple(tuple(row) for row in hooks)


def hook_lengths(partition: IntegerPartition) -> tuple[tuple[int, ...], ...]:
    """Return the hook length of every cell of a canonical partition.

    The hook length of cell ``(i, j)`` (0-indexed) is
    ``lambda_i - j + lambda'_j - i - 1``: one arm step plus the cell itself
    plus the number of cells below it in its column.
    """
    partition = _require_canonical_partition(partition)
    return _hook_lengths_canonical(partition.parts)


def _hook_length_product(hooks: tuple[tuple[int, ...], ...]) -> int:
    """Return the exact product of one hook-length array."""
    return prod(hook for row in hooks for hook in row)


def standard_young_tableaux_count(partition: IntegerPartition) -> int:
    """Count standard Young tableaux of a canonical partition.

    The number of standard Young tableaux of shape ``lambda`` is
    ``n! / prod_{(i,j) in lambda} h(i,j)`` where ``n = |lambda|`` and
    ``h(i,j)`` is the cell's hook length.
    """
    partition = _require_canonical_partition(partition)
    hooks = _hook_lengths_canonical(partition.parts)
    n = sum(partition.parts)
    return factorial(n) // _hook_length_product(hooks)


def semistandard_young_tableaux_count(
    partition: IntegerPartition, alphabet_size: int
) -> SemistandardYoungTableauCountResult:
    """Count SSYTs of ``partition`` over ``1..alphabet_size`` exactly.

    The hook-content factors are private kernel intermediates. The public
    result carries only the exact count and its source shape and alphabet.
    """
    partition = _require_canonical_partition(partition)
    resolved = _admit_hook_content(partition, alphabet_size)
    if resolved is not None:
        return SemistandardYoungTableauCountResult(
            partition=partition,
            alphabet_size=alphabet_size,
            count=resolved,
        )
    hooks = _hook_lengths_canonical(partition.parts)
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
    return SemistandardYoungTableauCountResult(
        partition=partition,
        alphabet_size=alphabet_size,
        count=quotient.numerator,
    )


def partition_dominance(
    left: IntegerPartition, right: IntegerPartition
) -> PartitionDominanceResult:
    """Compare partitions using leading-row prefix sums privately."""
    left = _require_canonical_partition(left)
    right = _require_canonical_partition(right)
    if sum(left.parts) != sum(right.parts):
        relation: DominanceRelation = "NOT_COMPARABLE_DIFFERENT_SIZE"
    else:
        length = max(len(left.parts), len(right.parts))
        left_total = right_total = 0
        left_ge = right_ge = True
        for index in range(length):
            left_total += left.parts[index] if index < len(left.parts) else 0
            right_total += right.parts[index] if index < len(right.parts) else 0
            left_ge = left_ge and left_total >= right_total
            right_ge = right_ge and left_total <= right_total
        if left_ge and right_ge:
            relation = "EQUAL"
        elif left_ge:
            relation = "LEFT_DOMINATES"
        elif right_ge:
            relation = "RIGHT_DOMINATES"
        else:
            relation = "INCOMPARABLE"
    return PartitionDominanceResult(left=left, right=right, relation=relation)


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


def _validate_tableau_carrier(
    carrier: type[StrictModel], rows: tuple[object, ...]
) -> StrictModel:
    """Revalidate a tableau, translating structural failures to domain errors."""

    try:
        return carrier.model_validate({"rows": rows})
    except (ValidationError, PydanticCustomError) as exc:
        raise OperationDomainValidationError(
            location=("tableau", "rows"),
            code="algebraic_combinatorics.tableau_carrier",
            message="tableau rows violate the canonical carrier contract",
        ) from exc


def _revalidate_candidate(tableau: object) -> TableauCandidate:
    """Return a freshly validated structural candidate, rejecting forged rows."""

    if not isinstance(
        tableau, (TableauCandidate, StandardYoungTableau, SemistandardYoungTableau)
    ):
        raise OperationDomainValidationError(
            location=("tableau",),
            code="algebraic_combinatorics.tableau_carrier",
            message="membership checks require a canonical tableau candidate",
        )
    rows = getattr(tableau, "rows", None)
    if not isinstance(rows, tuple):
        raise OperationDomainValidationError(
            location=("tableau", "rows"),
            code="algebraic_combinatorics.tableau_shape",
            message="a canonical tableau candidate has a tuple of rows",
        )
    return cast(TableauCandidate, _validate_tableau_carrier(TableauCandidate, rows))


def check_standard_tableau(
    tableau: TableauCandidate | StandardYoungTableau,
) -> StandardTableauCheckResult:
    """Return a source-bound membership decision for one standard candidate."""

    candidate = _revalidate_candidate(tableau)
    try:
        require_standard(candidate)
    except PydanticCustomError:
        return StandardTableauCheckResult(tableau=candidate, is_member=False)
    except ValidationError as error:
        if _is_shape_rejection(error):
            return StandardTableauCheckResult(tableau=candidate, is_member=False)
        raise
    return StandardTableauCheckResult(tableau=candidate, is_member=True)


def check_semistandard_tableau(
    tableau: TableauCandidate | SemistandardYoungTableau,
) -> SemistandardTableauCheckResult:
    """Return a source-bound membership decision for one semistandard candidate."""

    candidate = _revalidate_candidate(tableau)
    try:
        require_semistandard(candidate)
    except PydanticCustomError:
        return SemistandardTableauCheckResult(tableau=candidate, is_member=False)
    except ValidationError as error:
        if _is_shape_rejection(error):
            return SemistandardTableauCheckResult(tableau=candidate, is_member=False)
        raise
    return SemistandardTableauCheckResult(tableau=candidate, is_member=True)


def _skew_offsets(
    outer: IntegerPartition, inner: IntegerPartition
) -> tuple[int, ...] | None:
    """Return inner row offsets, or ``None`` when containment fails."""
    length = max(len(outer.parts), len(inner.parts))
    offsets: list[int] = []
    for index in range(length):
        outer_part = outer.parts[index] if index < len(outer.parts) else 0
        inner_part = inner.parts[index] if index < len(inner.parts) else 0
        if inner_part > outer_part:
            return None
        offsets.append(inner_part)
    return tuple(offsets)


def check_skew_littlewood_richardson(  # noqa: C901
    outer: IntegerPartition,
    inner: IntegerPartition,
    tableau: TableauCandidate,
    content: IntegerPartition,
    convention: SkewReadingConvention = "READING_WORD_RL_TOP_V1",
) -> SkewLittlewoodRichardsonCheckResult:
    """Replay skew-LR membership: coverage, semistandardity, content, lattice.

    The reading word concatenates rows top to bottom, each right to left,
    under READING_WORD_RL_TOP_V1. Every prefix must satisfy the Yamanouchi
    inequalities ``count(1) >= count(2) >= ...``. The result carries the
    first failed stage with its concrete prefix length or cell.
    """
    outer = _require_canonical_partition(outer)
    inner = _require_canonical_partition(inner)
    content = _require_canonical_partition(content)
    candidate = _revalidate_candidate(tableau)
    rows = candidate.rows

    offsets = _skew_offsets(outer, inner)
    if offsets is None:
        return SkewLittlewoodRichardsonCheckResult._from_kernel(
            outer,
            inner,
            tableau,
            content,
            convention,
            is_member=False,
            reading_word=(),
            failure_kind="CELL_COVERAGE",
        )
    skew_lengths = tuple(
        (outer.parts[index] if index < len(outer.parts) else 0) - offsets[index]
        for index in range(len(offsets))
    )
    # Candidate rows carry exactly the nonempty skew rows in top-to-bottom
    # order; empty skew rows are omitted (TableauRow carriers are nonempty).
    present = tuple(index for index, length in enumerate(skew_lengths) if length > 0)
    if len(rows) != len(present) or any(
        len(row) != skew_lengths[present[pos]] for pos, row in enumerate(rows)
    ):
        failed_row = next(
            (
                pos
                for pos in range(max(len(rows), len(present)))
                if pos >= len(rows)
                or pos >= len(present)
                or len(rows[pos]) != skew_lengths[present[pos]]
            ),
            -1,
        )
        return SkewLittlewoodRichardsonCheckResult._from_kernel(
            outer,
            inner,
            tableau,
            content,
            convention,
            is_member=False,
            reading_word=(),
            failure_kind="CELL_COVERAGE",
            failed_row=failed_row,
        )
    for pos, row in enumerate(rows):
        failed_column = next(
            (index for index in range(len(row) - 1) if row[index] > row[index + 1]),
            None,
        )
        if failed_column is not None:
            return SkewLittlewoodRichardsonCheckResult._from_kernel(
                outer,
                inner,
                tableau,
                content,
                convention,
                is_member=False,
                reading_word=(),
                failure_kind="SEMISTANDARD_ROW",
                failed_row=present[pos],
                failed_column=failed_column,
            )
    # Column strictness over vertically adjacent skew cells: collect each
    # diagram column top to bottom and require a strict increase.
    widest = max(skew_lengths, default=0) + max(offsets, default=0)
    for column in range(widest):
        column_cells: list[tuple[int, int]] = []
        for pos, true_row in enumerate(present):
            local = column - offsets[true_row]
            if 0 <= local < len(rows[pos]):
                column_cells.append((true_row, rows[pos][local]))
        for index in range(len(column_cells) - 1):
            if column_cells[index][1] >= column_cells[index + 1][1]:
                return SkewLittlewoodRichardsonCheckResult._from_kernel(
                    outer,
                    inner,
                    tableau,
                    content,
                    convention,
                    is_member=False,
                    reading_word=(),
                    failure_kind="SEMISTANDARD_COLUMN",
                    failed_row=column_cells[index][0],
                    failed_column=column,
                )
    multiplicities: dict[int, int] = {}
    for row in rows:
        for entry in row:
            multiplicities[entry] = multiplicities.get(entry, 0) + 1
    max_value = max(
        (*multiplicities, len(content.parts)),
        default=0,
    )
    for value in range(1, max_value + 1):
        expected = content.parts[value - 1] if value - 1 < len(content.parts) else 0
        if multiplicities.get(value, 0) != expected:
            return SkewLittlewoodRichardsonCheckResult._from_kernel(
                outer,
                inner,
                tableau,
                content,
                convention,
                is_member=False,
                reading_word=(),
                failure_kind="CONTENT",
                failed_value=value,
            )
    word = tuple(entry for row in rows for entry in reversed(row))
    counts: dict[int, int] = {}
    for prefix_length, entry in enumerate(word, start=1):
        counts[entry] = counts.get(entry, 0) + 1
        for value in range(2, max(counts, default=1) + 1):
            if counts.get(value - 1, 0) < counts.get(value, 0):
                return SkewLittlewoodRichardsonCheckResult._from_kernel(
                    outer,
                    inner,
                    tableau,
                    content,
                    convention,
                    is_member=False,
                    reading_word=word,
                    failure_kind="LATTICE",
                    failed_prefix_length=prefix_length,
                    failed_value=value,
                )
    return SkewLittlewoodRichardsonCheckResult._from_kernel(
        outer,
        inner,
        tableau,
        content,
        convention,
        is_member=True,
        reading_word=word,
        failure_kind="OK",
    )


def verify_skew_littlewood_richardson(
    claim: SkewLittlewoodRichardsonCheckResult,
) -> bool:
    """Replay a serialized skew-LR claim against its retained source."""

    try:
        return (
            check_skew_littlewood_richardson(
                claim.outer,
                claim.inner,
                claim.tableau,
                claim.content,
                claim.convention,
            )
            == claim
        )
    except (OperationDomainValidationError, TypeError, ValueError):
        return False
