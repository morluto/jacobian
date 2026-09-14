"""Native exact operations on finite integer sequences."""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from functools import reduce
from itertools import pairwise
from math import gcd

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.sequences.core._models import (
    AutocorrelationCell,
    AutocorrelationResult,
    FiniteIntegerSequence,
    FiniteRationalSequence,
    FrequencyEntry,
    IntegerSequenceBooleanResult,
    IntegerSequenceFrequenciesResult,
    IntegerSequenceIndexListResult,
    IntegerSequenceListResult,
    IntegerSequenceRationalResult,
    IntegerSequenceValueResult,
    SequenceLogConcavityRow,
    SequenceOrderShapeResult,
)
from jacobian.math.number_theory.sequences.core.values import (
    MAX_SEQUENCE_TOTAL_DIGITS,
    IntegerSequence,
)

MAX_AUTOCORRELATION_MULTIPLICATIONS = 4_000_000
MAX_AUTOCORRELATION_ADDITIONS = 4_000_000
MAX_ORDER_SHAPE_WORK = 4_000_000
MAX_ORDER_SHAPE_RESULT_ALLOCATIONS = 500_000


def _admit(
    request: IntegerSequence,
    *,
    output_digits: int,
    output_items: int = 1,
) -> list[int]:
    """Admit one operation's complete source/result envelope.

    The reusable sequence value checks only its canonical representation.  A
    native operation owns the derived output budget because sums, products,
    and materialized transforms have different growth envelopes.
    """

    values = _values(request)
    if output_items and output_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("values",),
            code="sequences.result_digits_exceeded",
            message=(
                "the exact result exceeds the "
                f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit bound"
            ),
        )
    total_output_digits = output_items * output_digits
    if output_items and total_output_digits > MAX_SEQUENCE_TOTAL_DIGITS:
        raise OperationDomainValidationError(
            location=("values",),
            code="sequences.result_representation_too_large",
            message=(
                "the exact result exceeds the "
                f"{MAX_SEQUENCE_TOTAL_DIGITS}-digit representation bound"
            ),
        )
    return values


def _digits(values: tuple[int, ...]) -> int:
    """Return a decimal-width bound from native source integers."""

    return max(
        (len(format_canonical_integer(abs(value))) for value in values), default=1
    )


def _multiplicative_digits(request: IntegerSequence, *, prefix: bool = False) -> int:
    """Bound product-like widths, accounting for the absorbing zero value."""

    total = 0
    maximum = 1
    for value in request.values:
        if value == 0:
            return maximum if prefix else 1
        total += len(format_canonical_integer(abs(value)))
        maximum = max(maximum, total)
    return maximum


def _values(request: IntegerSequence) -> list[int]:
    return list(request.values)


def _value_result(value: int) -> IntegerSequenceValueResult:
    return IntegerSequenceValueResult(value=value)


def _admit_autocorrelation(
    request: FiniteIntegerSequence | FiniteRationalSequence,
    *,
    output_items: int,
) -> tuple[tuple[Fraction, ...], int, int]:
    """Admit source, intermediate, and output bounds before expansion.

    A common denominator gives a sound width bound without computing any
    pairwise products.  Every coefficient is a sum of at most ``n`` products
    whose denominator divides ``D**2``, where ``D`` is the lcm of source
    denominators.  This also keeps cheap repeated rational inputs from being
    rejected merely because their syntactic denominators are nontrivial.
    """

    if isinstance(request, FiniteIntegerSequence):
        fractions = tuple(Fraction(value) for value in request.values)
    else:
        fractions = tuple(value.as_fraction() for value in request.values)
    if not fractions:
        return fractions, 1, 1

    denominators = tuple(value.denominator for value in fractions)
    common_denominator = 1
    common_denominator_digits = 1
    for denominator in denominators:
        denominator_digits = len(format_canonical_integer(denominator))
        if denominator_digits > MAX_CANONICAL_RATIONAL_DIGITS // 2:
            raise OperationDomainValidationError(
                location=("values",),
                code="sequences.autocorrelation_result_digits_exceeded",
                message="autocorrelation denominators exceed the exact rational digit bound",
            )
        divisor = gcd(common_denominator, denominator)
        quotient = denominator // divisor
        quotient_digits = len(format_canonical_integer(quotient))
        if common_denominator_digits + quotient_digits > (
            MAX_CANONICAL_RATIONAL_DIGITS // 2 + 1
        ):
            raise OperationDomainValidationError(
                location=("values",),
                code="sequences.autocorrelation_result_digits_exceeded",
                message="autocorrelation denominators exceed the exact rational digit bound",
            )
        common_denominator *= quotient
        common_denominator_digits = len(format_canonical_integer(common_denominator))
        if common_denominator_digits > MAX_CANONICAL_RATIONAL_DIGITS // 2:
            raise OperationDomainValidationError(
                location=("values",),
                code="sequences.autocorrelation_result_digits_exceeded",
                message="autocorrelation denominators exceed the exact rational digit bound",
            )
    if any(
        len(format_canonical_integer(abs(value.numerator))) + common_denominator_digits
        > MAX_CANONICAL_RATIONAL_DIGITS
        for value in fractions
    ):
        raise OperationDomainValidationError(
            location=("values",),
            code="sequences.autocorrelation_result_digits_exceeded",
            message="autocorrelation values exceed the exact rational digit bound",
        )
    widest_numerator: dict[int, int] = {}
    for value in fractions:
        numerator = abs(value.numerator)
        current = widest_numerator.get(value.denominator)
        if current is None or numerator > current:
            widest_numerator[value.denominator] = numerator
    common_numerator_digits = max(
        len(format_canonical_integer(numerator * (common_denominator // denominator)))
        for denominator, numerator in widest_numerator.items()
    )
    denominator_digits = len(format_canonical_integer(common_denominator))
    term_count_digits = len(str(len(fractions)))
    result_numerator_digits = 2 * common_numerator_digits + term_count_digits
    result_denominator_digits = 2 * denominator_digits
    result_digits = max(result_numerator_digits, result_denominator_digits)
    if result_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("values",),
            code="sequences.autocorrelation_result_digits_exceeded",
            message="autocorrelation values exceed the exact rational digit bound",
        )

    result_representation_digits = (
        result_numerator_digits + result_denominator_digits
        if isinstance(request, FiniteRationalSequence)
        else result_numerator_digits
    )
    if output_items * result_representation_digits > MAX_SEQUENCE_TOTAL_DIGITS:
        raise OperationDomainValidationError(
            location=("values",),
            code="sequences.autocorrelation.result_representation_too_large",
            message="autocorrelation output exceeds the exact representation bound",
        )
    operand_width = max(1, common_numerator_digits + denominator_digits)
    return fractions, result_digits, operand_width


def _order_shape_rationals(
    request: FiniteRationalSequence,
) -> tuple[CanonicalRational, ...]:
    if isinstance(request, FiniteRationalSequence):
        return request.values
    raise OperationDomainValidationError(
        location=("values",),
        code="sequences.order_shape.sequence_type",
        message="sequence_order_shape requires a FiniteRationalSequence value",
    )


def _product_component_digits(
    left: CanonicalRational, right: CanonicalRational
) -> tuple[int, int]:
    """Bound retained numerator and denominator digits of one exact product."""

    if left.num == 0 or right.num == 0:
        return 1, 1
    left_num = abs(left.num)
    right_num = abs(right.num)
    cancel_left = gcd(left_num, right.den)
    cancel_right = gcd(right_num, left.den)
    # Measure the actual reduced components so a unit factor cannot add a
    # spurious digit.
    numerator = (left_num // cancel_left) * (right_num // cancel_right)
    denominator = (left.den // cancel_right) * (right.den // cancel_left)
    return (
        max(1, len(format_canonical_integer(numerator))),
        max(1, len(format_canonical_integer(denominator))),
    )


def _admit_order_shape(
    request: FiniteRationalSequence,
) -> tuple[tuple[CanonicalRational, ...], tuple[int, ...]]:
    """Admit linear comparisons, exact products, and the complete profile.

    Returns the admitted rational values together with the exact weak-unimodal
    peak positions, so construction reuses the peak scan instead of replaying
    it after admission.
    """

    rational_values = _order_shape_rationals(request)
    fractions = tuple(value.as_fraction() for value in rational_values)
    size = len(fractions)
    source_digits = sum(
        len(format_canonical_integer(abs(value.num)))
        + len(format_canonical_integer(value.den))
        for value in rational_values
    )
    component_widths = tuple(
        max(
            len(format_canonical_integer(abs(value.num))),
            len(format_canonical_integer(value.den)),
        )
        for value in rational_values
    )
    # Charge the actual comparison operands: only adjacent and neighbouring
    # widths are compared, not the widest component across every entry. The
    # bound is computable from the widths, so it runs before the peak scan
    # rather than after the work it is meant to prevent.
    comparison_work = 0
    for index in range(size):
        if index % 512 == 0:
            request_checkpoint("during order-shape comparison admission")
        if index + 1 < size:
            comparison_work += max(component_widths[index], component_widths[index + 1])
        if 0 < index < size - 1:
            comparison_work += max(
                component_widths[index - 1], component_widths[index + 1]
            )
    if comparison_work > MAX_ORDER_SHAPE_WORK:
        raise OperationResourceAdmissionError(
            location=("values",),
            code="sequences.order_shape.work_bound",
            message="order-shape comparisons exceed the admitted exact-work bound",
        )
    # The mandatory peak scan is admitted, so compute it once here; the caller
    # retains this exact result instead of recomputing it.
    peaks = _order_shape_peaks(fractions)
    row_count = max(0, size - 2)
    # The result retains the source, one slot per actual peak position, and
    # four slots (index, two exact products, decision) per interior row.
    result_allocations = size + len(peaks) + 4 * row_count + 8
    if result_allocations > MAX_ORDER_SHAPE_RESULT_ALLOCATIONS:
        raise OperationResourceAdmissionError(
            location=("values",),
            code="sequences.order_shape.result_allocation_bound",
            message=(
                "the complete order-shape profile requires "
                f"{result_allocations} result allocations; maximum is "
                f"{MAX_ORDER_SHAPE_RESULT_ALLOCATIONS}"
            ),
        )
    # Integer axes are retained verbatim: every emitted peak position and every
    # interior row index contributes its own decimal width to the exact result.
    # Only the peak positions that are actually emitted are charged.
    peak_index_digits = sum(len(str(index)) for index in peaks)
    interior_index_digits = sum(len(str(index)) for index in range(1, max(size - 1, 1)))
    result_digits = source_digits + peak_index_digits + interior_index_digits + 5
    product_component_digits = 1
    for index in range(1, size - 1):
        if index % 128 == 0:
            request_checkpoint("during sequence order-shape product admission")
        square_numerator, square_denominator = _product_component_digits(
            rational_values[index], rational_values[index]
        )
        neighbor_numerator, neighbor_denominator = _product_component_digits(
            rational_values[index - 1], rational_values[index + 1]
        )
        product_component_digits = max(
            product_component_digits,
            square_numerator,
            square_denominator,
            neighbor_numerator,
            neighbor_denominator,
        )
        result_digits += (
            square_numerator
            + square_denominator
            + neighbor_numerator
            + neighbor_denominator
        )
    if row_count and product_component_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("values",),
            code="sequences.order_shape.result_digits_exceeded",
            message="order-shape cross-products exceed the exact rational digit bound",
        )
    if result_digits > MAX_SEQUENCE_TOTAL_DIGITS:
        raise OperationDomainValidationError(
            location=("values",),
            code="sequences.order_shape.result_representation_too_large",
            message=(
                "order-shape cross-products exceed the exact result representation "
                f"bound of {MAX_SEQUENCE_TOTAL_DIGITS} digits"
            ),
        )
    return rational_values, peaks


def _autocorrelation_scalar(
    value: Fraction,
    *,
    rational_output: bool,
) -> int | CanonicalRational:
    if rational_output or value.denominator != 1:
        return CanonicalRational.from_fraction(value)
    return value.numerator


def _require_autocorrelation_work(
    multiplications: int, additions: int, operand_width: int, *, convention: str
) -> None:
    if (
        multiplications * operand_width > MAX_AUTOCORRELATION_MULTIPLICATIONS
        or additions * operand_width > MAX_AUTOCORRELATION_ADDITIONS
    ):
        raise OperationResourceAdmissionError(
            location=("values",),
            code="sequences.autocorrelation.work_bound",
            message=(
                f"{convention} autocorrelation exceeds the admitted arithmetic-work bound"
            ),
        )


def aperiodic_autocorrelation(
    request: FiniteIntegerSequence | FiniteRationalSequence,
) -> AutocorrelationResult:
    size = len(request.values)
    multiplications = size * (size + 1) // 2
    additions = size * (size - 1) // 2
    values, _, operand_width = _admit_autocorrelation(
        request, output_items=max(2 * size - 1, 0)
    )
    _require_autocorrelation_work(
        multiplications, additions, operand_width, convention="aperiodic"
    )
    rational_output = isinstance(request, FiniteRationalSequence)
    cells = tuple(
        AutocorrelationCell(
            lag=lag,
            value=_autocorrelation_scalar(
                sum(
                    (
                        values[index] * values[index + lag]
                        for index in range(size - lag)
                    ),
                    Fraction(0),
                ),
                rational_output=rational_output,
            ),
        )
        for lag in range(size)
    )
    negative = tuple(
        AutocorrelationCell(lag=-cell.lag, value=cell.value)
        for cell in reversed(cells[1:])
    )
    return AutocorrelationResult(
        convention="aperiodic", source=request, cells=negative + cells
    )


def cyclic_autocorrelation(
    request: FiniteIntegerSequence | FiniteRationalSequence,
) -> AutocorrelationResult:
    size = len(request.values)
    multiplications = size * size
    additions = size * max(size - 1, 0)
    values, _, operand_width = _admit_autocorrelation(request, output_items=size)
    _require_autocorrelation_work(
        multiplications, additions, operand_width, convention="cyclic"
    )
    rational_output = isinstance(request, FiniteRationalSequence)
    return AutocorrelationResult(
        convention="cyclic",
        source=request,
        cells=tuple(
            AutocorrelationCell(
                lag=lag,
                value=_autocorrelation_scalar(
                    sum(
                        (
                            values[index] * values[(index + lag) % size]
                            for index in range(size)
                        ),
                        Fraction(0),
                    ),
                    rational_output=rational_output,
                ),
            )
            for lag in range(size)
        ),
    )


def _order_shape_scalar(
    value: Fraction,
) -> CanonicalRational:
    """Encode a product in the canonical rational result domain."""

    return CanonicalRational.from_fraction(value)


def _order_shape_monotonicity(
    fractions: tuple[Fraction, ...],
) -> tuple[int | None, int | None]:
    """Return the first weak-monotonicity violations in each direction."""

    nondecreasing_violation = None
    nonincreasing_violation = None
    for index in range(len(fractions) - 1):
        if index % 512 == 0:
            request_checkpoint("during order-shape monotonicity scan")
        if nondecreasing_violation is None and fractions[index] > fractions[index + 1]:
            nondecreasing_violation = index
        if nonincreasing_violation is None and fractions[index] < fractions[index + 1]:
            nonincreasing_violation = index
    return nondecreasing_violation, nonincreasing_violation


def _order_shape_peaks(fractions: tuple[Fraction, ...]) -> tuple[int, ...]:
    """Return every weak-unimodal peak position."""

    size = len(fractions)
    nondecreasing_prefix = [True] * size
    for index in range(1, size):
        if index % 512 == 0:
            request_checkpoint("during order-shape prefix scan")
        nondecreasing_prefix[index] = (
            nondecreasing_prefix[index - 1] and fractions[index - 1] <= fractions[index]
        )
    nonincreasing_suffix = [True] * size
    for index in range(size - 2, -1, -1):
        if index % 512 == 0:
            request_checkpoint("during order-shape suffix scan")
        nonincreasing_suffix[index] = (
            nonincreasing_suffix[index + 1] and fractions[index] >= fractions[index + 1]
        )
    peaks: list[int] = []
    for index in range(size):
        if index % 512 == 0:
            request_checkpoint("during order-shape peak scan")
        if nondecreasing_prefix[index] and nonincreasing_suffix[index]:
            peaks.append(index)
    return tuple(peaks)


def sequence_order_shape(
    request: FiniteRationalSequence,
) -> SequenceOrderShapeResult:
    admitted_values, peaks = _admit_order_shape(request)
    fractions = tuple(value.as_fraction() for value in admitted_values)
    nondecreasing_violation, nonincreasing_violation = _order_shape_monotonicity(
        fractions
    )
    log_row_list: list[SequenceLogConcavityRow] = []
    for index in range(1, len(fractions) - 1):
        if index % 512 == 0:
            request_checkpoint("during order-shape row construction")
        square = fractions[index] ** 2
        neighbor_product = fractions[index - 1] * fractions[index + 1]
        log_row_list.append(
            SequenceLogConcavityRow(
                index=index,
                square=_order_shape_scalar(square),
                neighbor_product=_order_shape_scalar(neighbor_product),
                holds=square >= neighbor_product,
            )
        )
    log_rows = tuple(log_row_list)
    first_log_violation = next((row.index for row in log_rows if not row.holds), None)
    first_negative = None
    nonzero: list[int] = []
    for index, value in enumerate(fractions):
        if index % 512 == 0:
            request_checkpoint("during order-shape witness scan")
        if first_negative is None and value < 0:
            first_negative = index
        if value != 0:
            nonzero.append(index)
    internal_zero_list: list[int] = []
    if nonzero:
        for index in range(nonzero[0] + 1, nonzero[-1]):
            if index % 512 == 0:
                request_checkpoint("during order-shape internal-zero scan")
            if fractions[index] == 0:
                internal_zero_list.append(index)
    internal_zero_indices = tuple(internal_zero_list)
    first_internal_zero = internal_zero_indices[0] if internal_zero_indices else None
    return SequenceOrderShapeResult(
        source=request,
        first_nondecreasing_violation=nondecreasing_violation,
        first_nonincreasing_violation=nonincreasing_violation,
        weak_unimodal_peak_positions=peaks,
        log_concavity_rows=log_rows,
        first_log_concavity_violation=first_log_violation,
        is_nonnegative=first_negative is None,
        first_negative_index=first_negative,
        has_internal_zero=bool(internal_zero_indices),
        first_internal_zero_index=first_internal_zero,
    )


def _list_result(values: list[int]) -> IntegerSequenceListResult:
    return IntegerSequenceListResult(values=tuple(values))


def sequence_sum(request: IntegerSequence) -> IntegerSequenceValueResult:
    values = _admit(
        request, output_digits=_digits(request.values) + len(str(len(request.values)))
    )
    return _value_result(sum(values))


def sequence_product(request: IntegerSequence) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_multiplicative_digits(request))
    return _value_result(math.prod(values))


def sequence_gcd(request: IntegerSequence) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_digits(request.values))
    return _value_result(reduce(math.gcd, values, 0))


def sequence_lcm(request: IntegerSequence) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_multiplicative_digits(request))
    return _value_result(reduce(math.lcm, values, 1))


def sequence_minimum(request: IntegerSequence) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_digits(request.values))
    return _value_result(min(values))


def sequence_maximum(request: IntegerSequence) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_digits(request.values))
    return _value_result(max(values))


def sequence_range(request: IntegerSequence) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_digits(request.values) + 1)
    return _value_result(max(values) - min(values))


def sequence_mean(request: IntegerSequence) -> IntegerSequenceRationalResult:
    values = _admit(
        request, output_digits=_digits(request.values) + len(str(len(request.values)))
    )
    fraction = Fraction(sum(values), len(values))
    return IntegerSequenceRationalResult(
        value=CanonicalRational(
            num=fraction.numerator,
            den=fraction.denominator,
        )
    )


def sequence_median(request: IntegerSequence) -> IntegerSequenceRationalResult:
    values = sorted(_admit(request, output_digits=_digits(request.values) + 1))
    middle = len(values) // 2
    if len(values) % 2:
        fraction = Fraction(values[middle])
    else:
        fraction = Fraction(values[middle - 1] + values[middle], 2)
    return IntegerSequenceRationalResult(
        value=CanonicalRational(
            num=fraction.numerator,
            den=fraction.denominator,
        )
    )


def sequence_distinct_count(
    request: IntegerSequence,
) -> IntegerSequenceValueResult:
    _admit(request, output_digits=len(str(len(request.values))))
    return _value_result(len(set(_values(request))))


def prefix_sums(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values) + len(str(len(request.values))),
        output_items=len(request.values),
    )
    total = 0
    result: list[int] = []
    for value in values:
        total += value
        result.append(total)
    return _list_result(result)


def first_differences(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values) + 1,
        output_items=max(len(request.values) - 1, 0),
    )
    return _list_result([right - left for left, right in pairwise(values)])


def second_differences(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values) + 2,
        output_items=max(len(request.values) - 2, 0),
    )
    first = [right - left for left, right in pairwise(values)]
    return _list_result([right - left for left, right in pairwise(first)])


def prefix_products(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_multiplicative_digits(request, prefix=True),
        output_items=len(request.values),
    )
    total = 1
    result: list[int] = []
    for value in values:
        total *= value
        result.append(total)
    return _list_result(result)


def prefix_minima(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    result = [values[0]]
    for value in values[1:]:
        result.append(min(result[-1], value))
    return _list_result(result)


def prefix_maxima(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    result = [values[0]]
    for value in values[1:]:
        result.append(max(result[-1], value))
    return _list_result(result)


def prefix_gcds(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    result = [abs(values[0])]
    for value in values[1:]:
        result.append(math.gcd(result[-1], value))
    return _list_result(result)


def prefix_lcms(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_multiplicative_digits(request, prefix=True),
        output_items=len(request.values),
    )
    result = [abs(values[0])]
    for value in values[1:]:
        result.append(math.lcm(result[-1], value))
    return _list_result(result)


def sorted_unique(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    return _list_result(sorted(set(values)))


def sort_sequence(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    return _list_result(sorted(values))


def reverse_sequence(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    return _list_result(list(reversed(values)))


def parities(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(request, output_digits=1, output_items=len(request.values))
    return _list_result([value % 2 for value in values])


def signs(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(request, output_digits=1, output_items=len(request.values))
    return _list_result([(value > 0) - (value < 0) for value in values])


def frequencies(request: IntegerSequence) -> IntegerSequenceFrequenciesResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    counts = Counter(values)
    entries = tuple(
        FrequencyEntry(value=value, count=counts[value]) for value in sorted(counts)
    )
    return IntegerSequenceFrequenciesResult(entries=entries)


def zero_indices(request: IntegerSequence) -> IntegerSequenceIndexListResult:
    values = _admit(
        request,
        output_digits=len(str(len(request.values))),
        output_items=len(request.values),
    )
    return IntegerSequenceIndexListResult(
        indices=tuple(index for index, value in enumerate(values) if value == 0),
    )


def decide_arithmetic(request: IntegerSequence) -> IntegerSequenceBooleanResult:
    values = _admit(request, output_digits=1)
    if len(values) < 2:
        return IntegerSequenceBooleanResult(holds=True)
    differences = {right - left for left, right in pairwise(values)}
    return IntegerSequenceBooleanResult(holds=len(differences) <= 1)


def decide_geometric(request: IntegerSequence) -> IntegerSequenceBooleanResult:
    values = _admit(request, output_digits=1)
    if len(values) < 2:
        return IntegerSequenceBooleanResult(holds=True)
    if values[0] == 0:
        return IntegerSequenceBooleanResult(holds=all(value == 0 for value in values))
    ratio = Fraction(values[1], values[0])
    return IntegerSequenceBooleanResult(
        holds=all(
            right * ratio.denominator == left * ratio.numerator
            for left, right in pairwise(values)
        )
    )


def decide_nondecreasing(
    request: IntegerSequence,
) -> IntegerSequenceBooleanResult:
    values = _admit(request, output_digits=1)
    return IntegerSequenceBooleanResult(
        holds=all(left <= right for left, right in pairwise(values))
    )


def decide_strictly_increasing(
    request: IntegerSequence,
) -> IntegerSequenceBooleanResult:
    values = _admit(request, output_digits=1)
    return IntegerSequenceBooleanResult(
        holds=all(left < right for left, right in pairwise(values))
    )


__all__ = [
    "decide_arithmetic",
    "decide_geometric",
    "decide_nondecreasing",
    "decide_strictly_increasing",
    "first_differences",
    "frequencies",
    "parities",
    "prefix_gcds",
    "prefix_lcms",
    "prefix_maxima",
    "prefix_minima",
    "prefix_products",
    "prefix_sums",
    "reverse_sequence",
    "second_differences",
    "sequence_distinct_count",
    "sequence_gcd",
    "sequence_lcm",
    "sequence_maximum",
    "sequence_mean",
    "sequence_median",
    "sequence_minimum",
    "sequence_product",
    "sequence_range",
    "sequence_sum",
    "signs",
    "sort_sequence",
    "sorted_unique",
    "zero_indices",
]
