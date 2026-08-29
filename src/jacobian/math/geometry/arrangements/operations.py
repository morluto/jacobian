"""Exact native operations for hyperplane arrangements."""

from __future__ import annotations

from jacobian.canonical import CanonicalLimits, format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.arrangements._models import (
    MAX_GENERIC_FORMULA_INDEX,
    ChamberCountResult,
    CharacteristicPolynomialResult,
    HyperplaneArrangementResult,
    RationalHyperplane,
)

MAX_GENERIC_FORMULA_WORK = 100_000
_FORMULA_RESULT_RESERVE_BYTES = 4_096


def _reject(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=f"hyperplane_arrangement.{code}",
        message=message,
    )


def _formula_indices(ambient_dimension: int, hyperplane_count: int) -> tuple[int, int]:
    for name, value in (
        ("ambient_dimension", ambient_dimension),
        ("hyperplane_count", hyperplane_count),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= MAX_GENERIC_FORMULA_INDEX
        ):
            _reject(
                (name,),
                f"{name}_out_of_range",
                f"{name} is outside the admitted formula range",
            )
    return ambient_dimension, hyperplane_count


def _bit_bound_to_decimal_digits(bit_bound: int) -> int:
    # 30103 / 100000 is a strict upper bound for log10(2).
    return (bit_bound * 30_103 + 99_999) // 100_000


def _decimal_limb_count(digit_bound: int) -> int:
    if digit_bound <= 0:
        return 0
    return (digit_bound + 8) // 9


def _canonical_integer_conversion_work(digit_bound: int) -> int:
    """Limb work for ``format_canonical_integer`` plus positivity reparse.

    Each helper walks base-``10**9`` chunks. Every ``divmod`` or multiply-add
    processes the whole remaining or growing operand, so the pair of
    conversions costs the sum of operand widths ``1..chunks`` twice, which
    is ``chunks * (chunks + 1)``.
    """
    chunks = _decimal_limb_count(digit_bound)
    return chunks * (chunks + 1)


def _binomial_prefix_bit_bound(upper: int, length: int) -> int:
    if length <= 1 or upper == 0:
        return 1
    largest_index = min(length - 1, upper // 2)
    sparse_product_bound = largest_index * max(1, upper.bit_length())
    return max(1, min(sparse_product_bound, upper))


def _chamber_recurrence_limb_work(upper: int, terms: int) -> int:
    """Mul/div limb visits while forming ``C(upper, 0), ..., C(upper, terms-1)``.

    Each of ``terms`` adjacent-binomial updates multiplies then divides the
    current coefficient. Both walks traverse the whole operand. Width is the
    largest requested prefix term, measured in the same base-``10**9`` limbs
    as canonical integer conversion.
    """
    if terms <= 0:
        return 0
    coefficient_digits = _bit_bound_to_decimal_digits(
        _binomial_prefix_bit_bound(upper, terms)
    )
    limbs = max(1, _decimal_limb_count(coefficient_digits))
    return 2 * terms * limbs


def _chamber_recurrence(upper: int, terms: int) -> int:
    coefficient = 1
    total = 0
    for index in range(terms):
        total += coefficient
        coefficient = coefficient * (upper - index) // (index + 1)
    return 2 * total


def _signed_binomial_prefix(upper: int, length: int) -> tuple[int, ...]:
    terms: list[int] = []
    coefficient = 1
    for index in range(length):
        terms.append(coefficient if index % 2 == 0 else -coefficient)
        if index >= upper:
            coefficient = 0
        else:
            coefficient = coefficient * (upper - index) // (index + 1)
    return tuple(terms)


def arrangement(
    ambient_dimension: int,
    hyperplanes: tuple[RationalHyperplane, ...],
) -> HyperplaneArrangementResult:
    """Check if an arrangement is central (all hyperplanes pass through origin)."""
    is_central = all(
        hyperplane.constant.as_fraction() == 0 for hyperplane in hyperplanes
    )
    return HyperplaneArrangementResult(
        hyperplane_count=len(hyperplanes),
        ambient_dimension=ambient_dimension,
        is_central=is_central,
    )


def characteristic_polynomial(
    ambient_dimension: int,
    hyperplane_count: int,
) -> CharacteristicPolynomialResult:
    r"""Compute the characteristic polynomial of a generic central arrangement."""
    n, m = _formula_indices(ambient_dimension, hyperplane_count)
    if n > MAX_GENERIC_FORMULA_WORK:
        _reject(
            ("ambient_dimension",),
            "characteristic_coefficient_work_exceeded",
            "characteristic polynomial exceeds the coefficient-work budget",
        )
    nonzero_coefficients = min(n, m) + 1
    zero_coefficients = n + 1 - nonzero_coefficients
    coefficient_digits = _bit_bound_to_decimal_digits(
        _binomial_prefix_bit_bound(m - 1, min(n, m)) + 1
    )
    predicted_bytes = (
        nonzero_coefficients * (coefficient_digits + 3)
        + zero_coefficients * 4
        + _FORMULA_RESULT_RESERVE_BYTES
    )
    if predicted_bytes > CanonicalLimits().max_output_bytes:
        _reject(
            ("ambient_dimension", "hyperplane_count"),
            "characteristic_result_bytes_exceeded",
            "characteristic polynomial exceeds the canonical output-byte limit",
        )
    conversion_work = (
        nonzero_coefficients * _canonical_integer_conversion_work(coefficient_digits)
        + zero_coefficients
    )
    if n + conversion_work > MAX_GENERIC_FORMULA_WORK:
        _reject(
            ("ambient_dimension", "hyperplane_count"),
            "characteristic_formatting_work_exceeded",
            "characteristic polynomial exceeds the canonical integer-conversion work budget",
        )

    inner = _signed_binomial_prefix(m - 1, n)
    descending = (
        inner[0],
        *(inner[index] - inner[index - 1] for index in range(1, n)),
        -inner[-1],
    )
    return CharacteristicPolynomialResult(
        coefficients=tuple(
            format_canonical_integer(coefficient)
            for coefficient in reversed(descending)
        ),
        degree=n,
    )


def chamber_count(ambient_dimension: int, hyperplane_count: int) -> ChamberCountResult:
    r"""Count chambers of a generic central arrangement."""
    n, m = _formula_indices(ambient_dimension, hyperplane_count)
    if n >= m:
        result_bit_bound = m + 1
        result_digits = _bit_bound_to_decimal_digits(result_bit_bound)
        if (
            result_digits + _FORMULA_RESULT_RESERVE_BYTES
            > CanonicalLimits().max_output_bytes
        ):
            _reject(
                ("hyperplane_count",),
                "chamber_result_bytes_exceeded",
                "chamber count exceeds the canonical output-byte limit",
            )
        # Closed-form ``2^m`` construction is cheap, but canonical formatting
        # and the positivity parse are quadratic in the 9-digit chunk count.
        conversion_work = _canonical_integer_conversion_work(result_digits)
        if conversion_work > MAX_GENERIC_FORMULA_WORK:
            _reject(
                ("hyperplane_count",),
                "chamber_formatting_work_exceeded",
                "chamber count exceeds the canonical integer-conversion work budget",
            )
        count = 1 << m
    elif n == m - 1:
        result_bit_bound = m + 1
        result_digits = _bit_bound_to_decimal_digits(result_bit_bound)
        if (
            result_digits + _FORMULA_RESULT_RESERVE_BYTES
            > CanonicalLimits().max_output_bytes
        ):
            _reject(
                ("hyperplane_count",),
                "chamber_result_bytes_exceeded",
                "chamber count exceeds the canonical output-byte limit",
            )
        conversion_work = _canonical_integer_conversion_work(result_digits)
        if conversion_work > MAX_GENERIC_FORMULA_WORK:
            _reject(
                ("hyperplane_count",),
                "chamber_formatting_work_exceeded",
                "chamber count exceeds the canonical integer-conversion work budget",
            )
        count = (1 << m) - 2
    else:
        prefix_bit_bound = _binomial_prefix_bit_bound(m - 1, n)
        result_bit_bound = prefix_bit_bound + n.bit_length() + 1
        result_digits = _bit_bound_to_decimal_digits(result_bit_bound)
        recurrence_work = _chamber_recurrence_limb_work(m - 1, n)
        conversion_work = _canonical_integer_conversion_work(result_digits)
        if n + recurrence_work + conversion_work > MAX_GENERIC_FORMULA_WORK:
            _reject(
                ("ambient_dimension", "hyperplane_count"),
                "chamber_summation_work_exceeded",
                "chamber count exceeds the binomial-summation work budget",
            )
        if (
            result_digits + _FORMULA_RESULT_RESERVE_BYTES
            > CanonicalLimits().max_output_bytes
        ):
            _reject(
                ("ambient_dimension", "hyperplane_count"),
                "chamber_result_bytes_exceeded",
                "chamber count exceeds the canonical output-byte limit",
            )
        count = _chamber_recurrence(m - 1, n)
    return ChamberCountResult(chamber_count=format_canonical_integer(count))


__all__ = [
    "arrangement",
    "chamber_count",
    "characteristic_polynomial",
]
