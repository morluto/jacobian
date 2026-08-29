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


def _power_of_two_digit_bound(exponent: int) -> int:
    # 30103 / 100000 is a strict upper bound for log10(2).
    return (exponent * 30_103) // 100_000 + 1


def _admit_formula_result_digits(hyperplane_count: int) -> int:
    digit_bound = _power_of_two_digit_bound(hyperplane_count)
    if digit_bound > CanonicalLimits().max_integer_digits:
        _reject(
            ("hyperplane_count",),
            "formula_result_digits_exceeded",
            "generic-arrangement values exceed the canonical integer digit limit",
        )
    return digit_bound


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
    digit_bound = _admit_formula_result_digits(m)
    predicted_bytes = (n + 1) * (digit_bound + 3) + _FORMULA_RESULT_RESERVE_BYTES
    if predicted_bytes > CanonicalLimits().max_output_bytes:
        _reject(
            ("ambient_dimension", "hyperplane_count"),
            "characteristic_result_bytes_exceeded",
            "characteristic polynomial exceeds the canonical output-byte limit",
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
    _admit_formula_result_digits(m)
    if n >= m:
        count = 1 << m
    else:
        if n > MAX_GENERIC_FORMULA_WORK:
            _reject(
                ("ambient_dimension",),
                "chamber_summation_work_exceeded",
                "chamber count exceeds the binomial-summation work budget",
            )
        coefficient = 1
        total = 0
        for index in range(n):
            total += coefficient
            coefficient = coefficient * (m - 1 - index) // (index + 1)
        count = 2 * total
    return ChamberCountResult(chamber_count=format_canonical_integer(count))


__all__ = [
    "arrangement",
    "chamber_count",
    "characteristic_polynomial",
]
