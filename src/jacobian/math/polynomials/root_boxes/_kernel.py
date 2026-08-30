"""Exact rational midpoint Krawczyk kernel for polynomial systems."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from jacobian.canonical import format_canonical_integer
from jacobian.math.analysis.intervals import RationalBox

from ._models import (
    MAX_ROOT_BOX_INTERMEDIATE_DIGITS,
    MAX_ROOT_BOX_RESULT_COMPONENT_DIGITS,
)

type FractionInterval = tuple[Fraction, Fraction]
type FractionMatrix = tuple[tuple[Fraction, ...], ...]
type FractionIntervalMatrix = tuple[tuple[FractionInterval, ...], ...]


class RootBoxKernelBudgetError(ValueError):
    """Exact interval-matrix arithmetic exceeded its admitted height bound."""


@dataclass(frozen=True, slots=True)
class MidpointKernelData:
    center: tuple[Fraction, ...]
    value_at_center: tuple[Fraction, ...]
    jacobian_at_center: FractionMatrix
    jacobian_enclosure: FractionIntervalMatrix


@dataclass(frozen=True, slots=True)
class KrawczykKernelData(MidpointKernelData):
    preconditioner: FractionMatrix
    krawczyk_image: tuple[FractionInterval, ...]


@dataclass(frozen=True, slots=True)
class ComponentExclusionKernelResult:
    status: Literal["NO_ROOT_COMPONENT"]
    component_index: int
    enclosure: FractionInterval


@dataclass(frozen=True, slots=True)
class SingularMidpointKernelResult:
    status: Literal["UNKNOWN_SINGULAR_MIDPOINT"]
    data: MidpointKernelData


@dataclass(frozen=True, slots=True)
class KrawczykKernelResult:
    status: Literal[
        "CERTIFIED_UNIQUE_NONSINGULAR",
        "NO_ROOT_KRAWCZYK",
        "UNKNOWN_KRAWCZYK",
    ]
    evidence: KrawczykKernelData


type RootBoxKernelResult = (
    ComponentExclusionKernelResult | SingularMidpointKernelResult | KrawczykKernelResult
)


def _component_digits(value: Fraction) -> int:
    return max(
        len(format_canonical_integer(abs(value.numerator))),
        len(format_canonical_integer(value.denominator)),
    )


def _numerator_digits(value: Fraction) -> int:
    return len(format_canonical_integer(abs(value.numerator)))


def _denominator_digits(value: Fraction) -> int:
    return len(format_canonical_integer(value.denominator))


def _require_height(value: Fraction, *, maximum: int, label: str) -> Fraction:
    if _component_digits(value) > maximum:
        raise RootBoxKernelBudgetError(
            f"{label} exceeds the {maximum:,}-digit exact-arithmetic bound"
        )
    return value


def _checked_add(left: Fraction, right: Fraction) -> Fraction:
    if (
        max(
            _denominator_digits(left) + _denominator_digits(right),
            _numerator_digits(left) + _denominator_digits(right) + 1,
            _numerator_digits(right) + _denominator_digits(left) + 1,
        )
        > MAX_ROOT_BOX_INTERMEDIATE_DIGITS
    ):
        raise RootBoxKernelBudgetError(
            "rational addition exceeds the admitted intermediate-height bound"
        )
    return _require_height(
        left + right,
        maximum=MAX_ROOT_BOX_INTERMEDIATE_DIGITS,
        label="rational sum",
    )


def _checked_multiply(left: Fraction, right: Fraction) -> Fraction:
    if _component_digits(left) + _component_digits(right) > (
        MAX_ROOT_BOX_INTERMEDIATE_DIGITS + 1
    ):
        raise RootBoxKernelBudgetError(
            "rational multiplication exceeds the admitted intermediate-height bound"
        )
    return _require_height(
        left * right,
        maximum=MAX_ROOT_BOX_INTERMEDIATE_DIGITS,
        label="rational product",
    )


def _add_intervals(
    left: FractionInterval,
    right: FractionInterval,
) -> FractionInterval:
    return (
        _checked_add(left[0], right[0]),
        _checked_add(left[1], right[1]),
    )


def _multiply_intervals(
    left: FractionInterval,
    right: FractionInterval,
) -> FractionInterval:
    products = tuple(
        _checked_multiply(left_value, right_value)
        for left_value in left
        for right_value in right
    )
    return min(products), max(products)


def _scale_interval(
    interval: FractionInterval,
    scalar: Fraction,
) -> FractionInterval:
    lower = _checked_multiply(interval[0], scalar)
    upper = _checked_multiply(interval[1], scalar)
    return (lower, upper) if scalar >= 0 else (upper, lower)


def _invert_matrix(matrix: FractionMatrix) -> FractionMatrix | None:
    """Invert a bounded exact matrix through python-flint, or report singularity."""

    from flint import fmpq, fmpq_mat

    order = len(matrix)
    backend = fmpq_mat(
        [[fmpq(value.numerator, value.denominator) for value in row] for row in matrix]
    )
    try:
        inverse = backend.inv()
    except ZeroDivisionError:
        return None
    result = tuple(
        tuple(
            _require_height(
                Fraction(
                    int(inverse[row, column].p),
                    int(inverse[row, column].q),
                ),
                maximum=MAX_ROOT_BOX_RESULT_COMPONENT_DIGITS,
                label="exact midpoint-Jacobian inverse component",
            )
            for column in range(order)
        )
        for row in range(order)
    )
    return result


def _matrix_interval_product(
    left: FractionMatrix,
    right: FractionIntervalMatrix,
) -> FractionIntervalMatrix:
    order = len(left)
    rows: list[tuple[FractionInterval, ...]] = []
    for row in range(order):
        output_row: list[FractionInterval] = []
        for column in range(order):
            total = (Fraction(0), Fraction(0))
            for inner in range(order):
                total = _add_intervals(
                    total,
                    _scale_interval(right[inner][column], left[row][inner]),
                )
            output_row.append(total)
        rows.append(tuple(output_row))
    return tuple(rows)


def _identity_minus(
    matrix: FractionIntervalMatrix,
) -> FractionIntervalMatrix:
    return tuple(
        tuple(
            (
                _checked_add(Fraction(row == column), -entry[1]),
                _checked_add(Fraction(row == column), -entry[0]),
            )
            for column, entry in enumerate(source_row)
        )
        for row, source_row in enumerate(matrix)
    )


def _matrix_interval_vector_product(
    matrix: FractionIntervalMatrix,
    vector: tuple[FractionInterval, ...],
) -> tuple[FractionInterval, ...]:
    result: list[FractionInterval] = []
    for row in matrix:
        total = (Fraction(0), Fraction(0))
        for entry, coordinate in zip(row, vector, strict=True):
            total = _add_intervals(total, _multiply_intervals(entry, coordinate))
        result.append(total)
    return tuple(result)


def _matrix_vector_product(
    matrix: FractionMatrix,
    vector: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    result: list[Fraction] = []
    for row in matrix:
        total = Fraction(0)
        for entry, value in zip(row, vector, strict=True):
            total = _checked_add(total, _checked_multiply(entry, value))
        result.append(total)
    return tuple(result)


def _krawczyk_image(
    box: RationalBox,
    data: MidpointKernelData,
    preconditioner: FractionMatrix,
) -> tuple[FractionInterval, ...]:
    centered_box = tuple(
        (
            interval.lower.as_fraction() - center,
            interval.upper.as_fraction() - center,
        )
        for interval, center in zip(box.intervals, data.center, strict=True)
    )
    if all(value == 0 for value in data.value_at_center):
        residual_center = data.center
    else:
        residual_center = tuple(
            _checked_add(center, -correction)
            for center, correction in zip(
                data.center,
                _matrix_vector_product(preconditioner, data.value_at_center),
                strict=True,
            )
        )
    constant_jacobian = all(
        enclosure == (data.jacobian_at_center[row][column],) * 2
        for row, entries in enumerate(data.jacobian_enclosure)
        for column, enclosure in enumerate(entries)
    )
    if constant_jacobian:
        remainder = ((Fraction(0), Fraction(0)),) * len(data.center)
    else:
        remainder = _matrix_interval_vector_product(
            _identity_minus(
                _matrix_interval_product(preconditioner, data.jacobian_enclosure)
            ),
            centered_box,
        )
    image = tuple(
        (
            _checked_add(center, interval[0]),
            _checked_add(center, interval[1]),
        )
        for center, interval in zip(residual_center, remainder, strict=True)
    )
    for interval in image:
        for endpoint in interval:
            _require_height(
                endpoint,
                maximum=MAX_ROOT_BOX_RESULT_COMPONENT_DIGITS,
                label="Krawczyk-image endpoint",
            )
    return image


def _strictly_inside(
    image: tuple[FractionInterval, ...],
    box: RationalBox,
) -> bool:
    return all(
        source.lower.as_fraction() < enclosure[0]
        and enclosure[1] < source.upper.as_fraction()
        for enclosure, source in zip(image, box.intervals, strict=True)
    )


def _is_disjoint(
    image: tuple[FractionInterval, ...],
    box: RationalBox,
) -> bool:
    return any(
        enclosure[1] < source.lower.as_fraction()
        or source.upper.as_fraction() < enclosure[0]
        for enclosure, source in zip(image, box.intervals, strict=True)
    )


def certify_root_box_kernel(
    box: RationalBox,
    prepared: ComponentExclusionKernelResult | MidpointKernelData,
) -> RootBoxKernelResult:
    """Run one exact deterministic midpoint Krawczyk attempt."""

    if isinstance(prepared, ComponentExclusionKernelResult):
        return prepared

    preconditioner = _invert_matrix(prepared.jacobian_at_center)
    if preconditioner is None:
        return SingularMidpointKernelResult(
            status="UNKNOWN_SINGULAR_MIDPOINT",
            data=prepared,
        )

    image = _krawczyk_image(box, prepared, preconditioner)
    evidence = KrawczykKernelData(
        center=prepared.center,
        value_at_center=prepared.value_at_center,
        jacobian_at_center=prepared.jacobian_at_center,
        jacobian_enclosure=prepared.jacobian_enclosure,
        preconditioner=preconditioner,
        krawczyk_image=image,
    )
    if _is_disjoint(image, box):
        return KrawczykKernelResult(status="NO_ROOT_KRAWCZYK", evidence=evidence)
    if _strictly_inside(image, box):
        return KrawczykKernelResult(
            status="CERTIFIED_UNIQUE_NONSINGULAR",
            evidence=evidence,
        )
    return KrawczykKernelResult(status="UNKNOWN_KRAWCZYK", evidence=evidence)


__all__ = [
    "ComponentExclusionKernelResult",
    "KrawczykKernelData",
    "KrawczykKernelResult",
    "MidpointKernelData",
    "RootBoxKernelBudgetError",
    "RootBoxKernelResult",
    "SingularMidpointKernelResult",
    "certify_root_box_kernel",
]
