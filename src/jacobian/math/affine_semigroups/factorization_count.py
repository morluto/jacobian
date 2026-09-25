"""Exact bounded counts of finite affine-semigroup fibers."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.affine_semigroups.semigroup import (
    MAX_AFFINE_DIGITS,
    MAX_AFFINE_FIBER_WORK,
    MAX_AFFINE_GENERATORS,
    MAX_AFFINE_ROWS,
    AffineConfiguration,
    PositiveAffineSemigroup,
    _admit_fiber,
    _admit_semigroup,
    _validate_target,
)
from jacobian.math.affine_semigroups.semigroup_models import (
    AffineFactorizationCountRequest,
)

MAX_AFFINE_FACTOR_COUNT_STATES = 250_000
MAX_AFFINE_FACTOR_COUNT_WORK = 2_000_000
MAX_AFFINE_FACTOR_COUNT_DIGITS = 128
MAX_AFFINE_FACTOR_COUNT_INPUT_DIGITS = 32
MAX_AFFINE_FACTOR_COUNT_GRADING_DIGITS = 64
MAX_AFFINE_FACTOR_COUNT_LABEL_CHARS = 4_096


class AffineFactorizationCount(StrictModel):
    """The exact number of nonnegative factorizations of one target."""

    semigroup: PositiveAffineSemigroup
    target: tuple[ExactInteger, ...]
    count: ExactInteger

    @model_validator(mode="after")
    def require_canonical_count_shape(self) -> Self:
        if len(self.target) != self.semigroup.configuration.rows:
            raise PydanticCustomError(
                "affine_semigroup.factorization_count_target",
                "the target must retain the semigroup ambient row axis",
            )
        if self.count < 0:
            raise PydanticCustomError(
                "affine_semigroup.factorization_count_sign",
                "a factorization count must be nonnegative",
            )
        return self


def _admit_count_source(value: object) -> PositiveAffineSemigroup:
    """Bound typed axes and scalars before revalidation copies their values."""
    if type(value) is not PositiveAffineSemigroup:
        raise OperationDomainValidationError(
            location=("semigroup",),
            code="affine_semigroup.semigroup",
            message="semigroup must be a canonical positive affine semigroup",
        )
    configuration = value.configuration
    if type(configuration) is not AffineConfiguration:
        raise OperationDomainValidationError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.configuration",
            message="configuration must be a canonical affine configuration",
        )
    if (
        type(configuration.row_labels) is not tuple
        or not 1 <= len(configuration.row_labels) <= MAX_AFFINE_ROWS
        or type(configuration.generator_labels) is not tuple
        or not 1 <= len(configuration.generator_labels) <= MAX_AFFINE_GENERATORS
        or type(configuration.entries) is not tuple
        or len(configuration.entries) != len(configuration.row_labels)
        or any(type(row) is not tuple for row in configuration.entries)
        or any(
            len(row) != len(configuration.generator_labels)
            for row in configuration.entries
        )
    ):
        raise OperationDomainValidationError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.factorization_count_shape",
            message="configuration axes exceed the bounded exact count shape",
        )
    labels = (*configuration.row_labels, *configuration.generator_labels)
    if (
        any(
            type(label) is not str
            or any(0xD800 <= ord(character) <= 0xDFFF for character in label)
            for label in labels
        )
        or sum(len(label) for label in labels) > MAX_AFFINE_FACTOR_COUNT_LABEL_CHARS
    ):
        raise OperationResourceAdmissionError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.factorization_count_labels",
            message="configuration labels exceed the admitted exact count input",
        )
    if any(
        type(entry) is not int or abs(entry) >= 10**MAX_AFFINE_DIGITS
        for row in configuration.entries
        for entry in row
    ):
        raise OperationResourceAdmissionError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.factorization_count_matrix_digits",
            message=f"configuration entries are limited to {MAX_AFFINE_DIGITS} digits",
        )
    if type(value.grading) is not tuple or len(value.grading) != len(
        configuration.row_labels
    ):
        raise OperationDomainValidationError(
            location=("semigroup", "grading"),
            code="affine_semigroup.factorization_count_grading",
            message="grading must retain one exact rational per ambient row",
        )
    grading_limit = 10**MAX_AFFINE_FACTOR_COUNT_GRADING_DIGITS
    for component in value.grading:
        if (
            type(component) is not CanonicalRational
            or type(component.num) is not int
            or type(component.den) is not int
            or component.den <= 0
        ):
            raise OperationDomainValidationError(
                location=("semigroup", "grading"),
                code="affine_semigroup.factorization_count_grading",
                message="grading components must be canonical exact rationals",
            )
        if abs(component.num) >= grading_limit or component.den >= grading_limit:
            raise OperationResourceAdmissionError(
                location=("semigroup", "grading"),
                code="affine_semigroup.factorization_count_grading_digits",
                message=(
                    "grading numerators and denominators are limited to "
                    f"{MAX_AFFINE_FACTOR_COUNT_GRADING_DIGITS} digits"
                ),
            )
    try:
        return _admit_semigroup(value)
    except OperationDomainValidationError:
        raise
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("semigroup",),
            code="affine_semigroup.semigroup",
            message="semigroup is malformed or not positive",
        ) from exc


def _count_one_row(semigroup: PositiveAffineSemigroup, target: tuple[int, ...]) -> int:
    """Use the exact univariate generating-function recurrence."""
    weights = tuple(vector[0] for vector in semigroup.configuration.columns_vectors)
    direction = 1 if weights[0] > 0 else -1
    if any(weight * direction <= 0 for weight in weights):
        raise OperationDomainValidationError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.factorization_count_not_positive",
            message="one-row generators must lie on one strictly positive ray",
        )
    signed_target = direction * target[0]
    if signed_target < 0:
        return 0
    if signed_target == 0:
        return 1

    divisor = gcd(*(abs(weight) for weight in weights))
    if signed_target % divisor:
        return 0
    target_degree = signed_target // divisor
    positive_weights = tuple(abs(weight) // divisor for weight in weights)
    state_count = target_degree + 1
    work_bound = sum(max(0, target_degree - weight + 1) for weight in positive_weights)
    if state_count > MAX_AFFINE_FACTOR_COUNT_STATES:
        raise OperationResourceAdmissionError(
            location=("target",),
            code="affine_semigroup.factorization_count_states",
            message=(
                f"univariate count requires {state_count} states; the limit is "
                f"{MAX_AFFINE_FACTOR_COUNT_STATES}"
            ),
        )
    if work_bound > MAX_AFFINE_FACTOR_COUNT_WORK:
        raise OperationResourceAdmissionError(
            location=("target",),
            code="affine_semigroup.factorization_count_work",
            message=(
                f"univariate count work {work_bound} exceeds "
                f"{MAX_AFFINE_FACTOR_COUNT_WORK} updates"
            ),
        )
    maximum_count = 1
    for weight in positive_weights:
        maximum_count *= target_degree // weight + 1
    if len(str(maximum_count)) > MAX_AFFINE_FACTOR_COUNT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("target",),
            code="affine_semigroup.factorization_count_digits",
            message=(
                "factorization count may exceed the "
                f"{MAX_AFFINE_FACTOR_COUNT_DIGITS}-digit result envelope"
            ),
        )

    return _univariate_coin_change(positive_weights, target_degree)


def _univariate_coin_change(weights: tuple[int, ...], target: int) -> int:
    """Count unbounded weighted partitions by coefficient recurrence."""
    counts = [0] * (target + 1)
    counts[0] = 1
    # [z^b] product_i (1-z^w_i)^(-1), processing each labelled generator
    # separately so duplicate columns count as distinct coordinates.
    for weight in weights:
        for degree in range(weight, target + 1):
            counts[degree] += counts[degree - weight]
    return counts[target]


def _count_general_fiber(
    semigroup: PositiveAffineSemigroup,
    target: tuple[int, ...],
    grades: tuple[Fraction, ...],
    target_grade: Fraction,
    maxima: tuple[int, ...],
) -> int:
    """Count an admitted coefficient box without materializing its fiber."""
    if target_grade < 0:
        return 0
    configuration = semigroup.configuration
    coefficient_box = 1
    for maximum in maxima:
        coefficient_box *= maximum + 1
    work_bound = coefficient_box * (
        configuration.columns + configuration.rows * configuration.columns
    )
    if work_bound > MAX_AFFINE_FACTOR_COUNT_WORK:
        raise OperationResourceAdmissionError(
            location=("target",),
            code="affine_semigroup.factorization_count_work",
            message=(
                f"general fiber count work {work_bound} exceeds "
                f"{MAX_AFFINE_FACTOR_COUNT_WORK} units"
            ),
        )
    if len(str(coefficient_box)) > MAX_AFFINE_FACTOR_COUNT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("target",),
            code="affine_semigroup.factorization_count_digits",
            message=(
                "factorization count may exceed the "
                f"{MAX_AFFINE_FACTOR_COUNT_DIGITS}-digit result envelope"
            ),
        )

    current = [0] * configuration.columns
    count = 0

    def visit(index: int, remaining_grade: Fraction) -> None:
        nonlocal count
        if index == configuration.columns:
            if all(
                sum(
                    current[column] * configuration.entries[row][column]
                    for column in range(configuration.columns)
                )
                == target[row]
                for row in range(configuration.rows)
            ):
                count += 1
            return
        maximum = min(maxima[index], int(remaining_grade // grades[index]))
        for coefficient in range(maximum + 1):
            current[index] = coefficient
            visit(index + 1, remaining_grade - coefficient * grades[index])
        current[index] = 0

    visit(0, target_grade)
    return count


def factorization_count(
    semigroup: PositiveAffineSemigroup, target: tuple[int, ...]
) -> AffineFactorizationCount:
    """Return the exact cardinality of a finite positive-semigroup fiber."""
    source = _admit_count_source(semigroup)
    _validate_target(source, target)
    target_limit = 10**MAX_AFFINE_FACTOR_COUNT_INPUT_DIGITS
    if any(abs(value) >= target_limit for value in target):
        raise OperationResourceAdmissionError(
            location=("target",),
            code="affine_semigroup.factorization_count_target_digits",
            message=(
                "target coordinates are limited to "
                f"{MAX_AFFINE_FACTOR_COUNT_INPUT_DIGITS} digits"
            ),
        )
    if source.configuration.rows == 1:
        count = _count_one_row(source, target)
    else:
        grades, target_grade, maxima = _admit_fiber(source, target)
        candidate_bound = 1
        for maximum in maxima:
            candidate_bound *= maximum + 1
        if candidate_bound > MAX_AFFINE_FIBER_WORK:
            raise OperationResourceAdmissionError(
                location=("target",),
                code="affine_semigroup.factorization_count_candidates",
                message=(
                    f"fiber coefficient box {candidate_bound} exceeds "
                    f"{MAX_AFFINE_FIBER_WORK} candidate states"
                ),
            )
        count = _count_general_fiber(source, target, grades, target_grade, maxima)
    return AffineFactorizationCount(semigroup=source, target=target, count=count)


__all__ = [
    "AffineFactorizationCount",
    "AffineFactorizationCountRequest",
    "factorization_count",
]
