"""Exact bounded i.i.d. Berry--Esseen bounds for finite rational laws."""

from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import Literal

from pydantic import Field, StrictInt

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.probability._distribution import (
    FiniteRationalDistribution,
    require_input_distribution,
)
from jacobian.math.probability._models import (
    MAX_INPUT_RATIONAL_DIGITS,
    MAX_RESULT_RATIONAL_DIGITS,
    _require_bounded_fraction,
)

MAX_BERRY_ESSEEN_ATOMS = 16_384
MAX_BERRY_ESSEEN_SAMPLE_COUNT = 10**12
BERRY_ESSEEN_CONSTANT = Fraction(14, 25)
BERRY_ESSEEN_BOUND_BITS = 64
BERRY_ESSEEN_THEOREM_VARIANT = "IID_BERRY_ESSEEN_C_05600"


class BerryEsseenRequest(StrictModel):
    """One finite rational law and a positive i.i.d. sample count."""

    distribution: FiniteRationalDistribution
    sample_count: StrictInt = Field(description="Positive i.i.d. sample count n.")


class BerryEsseenResult(StrictModel):
    """Exact source moments and an outward rational enclosure of the bound."""

    source: BerryEsseenRequest
    theorem_variant: Literal["IID_BERRY_ESSEEN_C_05600"]
    universal_constant: CanonicalRational
    mean: CanonicalRational
    variance: CanonicalRational
    third_absolute_central_moment: CanonicalRational
    bound_squared: CanonicalRational
    bound_lower: CanonicalRational
    bound_upper: CanonicalRational
    bound_precision_bits: int = Field(ge=1, le=256, strict=True)


def _admission_fraction(
    value: Fraction,
    *,
    location: tuple[str | int, ...],
    label: str,
) -> Fraction:
    try:
        _require_bounded_fraction(
            value,
            max_digits=MAX_RESULT_RATIONAL_DIGITS,
            label=label,
        )
    except ValueError as exc:
        raise OperationResourceAdmissionError(
            location=location,
            code="probability.berry_esseen.rational_height_bound",
            message=str(exc),
        ) from exc
    return value


def _add(
    left: Fraction,
    right: Fraction,
    *,
    location: tuple[str | int, ...],
    label: str,
) -> Fraction:
    return _admission_fraction(left + right, location=location, label=label)


def _mul(
    left: Fraction,
    right: Fraction,
    *,
    location: tuple[str | int, ...],
    label: str,
) -> Fraction:
    return _admission_fraction(left * right, location=location, label=label)


def _sqrt_interval(value: Fraction) -> tuple[Fraction, Fraction]:
    """Return a deterministic dyadic interval containing ``sqrt(value)``."""

    if value == 0:
        return Fraction(), Fraction()
    numerator_root = isqrt(value.numerator)
    denominator_root = isqrt(value.denominator)
    if numerator_root * numerator_root == value.numerator and (
        denominator_root * denominator_root == value.denominator
    ):
        exact = Fraction(numerator_root, denominator_root)
        return exact, exact
    scale = 1 << BERRY_ESSEEN_BOUND_BITS
    scaled_floor = (value.numerator * scale * scale) // value.denominator
    lower_numerator = isqrt(scaled_floor)
    lower = Fraction(lower_numerator, scale)
    return lower, Fraction(lower_numerator + 1, scale)


def berry_esseen_bound(request: BerryEsseenRequest) -> BerryEsseenResult:
    """Compute the exact i.i.d. Berry--Esseen upper bound."""

    if request.sample_count < 1:
        raise OperationDomainValidationError(
            location=("sample_count",),
            code="probability.berry_esseen.nonpositive_sample_count",
            message="Berry--Esseen sample_count must be positive",
        )
    if request.sample_count > MAX_BERRY_ESSEEN_SAMPLE_COUNT:
        raise OperationResourceAdmissionError(
            location=("sample_count",),
            code="probability.berry_esseen.sample_count_bound",
            message=(
                "Berry--Esseen admission allows sample_count in [1, "
                f"{MAX_BERRY_ESSEEN_SAMPLE_COUNT}]"
            ),
        )
    if len(request.distribution.atoms) > MAX_BERRY_ESSEEN_ATOMS:
        raise OperationResourceAdmissionError(
            location=("distribution", "atoms"),
            code="probability.berry_esseen.atom_work_bound",
            message=(
                "Berry--Esseen admission allows at most "
                f"{MAX_BERRY_ESSEEN_ATOMS} input atoms"
            ),
        )

    location = ("distribution",)
    try:
        require_input_distribution(
            request.distribution.atoms,
            require_canonical=True,
            max_digits=MAX_INPUT_RATIONAL_DIGITS,
        )
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=location,
            code="probability.berry_esseen.input_distribution",
            message=str(exc),
        ) from exc

    mean = Fraction()
    for atom in request.distribution.atoms:
        mean = _add(
            mean,
            _mul(
                atom.value.as_fraction(),
                atom.probability.as_fraction(),
                location=location,
                label="Berry--Esseen mean contribution",
            ),
            location=location,
            label="Berry--Esseen mean",
        )

    variance = Fraction()
    third = Fraction()
    for atom in request.distribution.atoms:
        centered = _admission_fraction(
            atom.value.as_fraction() - mean,
            location=location,
            label="Berry--Esseen centered value",
        )
        squared = _mul(
            centered,
            centered,
            location=location,
            label="Berry--Esseen centered square",
        )
        cubed_absolute = _mul(
            squared,
            abs(centered),
            location=location,
            label="Berry--Esseen centered third absolute power",
        )
        variance = _add(
            variance,
            _mul(
                atom.probability.as_fraction(),
                squared,
                location=location,
                label="Berry--Esseen variance contribution",
            ),
            location=location,
            label="Berry--Esseen variance",
        )
        third = _add(
            third,
            _mul(
                atom.probability.as_fraction(),
                cubed_absolute,
                location=location,
                label="Berry--Esseen third absolute contribution",
            ),
            location=location,
            label="Berry--Esseen third absolute moment",
        )
    if variance <= 0:
        raise OperationDomainValidationError(
            location=location,
            code="probability.berry_esseen.zero_variance_distribution",
            message="the Berry--Esseen theorem requires positive variance",
        )

    variance_squared = _mul(
        variance,
        variance,
        location=location,
        label="Berry--Esseen variance square",
    )
    variance_cubed = _mul(
        variance_squared,
        variance,
        location=location,
        label="Berry--Esseen variance cube",
    )
    third_squared = _mul(
        third,
        third,
        location=location,
        label="Berry--Esseen third absolute moment square",
    )
    denominator = _mul(
        variance_cubed,
        Fraction(request.sample_count),
        location=("sample_count",),
        label="Berry--Esseen variance cube times sample count",
    )
    numerator = _mul(
        BERRY_ESSEEN_CONSTANT * BERRY_ESSEEN_CONSTANT,
        third_squared,
        location=location,
        label="Berry--Esseen bound numerator",
    )
    bound_squared = _admission_fraction(
        numerator / denominator,
        location=("sample_count",),
        label="Berry--Esseen squared bound",
    )
    bound_lower, bound_upper = _sqrt_interval(bound_squared)
    _admission_fraction(
        bound_lower,
        location=("sample_count",),
        label="Berry--Esseen lower bound",
    )
    _admission_fraction(
        bound_upper,
        location=("sample_count",),
        label="Berry--Esseen upper bound",
    )

    return BerryEsseenResult(
        source=request,
        theorem_variant=BERRY_ESSEEN_THEOREM_VARIANT,
        universal_constant=CanonicalRational.from_fraction(BERRY_ESSEEN_CONSTANT),
        mean=CanonicalRational.from_fraction(mean),
        variance=CanonicalRational.from_fraction(variance),
        third_absolute_central_moment=CanonicalRational.from_fraction(third),
        bound_squared=CanonicalRational.from_fraction(bound_squared),
        bound_lower=CanonicalRational.from_fraction(bound_lower),
        bound_upper=CanonicalRational.from_fraction(bound_upper),
        bound_precision_bits=BERRY_ESSEEN_BOUND_BITS,
    )


__all__ = [
    "BERRY_ESSEEN_BOUND_BITS",
    "BERRY_ESSEEN_CONSTANT",
    "BERRY_ESSEEN_THEOREM_VARIANT",
    "MAX_BERRY_ESSEEN_ATOMS",
    "MAX_BERRY_ESSEEN_SAMPLE_COUNT",
    "BerryEsseenRequest",
    "BerryEsseenResult",
    "berry_esseen_bound",
]
