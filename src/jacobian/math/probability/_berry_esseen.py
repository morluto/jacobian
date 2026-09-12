"""Exact bounded Berry--Esseen bounds for finite rational summands.

The operation uses the published general independent-summand constant
``C = 0.5600 = 14/25`` from Shevtsova, *On the asymptotically exact constants
in the Berry--Esseen--Katz inequality*, Theory of Probability and its
Applications 55 (2011), DOI: 10.4213/tvp4201.  The result is the theorem's
explicit upper bound; it is not a claim that the Kolmogorov distance was
computed.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import Literal

from pydantic import Field

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

MAX_BERRY_ESSEEN_SUMMANDS = 256
MAX_BERRY_ESSEEN_ATOMS = 16_384
BERRY_ESSEEN_CONSTANT = Fraction(14, 25)
BERRY_ESSEEN_BOUND_BITS = 64
BERRY_ESSEEN_THEOREM_VARIANT = (
    "INDEPENDENT_NON_IDENTICALLY_DISTRIBUTED_BERRY_ESSEEN_C_05600"
)


class BerryEsseenRequest(StrictModel):
    """Independent finite rational laws, one law for each summand."""

    summands: tuple[FiniteRationalDistribution, ...] = Field(
        min_length=1,
        max_length=MAX_BERRY_ESSEEN_SUMMANDS,
        description=(
            "Independent summands. Repeating one law gives the i.i.d. case; "
            "different laws are allowed by the published independent-summand theorem."
        ),
    )


class BerryEsseenResult(StrictModel):
    """Exact moments and an outward rational enclosure of the theorem bound."""

    source: BerryEsseenRequest
    theorem_variant: Literal[
        "INDEPENDENT_NON_IDENTICALLY_DISTRIBUTED_BERRY_ESSEEN_C_05600"
    ]
    universal_constant: CanonicalRational
    total_mean: CanonicalRational
    total_variance: CanonicalRational
    total_third_absolute_central_moment: CanonicalRational
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
    if lower * lower == value:
        return lower, lower
    return lower, Fraction(lower_numerator + 1, scale)


def berry_esseen_bound(request: BerryEsseenRequest) -> BerryEsseenResult:
    """Compute the exact independent-summand Berry--Esseen upper bound."""

    total_atoms = sum(len(summand.atoms) for summand in request.summands)
    if total_atoms > MAX_BERRY_ESSEEN_ATOMS:
        raise OperationResourceAdmissionError(
            location=("summands",),
            code="probability.berry_esseen.atom_work_bound",
            message=(
                "Berry--Esseen admission allows at most "
                f"{MAX_BERRY_ESSEEN_ATOMS} input atoms across all summands"
            ),
        )

    means: list[Fraction] = []
    variances: list[Fraction] = []
    third_moments: list[Fraction] = []
    for index, summand in enumerate(request.summands):
        location = ("summands", index)
        try:
            require_input_distribution(
                summand.atoms,
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
        for atom in summand.atoms:
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
        for atom in summand.atoms:
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
                code="probability.berry_esseen.zero_variance_summand",
                message=(
                    "the pinned independent-summand theorem requires every "
                    "summand to have positive variance"
                ),
            )
        means.append(mean)
        variances.append(variance)
        third_moments.append(third)

    total_mean = Fraction()
    total_variance = Fraction()
    total_third = Fraction()
    for mean, variance, third in zip(means, variances, third_moments, strict=True):
        total_mean = _add(
            total_mean,
            mean,
            location=("summands",),
            label="Berry--Esseen total mean",
        )
        total_variance = _add(
            total_variance,
            variance,
            location=("summands",),
            label="Berry--Esseen total variance",
        )
        total_third = _add(
            total_third,
            third,
            location=("summands",),
            label="Berry--Esseen total third absolute moment",
        )

    variance_squared = _mul(
        total_variance,
        total_variance,
        location=("summands",),
        label="Berry--Esseen variance square",
    )
    variance_cubed = _mul(
        variance_squared,
        total_variance,
        location=("summands",),
        label="Berry--Esseen variance cube",
    )
    third_squared = _mul(
        total_third,
        total_third,
        location=("summands",),
        label="Berry--Esseen third absolute moment square",
    )
    numerator = _mul(
        BERRY_ESSEEN_CONSTANT * BERRY_ESSEEN_CONSTANT,
        third_squared,
        location=("summands",),
        label="Berry--Esseen bound numerator",
    )
    bound_squared = _admission_fraction(
        numerator / variance_cubed,
        location=("summands",),
        label="Berry--Esseen squared bound",
    )
    bound_lower, bound_upper = _sqrt_interval(bound_squared)
    _admission_fraction(
        bound_lower,
        location=("summands",),
        label="Berry--Esseen lower bound",
    )
    _admission_fraction(
        bound_upper,
        location=("summands",),
        label="Berry--Esseen upper bound",
    )

    return BerryEsseenResult(
        source=request,
        theorem_variant=BERRY_ESSEEN_THEOREM_VARIANT,
        universal_constant=CanonicalRational.from_fraction(BERRY_ESSEEN_CONSTANT),
        total_mean=CanonicalRational.from_fraction(total_mean),
        total_variance=CanonicalRational.from_fraction(total_variance),
        total_third_absolute_central_moment=CanonicalRational.from_fraction(
            total_third
        ),
        bound_squared=CanonicalRational.from_fraction(bound_squared),
        bound_lower=CanonicalRational.from_fraction(bound_lower),
        bound_upper=CanonicalRational.from_fraction(bound_upper),
        bound_precision_bits=BERRY_ESSEEN_BOUND_BITS,
    )


__all__ = [
    "BERRY_ESSEEN_BOUND_BITS",
    "BERRY_ESSEEN_CONSTANT",
    "BERRY_ESSEEN_THEOREM_VARIANT",
    "BerryEsseenRequest",
    "BerryEsseenResult",
    "berry_esseen_bound",
]
