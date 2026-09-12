"""Exact bounded i.i.d. Berry--Esseen bounds for finite rational laws.

The theorem variant is Shevtsova's general-independent Berry--Esseen
constant ``C = 0.5600 = 14/25``, specialized to repeated i.i.d. summands.
The operation deliberately retains that explicit, conservative constant
rather than claiming the sharper i.i.d.-specific constants from later work.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, DecimalIntegerEncoding
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
    _validation_error,
)

MAX_BERRY_ESSEEN_ATOMS = 16_384
# The pinned bound contains the sample count only as the exact factor n in
# variance**3 * n. Keep n within the result carrier's integer height; the
# resulting products and the bound itself are admitted separately below.
MAX_BERRY_ESSEEN_SAMPLE_COUNT = 10**MAX_RESULT_RATIONAL_DIGITS - 1
BERRY_ESSEEN_CONSTANT = Fraction(14, 25)
BERRY_ESSEEN_BOUND_BITS = 64
BERRY_ESSEEN_THEOREM_VARIANT: Literal[
    "IID_SPECIALIZATION_OF_GENERAL_INDEPENDENT_BERRY_ESSEEN_C_05600"
] = "IID_SPECIALIZATION_OF_GENERAL_INDEPENDENT_BERRY_ESSEEN_C_05600"


class BerryEsseenRequest(StrictModel):
    """One finite rational law and a positive i.i.d. sample count."""

    distribution: FiniteRationalDistribution
    sample_count: Annotated[
        int, DecimalIntegerEncoding(max_digits=MAX_RESULT_RATIONAL_DIGITS)
    ] = Field(
        description=(
            "Positive i.i.d. sample count n with at most "
            f"{MAX_RESULT_RATIONAL_DIGITS} decimal digits. Admission uses the "
            "reduced height of the standardized bound, not an independent "
            "cutoff on n; the field digit envelope is the representable limit."
        ),
        json_schema_extra={
            "pattern": (
                f"^[1-9][0-9]{{0,{MAX_RESULT_RATIONAL_DIGITS - 1}}}(?![\\s\\S])"
            )
        },
    )

    @model_validator(mode="after")
    def require_positive_sample_count(self) -> Self:
        """Keep positivity in the value contract, independent of wire encoding."""

        if self.sample_count < 1:
            raise _validation_error("Berry--Esseen sample_count must be positive")
        return self


class BerryEsseenResult(StrictModel):
    """Exact source moments and an outward rational enclosure of the bound."""

    source: BerryEsseenRequest
    theorem_variant: Literal[
        "IID_SPECIALIZATION_OF_GENERAL_INDEPENDENT_BERRY_ESSEEN_C_05600"
    ]
    universal_constant: CanonicalRational
    mean: CanonicalRational
    variance: CanonicalRational
    third_absolute_central_moment: CanonicalRational
    bound_squared: CanonicalRational
    bound_lower: CanonicalRational
    bound_upper: CanonicalRational
    bound_precision_bits: int = Field(ge=1, le=256, strict=True)

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: BerryEsseenRequest,
        theorem_variant: Literal[
            "IID_SPECIALIZATION_OF_GENERAL_INDEPENDENT_BERRY_ESSEEN_C_05600"
        ],
        universal_constant: CanonicalRational,
        mean: CanonicalRational,
        variance: CanonicalRational,
        third_absolute_central_moment: CanonicalRational,
        bound_squared: CanonicalRational,
        bound_lower: CanonicalRational,
        bound_upper: CanonicalRational,
        bound_precision_bits: int,
    ) -> Self:
        """Build the result after the operation establishes its invariants.

        The operation has already admitted the source distribution and all
        intermediate and output rational heights.  Trusted construction keeps
        that work from being replayed by the structural result validator;
        caller-authored and deserialized results still use normal validation.
        """

        return cls.model_construct(
            source=source,
            theorem_variant=theorem_variant,
            universal_constant=universal_constant,
            mean=mean,
            variance=variance,
            third_absolute_central_moment=third_absolute_central_moment,
            bound_squared=bound_squared,
            bound_lower=bound_lower,
            bound_upper=bound_upper,
            bound_precision_bits=bound_precision_bits,
        )

    @model_validator(mode="after")
    def require_structural_bound_invariants(self) -> Self:
        """Validate shape and source metadata without replaying moments."""

        if self.theorem_variant != BERRY_ESSEEN_THEOREM_VARIANT:
            raise _validation_error("Berry--Esseen theorem variant is not supported")
        if self.universal_constant.as_fraction() != BERRY_ESSEEN_CONSTANT:
            raise _validation_error(
                "Berry--Esseen universal constant does not match the theorem variant"
            )
        if self.source.sample_count < 1:
            raise _validation_error(
                "Berry--Esseen source sample_count is out of bounds"
            )
        if len(self.source.distribution.atoms) > MAX_BERRY_ESSEEN_ATOMS:
            raise _validation_error("Berry--Esseen source atom count is out of bounds")
        # Re-admit the serialized source law, but do not replay the reported
        # moments or bound. The source must retain the operation's normalized
        # finite-law and input rational-height contract after transport.
        require_input_distribution(
            self.source.distribution.atoms,
            require_canonical=True,
            max_digits=MAX_INPUT_RATIONAL_DIGITS,
        )
        if self.bound_precision_bits != BERRY_ESSEEN_BOUND_BITS:
            raise _validation_error("Berry--Esseen bound precision is not supported")

        for label, value in (
            ("Berry--Esseen universal constant", self.universal_constant),
            ("Berry--Esseen mean", self.mean),
            ("Berry--Esseen variance", self.variance),
            (
                "Berry--Esseen third absolute central moment",
                self.third_absolute_central_moment,
            ),
            ("Berry--Esseen squared bound", self.bound_squared),
            ("Berry--Esseen lower bound", self.bound_lower),
            ("Berry--Esseen upper bound", self.bound_upper),
        ):
            _require_bounded_fraction(
                value.as_fraction(),
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label=label,
            )

        variance = self.variance.as_fraction()
        third = self.third_absolute_central_moment.as_fraction()
        bound_squared = self.bound_squared.as_fraction()
        lower = self.bound_lower.as_fraction()
        upper = self.bound_upper.as_fraction()
        if variance <= 0:
            raise _validation_error("Berry--Esseen variance must be positive")
        if third <= 0:
            raise _validation_error(
                "Berry--Esseen third absolute central moment must be positive"
            )
        if bound_squared <= 0:
            raise _validation_error("Berry--Esseen squared bound must be positive")
        if lower < 0 or upper < lower:
            raise _validation_error(
                "Berry--Esseen outward bound interval must be ordered and nonnegative"
            )
        if not lower * lower <= bound_squared <= upper * upper:
            raise _validation_error(
                "Berry--Esseen outward interval must enclose the squared bound"
            )
        grid_scale = 1 << self.bound_precision_bits
        if (lower * grid_scale).denominator != 1 or (
            upper * grid_scale
        ).denominator != 1:
            raise _validation_error(
                "Berry--Esseen bound endpoints must lie on the "
                "2^-bound_precision_bits dyadic grid"
            )
        if lower == upper:
            return self
        if upper - lower != Fraction(1, grid_scale):
            raise _validation_error(
                "Berry--Esseen non-singleton bound endpoints must be consecutive "
                "points on the 2^-bound_precision_bits grid"
            )
        return self


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
    scale = 1 << BERRY_ESSEEN_BOUND_BITS
    scaled_numerator = value.numerator * scale * scale
    if scaled_numerator % value.denominator == 0:
        exact_scaled = scaled_numerator // value.denominator
        root = isqrt(exact_scaled)
        if root * root == exact_scaled:
            exact = Fraction(root, scale)
            return exact, exact
    scaled_floor = scaled_numerator // value.denominator
    lower_numerator = isqrt(scaled_floor)
    return Fraction(lower_numerator, scale), Fraction(lower_numerator + 1, scale)


def berry_esseen_bound(request: BerryEsseenRequest) -> BerryEsseenResult:
    """Compute the exact i.i.d. Berry--Esseen upper bound."""

    if request.sample_count < 1:
        raise OperationDomainValidationError(
            location=("sample_count",),
            code="probability.berry_esseen.nonpositive_sample_count",
            message="Berry--Esseen sample_count must be positive",
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

    # Form the scale-invariant ratio with cancellation before height checks.
    # Affine rescaling inflates variance^3 and rho^2 equally; admitting those
    # unreduced intermediates would reject cheap standardized bounds.
    bound_squared = _admission_fraction(
        (BERRY_ESSEEN_CONSTANT * BERRY_ESSEEN_CONSTANT * third * third)
        / (variance * variance * variance * Fraction(request.sample_count)),
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

    return BerryEsseenResult._from_kernel(
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
