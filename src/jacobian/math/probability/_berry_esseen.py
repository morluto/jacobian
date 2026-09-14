"""Exact bounded i.i.d. Berry--Esseen bounds for finite rational laws.

The theorem variant is Shevtsova's general-independent Berry--Esseen
constant ``C = 0.5600 = 14/25``, specialized to repeated i.i.d. summands.
The operation deliberately retains that explicit, conservative constant
rather than claiming the sharper i.i.d.-specific constants from later work.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import Annotated, Any, Literal, Self

from pydantic import Field, ValidationError, model_validator
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    CanonicalRational,
    DecimalIntegerEncoding,
    require_bounded_rational,
)
from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.probability._distribution import (
    FiniteDistributionAtom,
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

    @classmethod
    def model_json_schema(
        cls,
        by_alias: bool = True,
        ref_template: str = "#/$defs/{model}",
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = "validation",
        *,
        union_format: Literal["any_of", "primitive_type_array"] = "any_of",
    ) -> dict[str, Any]:
        schema = super().model_json_schema(
            by_alias,
            ref_template,
            schema_generator,
            mode,
            union_format=union_format,
        )
        defs = schema.get("$defs")
        properties = schema.get("properties")
        if not isinstance(defs, dict) or not isinstance(properties, dict):
            return schema
        shared = defs.get("FiniteRationalDistribution")
        if not isinstance(shared, dict):
            return schema
        cloned = dict(shared)
        cloned_properties = dict(cloned.get("properties", {}))
        atoms = cloned_properties.get("atoms")
        if isinstance(atoms, dict):
            cloned_atoms = dict(atoms)
            cloned_atoms["maxItems"] = MAX_BERRY_ESSEEN_ATOMS
            cloned_properties["atoms"] = cloned_atoms
            cloned["properties"] = cloned_properties
        defs = dict(defs)
        defs["BerryEsseenFiniteDistribution"] = cloned
        schema["$defs"] = defs
        properties = dict(properties)
        properties["distribution"] = {"$ref": "#/$defs/BerryEsseenFiniteDistribution"}
        schema["properties"] = properties
        return schema


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
    def model_json_schema(
        cls,
        by_alias: bool = True,
        ref_template: str = "#/$defs/{model}",
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = "validation",
        *,
        union_format: Literal["any_of", "primitive_type_array"] = "any_of",
    ) -> dict[str, Any]:
        schema = super().model_json_schema(
            by_alias,
            ref_template,
            schema_generator,
            mode,
            union_format=union_format,
        )
        defs = schema.get("$defs")
        if not isinstance(defs, dict):
            return schema
        shared = defs.get("FiniteRationalDistribution")
        if not isinstance(shared, dict):
            return schema
        cloned = dict(shared)
        cloned_properties = dict(cloned.get("properties", {}))
        atoms = cloned_properties.get("atoms")
        if isinstance(atoms, dict):
            cloned_atoms = dict(atoms)
            cloned_atoms["maxItems"] = MAX_BERRY_ESSEEN_ATOMS
            cloned_properties["atoms"] = cloned_atoms
            cloned["properties"] = cloned_properties
        defs = dict(defs)
        defs["BerryEsseenFiniteDistribution"] = cloned
        request_schema = defs.get("BerryEsseenRequest")
        if isinstance(request_schema, dict):
            request_schema = dict(request_schema)
            request_properties = dict(request_schema.get("properties", {}))
            request_properties["distribution"] = {
                "$ref": "#/$defs/BerryEsseenFiniteDistribution"
            }
            request_schema["properties"] = request_properties
            defs["BerryEsseenRequest"] = request_schema
        schema["$defs"] = defs
        properties = schema.get("properties")
        if isinstance(properties, dict):
            properties = dict(properties)
            source = properties.get("source")
            if isinstance(source, dict) and "$ref" not in source:
                source_properties = dict(source.get("properties", {}))
                source_properties["distribution"] = {
                    "$ref": "#/$defs/BerryEsseenFiniteDistribution"
                }
                source = dict(source)
                source["properties"] = source_properties
                properties["source"] = source
            schema["properties"] = properties
        return schema

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
        _require_source_sample_count(self.source.sample_count)
        if len(self.source.distribution.atoms) > MAX_BERRY_ESSEEN_ATOMS:
            raise _validation_error("Berry--Esseen source atom count is out of bounds")
        # Restore canonical structure only. Exact probability normalization and
        # the aggregate input-height admission are operation-boundary work, not
        # deserialization work.
        _require_structural_source_atoms(self.source.distribution.atoms)
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


def _require_structural_source_atoms(
    atoms: tuple[FiniteDistributionAtom, ...],
) -> None:
    """Revalidate each source atom's canonical carrier without normalization."""

    for atom in atoms:
        try:
            FiniteDistributionAtom.model_validate(
                {
                    "value": {"num": atom.value.num, "den": atom.value.den},
                    "probability": {
                        "num": atom.probability.num,
                        "den": atom.probability.den,
                    },
                }
            )
        except (ValidationError, PydanticCustomError) as exc:
            raise _validation_error(
                "Berry--Esseen source atoms must be canonical nonnegative masses"
            ) from exc


def _require_source_sample_count(sample_count: object) -> None:
    """Validate the source sample-count contract on a deserialized result."""

    if type(sample_count) is not int:
        raise _validation_error("Berry--Esseen source sample_count must be an integer")
    if sample_count < 1:
        raise _validation_error("Berry--Esseen source sample_count is out of bounds")
    if sample_count > MAX_BERRY_ESSEEN_SAMPLE_COUNT:
        raise _validation_error(
            "Berry--Esseen source sample_count exceeds the representable height"
        )


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


def _require_native_berry_request(request: BerryEsseenRequest) -> None:
    if not isinstance(request, BerryEsseenRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="probability.berry_esseen.request_type",
            message="Berry--Esseen requires a BerryEsseenRequest value",
        )
    if type(request.sample_count) is not int:
        raise OperationDomainValidationError(
            location=("sample_count",),
            code="probability.berry_esseen.sample_count_type",
            message="Berry--Esseen sample_count must be an integer",
        )
    if request.sample_count < 1:
        raise OperationDomainValidationError(
            location=("sample_count",),
            code="probability.berry_esseen.nonpositive_sample_count",
            message="Berry--Esseen sample_count must be positive",
        )
    if request.sample_count > MAX_BERRY_ESSEEN_SAMPLE_COUNT:
        raise OperationDomainValidationError(
            location=("sample_count",),
            code="probability.berry_esseen.sample_count_digits",
            message=(
                "Berry--Esseen sample_count must have at most "
                f"{MAX_RESULT_RATIONAL_DIGITS} decimal digits"
            ),
        )
    if not isinstance(request.distribution, FiniteRationalDistribution):
        raise OperationDomainValidationError(
            location=("distribution",),
            code="probability.berry_esseen.distribution_type",
            message="Berry--Esseen distribution must be a finite rational law",
        )
    atoms = getattr(request.distribution, "atoms", None)
    # Split the container-shape test from the element-type scan and put the
    # length cap between them: a forged distribution may hold an arbitrarily
    # large tuple of genuine atoms, and an element-wise `any(...)` ahead of
    # `len(atoms)` would traverse all of it before issuing the resource refusal
    # that is supposed to bound it.
    if not isinstance(atoms, tuple):
        raise OperationDomainValidationError(
            location=("distribution", "atoms"),
            code="probability.berry_esseen.distribution_type",
            message="Berry--Esseen distribution must be a finite rational law",
        )
    if len(atoms) > MAX_BERRY_ESSEEN_ATOMS:
        raise OperationResourceAdmissionError(
            location=("distribution", "atoms"),
            code="probability.berry_esseen.atom_work_bound",
            message=(
                "Berry--Esseen admission allows at most "
                f"{MAX_BERRY_ESSEEN_ATOMS} input atoms"
            ),
        )
    if any(not isinstance(atom, FiniteDistributionAtom) for atom in atoms):
        raise OperationDomainValidationError(
            location=("distribution", "atoms"),
            code="probability.berry_esseen.distribution_type",
            message="Berry--Esseen distribution must be a finite rational law",
        )
    # A constructed atom is returned unchanged by the default Pydantic
    # configuration, so validate a fresh payload with raw components to rerun
    # the nested canonical-rational and nonnegative-probability contract.
    for index, atom in enumerate(atoms):
        if index % 256 == 0:
            request_checkpoint("during Berry--Esseen atom revalidation")
        try:
            value = atom.value
            probability = atom.probability
            payload = {
                "value": {"num": value.num, "den": value.den},
                "probability": {"num": probability.num, "den": probability.den},
            }
        except AttributeError as exc:
            # A forged atom can carry a wrongly typed nested carrier, such as
            # `value="bad"`. That is a malformed input contract, not an
            # implementation leak: report the stable domain diagnostic.
            raise OperationDomainValidationError(
                location=("distribution", "atoms", index, "value"),
                code="probability.berry_esseen.atom_contract",
                message="Berry--Esseen atoms must be canonical nonnegative masses",
            ) from exc
        try:
            FiniteDistributionAtom.model_validate(payload)
        except (ValidationError, PydanticCustomError, TypeError) as exc:
            raise OperationDomainValidationError(
                location=("distribution", "atoms", index),
                code="probability.berry_esseen.atom_contract",
                message="Berry--Esseen atoms must be canonical nonnegative masses",
            ) from exc


def berry_esseen_bound(request: BerryEsseenRequest) -> BerryEsseenResult:
    """Compute the exact i.i.d. Berry--Esseen upper bound."""

    _require_native_berry_request(request)

    location = ("distribution",)
    try:
        for index, atom in enumerate(request.distribution.atoms):
            if index % 256 == 0:
                request_checkpoint("during Berry--Esseen input-height admission")
            require_bounded_rational(
                atom.value,
                max_digits=MAX_INPUT_RATIONAL_DIGITS,
                label="finite-distribution input atom",
            )
            require_bounded_rational(
                atom.probability,
                max_digits=MAX_INPUT_RATIONAL_DIGITS,
                label="finite-distribution input probability",
            )
    except ValueError as exc:
        raise OperationResourceAdmissionError(
            location=location,
            code="probability.berry_esseen.input_height",
            message=str(exc),
        ) from exc
    try:
        require_input_distribution(
            request.distribution.atoms,
            require_canonical=True,
            max_digits=None,
        )
    except ValueError as exc:
        message = str(exc)
        if "exceeds" in message:
            raise OperationResourceAdmissionError(
                location=location,
                code="probability.berry_esseen.normalization_height",
                message=message,
            ) from exc
        raise OperationDomainValidationError(
            location=location,
            code="probability.berry_esseen.input_distribution",
            message=message,
        ) from exc

    mean = Fraction()
    for index, atom in enumerate(request.distribution.atoms):
        if index % 256 == 0:
            request_checkpoint("during Berry--Esseen mean scan")
        if atom.probability.as_fraction() == 0:
            # A zero-mass atom shifts this law by nothing.
            continue
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
    for index, atom in enumerate(request.distribution.atoms):
        if index % 256 == 0:
            request_checkpoint("during Berry--Esseen moment scan")
        weight = atom.probability.as_fraction()
        if weight == 0:
            # Skip before any centered power is formed. An atom of zero mass
            # contributes nothing to the mean, the variance, or the third
            # absolute moment, but centering it still divides by that atom's
            # denominator: on `x^0 + (1/q2)x^0 + (1/q1)x^1`-shaped laws with
            # coprime 128-digit `q1`, `q2` the zero-mass point has a 255-digit
            # centered denominator and a 763-digit cube, so an irrelevant
            # support point could reject a law whose own moments all fit the
            # 512-digit envelope.
            continue
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
                weight,
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
                weight,
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
