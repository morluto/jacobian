"""Exact enumeration of rational affine places on admitted hyperelliptic models."""

from __future__ import annotations

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields.values import FiniteFieldPresentation
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    HyperellipticAffinePlace,
)
from jacobian.math.function_fields.operations import (
    _admit_field,
    _canonical_field,
    _hyperelliptic_branch_polynomial,
    _validated_field,
)

MAX_AFFINE_PLACE_ENUMERATION_WORK = 5_000_000
MAX_AFFINE_PLACE_ENUMERATION_OUTPUT = 514
MAX_AFFINE_PLACE_ENUMERATION_RESULT_BYTES = 2_000_000
# The place's non-field JSON fields (coordinates, local parameter, and GF(p)
# residue presentation), including object punctuation, fit within this cap.
_PLACE_RESULT_OVERHEAD_BYTES = 256
_PLACE_RESULT_FIXED_WORK = 64
_FIELD_COEFFICIENT_VALIDATION_WORK = 4


class HyperellipticAffinePlacesRequest(StrictModel):
    """A supported odd-characteristic squarefree model ``y^2=f(x)``."""

    field: FiniteFunctionField


class HyperellipticAffinePlacesResult(StrictModel):
    """All GF(p)-rational affine places, ordered by ``(x,y)``."""

    field: FiniteFunctionField
    places: tuple[HyperellipticAffinePlace, ...] = Field(
        max_length=MAX_AFFINE_PLACE_ENUMERATION_OUTPUT
    )

    @model_validator(mode="after")
    def require_parent_and_canonical_order(self) -> HyperellipticAffinePlacesResult:
        previous: tuple[int, int] | None = None
        for place in self.places:
            if place.field != self.field:
                raise ValueError(
                    "every affine place must retain the result function field"
                )
            coordinate = (place.x, place.y)
            if previous is not None and coordinate <= previous:
                raise ValueError("affine places must be unique and ordered by (x,y)")
            previous = coordinate
        return self


def enumerate_hyperelliptic_affine_places(
    field: FiniteFunctionField,
) -> HyperellipticAffinePlacesResult:
    """Enumerate exactly the rational affine points of an admitted ``y^2=f(x)``.

    This is an affine rational-point enumeration, not a complete enumeration of
    places of the global function field. Points at infinity and places with a
    non-prime residue field require other carriers.
    """

    field = _canonical_field(_validated_field(field))
    if field.characteristic == 2:
        raise OperationDomainValidationError(
            location=("field", "characteristic"),
            code="function_field.affine_enumeration_characteristic",
            message="affine y^2=f(x) enumeration requires odd characteristic",
        )
    _admit_field(field)
    branch = _hyperelliptic_branch_polynomial(field)
    if branch is None:
        raise OperationDomainValidationError(
            location=("field",),
            code="function_field.affine_enumeration_model",
            message="field must be an odd-characteristic squarefree y^2=f(x) model",
        )

    prime = field.characteristic
    # Admission includes the square table and Horner scan, up to two output
    # records per x, structural validation of each repeated field/residue, and
    # the maximum serialized output size. All estimates precede either result
    # collection, so transport serialization cannot exceed the admitted bound.
    output_count_bound = 2 * prime
    field_json_bytes = len(field.model_dump_json().encode("utf-8"))
    result_bytes_bound = (
        field_json_bytes
        + output_count_bound * (field_json_bytes + _PLACE_RESULT_OVERHEAD_BYTES)
        + _PLACE_RESULT_OVERHEAD_BYTES
    )
    field_coefficient_count = sum(
        len(polynomial.coefficients)
        for coefficient in field.defining_polynomial
        for polynomial in (coefficient.numerator, coefficient.denominator)
    )
    work = (
        prime * (len(branch) + 1)
        + output_count_bound
        * (
            _PLACE_RESULT_FIXED_WORK
            + _FIELD_COEFFICIENT_VALIDATION_WORK * field_coefficient_count
        )
        + result_bytes_bound
    )
    if result_bytes_bound > MAX_AFFINE_PLACE_ENUMERATION_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="function_field.affine_enumeration_output_exceeds_envelope",
            message="affine place output exceeds its admitted serialized-byte bound",
        )
    if work > MAX_AFFINE_PLACE_ENUMERATION_WORK:
        raise OperationResourceAdmissionError(
            location=("field",),
            code="function_field.affine_enumeration_work_exceeds_envelope",
            message=(
                "affine place scan, result construction, validation, and output "
                "work exceed the admitted bound"
            ),
        )
    roots: dict[int, list[int]] = {}
    for y in range(prime):
        roots.setdefault(y * y % prime, []).append(y)

    residue = FiniteFieldPresentation(
        characteristic=prime,
        modulus_coefficients=(0, 1),
        generator="z",
    )
    places: list[HyperellipticAffinePlace] = []
    for x in range(prime):
        value = 0
        for coefficient in reversed(branch):
            value = (value * x + coefficient) % prime
        for y in roots.get(value, ()):
            places.append(
                HyperellipticAffinePlace(
                    field=field,
                    x=x,
                    y=y,
                    local_parameter="y" if y == 0 else "x_minus_x0",
                    residue_field=residue,
                )
            )
    # Each x has at most two square roots in an odd prime field, so this bound
    # follows from the admitted characteristic, before the output is built.
    return HyperellipticAffinePlacesResult(field=field, places=tuple(places))


def _run(request: HyperellipticAffinePlacesRequest) -> HyperellipticAffinePlacesResult:
    return enumerate_hyperelliptic_affine_places(request.field)
