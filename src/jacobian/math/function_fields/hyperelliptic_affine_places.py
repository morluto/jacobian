"""Exact enumeration of rational affine places on admitted hyperelliptic models."""

from __future__ import annotations

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields.values import FiniteFieldPresentation
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    HyperellipticAffinePlace,
)
from jacobian.math.function_fields.operations import (
    _admit_field,
    _hyperelliptic_branch_polynomial,
    _validated_field,
)

MAX_AFFINE_PLACE_ENUMERATION_WORK = 100_000
MAX_AFFINE_PLACE_ENUMERATION_OUTPUT = 514


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
        coordinates = tuple((place.x, place.y) for place in self.places)
        if any(place.field != self.field for place in self.places):
            raise ValueError("every affine place must retain the result function field")
        if coordinates != tuple(sorted(set(coordinates))):
            raise ValueError("affine places must be unique and ordered by (x,y)")
        return self


def enumerate_hyperelliptic_affine_places(
    field: FiniteFunctionField,
) -> HyperellipticAffinePlacesResult:
    """Enumerate exactly the rational affine points of an admitted ``y^2=f(x)``.

    This is an affine rational-point enumeration, not a complete enumeration of
    places of the global function field. Points at infinity and places with a
    non-prime residue field require other carriers.
    """

    field = _validated_field(field)
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
    # Build the square table once and evaluate f at each x by Horner. The
    # count bounds those exact loops before either result collection is built.
    work = prime * (len(branch) + 1)
    if work > MAX_AFFINE_PLACE_ENUMERATION_WORK:
        raise OperationResourceAdmissionError(
            location=("field",),
            code="function_field.affine_enumeration_work_exceeds_envelope",
            message="affine place enumeration exceeds its admitted work bound",
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


HYPERELLIPTIC_AFFINE_PLACES_TOOL = MathTool(
    operation_id="function_field.hyperelliptic_affine_places.enumerate",
    title="Enumerate rational affine hyperelliptic places",
    description=(
        "Enumerate all GF(p)-rational affine points on an admitted odd-"
        "characteristic squarefree model y^2=f(x), returning the existing "
        "source-bound affine place values in lexicographic (x,y) order. This "
        "does not enumerate points at infinity or places with larger residue fields."
    ),
    request_type=HyperellipticAffinePlacesRequest,
    result_type=HyperellipticAffinePlacesResult,
    run=_run,
    tags=("function-field", "hyperelliptic", "affine-place", "enumeration", "exact"),
    examples=(
        OperationExample(
            name="elliptic_model_over_gf5",
            description="Enumerate the rational affine places on y^2=x^3-x over GF(5).",
            input={
                "field": {
                    "characteristic": 5,
                    "variable": "x",
                    "generator": "y",
                    "defining_polynomial": [
                        {
                            "numerator": {
                                "characteristic": 5,
                                "coefficients": [0, 1, 0, 4],
                            },
                            "denominator": {
                                "characteristic": 5,
                                "coefficients": [1],
                            },
                        },
                        {
                            "numerator": {"characteristic": 5, "coefficients": [0]},
                            "denominator": {
                                "characteristic": 5,
                                "coefficients": [1],
                            },
                        },
                        {
                            "numerator": {"characteristic": 5, "coefficients": [1]},
                            "denominator": {
                                "characteristic": 5,
                                "coefficients": [1],
                            },
                        },
                    ],
                }
            },
        ),
    ),
)
