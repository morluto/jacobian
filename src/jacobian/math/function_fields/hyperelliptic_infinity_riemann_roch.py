"""Exact Riemann-Roch spaces for multiples of odd-degree infinity places."""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.function_fields._models import (
    MAX_POLYNOMIAL_X_DEGREE,
    MAX_RIEMANN_ROCH_BASIS_DIMENSION,
    FiniteFunctionField,
    FiniteFunctionFieldElement,
    HyperellipticInfinityPlace,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields.operations import (
    _admit_field_resources,
    _canonical_field,
    _hyperelliptic_branch_polynomial,
    _validated_field,
)

MAX_HYPERELLIPTIC_INFINITY_RR_OUTPUT_BYTES = 2_000_000
MAX_HYPERELLIPTIC_INFINITY_RR_WORK = 20_000_000
_BASIS_ENTRY_OVERHEAD_BYTES = 4_096


class HyperellipticInfinityRiemannRochRequest(StrictModel):
    """Request ``L(m P_infinity)`` for one supported hyperelliptic field."""

    place: HyperellipticInfinityPlace
    multiplicity: int = Field(description="The signed coefficient m in m*P_infinity.")


class HyperellipticInfinityRiemannRochSpace(StrictModel):
    """A complete exact basis for ``L(m P_infinity)``."""

    field: FiniteFunctionField
    place: HyperellipticInfinityPlace
    multiplicity: int
    genus: int = Field(ge=1)
    dimension: int = Field(ge=0, le=MAX_RIEMANN_ROCH_BASIS_DIMENSION)
    basis: tuple[FiniteFunctionFieldElement, ...] = Field(
        max_length=MAX_RIEMANN_ROCH_BASIS_DIMENSION
    )

    @model_validator(mode="after")
    def require_parent_and_dimension(self) -> HyperellipticInfinityRiemannRochSpace:
        if self.place.field != self.field:
            raise PydanticCustomError(
                "function_field.hyperelliptic_rr_parent",
                "the infinity place must belong to the result field",
            )
        if self.dimension != len(self.basis) or any(
            element.field != self.field for element in self.basis
        ):
            raise PydanticCustomError(
                "function_field.hyperelliptic_rr_basis",
                "basis length and parents must match the declared space",
            )
        return self


def _dimension(multiplicity: int, branch_degree: int) -> int:
    if multiplicity < 0:
        return 0
    x_count = multiplicity // 2 + 1
    y_count = max(0, (multiplicity - branch_degree) // 2 + 1)
    return x_count + y_count


def _admit(
    request: HyperellipticInfinityRiemannRochRequest,
) -> tuple[
    FiniteFunctionField,
    HyperellipticInfinityPlace,
    tuple[int, ...],
    int,
    int,
]:
    try:
        request = HyperellipticInfinityRiemannRochRequest.model_validate(
            request.model_dump()
        )
        place = HyperellipticInfinityPlace.model_validate(request.place.model_dump())
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("place",),
            code="function_field.hyperelliptic_rr_invalid_request",
            message="request must contain a canonical typed infinity place",
        ) from exc
    if request.multiplicity.bit_length() > 4096:
        raise OperationResourceAdmissionError(
            location=("multiplicity",),
            code="function_field.hyperelliptic_rr_multiplicity_exceeds_envelope",
            message="the infinity coefficient exceeds the 4096-bit input envelope",
        )
    field = _validated_field(place.field)
    _admit_field_resources(field)
    field = _canonical_field(field)
    branch = _hyperelliptic_branch_polynomial(field)
    if branch is None or (len(branch) - 1) % 2 == 0:
        raise OperationDomainValidationError(
            location=("place", "field"),
            code="function_field.hyperelliptic_rr_requires_odd_model",
            message=(
                "the space requires a squarefree odd-characteristic model "
                "y^2=f(x) with odd degree 3 through the admitted bound"
            ),
        )
    degree = len(branch) - 1
    genus = (degree - 1) // 2
    dimension = _dimension(request.multiplicity, degree)
    if dimension > MAX_RIEMANN_ROCH_BASIS_DIMENSION:
        raise OperationResourceAdmissionError(
            location=("multiplicity",),
            code="function_field.hyperelliptic_rr_basis_exceeds_envelope",
            message=(
                "the complete Riemann-Roch basis exceeds the admitted "
                f"dimension {MAX_RIEMANN_ROCH_BASIS_DIMENSION}"
            ),
        )
    maximum_exponent = max(
        0,
        request.multiplicity // 2,
        (request.multiplicity - degree) // 2,
    )
    if maximum_exponent > MAX_POLYNOMIAL_X_DEGREE:
        raise OperationResourceAdmissionError(
            location=("multiplicity",),
            code="function_field.hyperelliptic_rr_coefficient_degree_exceeds_envelope",
            message=(
                "basis coefficients exceed the admitted degree-"
                f"{MAX_POLYNOMIAL_X_DEGREE} polynomial bound"
            ),
        )
    field_bytes = len(field.model_dump_json().encode("utf-8"))
    output_bytes = (
        2 * field_bytes
        + dimension * (field_bytes + _BASIS_ENTRY_OVERHEAD_BYTES)
        + 1_024
    )
    if output_bytes > MAX_HYPERELLIPTIC_INFINITY_RR_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="function_field.hyperelliptic_rr_output_exceeds_envelope",
            message="the exact basis exceeds its admitted serialized-byte bound",
        )
    work = dimension * (field_bytes + maximum_exponent + 1)
    if work > MAX_HYPERELLIPTIC_INFINITY_RR_WORK:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="function_field.hyperelliptic_rr_work_exceeds_envelope",
            message="basis construction and field-bound result validation exceed the admitted work bound",
        )
    # Since f is squarefree of odd degree over odd characteristic, it has a
    # simple root over the algebraic closure. Thus f is not a square in
    # GF(p)(x), so y^2-f is irreducible and separable without a factor search.
    canonical_place = HyperellipticInfinityPlace(
        field=field, residue_field=place.residue_field
    )
    return field, canonical_place, branch, genus, dimension


def hyperelliptic_infinity_riemann_roch_space(
    place: HyperellipticInfinityPlace, multiplicity: int
) -> HyperellipticInfinityRiemannRochSpace:
    if not isinstance(place, HyperellipticInfinityPlace):
        raise OperationDomainValidationError(
            location=("place",),
            code="function_field.hyperelliptic_rr_place_type",
            message="place must be a typed odd-degree hyperelliptic infinity place",
        )
    try:
        request = HyperellipticInfinityRiemannRochRequest.model_validate(
            {"place": place.model_dump(), "multiplicity": multiplicity}
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="function_field.hyperelliptic_rr_invalid_request",
            message="request must contain a valid place and strict integer multiplicity",
        ) from exc
    field, place, branch, genus, dimension = _admit(request)
    degree = len(branch) - 1
    prime = field.characteristic
    zero_poly = PrimeFieldPolynomial(characteristic=prime, coefficients=(0,))
    one_poly = PrimeFieldPolynomial(characteristic=prime, coefficients=(1,))
    zero = PrimeFieldRationalFunction(numerator=zero_poly, denominator=one_poly)

    def monomial(exponent: int) -> PrimeFieldRationalFunction:
        coefficients = (0,) * exponent + (1,)
        return PrimeFieldRationalFunction(
            numerator=PrimeFieldPolynomial(
                characteristic=prime, coefficients=coefficients
            ),
            denominator=one_poly,
        )

    basis: list[FiniteFunctionFieldElement] = []
    if multiplicity >= 0:
        for exponent in range(multiplicity // 2 + 1):
            basis.append(
                FiniteFunctionFieldElement(
                    field=field,
                    coordinates=(monomial(exponent), zero),
                )
            )
        for exponent in range(max(0, (multiplicity - degree) // 2 + 1)):
            basis.append(
                FiniteFunctionFieldElement(
                    field=field,
                    coordinates=(zero, monomial(exponent)),
                )
            )
    if len(basis) != dimension:
        raise ArithmeticError(
            "hyperelliptic Riemann-Roch basis count changed after admission"
        )
    return HyperellipticInfinityRiemannRochSpace(
        field=field,
        place=place,
        multiplicity=multiplicity,
        genus=genus,
        dimension=dimension,
        basis=tuple(basis),
    )


_GF5_ELLIPTIC = {
    "characteristic": 5,
    "variable": "x",
    "generator": "y",
    "defining_polynomial": [
        {
            "numerator": {"characteristic": 5, "coefficients": [0, 4, 0, 1]},
            "denominator": {"characteristic": 5, "coefficients": [1]},
        },
        {
            "numerator": {"characteristic": 5, "coefficients": [0]},
            "denominator": {"characteristic": 5, "coefficients": [1]},
        },
        {
            "numerator": {"characteristic": 5, "coefficients": [1]},
            "denominator": {"characteristic": 5, "coefficients": [1]},
        },
    ],
}


HYPERELLIPTIC_INFINITY_RIEMANN_ROCH_TOOL = MathTool(
    operation_id="function_field.hyperelliptic_infinity_riemann_roch_space.compute",
    title="Compute a hyperelliptic Riemann-Roch space at infinity",
    description=(
        "Return the complete exact basis and dimension of L(m P_infinity) "
        "for an odd-degree squarefree model y^2=f(x) over GF(p). The exact "
        "infinity place must retain the same field. The basis dimension and "
        "polynomial degrees are bounded before construction."
    ),
    request_type=HyperellipticInfinityRiemannRochRequest,
    result_type=HyperellipticInfinityRiemannRochSpace,
    run=lambda request: hyperelliptic_infinity_riemann_roch_space(
        request.place, request.multiplicity
    ),
    tags=("function-field", "hyperelliptic", "riemann-roch", "basis", "exact"),
    discovery_terms=(
        "Riemann-Roch space of a multiple of the hyperelliptic infinity place",
        "basis of L(m infinity) for y squared equals f(x)",
    ),
    examples=(
        OperationExample(
            name="elliptic_cubic_three_infinity",
            description="For y^2=x^3-x over GF(5), L(3 infinity) has basis 1,x,y.",
            input={
                "place": {
                    "field": _GF5_ELLIPTIC,
                    "residue_field": {
                        "characteristic": "5",
                        "modulus_coefficients": ["0", "1"],
                        "generator": "z",
                    },
                },
                "multiplicity": 3,
            },
        ),
    ),
)


__all__ = [
    "HYPERELLIPTIC_INFINITY_RIEMANN_ROCH_TOOL",
    "HyperellipticInfinityRiemannRochRequest",
    "HyperellipticInfinityRiemannRochSpace",
    "hyperelliptic_infinity_riemann_roch_space",
]
