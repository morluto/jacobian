"""Exact short-Weierstrass discriminant over one declared finite field.

Only odd characteristic ``p > 3`` is admitted: characteristics 2 and 3 need
generalized Weierstrass models with different formulas and are rejected
structurally before any arithmetic.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields._admission import require_field
from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"elliptic_curve.finite_field.{reason}", message)


class FiniteFieldShortWeierstrassCurve(StrictModel):
    """A short Weierstrass curve ``y^2 = x^3 + A x + B`` over one finite field.

    Coefficients are bound to exactly the declared field presentation; the
    model tag records that short-Weierstrass formulas apply only in odd
    characteristic above 3. A valid curve value additionally requires
    nonzero discriminant, which the discriminant operation reports rather
    than this structural carrier.
    """

    field: FiniteFieldPresentation
    coefficient_a: FiniteFieldElement
    coefficient_b: FiniteFieldElement
    model: Literal["SHORT_WEIERSTRASS_ODD_CHAR_GT_3"] = (
        "SHORT_WEIERSTRASS_ODD_CHAR_GT_3"
    )

    @model_validator(mode="after")
    def require_shared_presentation(self) -> Self:
        if (
            self.coefficient_a.presentation != self.field
            or self.coefficient_b.presentation != self.field
        ):
            raise _validation_error(
                "coefficient_presentation_mismatch",
                "curve coefficients must use the declared field presentation",
            )
        return self


class FiniteFieldDiscriminantRequest(StrictModel):
    """Inspect one short-Weierstrass coefficient pair, singular or not.

    Structural admission (characteristic above 3, prime characteristic,
    irreducible modulus, shared presentation) runs in
    ``finite_field_discriminant`` before any field arithmetic.
    """

    field: FiniteFieldPresentation
    coefficient_a: FiniteFieldElement = Field(
        description="Coefficient A bound to the declared field presentation."
    )
    coefficient_b: FiniteFieldElement = Field(
        description="Coefficient B bound to the declared field presentation."
    )

    @model_validator(mode="after")
    def require_shared_presentation(self) -> Self:
        if (
            self.coefficient_a.presentation != self.field
            or self.coefficient_b.presentation != self.field
        ):
            raise _validation_error(
                "coefficient_presentation_mismatch",
                "discriminant coefficients must use the declared field presentation",
            )
        return self


class FiniteFieldDiscriminantResult(StrictModel):
    """Source-bound discriminant data of one finite-field coefficient pair.

    ``four_a_cubed`` is ``4A^3``, ``twentyseven_b_squared`` is ``27B^2``,
    and ``discriminant`` is ``Delta = -16(4A^3 + 27B^2)``. The ``j``-invariant
    ``1728 * 4A^3 / (4A^3 + 27B^2)`` is present exactly when the pair is
    nonsingular (``Delta != 0``).
    """

    field: FiniteFieldPresentation
    coefficient_a: FiniteFieldElement
    coefficient_b: FiniteFieldElement
    four_a_cubed: FiniteFieldElement
    twentyseven_b_squared: FiniteFieldElement
    discriminant: FiniteFieldElement
    is_nonsingular: bool
    j_invariant: FiniteFieldElement | None = Field(
        default=None,
        description="j-invariant, present exactly for nonsingular pairs.",
    )

    @model_validator(mode="after")
    def require_discriminant_branches(self) -> Self:
        for label in (
            "coefficient_a",
            "coefficient_b",
            "four_a_cubed",
            "twentyseven_b_squared",
            "discriminant",
        ):
            if getattr(self, label).presentation != self.field:
                raise _validation_error(
                    "result_presentation_mismatch",
                    "discriminant data must use the declared field presentation",
                )
        if self.j_invariant is not None and self.j_invariant.presentation != self.field:
            raise _validation_error(
                "result_presentation_mismatch",
                "discriminant data must use the declared field presentation",
            )
        if self.is_nonsingular == self.discriminant.is_zero:
            raise _validation_error(
                "singularity_mismatch",
                "is_nonsingular must be the negation of discriminant vanishing",
            )
        if (self.j_invariant is None) == self.is_nonsingular:
            raise _validation_error(
                "j_invariant_branch_mismatch",
                "j is present exactly for nonsingular coefficient pairs",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: FiniteFieldDiscriminantRequest,
        *,
        four_a_cubed: FiniteFieldElement,
        twentyseven_b_squared: FiniteFieldElement,
        discriminant: FiniteFieldElement,
        is_nonsingular: bool,
        j_invariant: FiniteFieldElement | None,
    ) -> Self:
        """Build one result after the admitted kernel established its values."""

        return cls.model_construct(
            field=request.field,
            coefficient_a=request.coefficient_a,
            coefficient_b=request.coefficient_b,
            four_a_cubed=four_a_cubed,
            twentyseven_b_squared=twentyseven_b_squared,
            discriminant=discriminant,
            is_nonsingular=is_nonsingular,
            j_invariant=j_invariant,
        )


def _coordinates(element: FiniteFieldElement) -> tuple[int, ...]:
    return tuple(element.coordinates)


def _element(
    field: FiniteFieldPresentation, coordinates: tuple[int, ...]
) -> FiniteFieldElement:
    return FiniteFieldElement(presentation=field, coordinates=coordinates)


def _add(
    modulus: int, left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple((a + b) % modulus for a, b in zip(left, right, strict=True))


def _multiply(
    presentation: FiniteFieldPresentation,
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    """Multiply power-basis coordinates modulo the presentation modulus."""

    modulus = presentation.characteristic
    degree = presentation.degree
    modulus_coefficients = presentation.modulus_coefficients
    raw = [0] * (2 * degree - 1)
    for i, a in enumerate(left):
        if a:
            for j, b in enumerate(right):
                raw[i + j] = (raw[i + j] + a * b) % modulus
    for position in range(2 * degree - 2, degree - 1, -1):
        factor = raw[position] % modulus
        if factor:
            for offset in range(degree + 1):
                raw[position - degree + offset] = (
                    raw[position - degree + offset]
                    - factor * modulus_coefficients[offset]
                ) % modulus
    return tuple(raw[:degree])


def _power(
    presentation: FiniteFieldPresentation,
    value: tuple[int, ...],
    exponent: int,
) -> tuple[int, ...]:
    result = (1,) + (0,) * (presentation.degree - 1)
    base = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _multiply(presentation, result, base)
        base = _multiply(presentation, base, base)
        remaining >>= 1
    return result


def _inverse(
    presentation: FiniteFieldPresentation, value: tuple[int, ...]
) -> tuple[int, ...]:
    """Invert a nonzero element by the extended Euclidean algorithm over Fp."""

    modulus = presentation.characteristic
    modulus_poly = list(presentation.modulus_coefficients)

    def degree(poly: list[int]) -> int:
        index = len(poly) - 1
        while index > 0 and poly[index] % modulus == 0:
            index -= 1
        return index

    def normalize(poly: list[int]) -> list[int]:
        trimmed = [c % modulus for c in poly[: degree(poly) + 1]]
        while len(trimmed) > 1 and trimmed[-1] == 0:
            trimmed.pop()
        return trimmed

    remainder, current = normalize(modulus_poly), normalize(list(value))
    old, new = [0], [1]
    while not (len(current) == 1 and current[0] % modulus == 0):
        quotient, leftover = _poly_divmod(remainder, current, modulus)
        remainder, current = current, leftover
        old, new = new, _poly_sub(old, _poly_mul(quotient, new, modulus), modulus)
    if len(remainder) != 1 or remainder[0] % modulus == 0:
        raise ArithmeticError("element is not invertible")
    scale = pow(remainder[0] % modulus, -1, modulus)
    inverse = [(c * scale) % modulus for c in old]
    return tuple(
        inverse[i] if i < len(inverse) else 0 for i in range(presentation.degree)
    )


def _poly_mul(left: list[int], right: list[int], modulus: int) -> list[int]:
    if not left or not right:
        return [0]
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] = (result[i + j] + a * b) % modulus
    return result


def _poly_sub(left: list[int], right: list[int], modulus: int) -> list[int]:
    width = max(len(left), len(right))
    return [
        ((left[i] if i < len(left) else 0) - (right[i] if i < len(right) else 0))
        % modulus
        for i in range(width)
    ]


def _poly_divmod(
    dividend: list[int], divisor: list[int], modulus: int
) -> tuple[list[int], list[int]]:
    remainder = [c % modulus for c in dividend]
    divisor = [c % modulus for c in divisor]
    while len(divisor) > 1 and divisor[-1] == 0:
        divisor.pop()
    divisor_degree = len(divisor) - 1
    leading_inverse = pow(divisor[-1], -1, modulus)
    quotient = [0] * max(len(remainder) - divisor_degree, 0)
    while len(remainder) - 1 >= divisor_degree and any(c % modulus for c in remainder):
        shift = len(remainder) - 1 - divisor_degree
        factor = remainder[-1] * leading_inverse % modulus
        if shift < len(quotient):
            quotient[shift] = (quotient[shift] + factor) % modulus
        for i, c in enumerate(divisor):
            remainder[shift + i] = (remainder[shift + i] - factor * c) % modulus
        while len(remainder) > 1 and remainder[-1] == 0:
            remainder.pop()
    return quotient, remainder


def _scale(modulus: int, scalar: int, value: tuple[int, ...]) -> tuple[int, ...]:
    return tuple((scalar * a) % modulus for a in value)


def require_discriminant_admission(
    field: FiniteFieldPresentation,
    coefficient_a: FiniteFieldElement,
    coefficient_b: FiniteFieldElement,
) -> None:
    """Admit the finite-field discriminant domain once per call."""

    if not isinstance(field, FiniteFieldPresentation):
        raise OperationDomainValidationError(
            location=("field",),
            code="elliptic_curve.finite_field.field_type",
            message="field must be a finite-field presentation value",
        )
    for label, coefficient in (
        ("coefficient_a", coefficient_a),
        ("coefficient_b", coefficient_b),
    ):
        if not isinstance(coefficient, FiniteFieldElement):
            raise OperationDomainValidationError(
                location=(label,),
                code="elliptic_curve.finite_field.coefficient_type",
                message="curve coefficients must be finite-field element values",
            )
        if coefficient.presentation != field:
            raise OperationDomainValidationError(
                location=(label,),
                code="elliptic_curve.finite_field.coefficient_presentation_mismatch",
                message="curve coefficients must use the declared field presentation",
            )
    if field.characteristic <= 3:
        raise OperationDomainValidationError(
            location=("field", "characteristic"),
            code="elliptic_curve.finite_field.characteristic_above_three",
            message=(
                "short Weierstrass discriminant requires odd characteristic "
                "above 3; characteristics 2 and 3 need generalized models"
            ),
        )
    # Establish the caller-authored field claim (prime characteristic and
    # irreducible modulus) before any quotient-ring arithmetic relies on it.
    require_field(field)


def finite_field_discriminant(
    field: FiniteFieldPresentation,
    coefficient_a: FiniteFieldElement,
    coefficient_b: FiniteFieldElement,
) -> FiniteFieldDiscriminantResult:
    """Compute 4A^3, 27B^2, Delta, singularity, and j over one finite field."""

    require_discriminant_admission(field, coefficient_a, coefficient_b)
    modulus = field.characteristic
    a = _coordinates(coefficient_a)
    b = _coordinates(coefficient_b)
    four_a_cubed = _scale(modulus, 4, _power(field, a, 3))
    twentyseven_b_squared = _scale(modulus, 27, _multiply(field, b, b))
    total = _add(modulus, four_a_cubed, twentyseven_b_squared)
    discriminant = _scale(modulus, modulus - 16, total)
    is_nonsingular = any(discriminant)
    j_invariant: FiniteFieldElement | None = None
    if is_nonsingular:
        ratio = _multiply(
            field, _scale(modulus, 1728, four_a_cubed), _inverse(field, total)
        )
        j_invariant = _element(field, ratio)
    return FiniteFieldDiscriminantResult._from_kernel(
        FiniteFieldDiscriminantRequest(
            field=field, coefficient_a=coefficient_a, coefficient_b=coefficient_b
        ),
        four_a_cubed=_element(field, four_a_cubed),
        twentyseven_b_squared=_element(field, twentyseven_b_squared),
        discriminant=_element(field, discriminant),
        is_nonsingular=is_nonsingular,
        j_invariant=j_invariant,
    )


__all__ = [
    "FiniteFieldDiscriminantRequest",
    "FiniteFieldDiscriminantResult",
    "FiniteFieldShortWeierstrassCurve",
    "finite_field_discriminant",
    "require_discriminant_admission",
]
