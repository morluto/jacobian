"""Public ring-of-integers value and kernel for presented simple number fields.

The kernel reuses the owner-local SymPy ``round_two`` adapter.  SymPy returns
an integral basis in the monic norm basis ``ZZ[beta]`` with ``beta = L*alpha``
for the presented leading coefficient ``L``; the kernel pulls each basis
element back to rational coordinates on the presentation's own ascending power
basis so the returned value composes with the existing field-element carrier.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.number_fields._integral_basis import (
    recognized_integral_basis,
)
from jacobian.math.number_theory.number_fields.values import (
    MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
    NumberFieldDiscriminantInteger,
    SimpleNumberFieldPresentation,
)

MAX_INTEGRAL_BASIS_DEGREE = 31


class NumberFieldRingOfIntegersResult(StrictModel):
    """Deterministic integral-basis witness for ``O_K`` on the power basis.

    Integral bases are not mathematical singletons.  The operation's pinned
    SymPy kernel chooses one deterministic basis, while this value records the
    exact witness and the field it belongs to.
    """

    field: SimpleNumberFieldPresentation
    basis: tuple[tuple[CanonicalRational, ...], ...]
    field_discriminant: NumberFieldDiscriminantInteger

    @model_validator(mode="after")
    def validate_basis_shape(self) -> NumberFieldRingOfIntegersResult:
        """Keep malformed authored results from crossing the value boundary."""

        degree = self.field.degree
        if len(self.basis) != degree:
            raise PydanticCustomError(
                "number_field.ring_of_integers_basis_length",
                "an integral basis needs exactly one vector per field degree",
            )
        if self.basis[0] != _one_coordinates(degree):
            raise PydanticCustomError(
                "number_field.ring_of_integers_unit_basis",
                "every integral basis contains the unit as its first vector",
            )
        if any(len(vector) != degree for vector in self.basis):
            raise PydanticCustomError(
                "number_field.ring_of_integers_vector_length",
                "every integral basis vector spans the complete power basis",
            )
        return self

    def require_canonical_basis(self) -> None:
        if len(self.basis) != self.field.degree:
            raise OperationDomainValidationError(
                location=("basis",),
                code="number_field.ring_of_integers_basis_length",
                message="an integral basis needs exactly one vector per field degree",
            )
        if self.basis[0] != _one_coordinates(self.field.degree):
            raise OperationDomainValidationError(
                location=("basis",),
                code="number_field.ring_of_integers_unit_basis",
                message="every integral basis contains the unit as its first vector",
            )
        for vector in self.basis:
            if len(vector) != self.field.degree:
                raise OperationDomainValidationError(
                    location=("basis",),
                    code="number_field.ring_of_integers_vector_length",
                    message="every integral basis vector spans the complete power basis",
                )


def _one_coordinates(degree: int) -> tuple[CanonicalRational, ...]:
    return (
        CanonicalRational(num=1, den=1),
        *(CanonicalRational(num=0, den=1) for _ in range(degree - 1)),
    )


def _rational(value: Any) -> CanonicalRational:
    fraction = Fraction(value)
    return CanonicalRational(num=fraction.numerator, den=fraction.denominator)


def ring_of_integers(
    field: SimpleNumberFieldPresentation,
) -> NumberFieldRingOfIntegersResult:
    """Return the integral basis in the presentation's own power basis."""

    if field.degree > MAX_INTEGRAL_BASIS_DEGREE:
        raise OperationDomainValidationError(
            location=("field",),
            code="number_field.ring_of_integers_degree_bound",
            message=(
                "the ring-of-integers operation is limited to degree "
                f"{MAX_INTEGRAL_BASIS_DEGREE}"
            ),
        )
    recognized = recognized_integral_basis(field)
    if recognized is None:
        raise OperationDomainValidationError(
            location=("field",),
            code="number_field.defining_polynomial_must_be_irreducible",
            message="a number field requires an irreducible defining polynomial",
        )
    ring, field_discriminant, alpha, leading = recognized
    basis: list[tuple[CanonicalRational, ...]] = []
    for index in range(field.degree):
        element = ring.basis_element_pullbacks()[index]
        expression = element.as_expr().subs(alpha, leading * alpha).expand()
        polynomial = _as_poly_in_alpha(expression, alpha)
        coefficients = polynomial.all_coeffs()[::-1]
        padded = [
            *(_rational(coefficient) for coefficient in coefficients),
            *(
                CanonicalRational(num=0, den=1)
                for _ in range(field.degree - len(coefficients))
            ),
        ]
        if len(padded) != field.degree:
            raise OperationDomainValidationError(
                location=("basis",),
                code="number_field.ring_of_integers_vector_length",
                message="an integral basis vector must span the complete power basis",
            )
        for coefficient in padded:
            require_bounded_rational(
                coefficient,
                max_digits=MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
                label="integral basis",
            )
        basis.append(tuple(padded))
    result = NumberFieldRingOfIntegersResult(
        field=field,
        basis=tuple(basis),
        field_discriminant=int(field_discriminant),
    )
    result.require_canonical_basis()
    return result


def _as_poly_in_alpha(expression: Any, alpha: Any) -> Any:
    import sympy

    return sympy.Poly(expression, alpha)


__all__ = [
    "MAX_INTEGRAL_BASIS_DEGREE",
    "NumberFieldRingOfIntegersResult",
    "ring_of_integers",
]
