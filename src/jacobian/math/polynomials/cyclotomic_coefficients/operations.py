"""Exact coefficient-ring maps into rational cyclotomic polynomial values."""

from __future__ import annotations

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian.catalog.models import (
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.polynomials.cyclotomic_coefficients._models import (
    MAX_CYCLOTOMIC_POLYNOMIAL_COORDINATES,
    CyclotomicPolynomial,
    CyclotomicPolynomialTerm,
)
from jacobian.math.polynomials.values import RationalPolynomial


def _resource_error(code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("polynomial",), code=code, message=message
    )


def _admit_embedding(
    polynomial: RationalPolynomial,
    field: RationalCyclotomicField,
) -> tuple[int, int]:
    degree = field.degree
    coordinate_count = len(polynomial.polynomial.terms) * degree
    if coordinate_count > MAX_CYCLOTOMIC_POLYNOMIAL_COORDINATES:
        _resource_error(
            "coordinate_bound",
            "embedded polynomial coefficient coordinates exceed the admitted bound",
        )

    for term in polynomial.polynomial.terms:
        if (
            canonical_rational_component_digits(term.coefficient)
            > MAX_CYCLIC_FIELD_ELEMENT_DIGITS
        ):
            _resource_error(
                "coefficient_height_bound",
                "source coefficient exceeds the cyclotomic element coordinate height bound",
            )
    return degree, coordinate_count


def embed_rational_polynomial(
    polynomial: RationalPolynomial,
    field: RationalCyclotomicField,
) -> CyclotomicPolynomial:
    """Apply the canonical inclusion QQ -> QQ(zeta_n) to every coefficient."""

    degree, _ = _admit_embedding(polynomial, field)
    zero = CanonicalRational(num=0, den=1)
    terms = tuple(
        CyclotomicPolynomialTerm(
            coefficient=RationalCyclotomicElement(
                field=field,
                coefficients_ascending=(
                    term.coefficient,
                    *(zero for _ in range(degree - 1)),
                ),
            ),
            exponents=term.exponents,
        )
        for term in polynomial.polynomial.terms
    )
    return CyclotomicPolynomial(
        field=field,
        variables=polynomial.variables,
        terms=terms,
    )


__all__ = ["embed_rational_polynomial"]
