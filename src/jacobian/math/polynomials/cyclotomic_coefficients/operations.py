"""Exact coefficient-ring maps into rational cyclotomic polynomial values."""

from __future__ import annotations

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
    RationalCyclotomicElement,
)
from jacobian.math.polynomials.cyclotomic_coefficients._models import (
    MAX_CYCLOTOMIC_POLYNOMIAL_COORDINATES,
    CyclotomicPolynomial,
    CyclotomicPolynomialTerm,
    RationalPolynomialCyclotomicEmbeddingRequest,
)

_MAX_OUTPUT_BYTES = 10 * 1024 * 1024


def _domain_error(code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=("polynomial",), code=code, message=message
    )


def _resource_error(code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("polynomial",), code=code, message=message
    )


def _integer_digits(value: int) -> int:
    return len(format_canonical_integer(abs(value)))


def _admit_embedding(
    request: RationalPolynomialCyclotomicEmbeddingRequest,
) -> tuple[int, int]:
    polynomial = request.polynomial
    field = request.field
    if polynomial.domain != "QQ":
        _domain_error(
            "coefficient_domain", "source polynomial must have coefficients in QQ"
        )
    degree = field.degree
    coordinate_count = len(polynomial.polynomial.terms) * degree
    if coordinate_count > MAX_CYCLOTOMIC_POLYNOMIAL_COORDINATES:
        _resource_error(
            "coordinate_bound",
            "embedded polynomial coefficient coordinates exceed the admitted bound",
        )

    source_digits = 0
    for term in polynomial.polynomial.terms:
        numerator_digits = _integer_digits(term.coefficient.num)
        denominator_digits = _integer_digits(term.coefficient.den)
        if max(numerator_digits, denominator_digits) > MAX_CYCLIC_FIELD_ELEMENT_DIGITS:
            _resource_error(
                "coefficient_height_bound",
                "source coefficient exceeds the cyclotomic element coordinate height bound",
            )
        source_digits += numerator_digits + denominator_digits

    zero_coordinates = coordinate_count - len(polynomial.polynomial.terms)
    # Each rational coordinate contributes 19 JSON punctuation/key bytes plus
    # its numerator/denominator digits.  Include generous per-element and
    # per-term overhead for the repeated field parent, axes, and exponents.
    output_bytes = (
        2_048
        + 19 * coordinate_count
        + source_digits
        + 2 * zero_coordinates
        + 256 * len(polynomial.polynomial.terms)
        + 40 * len(polynomial.variables)
    )
    if output_bytes > _MAX_OUTPUT_BYTES:
        _resource_error(
            "output_bound", "embedded polynomial exceeds the exact output-byte bound"
        )
    return degree, coordinate_count


def embed_rational_polynomial(
    request: RationalPolynomialCyclotomicEmbeddingRequest,
) -> CyclotomicPolynomial:
    """Apply the canonical inclusion QQ -> QQ(zeta_n) to every coefficient."""

    degree, _ = _admit_embedding(request)
    field = request.field
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
        for term in request.polynomial.polynomial.terms
    )
    return CyclotomicPolynomial(
        field=field,
        variables=request.polynomial.variables,
        terms=terms,
    )


__all__ = ["embed_rational_polynomial"]
