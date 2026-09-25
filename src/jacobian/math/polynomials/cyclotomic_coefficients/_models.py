"""Canonical polynomial values over rational cyclotomic fields."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import format_canonical_integer
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
    MAX_CYCLIC_PERIOD,
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
    RationalPolynomial,
)

MAX_CYCLOTOMIC_POLYNOMIAL_COORDINATES = 16_384
MAX_CYCLOTOMIC_POLYNOMIAL_COORDINATE_DIGITS = 2 * MAX_CYCLIC_FIELD_ELEMENT_DIGITS
MAX_CYCLOTOMIC_POLYNOMIAL_TOTAL_DIGITS = (
    MAX_CYCLOTOMIC_POLYNOMIAL_COORDINATES * MAX_CYCLOTOMIC_POLYNOMIAL_COORDINATE_DIGITS
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial.cyclotomic.{reason}", message)


def _euler_phi(value: int) -> int:
    remaining = value
    result = value
    prime = 2
    while prime * prime <= remaining:
        if remaining % prime == 0:
            while remaining % prime == 0:
                remaining //= prime
            result -= result // prime
        prime += 1 if prime == 2 else 2
    if remaining > 1:
        result -= result // remaining
    return result


def _raw_coordinate_count(data: object) -> int | None:
    if not isinstance(data, Mapping):
        return None
    field = data.get("field")
    terms = data.get("terms")
    if not isinstance(field, Mapping) or not isinstance(terms, (list, tuple)):
        return None
    order = field.get("order")
    if type(order) is not int or not 1 <= order <= MAX_CYCLIC_PERIOD:
        return None
    return len(terms) * _euler_phi(order)


class CyclotomicPolynomialTerm(StrictModel):
    """One nonzero monomial with an exact cyclotomic coefficient."""

    coefficient: RationalCyclotomicElement
    exponents: tuple[StrictInt, ...] = Field(
        min_length=0, max_length=MAX_POLYNOMIAL_VARIABLES
    )

    @model_validator(mode="after")
    def require_nonzero_bounded_term(self) -> Self:
        if all(value.num == 0 for value in self.coefficient.coefficients_ascending):
            raise _validation_error("zero_term", "zero terms must be omitted")
        if any(
            exponent < 0 or exponent > MAX_POLYNOMIAL_EXPONENT
            for exponent in self.exponents
        ):
            raise _validation_error(
                "exponent_bound",
                "polynomial exponents exceed the shared representation limit",
            )
        return self


class CyclotomicPolynomial(StrictModel):
    """A sparse polynomial over one explicitly presented field QQ(zeta_n).

    Coefficients use the shared ``RationalCyclotomicElement`` type and its
    ascending power basis.  The polynomial binds one coefficient field and an
    ordered variable tuple; each term repeats that field in its coefficient so
    independently serialized coefficients retain their parent.
    """

    domain: Literal["QQ_CYCLOTOMIC_POLYNOMIAL"] = "QQ_CYCLOTOMIC_POLYNOMIAL"
    field: RationalCyclotomicField
    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=0, max_length=MAX_POLYNOMIAL_VARIABLES
    )
    terms: tuple[CyclotomicPolynomialTerm, ...] = Field(
        default=(), max_length=MAX_POLYNOMIAL_TERMS
    )

    @model_validator(mode="before")
    @classmethod
    def admit_raw_coordinate_shape(cls, data: object) -> object:
        data = canonicalize_json_containers(data)
        coordinate_count = _raw_coordinate_count(data)
        if (
            coordinate_count is not None
            and coordinate_count > MAX_CYCLOTOMIC_POLYNOMIAL_COORDINATES
        ):
            raise _validation_error(
                "coordinate_bound",
                "polynomial coefficient coordinates exceed the shared representation bound",
            )
        return data

    @model_validator(mode="after")
    def require_canonical_polynomial(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error("duplicate_variables", "variables must be distinct")
        if any(term.coefficient.field != self.field for term in self.terms):
            raise _validation_error(
                "coefficient_parent", "every coefficient must use the polynomial field"
            )
        support = tuple(term.exponents for term in self.terms)
        if any(len(exponents) != len(self.variables) for exponents in support):
            raise _validation_error(
                "monomial_shape", "every monomial must match the ordered variables"
            )
        if len(set(support)) != len(support):
            raise _validation_error(
                "duplicate_exponents", "monomial exponents must be unique"
            )
        if support != tuple(sorted(support, reverse=True)):
            raise _validation_error(
                "term_order", "terms must use descending lexicographic exponent order"
            )
        coordinates = sum(
            len(term.coefficient.coefficients_ascending) for term in self.terms
        )
        if coordinates > MAX_CYCLOTOMIC_POLYNOMIAL_COORDINATES:
            raise _validation_error(
                "coordinate_bound",
                "polynomial coefficient coordinates exceed the shared representation bound",
            )
        total_digits = sum(
            len(format_canonical_integer(abs(value.num)))
            + len(format_canonical_integer(value.den))
            for term in self.terms
            for value in term.coefficient.coefficients_ascending
        )
        if total_digits > MAX_CYCLOTOMIC_POLYNOMIAL_TOTAL_DIGITS:
            raise _validation_error(
                "digit_bound",
                "polynomial coefficient digits exceed the shared representation bound",
            )
        return self


class RationalPolynomialCyclotomicEmbeddingRequest(StrictModel):
    """Embed one rational polynomial in an explicitly selected cyclotomic field."""

    polynomial: RationalPolynomial
    field: RationalCyclotomicField


__all__ = [
    "MAX_CYCLOTOMIC_POLYNOMIAL_COORDINATES",
    "MAX_CYCLOTOMIC_POLYNOMIAL_COORDINATE_DIGITS",
    "MAX_CYCLOTOMIC_POLYNOMIAL_TOTAL_DIGITS",
    "CyclotomicPolynomial",
    "CyclotomicPolynomialTerm",
    "RationalPolynomialCyclotomicEmbeddingRequest",
]
