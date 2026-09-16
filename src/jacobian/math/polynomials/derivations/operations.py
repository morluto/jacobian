"""Native exact polynomial-derivation operations."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.derivations._models import (
    MAX_DERIVATION_CONTRIBUTION_CELLS,
    MAX_DERIVATION_IMAGE_TERMS,
    MAX_DERIVATION_SOURCE_TERMS,
    DerivationApplyResult,
    PolynomialDerivation,
    _require_derivation_polynomial,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

_TermMap = dict[tuple[int, ...], Fraction]


def _as_derivation(
    value: PolynomialDerivation | Mapping[str, Any],
) -> PolynomialDerivation:
    return (
        value
        if isinstance(value, PolynomialDerivation)
        else PolynomialDerivation.model_validate(value)
    )


def _as_polynomial(value: RationalPolynomial | Mapping[str, Any]) -> RationalPolynomial:
    return (
        value
        if isinstance(value, RationalPolynomial)
        else RationalPolynomial.model_validate(value)
    )


def _run_admission(admission: Any, *, location: tuple[str | int, ...]) -> None:
    try:
        admission()
    except OperationDomainValidationError:
        raise
    except OperationResourceAdmissionError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=location,
            code="polynomial_derivation.admission",
            message=str(exc),
        ) from exc


def _admit_derivation_apply(
    derivation: PolynomialDerivation, polynomial: RationalPolynomial
) -> None:
    """Enforce the shared ring and the pre-expansion work/output envelope."""
    if polynomial.variables != derivation.variables:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial_derivation.ordered_ring",
            message="the polynomial must use the derivation's ordered ring",
        )
    _run_admission(
        lambda: _require_derivation_polynomial(
            polynomial,
            label="source polynomial",
            maximum_terms=MAX_DERIVATION_SOURCE_TERMS,
        ),
        location=("polynomial",),
    )
    for index, image in enumerate(derivation.images):
        _run_admission(
            lambda image=image, index=index: _require_derivation_polynomial(
                image,
                label=f"generator image {index}",
                maximum_terms=MAX_DERIVATION_IMAGE_TERMS,
            ),
            location=("derivation", "images", index),
        )
    predicted_cells = 0
    for axis, image in enumerate(derivation.images):
        surviving = sum(
            1 for term in polynomial.polynomial.terms if term.exponents[axis] > 0
        )
        predicted_cells += len(image.polynomial.terms) * surviving
    if predicted_cells > MAX_DERIVATION_CONTRIBUTION_CELLS:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial_derivation.contribution_budget",
            message="derivation application exceeds the contribution-cell budget",
        )


def _term_map(polynomial: RationalPolynomial) -> _TermMap:
    return {
        tuple(term.exponents): term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }


def _partial(terms: _TermMap, axis: int) -> _TermMap:
    derived: _TermMap = {}
    for exponents, coefficient in terms.items():
        if exponents[axis] > 0:
            lowered = list(exponents)
            lowered[axis] -= 1
            derived[tuple(lowered)] = coefficient * exponents[axis]
    return derived


def _multiply(left: _TermMap, right: _TermMap) -> _TermMap:
    product: _TermMap = {}
    for left_exponents, left_coefficient in left.items():
        for right_exponents, right_coefficient in right.items():
            exponents = tuple(
                left_exponent + right_exponent
                for left_exponent, right_exponent in zip(
                    left_exponents, right_exponents, strict=True
                )
            )
            product[exponents] = (
                product.get(exponents, Fraction(0))
                + left_coefficient * right_coefficient
            )
    return {exponents: value for exponents, value in product.items() if value != 0}


def _add(into: _TermMap, extra: _TermMap) -> None:
    for exponents, value in extra.items():
        combined = into.get(exponents, Fraction(0)) + value
        if combined == 0:
            into.pop(exponents, None)
        else:
            into[exponents] = combined


def _encode(variables: tuple[str, ...], terms: _TermMap) -> RationalPolynomial:
    ordered = sorted(terms.items(), key=lambda item: item[0], reverse=True)
    return RationalPolynomial(
        domain="QQ",
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(coefficient),
                    exponents=exponents,
                )
                for exponents, coefficient in ordered
            )
        ),
    )


def apply_derivation(
    derivation: PolynomialDerivation | Mapping[str, Any],
    polynomial: RationalPolynomial | Mapping[str, Any],
) -> DerivationApplyResult:
    """Compute D(f) = sum_i D(x_i) * partial_i(f) with a per-variable ledger."""
    derivation_value = _as_derivation(derivation)
    polynomial_value = _as_polynomial(polynomial)
    _admit_derivation_apply(derivation_value, polynomial_value)
    source = _term_map(polynomial_value)
    total: _TermMap = {}
    contributions: list[RationalPolynomial] = []
    for axis, image in enumerate(derivation_value.images):
        contribution = _multiply(_term_map(image), _partial(source, axis))
        contributions.append(_encode(derivation_value.variables, contribution))
        _add(total, contribution)
    return DerivationApplyResult._from_kernel(
        derivation_value,
        polynomial_value,
        result=_encode(derivation_value.variables, total),
        contributions=tuple(contributions),
    )


__all__ = ["apply_derivation"]
