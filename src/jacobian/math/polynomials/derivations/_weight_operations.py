"""Exact polynomial coactions from bounded diagonal integer weights."""

from __future__ import annotations

from collections.abc import Iterator
from math import comb
from typing import Any

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.derivations._weight_models import (
    MAX_DIAGONAL_WEIGHT,
    MAX_GM_INVARIANT_MONOMIALS,
    MAX_WEIGHT_ACTION_DEGREE,
    MAX_WEIGHT_ACTION_TERMS,
    PolynomialWeightAction,
    PolynomialWeightActionRequest,
    PolynomialWeightActionResult,
    PolynomialWeightComponent,
    PolynomialWeightDegreeDimension,
    PolynomialWeightInvariantRequest,
    PolynomialWeightInvariantResult,
)
from jacobian.math.polynomials.values import (
    RationalLaurentPolynomial,
    RationalLaurentPolynomialTerm,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _admit_request(
    request: PolynomialWeightActionRequest | dict[str, Any],
) -> PolynomialWeightActionRequest:
    try:
        payload = (
            request.model_dump()
            if isinstance(request, PolynomialWeightActionRequest)
            else request
        )
        checked = PolynomialWeightActionRequest.model_validate(payload)
        action = PolynomialWeightAction.model_validate(checked.action.model_dump())
        source = RationalPolynomial.model_validate(checked.polynomial.model_dump())
        checked = checked.model_copy(update={"action": action, "polynomial": source})
    except (ValidationError, TypeError, ValueError, PydanticCustomError) as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="polynomial_weight_action.request_shape",
            message="the action request must contain a canonical polynomial in its bound ring",
        ) from exc

    if len(source.polynomial.terms) > MAX_WEIGHT_ACTION_TERMS:
        raise OperationResourceAdmissionError(
            location=("polynomial", "polynomial", "terms"),
            code="polynomial_weight_action.term_budget",
            message="source polynomial exceeds the diagonal-action term budget",
        )
    # Admit all growth before deriving Laurent exponent tuples or projections.
    for term in source.polynomial.terms:
        degree = sum(term.exponents)
        if degree > MAX_WEIGHT_ACTION_DEGREE:
            raise OperationResourceAdmissionError(
                location=("polynomial", "polynomial", "terms"),
                code="polynomial_weight_action.degree_budget",
                message="source polynomial exceeds the total-degree budget",
            )
        weight = sum(w * e for w, e in zip(action.weights, term.exponents, strict=True))
        # This follows from the admitted degree and coordinate-weight bounds,
        # but retain the explicit carrier check at the representation boundary.
        if abs(weight) > MAX_DIAGONAL_WEIGHT * MAX_WEIGHT_ACTION_DEGREE:
            raise OperationResourceAdmissionError(
                location=("polynomial", "polynomial", "terms"),
                code="polynomial_weight_action.laurent_exponent_budget",
                message="the induced Laurent exponent exceeds the admitted range",
            )
        try:
            require_bounded_rational(
                term.coefficient,
                max_digits=128,
                label="weight-action source coefficient",
            )
        except ValueError as exc:
            raise OperationResourceAdmissionError(
                location=("polynomial", "polynomial", "terms"),
                code="polynomial_weight_action.coefficient_budget",
                message=str(exc),
            ) from exc
    return checked


def diagonal_weight_action(
    request: PolynomialWeightActionRequest | dict[str, Any],
) -> PolynomialWeightActionResult:
    """Return ``rho(f)`` and its exact integer-weight decomposition.

    Each source monomial ``c*x**e`` maps to ``c*t**(w.e)*x**e``. The sparse
    image has no more terms than the admitted source, so the preflight above
    bounds its complete expansion before the first output term is built.
    """
    checked = _admit_request(request)
    action, source, parameter = checked.action, checked.polynomial, checked.parameter
    grouped: dict[int, list[RationalPolynomialTerm]] = {}
    coaction_terms: list[RationalLaurentPolynomialTerm] = []
    for term in source.polynomial.terms:
        weight = sum(w * e for w, e in zip(action.weights, term.exponents, strict=True))
        grouped.setdefault(weight, []).append(term)
        coaction_terms.append(
            RationalLaurentPolynomialTerm(
                coefficient=term.coefficient,
                exponents=(*term.exponents, weight),
            )
        )

    components = tuple(
        PolynomialWeightComponent(
            weight=weight,
            polynomial=RationalPolynomial(
                variables=action.variables,
                polynomial=SparseRationalPolynomial(
                    terms=tuple(
                        sorted(terms, key=lambda term: term.exponents, reverse=True)
                    )
                ),
            ),
        )
        for weight, terms in sorted(grouped.items())
    )
    zero = next(
        (component.polynomial for component in components if component.weight == 0),
        RationalPolynomial(
            variables=action.variables, polynomial=SparseRationalPolynomial(terms=())
        ),
    )
    return PolynomialWeightActionResult(
        action=action,
        source=source,
        parameter=parameter,
        coaction=RationalLaurentPolynomial(
            variables=(*action.variables, parameter),
            terms=tuple(
                sorted(coaction_terms, key=lambda term: term.exponents, reverse=True)
            ),
        ),
        components=components,
        weight_zero=zero,
    )


def _degree_compositions(variable_count: int, degree: int) -> Iterator[tuple[int, ...]]:
    """Yield a homogeneous monomial exponent basis in descending lex order."""
    if variable_count == 1:
        yield (degree,)
        return
    for first in range(degree, -1, -1):
        for tail in _degree_compositions(variable_count - 1, degree - first):
            yield (first, *tail)


def gm_invariants_through_degree(
    request: PolynomialWeightInvariantRequest | dict[str, Any],
) -> PolynomialWeightInvariantResult:
    """Return the complete weight-zero monomial basis through one degree bound."""
    try:
        payload = (
            request.model_dump()
            if isinstance(request, PolynomialWeightInvariantRequest)
            else request
        )
        checked = PolynomialWeightInvariantRequest.model_validate(payload)
        action = PolynomialWeightAction.model_validate(checked.action.model_dump())
        checked = checked.model_copy(update={"action": action})
    except (ValidationError, TypeError, ValueError, PydanticCustomError) as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="polynomial_weight_invariant.request_shape",
            message="invariant-slice request must use a canonical bounded integer-weight action",
        ) from exc

    degree = checked.degree
    variable_count = len(action.variables)
    candidate_count = comb(variable_count + degree, degree)
    if candidate_count > MAX_GM_INVARIANT_MONOMIALS:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="polynomial_weight_invariant.monomial_budget",
            message=(
                "the complete polynomial degree slice has "
                f"{candidate_count} monomials, exceeding the "
                f"{MAX_GM_INVARIANT_MONOMIALS}-monomial envelope"
            ),
        )
    # All candidate tuples and result rows are bounded before enumeration.
    output_cells = candidate_count * (variable_count + 2) + degree + 1
    if output_cells > 50_000:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="polynomial_weight_invariant.output_budget",
            message="the exact invariant basis and Hilbert prefix exceed the output envelope",
        )

    basis = []
    profile = []
    for homogeneous_degree in range(degree + 1):
        count = 0
        for exponents in _degree_compositions(variable_count, homogeneous_degree):
            weight = sum(
                variable_weight * exponent
                for variable_weight, exponent in zip(
                    action.weights, exponents, strict=True
                )
            )
            if weight != 0:
                continue
            count += 1
            basis.append(
                RationalPolynomial(
                    variables=action.variables,
                    polynomial=SparseRationalPolynomial(
                        terms=(
                            RationalPolynomialTerm(
                                coefficient=CanonicalRational(num=1, den=1),
                                exponents=exponents,
                            ),
                        )
                    ),
                )
            )
        profile.append(
            PolynomialWeightDegreeDimension(degree=homogeneous_degree, dimension=count)
        )
    return PolynomialWeightInvariantResult(
        action=action,
        degree=degree,
        basis=tuple(basis),
        hilbert_prefix=tuple(profile),
        dimension=len(basis),
    )


__all__ = ["diagonal_weight_action", "gm_invariants_through_degree"]
