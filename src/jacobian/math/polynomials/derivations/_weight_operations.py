"""Exact polynomial coactions from bounded diagonal integer weights."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
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
    MAX_GM_INVARIANT_DEGREE,
    MAX_GM_INVARIANT_MONOMIALS,
    MAX_WEIGHT_ACTION_DEGREE,
    MAX_WEIGHT_ACTION_TERMS,
    MAX_WEIGHT_ACTION_VARIABLES,
    PolynomialWeightAction,
    PolynomialWeightActionResult,
    PolynomialWeightComponent,
    PolynomialWeightDegreeDimension,
    PolynomialWeightInvariantResult,
)
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalLaurentPolynomial,
    RationalLaurentPolynomialTerm,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _as_weight_action(
    action: PolynomialWeightAction | Mapping[str, Any], *, code: str
) -> PolynomialWeightAction:
    try:
        payload = (
            action.model_dump()
            if isinstance(action, PolynomialWeightAction)
            else action
        )
        return PolynomialWeightAction.model_validate(payload)
    except (ValidationError, TypeError, ValueError, PydanticCustomError) as exc:
        raise OperationDomainValidationError(
            location=("action",),
            code=code,
            message="the request must carry one bounded integer weight per variable",
        ) from exc


def _as_decoded_polynomial(
    value: RationalPolynomial | Mapping[str, Any],
) -> RationalPolynomial:
    try:
        payload = value.model_dump() if isinstance(value, RationalPolynomial) else value
        return RationalPolynomial.model_validate(payload)
    except (ValidationError, TypeError, ValueError, PydanticCustomError) as exc:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial_weight_action.request_shape",
            message="the action request must contain a canonical polynomial in its bound ring",
        ) from exc


def _admit_weight_request(
    action: PolynomialWeightAction,
    source: RationalPolynomial,
    parameter: PolynomialVariable,
) -> None:
    if source.variables != action.variables:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial_weight_action.request_shape",
            message="polynomial and action must use the same ordered QQ ring",
        )
    if parameter in action.variables:
        raise OperationDomainValidationError(
            location=("parameter",),
            code="polynomial_weight_action.request_shape",
            message="the Laurent parameter must be distinct from ring variables",
        )
    if len(action.variables) > MAX_WEIGHT_ACTION_VARIABLES:
        raise OperationDomainValidationError(
            location=("action",),
            code="polynomial_weight_action.request_shape",
            message=(
                f"the diagonal action is bounded to {MAX_WEIGHT_ACTION_VARIABLES} "
                "source variables because the Laurent coaction carrier reserves "
                "its eighth axis for the parameter"
            ),
        )

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


def diagonal_weight_action(
    action: PolynomialWeightAction | Mapping[str, Any],
    polynomial: RationalPolynomial | Mapping[str, Any],
    parameter: PolynomialVariable = "t",
) -> PolynomialWeightActionResult:
    """Return ``rho(f)`` and its exact integer-weight decomposition.

    Each source monomial ``c*x**e`` maps to ``c*t**(w.e)*x**e``. The sparse
    image has no more terms than the admitted source, so the preflight below
    bounds its complete expansion before the first output term is built.
    """
    checked_action = _as_weight_action(
        action, code="polynomial_weight_action.request_shape"
    )
    checked_source = _as_decoded_polynomial(polynomial)
    _admit_weight_request(checked_action, checked_source, parameter)
    action, source = checked_action, checked_source
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
    action: PolynomialWeightAction | Mapping[str, Any], degree: int
) -> PolynomialWeightInvariantResult:
    """Return the complete weight-zero monomial basis through one degree bound."""
    action = _as_weight_action(action, code="polynomial_weight_invariant.request_shape")
    if (
        isinstance(degree, bool)
        or not isinstance(degree, int)
        or not 0 <= degree <= MAX_GM_INVARIANT_DEGREE
    ):
        raise OperationDomainValidationError(
            location=("degree",),
            code="polynomial_weight_invariant.request_shape",
            message=(
                "the invariant-slice degree must be an integer between 0 and "
                f"{MAX_GM_INVARIANT_DEGREE}"
            ),
        )

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
