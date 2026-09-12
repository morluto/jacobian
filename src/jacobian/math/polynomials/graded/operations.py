"""Exact bounded initial-ideal and standard-graded quotient operations."""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Literal

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._conversions import (
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.graded._models import (
    MAX_STANDARD_MONOMIALS,
    HilbertFunctionResult,
    InitialMonomialIdealResult,
    StandardMonomialsResult,
)
from jacobian.math.polynomials.ideals._models import (
    IdealComputationBudget,
    _require_ideal_budget,
)
from jacobian.math.polynomials.ideals.operations import groebner_basis
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _require_homogeneous(ideal: RationalPolynomialIdeal) -> None:
    try:
        _require_ideal_budget(ideal, label="graded ideal")
    except ValueError as error:
        raise OperationResourceAdmissionError(
            location=("ideal",), code="graded_ideal.input_budget", message=str(error)
        ) from error
    for generator in ideal.generators:
        degrees = {sum(term.exponents) for term in generator.polynomial.terms}
        if len(degrees) > 1:
            raise OperationDomainValidationError(
                location=("ideal", "generators"),
                code="graded_ideal.nonhomogeneous",
                message="every ideal generator must be homogeneous",
            )


def _unit_monomial(variables: tuple[str, ...], exponents: tuple[int, ...]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1), exponents=exponents
                ),
            )
        ),
    )


def initial_monomial_ideal(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> InitialMonomialIdealResult:
    """Project the existing exact Gröbner result to its initial monomial ideal."""

    _require_homogeneous(ideal)
    basis_result = groebner_basis(ideal, monomial_order, resource_budget=resource_budget)
    variables = ideal.variables
    order = {"lex": "lex", "grlex": "grlex", "grevlex": "grevlex"}[monomial_order]
    from sympy import Poly

    exponents: set[tuple[int, ...]] = set()
    zero_basis = False
    for generator in basis_result.basis.generators:
        if not generator.polynomial.terms:
            zero_basis = True
            continue
        leading = Poly(
            rational_polynomial_to_sympy(generator), *symbols_for_variables(variables), domain="QQ"
        ).LM(order=order)
        exponents.add(tuple(leading.exponents))
    minimal = tuple(sorted(
        (
            exponent
            for exponent in exponents
            if not any(
                other != exponent
                and all(left <= right for left, right in zip(other, exponent, strict=True))
                for other in exponents
            )
        ),
        reverse=True,
    ))
    if zero_basis and not minimal:
        generators = (
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(terms=()),
            ),
        )
    else:
        generators = tuple(_unit_monomial(variables, exponent) for exponent in minimal)
    return InitialMonomialIdealResult(
        ideal=ideal,
        groebner_basis=basis_result.basis,
        initial_ideal=RationalPolynomialIdeal(variables=variables, generators=generators),
        monomial_order=monomial_order,
    )


def _require_monomial_ideal(ideal: RationalPolynomialIdeal) -> tuple[tuple[int, ...], ...]:
    generators: list[tuple[int, ...]] = []
    for generator in ideal.generators:
        terms = generator.polynomial.terms
        if not terms:
            continue
        if len(terms) != 1 or terms[0].coefficient != CanonicalRational(num=1, den=1):
            raise OperationDomainValidationError(
                location=("initial_ideal",),
                code="graded_ideal.monomial_shape",
                message="standard-monomial operations require unit monomial generators",
            )
        generators.append(terms[0].exponents)
    return tuple(generators)


def _compositions(degree: int, variables: int) -> tuple[tuple[int, ...], ...]:
    if variables == 1:
        return ((degree,),)
    return tuple(
        tuple(
            boundaries[index + 1] - boundaries[index] - 1
            for index in range(variables)
        )
        for cuts in combinations(range(1, degree + variables), variables - 1)
        for boundaries in ((0, *cuts, degree + variables),)
    )


def standard_monomials(initial_ideal: RationalPolynomialIdeal, degree: int) -> StandardMonomialsResult:
    generators = _require_monomial_ideal(initial_ideal)
    variables = len(initial_ideal.variables)
    domain_size = comb(degree + variables - 1, variables - 1)
    if domain_size > MAX_STANDARD_MONOMIALS:
        raise OperationResourceAdmissionError(
            location=("degree",), code="graded_ideal.monomial_domain_budget", message="standard-monomial domain exceeds the bounded enumeration envelope"
        )
    monomials = tuple(
        monomial
        for monomial in _compositions(degree, variables)
        if not any(all(left <= right for left, right in zip(generator, monomial, strict=True)) for generator in generators)
    )
    return StandardMonomialsResult(
        initial_ideal=initial_ideal, degree=degree, monomials=tuple(sorted(monomials, reverse=True)), count=len(monomials)
    )


def hilbert_function(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
    max_degree: int = 0,
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> HilbertFunctionResult:
    initial = initial_monomial_ideal(ideal, monomial_order, resource_budget=resource_budget)
    values = tuple(
        standard_monomials(initial.initial_ideal, degree).count
        for degree in range(max_degree + 1)
    )
    return HilbertFunctionResult(
        ideal=ideal,
        initial_ideal=initial.initial_ideal,
        monomial_order=monomial_order,
        values=values,
    )


__all__ = ["hilbert_function", "initial_monomial_ideal", "standard_monomials"]
