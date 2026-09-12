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
    rational_function_from_sympy,
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.graded._models import (
    MAX_GRADED_DEGREE,
    MAX_HILBERT_PREFIX,
    MAX_HILBERT_SERIES_GENERATORS,
    MAX_STANDARD_MONOMIALS,
    HilbertDimensionResult,
    HilbertFunctionResult,
    HilbertMultiplicityResult,
    HilbertPolynomialResult,
    HilbertSeriesResult,
    HVectorResult,
    InitialMonomialIdealResult,
    StandardMonomialsResult,
)
from jacobian.math.polynomials.ideals._models import (
    IdealComputationBudget,
    _require_ideal_budget,
)
from jacobian.math.polynomials.ideals.operations import groebner_basis
from jacobian.math.polynomials.values import (
    MAX_RATIONAL_FUNCTION_EXPONENT,
    MAX_RATIONAL_FUNCTION_TERMS,
    RationalFunction,
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


def _unit_monomial(
    variables: tuple[str, ...], exponents: tuple[int, ...]
) -> RationalPolynomial:
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
    basis_result = groebner_basis(
        ideal, monomial_order, resource_budget=resource_budget
    )
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
            rational_polynomial_to_sympy(generator),
            *symbols_for_variables(variables),
            domain="QQ",
        ).LM(order=order)
        exponents.add(tuple(leading.exponents))
    minimal = tuple(
        sorted(
            (
                exponent
                for exponent in exponents
                if not any(
                    other != exponent
                    and all(
                        left <= right
                        for left, right in zip(other, exponent, strict=True)
                    )
                    for other in exponents
                )
            ),
            reverse=True,
        )
    )
    generators: tuple[RationalPolynomial, ...]
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
        initial_ideal=RationalPolynomialIdeal(
            variables=variables, generators=generators
        ),
        monomial_order=monomial_order,
    )


def _require_monomial_ideal(
    ideal: RationalPolynomialIdeal,
) -> tuple[tuple[int, ...], ...]:
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
            boundaries[index + 1] - boundaries[index] - 1 for index in range(variables)
        )
        for cuts in combinations(range(1, degree + variables), variables - 1)
        for boundaries in ((0, *cuts, degree + variables),)
    )


def standard_monomials(
    initial_ideal: RationalPolynomialIdeal, degree: int
) -> StandardMonomialsResult:
    if degree < 0 or degree > MAX_GRADED_DEGREE:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="graded_ideal.monomial_degree_budget",
            message="standard monomials support degrees from 0 through 32",
        )
    generators = _require_monomial_ideal(initial_ideal)
    variables = len(initial_ideal.variables)
    domain_size = comb(degree + variables - 1, variables - 1)
    if domain_size > MAX_STANDARD_MONOMIALS:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="graded_ideal.monomial_domain_budget",
            message="standard-monomial domain exceeds the bounded enumeration envelope",
        )
    monomials = tuple(
        monomial
        for monomial in _compositions(degree, variables)
        if not any(
            all(left <= right for left, right in zip(generator, monomial, strict=True))
            for generator in generators
        )
    )
    return StandardMonomialsResult(
        initial_ideal=initial_ideal,
        degree=degree,
        monomials=tuple(sorted(monomials, reverse=True)),
        count=len(monomials),
    )


def hilbert_function(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
    max_degree: int = 0,
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> HilbertFunctionResult:
    if max_degree < 0 or max_degree > MAX_GRADED_DEGREE:
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="graded_ideal.function_degree_budget",
            message="Hilbert-function prefixes support degrees from 0 through 32",
        )
    initial = initial_monomial_ideal(
        ideal, monomial_order, resource_budget=resource_budget
    )
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


def _admit_hilbert_series(
    initial_ideal: RationalPolynomialIdeal,
) -> tuple[tuple[int, ...], ...]:
    generators = _require_monomial_ideal(initial_ideal)
    if len(generators) > MAX_HILBERT_SERIES_GENERATORS:
        raise OperationResourceAdmissionError(
            location=("initial_ideal",),
            code="graded_ideal.series_generator_budget",
            message="Hilbert-series inclusion-exclusion supports at most 8 minimal generators",
        )
    if any(len(generator) != len(initial_ideal.variables) for generator in generators):
        raise OperationDomainValidationError(
            location=("initial_ideal",),
            code="graded_ideal.axis",
            message="initial-ideal generators must use the complete ordered axis",
        )
    maximum_degree = sum(
        max((generator[axis] for generator in generators), default=0)
        for axis in range(len(initial_ideal.variables))
    )
    if maximum_degree > MAX_RATIONAL_FUNCTION_EXPONENT:
        raise OperationResourceAdmissionError(
            location=("initial_ideal",),
            code="graded_ideal.series_degree_budget",
            message="Hilbert-series numerator degree exceeds the rational-function envelope",
        )
    return generators


def _polynomial_from_integer_coefficients(
    coefficients: dict[int, int],
) -> RationalPolynomial:
    variables = ("t",)
    terms = tuple(
        RationalPolynomialTerm(
            coefficient=CanonicalRational(num=value, den=1),
            exponents=(degree,),
        )
        for degree, value in sorted(coefficients.items(), reverse=True)
        if value
    )
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(terms=terms),
    )


def _series_data(
    initial_ideal: RationalPolynomialIdeal,
    prefix_degree: int,
) -> tuple[
    RationalPolynomial,
    RationalFunction,
    RationalPolynomial,
    RationalPolynomial,
    int,
    tuple[int, ...],
    tuple[int, ...],
]:
    generators = _admit_hilbert_series(initial_ideal)
    variable_count = len(initial_ideal.variables)
    subset_coefficients: dict[int, int] = {0: 1}
    for mask in range(1, 1 << len(generators)):
        lcm_exponents = [0] * variable_count
        cardinality = 0
        for index, generator in enumerate(generators):
            if mask & (1 << index):
                cardinality += 1
                for axis, exponent in enumerate(generator):
                    lcm_exponents[axis] = max(lcm_exponents[axis], exponent)
        degree = sum(lcm_exponents)
        subset_coefficients[degree] = subset_coefficients.get(degree, 0) + (
            -1 if cardinality % 2 else 1
        )
    subset_coefficients = {
        degree: value for degree, value in subset_coefficients.items() if value
    }
    ambient_numerator = _polynomial_from_integer_coefficients(subset_coefficients)
    from sympy import Symbol

    t = Symbol("t")
    ambient_expression = rational_polynomial_to_sympy(ambient_numerator).as_expr()
    series = rational_function_from_sympy(
        ambient_expression / (1 - t) ** variable_count,
        ("t",),
        maximum_terms=MAX_RATIONAL_FUNCTION_TERMS,
    )
    denominator_exponent = max(
        (term.exponents[0] for term in series.denominator.terms),
        default=0,
    )
    sign = -1 if denominator_exponent % 2 else 1
    raw_coefficients = {
        term.exponents[0]: int(term.coefficient.as_fraction())
        for term in series.numerator.terms
    }
    reduced_numerator = _polynomial_from_integer_coefficients(raw_coefficients)
    h_coefficients = {
        degree: sign * coefficient for degree, coefficient in raw_coefficients.items()
    }
    h_vector = tuple(
        h_coefficients.get(index, 0)
        for index in range(max(h_coefficients, default=0) + 1)
    )
    h_numerator = _polynomial_from_integer_coefficients(h_coefficients)
    prefix = []
    from math import comb

    for degree in range(prefix_degree + 1):
        if denominator_exponent == 0:
            value = h_coefficients.get(degree, 0)
        else:
            value = sum(
                coefficient
                * comb(
                    degree - shift + denominator_exponent - 1,
                    denominator_exponent - 1,
                )
                for shift, coefficient in h_coefficients.items()
                if shift <= degree
            )
        prefix.append(value)
    return (
        ambient_numerator,
        series,
        reduced_numerator,
        h_numerator,
        denominator_exponent,
        tuple(prefix),
        h_vector,
    )


def hilbert_series(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
    prefix_degree: int = 0,
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> HilbertSeriesResult:
    if prefix_degree < 0 or prefix_degree > MAX_HILBERT_PREFIX:
        raise OperationResourceAdmissionError(
            location=("prefix_degree",),
            code="graded_ideal.series_prefix_budget",
            message="Hilbert-series prefixes support degrees from 0 through 16",
        )
    initial = initial_monomial_ideal(
        ideal, monomial_order, resource_budget=resource_budget
    )
    data = _series_data(initial.initial_ideal, prefix_degree)
    return HilbertSeriesResult(
        ideal=ideal,
        initial_ideal=initial.initial_ideal,
        monomial_order=monomial_order,
        ambient_numerator=data[0],
        ambient_denominator_exponent=len(ideal.variables),
        series=data[1],
        reduced_numerator=data[2],
        h_numerator=data[3],
        denominator_exponent=data[4],
        prefix=data[5],
    )


def _series_projection(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"],
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> HilbertSeriesResult:
    return hilbert_series(ideal, monomial_order, resource_budget=resource_budget)


def hilbert_polynomial(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> HilbertPolynomialResult:
    data = _series_projection(ideal, monomial_order, resource_budget=resource_budget)
    dimension = data.denominator_exponent
    stabilization = max(
        (term.exponents[0] for term in data.h_numerator.polynomial.terms),
        default=-1,
    ) + (1 if data.denominator_exponent == 0 else 0)

    from sympy import QQ, Poly, Symbol, binomial, expand_func

    m = Symbol("m")
    if dimension == 0:
        polynomial = RationalPolynomial(
            variables=("m",),
            polynomial=SparseRationalPolynomial(terms=()),
        )
    else:
        expression = sum(
            int(term.coefficient.as_fraction())
            * expand_func(
                binomial(m - term.exponents[0] + dimension - 1, dimension - 1)
            )
            for term in data.h_numerator.polynomial.terms
        )
        polynomial = rational_polynomial_from_sympy(
            Poly(expression.expand(), m, domain=QQ), ("m",), maximum_terms=64
        )
    return HilbertPolynomialResult(
        ideal=ideal,
        initial_ideal=data.initial_ideal,
        monomial_order=monomial_order,
        dimension=dimension,
        polynomial=polynomial,
        stabilization_degree=stabilization,
    )


def hilbert_dimension(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> HilbertDimensionResult:
    data = _series_projection(ideal, monomial_order, resource_budget=resource_budget)
    return HilbertDimensionResult(
        ideal=ideal,
        initial_ideal=data.initial_ideal,
        monomial_order=monomial_order,
        dimension=data.denominator_exponent,
    )


def hilbert_multiplicity(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> HilbertMultiplicityResult:
    data = _series_projection(ideal, monomial_order, resource_budget=resource_budget)
    multiplicity = sum(
        term.coefficient.as_fraction() for term in data.h_numerator.polynomial.terms
    )
    if multiplicity.denominator != 1 or multiplicity < 0:
        raise OperationDomainValidationError(
            location=("series",),
            code="graded_ideal.multiplicity",
            message="Hilbert multiplicity did not reduce to a nonnegative integer",
        )
    return HilbertMultiplicityResult(
        ideal=ideal,
        initial_ideal=data.initial_ideal,
        monomial_order=monomial_order,
        dimension=data.denominator_exponent,
        multiplicity=multiplicity.numerator,
    )


def h_vector(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> HVectorResult:
    data = _series_projection(ideal, monomial_order, resource_budget=resource_budget)
    h_coefficients = {
        term.exponents[0]: int(term.coefficient.as_fraction())
        for term in data.h_numerator.polynomial.terms
    }
    values = tuple(
        h_coefficients.get(index, 0)
        for index in range(max(h_coefficients, default=0) + 1)
    )
    return HVectorResult(
        ideal=ideal,
        initial_ideal=data.initial_ideal,
        monomial_order=monomial_order,
        dimension=data.denominator_exponent,
        h_vector=values,
    )


__all__ = [
    "h_vector",
    "hilbert_dimension",
    "hilbert_function",
    "hilbert_multiplicity",
    "hilbert_polynomial",
    "hilbert_series",
    "initial_monomial_ideal",
    "standard_monomials",
]
