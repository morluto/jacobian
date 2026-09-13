"""Exact bounded initial-ideal and standard-graded quotient operations."""

from __future__ import annotations

import time
from collections.abc import Iterator
from math import comb
from typing import Literal

from sympy import QQ, Poly, Symbol, binomial, cancel, expand_func, fraction

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    execution_deadline,
    request_checkpoint,
    require_execution_deadline,
)
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
from jacobian.math.polynomials.ideals._models import IdealComputationBudget
from jacobian.math.polynomials.ideals.operations import _admit_source, groebner_basis
from jacobian.math.polynomials.values import (
    MAX_RATIONAL_FUNCTION_EXPONENT,
    MAX_RATIONAL_FUNCTION_TERMS,
    RationalFunction,
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

# An exact degree-slice monomial count is only enumerated when the ambient
# slice is small enough to walk; above this the pure-power bound is used.
_MONOMIAL_ENUMERATION_CEILING = 200_000


def _require_monomial_order(
    monomial_order: object,
) -> Literal["lex", "grlex", "grevlex"]:
    if type(monomial_order) is not str or monomial_order not in {
        "lex",
        "grlex",
        "grevlex",
    }:
        raise OperationDomainValidationError(
            location=("monomial_order",),
            code="graded_ideal.monomial_order",
            message="monomial_order must be lex, grlex, or grevlex",
        )
    return monomial_order  # type: ignore[return-value]


def _is_explicit_unit_ideal(ideal: RationalPolynomialIdeal) -> bool:
    return any(
        len(generator.polynomial.terms) == 1
        and not any(generator.polynomial.terms[0].exponents)
        and generator.polynomial.terms[0].coefficient.num != 0
        for generator in ideal.generators
    )


def _require_homogeneous(ideal: RationalPolynomialIdeal) -> None:
    if _is_explicit_unit_ideal(ideal):
        return
    try:
        _admit_source(ideal, label="graded ideal")
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


def _unit_initial_ideal(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"],
) -> InitialMonomialIdealResult:
    unit = _unit_monomial(ideal.variables, (0,) * len(ideal.variables))
    unit_ideal = RationalPolynomialIdeal(variables=ideal.variables, generators=(unit,))
    return InitialMonomialIdealResult(
        ideal=ideal,
        groebner_basis=unit_ideal,
        initial_ideal=unit_ideal,
        monomial_order=monomial_order,
    )


def initial_monomial_ideal(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
    *,
    resource_budget: IdealComputationBudget | None = None,
    _outer_deadline: float | None = None,
) -> InitialMonomialIdealResult:
    """Project the existing exact Gröbner result to its initial monomial ideal."""

    monomial_order = _require_monomial_order(monomial_order)
    _require_homogeneous(ideal)
    resource_budget = resource_budget or IdealComputationBudget()
    deadline = execution_deadline(float(resource_budget.wall_seconds))
    if _outer_deadline is not None:
        # Inherit an earlier absolute deadline so a native caller's shared wall
        # limit is not restarted by a fresh sub-window here.
        deadline = min(deadline, _outer_deadline)
    if _is_explicit_unit_ideal(ideal):
        require_execution_deadline(deadline)
        return _unit_initial_ideal(ideal, monomial_order)
    # Pass the already-bound absolute deadline to the nested Groebner call so a
    # native call without a request envelope does not restart a fresh window.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        require_execution_deadline(deadline)
    nested_budget = resource_budget.model_copy(
        update={"wall_seconds": max(1.0, remaining)}
    )
    basis_result = groebner_basis(ideal, monomial_order, resource_budget=nested_budget)
    require_execution_deadline(deadline)
    variables = ideal.variables
    order = {"lex": "lex", "grlex": "grlex", "grevlex": "grevlex"}[monomial_order]
    exponents: set[tuple[int, ...]] = set()
    zero_basis = False
    for generator in basis_result.basis.generators:
        if not generator.polynomial.terms:
            zero_basis = True
            continue
        leading_terms = generator.polynomial.terms
        if len(leading_terms) == 1 and not any(leading_terms[0].exponents):
            exponents.add(leading_terms[0].exponents)
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


def _divides_monomial(generator: tuple[int, ...], monomial: tuple[int, ...]) -> bool:
    return all(left <= right for left, right in zip(generator, monomial, strict=True))


def _prefix_already_nonstandard(
    prefix: list[int],
    generators: tuple[tuple[int, ...], ...],
    variable_count: int,
) -> bool:
    assigned = len(prefix)
    return any(
        all(generator[index] <= prefix[index] for index in range(assigned))
        and all(generator[index] == 0 for index in range(assigned, variable_count))
        for generator in generators
    )


def _enumerate_standard_monomials(
    degree: int,
    generators: tuple[tuple[int, ...], ...],
    caps: tuple[int | None, ...],
    deadline: float | None = None,
) -> Iterator[tuple[int, ...]]:
    variable_count = len(caps)
    if variable_count == 0:
        if degree == 0 and not any(not any(generator) for generator in generators):
            yield ()
        return
    prefix: list[int] = []
    steps = 0

    def visit(axis: int, remaining: int) -> Iterator[tuple[int, ...]]:
        nonlocal steps
        steps += 1
        if steps % 256 == 0:
            request_checkpoint("during standard-monomial composition enumeration")
            if deadline is not None:
                require_execution_deadline(deadline)
        if _prefix_already_nonstandard(prefix, generators, variable_count):
            return
        cap = caps[axis]
        if axis == variable_count - 1:
            maximum = remaining if cap is None else cap - 1
            if remaining < 0 or remaining > maximum:
                return
            monomial = (*prefix, remaining)
            if not any(
                _divides_monomial(generator, monomial) for generator in generators
            ):
                yield monomial
            return
        maximum = remaining if cap is None else min(remaining, cap - 1)
        for exponent in range(maximum + 1):
            prefix.append(exponent)
            yield from visit(axis + 1, remaining - exponent)
            prefix.pop()

    yield from visit(0, degree)


def standard_monomials(
    initial_ideal: RationalPolynomialIdeal,
    degree: int,
    *,
    _deadline: float | None = None,
) -> StandardMonomialsResult:
    if type(degree) is not int:
        raise OperationDomainValidationError(
            location=("degree",),
            code="graded_ideal.degree_type",
            message="standard-monomial degree must be an integer",
        )
    if degree < 0 or degree > MAX_GRADED_DEGREE:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="graded_ideal.monomial_degree_budget",
            message="standard monomials support degrees from 0 through 32",
        )
    generators = _require_monomial_ideal(initial_ideal)
    if any(not any(exponent) for exponent in generators):
        return StandardMonomialsResult(
            initial_ideal=initial_ideal,
            degree=degree,
            monomials=(),
            count=0,
        )
    variables = len(initial_ideal.variables)
    domain_size = _pruned_standard_monomial_bound(
        generators, variables, degree, _deadline
    )
    if domain_size > MAX_STANDARD_MONOMIALS:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="graded_ideal.monomial_domain_budget",
            message="standard-monomial domain exceeds the bounded enumeration envelope",
        )
    monomials = tuple(
        sorted(
            _enumerate_standard_monomials(
                degree, generators, _pure_power_caps(generators, variables), _deadline
            ),
            reverse=True,
        )
    )
    return StandardMonomialsResult(
        initial_ideal=initial_ideal,
        degree=degree,
        monomials=monomials,
        count=len(monomials),
    )


def _pure_power_caps(
    generators: tuple[tuple[int, ...], ...], variable_count: int
) -> tuple[int | None, ...]:
    caps: list[int | None] = [None] * variable_count
    for generator in generators:
        nonzero = [index for index, exponent in enumerate(generator) if exponent]
        if len(nonzero) != 1:
            continue
        axis = nonzero[0]
        current = caps[axis]
        caps[axis] = (
            generator[axis] if current is None else min(current, generator[axis])
        )
    return tuple(caps)


def _pruned_standard_monomial_bound(
    generators: tuple[tuple[int, ...], ...],
    variable_count: int,
    degree: int,
    deadline: float | None = None,
) -> int:
    if variable_count == 0:
        return 1 if degree == 0 else 0
    if any(not any(exponent) for exponent in generators):
        return 0
    caps = _pure_power_caps(generators, variable_count)
    ways = [0] * (degree + 1)
    ways[0] = 1
    for cap in caps:
        next_ways = [0] * (degree + 1)
        max_exponent = degree if cap is None else cap - 1
        for current in range(degree + 1):
            count = ways[current]
            if count == 0:
                continue
            limit = current + max_exponent
            if limit > degree:
                limit = degree
            for target in range(current, limit + 1):
                next_ways[target] += count
        ways = next_ways
    cap_bound = ways[degree]
    # Mixed generators (more than one nonzero axis) are not captured by the
    # per-axis caps. Count the degree slice exactly when the ambient slice is
    # small enough to enumerate; otherwise fall back to a support relaxation
    # that still accounts for the mixed constraints without the ambient count.
    ambient = comb(degree + variable_count - 1, variable_count - 1)
    if cap_bound <= MAX_STANDARD_MONOMIALS:
        return cap_bound
    # When every generator is a pure power, the per-axis caps are the exact
    # standard-monomial count, so an exact scan would only repeat the bound.
    if not any(
        sum(1 for exponent in generator if exponent) > 1 for generator in generators
    ):
        return cap_bound
    if ambient <= _MONOMIAL_ENUMERATION_CEILING:
        return _enumerated_standard_monomial_count(
            generators, variable_count, degree, deadline
        )
    relaxation = _support_relaxation_bound(generators, variable_count, degree)
    if relaxation is not None and relaxation < cap_bound:
        return relaxation
    return cap_bound


def _support_relaxation_bound(
    generators: tuple[tuple[int, ...], ...], variable_count: int, degree: int
) -> int | None:
    """Sound upper bound counting only support-avoiding monomials.

    For squarefree generators, a monomial is divisible by a generator exactly
    when its support contains the generator's support, so the standard
    monomials of degree ``d`` are those whose support contains no generator
    support. Summing compositions over those admissible supports is then an
    exact count. A non-squarefree generator has an exponent threshold, so
    support containment no longer implies divisibility and this relaxation is
    not an upper bound; return ``None`` so the caller keeps the sound ambient
    bound.
    """

    if any(any(exponent > 1 for exponent in generator) for generator in generators):
        return None
    if degree == 0:
        return 0 if any(not any(exponent) for exponent in generators) else 1
    if variable_count > 20:
        return None
    forbidden = tuple(
        frozenset(index for index, exponent in enumerate(generator) if exponent)
        for generator in generators
    )
    bound = 0
    for mask in range(1, 1 << variable_count):
        support = frozenset(
            index for index in range(variable_count) if mask >> index & 1
        )
        if any(generator_support <= support for generator_support in forbidden):
            continue
        # Positive compositions of ``degree`` into ``len(support)`` parts.
        size = len(support)
        if size <= degree:
            bound += comb(degree - 1, size - 1)
        if bound > MAX_STANDARD_MONOMIALS:
            return bound
    return bound


def _enumerated_standard_monomial_count(
    generators: tuple[tuple[int, ...], ...],
    variable_count: int,
    degree: int,
    deadline: float | None = None,
) -> int:
    """Count degree-slice monomials not divisible by any generator."""

    count = 0
    for index, composition in enumerate(
        _degree_compositions(variable_count, degree), start=1
    ):
        if index % 4096 == 0:
            request_checkpoint("during graded standard-monomial counting")
            if deadline is not None:
                require_execution_deadline(deadline)
        if not any(
            all(composition[axis] >= generator[axis] for axis in range(variable_count))
            for generator in generators
        ):
            count += 1
    return count


def _degree_compositions(variable_count: int, degree: int) -> Iterator[tuple[int, ...]]:
    """Yield monomial exponent tuples of one total degree."""

    if variable_count == 1:
        yield (degree,)
        return
    for first in range(degree + 1):
        for rest in _degree_compositions(variable_count - 1, degree - first):
            yield (first, *rest)


def _require_hilbert_function_slices(
    generators: tuple[tuple[int, ...], ...] | None,
    variable_count: int,
    max_degree: int,
    *,
    deadline: float,
) -> None:
    for degree in range(max_degree + 1):
        require_execution_deadline(deadline)
        if generators is None:
            domain_size = (
                1
                if variable_count == 0
                else comb(degree + variable_count - 1, variable_count - 1)
            )
        else:
            domain_size = _pruned_standard_monomial_bound(
                generators, variable_count, degree, deadline
            )
        if domain_size > MAX_STANDARD_MONOMIALS:
            raise OperationResourceAdmissionError(
                location=("max_degree",),
                code="graded_ideal.monomial_domain_budget",
                message="standard-monomial domain exceeds the bounded enumeration envelope",
            )


def _monomial_greater(
    left: tuple[int, ...],
    right: tuple[int, ...],
    monomial_order: Literal["lex", "grlex", "grevlex"],
) -> bool:
    if monomial_order == "lex":
        return left > right
    left_degree = sum(left)
    right_degree = sum(right)
    if left_degree != right_degree:
        return left_degree > right_degree
    if monomial_order == "grlex":
        return left > right
    for left_exponent, right_exponent in zip(
        reversed(left), reversed(right), strict=True
    ):
        if left_exponent != right_exponent:
            return left_exponent < right_exponent
    return False


def _leading_source_monomials(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"],
) -> tuple[tuple[int, ...], ...]:
    exponents: list[tuple[int, ...]] = []
    for generator in ideal.generators:
        terms = generator.polynomial.terms
        if not terms:
            continue
        leading = terms[0].exponents
        for term in terms[1:]:
            if _monomial_greater(term.exponents, leading, monomial_order):
                leading = term.exponents
        exponents.append(leading)
    unique = tuple(dict.fromkeys(exponents))
    return tuple(
        exponent
        for exponent in unique
        if not any(
            other != exponent
            and all(left <= right for left, right in zip(other, exponent, strict=True))
            for other in unique
        )
    )


def _all_source_generators_are_unit_monomials(ideal: RationalPolynomialIdeal) -> bool:
    for generator in ideal.generators:
        terms = generator.polynomial.terms
        if not terms:
            continue
        if len(terms) != 1 or terms[0].coefficient.num == 0:
            return False
    return True


def _preflight_monomial_series_generator_bound(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"],
) -> None:
    if not _all_source_generators_are_unit_monomials(ideal):
        return
    monomials = _leading_source_monomials(ideal, monomial_order)
    if len(monomials) > MAX_HILBERT_SERIES_GENERATORS:
        raise OperationResourceAdmissionError(
            location=("ideal",),
            code="graded_ideal.series_generator_budget",
            message="Hilbert-series inclusion-exclusion supports at most 8 minimal generators",
        )


def hilbert_function(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
    max_degree: int = 0,
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> HilbertFunctionResult:
    monomial_order = _require_monomial_order(monomial_order)
    if type(max_degree) is not int:
        raise OperationDomainValidationError(
            location=("max_degree",),
            code="graded_ideal.degree_type",
            message="Hilbert-function max_degree must be an integer",
        )
    if max_degree < 0 or max_degree > MAX_GRADED_DEGREE:
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="graded_ideal.function_degree_budget",
            message="Hilbert-function prefixes support degrees from 0 through 32",
        )
    resource_budget = resource_budget or IdealComputationBudget()
    deadline = execution_deadline(float(resource_budget.wall_seconds))
    if not _is_explicit_unit_ideal(ideal):
        # Validate source applicability and homogeneity before any kernel work,
        # so a nonhomogeneous presentation reports the domain error rather than
        # an oversized-computation resource error. The deadline is bound first
        # because slice admission may count monomials.
        _require_homogeneous(ideal)
        require_execution_deadline(deadline)
        # When every generator is already a monomial, the source leading terms
        # are the exact initial ideal and can decide the slice budget before the
        # Groebner step. Otherwise the source leadings are only an upper bound
        # that Groebner reduction can strengthen, so the computed initial ideal
        # decides instead.
        if _all_source_generators_are_unit_monomials(ideal):
            leading = _leading_source_monomials(ideal, monomial_order)
            if not any(not any(exponent) for exponent in leading):
                _require_hilbert_function_slices(
                    leading, len(ideal.variables), max_degree, deadline=deadline
                )
            initial = initial_monomial_ideal(
                ideal,
                monomial_order,
                resource_budget=resource_budget,
                _outer_deadline=deadline,
            )
            require_execution_deadline(deadline)
            return _hilbert_function_values(
                ideal, monomial_order, initial, max_degree, deadline
            )
    initial = initial_monomial_ideal(
        ideal, monomial_order, resource_budget=resource_budget, _outer_deadline=deadline
    )
    require_execution_deadline(deadline)
    computed_leading = _leading_monomials_of_ideal(initial.initial_ideal)
    if not any(not any(exponent) for exponent in computed_leading):
        _require_hilbert_function_slices(
            computed_leading,
            len(ideal.variables),
            max_degree,
            deadline=deadline,
        )
    return _hilbert_function_values(
        ideal, monomial_order, initial, max_degree, deadline
    )


def _hilbert_function_values(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"],
    initial: InitialMonomialIdealResult,
    max_degree: int,
    deadline: float,
) -> HilbertFunctionResult:
    values = []
    for degree in range(max_degree + 1):
        request_checkpoint("during Hilbert-function enumeration")
        require_execution_deadline(deadline)
        values.append(
            standard_monomials(initial.initial_ideal, degree, _deadline=deadline).count
        )
    require_execution_deadline(deadline)
    return HilbertFunctionResult(
        ideal=ideal,
        initial_ideal=initial.initial_ideal,
        monomial_order=monomial_order,
        values=tuple(values),
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
    deadline: float,
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
    request_checkpoint("during Hilbert-series cancellation")
    require_execution_deadline(deadline)
    t = Symbol("t")
    ambient_expression = rational_polynomial_to_sympy(ambient_numerator).as_expr()
    unreduced = ambient_expression / (1 - t) ** variable_count
    numerator_expression, denominator_expression = fraction(cancel(unreduced))
    request_checkpoint("during Hilbert-series degree inspection")
    require_execution_deadline(deadline)
    numerator_poly = Poly(numerator_expression, t, domain=QQ)
    denominator_poly = Poly(denominator_expression, t, domain=QQ)
    reduced_degree = max(
        0 if numerator_poly.is_zero else int(numerator_poly.degree()),
        0 if denominator_poly.is_zero else int(denominator_poly.degree()),
    )
    if reduced_degree > MAX_RATIONAL_FUNCTION_EXPONENT:
        raise OperationResourceAdmissionError(
            location=("initial_ideal",),
            code="graded_ideal.series_degree_budget",
            message="Hilbert-series numerator degree exceeds the rational-function envelope",
        )
    series = rational_function_from_sympy(
        unreduced,
        ("t",),
        maximum_terms=MAX_RATIONAL_FUNCTION_TERMS,
        deadline_check=lambda: require_execution_deadline(deadline),
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
    monomial_order = _require_monomial_order(monomial_order)
    if type(prefix_degree) is not int:
        raise OperationDomainValidationError(
            location=("prefix_degree",),
            code="graded_ideal.degree_type",
            message="Hilbert-series prefix_degree must be an integer",
        )
    if prefix_degree < 0 or prefix_degree > MAX_HILBERT_PREFIX:
        raise OperationResourceAdmissionError(
            location=("prefix_degree",),
            code="graded_ideal.series_prefix_budget",
            message="Hilbert-series prefixes support degrees from 0 through 16",
        )
    _preflight_monomial_series_generator_bound(ideal, monomial_order)
    resource_budget = resource_budget or IdealComputationBudget()
    deadline = execution_deadline(float(resource_budget.wall_seconds))
    initial = initial_monomial_ideal(
        ideal, monomial_order, resource_budget=resource_budget
    )
    require_execution_deadline(deadline)
    data = _series_data(initial.initial_ideal, prefix_degree, deadline)
    require_execution_deadline(deadline)
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
    monomial_order = _require_monomial_order(monomial_order)
    resource_budget = resource_budget or IdealComputationBudget()
    deadline = execution_deadline(float(resource_budget.wall_seconds))
    data = _series_projection(ideal, monomial_order, resource_budget=resource_budget)
    dimension = data.denominator_exponent
    h_degree = max(
        (term.exponents[0] for term in data.h_numerator.polynomial.terms),
        default=-1,
    )
    if dimension == 0:
        stabilization = max(0, h_degree + 1)
    else:
        stabilization = max(0, h_degree - dimension + 1)

    require_execution_deadline(deadline)
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
    monomial_order = _require_monomial_order(monomial_order)
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
    monomial_order = _require_monomial_order(monomial_order)
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
    monomial_order = _require_monomial_order(monomial_order)
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


def _leading_monomials_of_ideal(
    ideal: RationalPolynomialIdeal,
) -> tuple[tuple[int, ...], ...]:
    """Exponent vectors of a monomial ideal's generators (its leading terms)."""

    exponents: list[tuple[int, ...]] = []
    for generator in ideal.generators:
        terms = generator.polynomial.terms
        if not terms:
            continue
        exponents.append(terms[0].exponents)
    return tuple(dict.fromkeys(exponents))
