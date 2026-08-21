"""Exact SymPy-backed polynomial computations over ``QQ``."""

from __future__ import annotations

import contextlib
from typing import Any

from jacobian.math import polynomials
from jacobian.math.polynomials._conversions import (
    rational_from_sympy,
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials._models import (
    PolynomialBezoutIdentity,
    PolynomialDiscriminantRequest,
    PolynomialDiscriminantResult,
    PolynomialFactorizationResult,
    PolynomialFactorRequest,
    PolynomialGcdRequest,
    PolynomialGcdResult,
    PolynomialGroebnerBasisRequest,
    PolynomialGroebnerBasisResult,
    PolynomialInvariantValue,
    PolynomialIrreducibleFactor,
    PolynomialResultantRequest,
    PolynomialResultantResult,
    PolynomialScalarValue,
    PolynomialSquareFreeDecompositionResult,
    PolynomialSquareFreeFactor,
    PolynomialSquareFreeRequest,
    PolynomialValue,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    SparseRationalPolynomial,
)

_MAX_OUTPUT_TERMS = 1024


class PolynomialOutputBudgetError(RuntimeError):
    """A valid computation produced more output than its public contract permits."""


def _result_polynomial(poly: object, variables: tuple[str, ...]) -> RationalPolynomial:
    try:
        return rational_polynomial_from_sympy(
            poly,
            variables,
            maximum_terms=_MAX_OUTPUT_TERMS,
        )
    except ValueError as exc:
        if "term operation budget" in str(exc):
            raise PolynomialOutputBudgetError(str(exc)) from exc
        raise


def _invariant_value(
    expression: Any,
    remaining_variables: tuple[str, ...],
) -> PolynomialInvariantValue:
    from sympy import QQ, Poly

    if not remaining_variables:
        return PolynomialScalarValue(value=rational_from_sympy(expression))
    return PolynomialValue(
        value=_result_polynomial(
            Poly(expression, *symbols_for_variables(remaining_variables), domain=QQ),
            remaining_variables,
        )
    )


def polynomial_gcd(request: PolynomialGcdRequest) -> PolynomialGcdResult:
    left = rational_polynomial_to_sympy(request.left)
    right = rational_polynomial_to_sympy(request.right)
    left_multiplier, right_multiplier, gcd = polynomials.gcdex(left, right)
    variables = request.left.variables
    return PolynomialGcdResult(
        gcd=_result_polynomial(gcd, variables),
        bezout=PolynomialBezoutIdentity(
            left_multiplier=_result_polynomial(left_multiplier, variables),
            right_multiplier=_result_polynomial(right_multiplier, variables),
        ),
    )


def polynomial_resultant(
    request: PolynomialResultantRequest,
) -> PolynomialResultantResult:
    variables = request.left.variables
    elimination_index = variables.index(request.elimination_variable)
    generator = symbols_for_variables(variables)[elimination_index]
    value = polynomials.resultant(
        rational_polynomial_to_sympy(request.left),
        rational_polynomial_to_sympy(request.right),
        generator,
    )
    remaining_variables = tuple(
        variable for variable in variables if variable != request.elimination_variable
    )
    return PolynomialResultantResult(
        elimination_variable=request.elimination_variable,
        resultant=_invariant_value(value, remaining_variables),
    )


def polynomial_discriminant(
    request: PolynomialDiscriminantRequest,
) -> PolynomialDiscriminantResult:
    variables = request.polynomial.variables
    variable_index = variables.index(request.variable)
    generator = symbols_for_variables(variables)[variable_index]
    value = polynomials.discriminant(
        rational_polynomial_to_sympy(request.polynomial), generator
    )
    remaining_variables = tuple(
        variable for variable in variables if variable != request.variable
    )
    return PolynomialDiscriminantResult(
        variable=request.variable,
        discriminant=_invariant_value(value, remaining_variables),
    )


def polynomial_square_free_decomposition(
    request: PolynomialSquareFreeRequest,
) -> PolynomialSquareFreeDecompositionResult:
    source = rational_polynomial_to_sympy(request.polynomial)
    coefficient, canonical_factors, reconstructed = (
        polynomials.square_free_decomposition(source)
    )
    factors = tuple(
        PolynomialSquareFreeFactor(
            factor=_result_polynomial(factor, request.polynomial.variables),
            multiplicity=multiplicity,
        )
        for factor, multiplicity in sorted(canonical_factors, key=lambda item: item[1])
    )
    return PolynomialSquareFreeDecompositionResult(
        coefficient=rational_from_sympy(coefficient),
        factors=factors,
        reconstructed=_result_polynomial(reconstructed, request.polynomial.variables),
    )


def _irreducible_factor_sort_key(
    record: PolynomialIrreducibleFactor,
) -> tuple[int, int, tuple[tuple[tuple[int, ...], str, str], ...]]:
    return (
        record.multiplicity,
        max(
            (sum(term.exponents) for term in record.factor.polynomial.terms),
            default=0,
        ),
        tuple(
            (term.exponents, term.coefficient.num, term.coefficient.den)
            for term in record.factor.polynomial.terms
        ),
    )


def polynomial_factorization(
    request: PolynomialFactorRequest,
) -> PolynomialFactorizationResult:
    source = rational_polynomial_to_sympy(request.polynomial)
    coefficient, canonical_factors, reconstructed = polynomials.factorization(source)
    factors = tuple(
        sorted(
            (
                PolynomialIrreducibleFactor(
                    factor=_result_polynomial(factor, request.polynomial.variables),
                    multiplicity=multiplicity,
                )
                for factor, multiplicity in canonical_factors
            ),
            key=_irreducible_factor_sort_key,
        )
    )
    return PolynomialFactorizationResult(
        coefficient=rational_from_sympy(coefficient),
        factors=factors,
        reconstructed=_result_polynomial(reconstructed, request.polynomial.variables),
    )


def _groebner_worker(
    ideal_payload: dict[str, Any],
    monomial_order: str,
    queue: Any,
) -> None:
    """Entry point for the isolated Gröbner worker process."""

    try:
        from jacobian.math.polynomials.values import RationalPolynomialIdeal

        ideal = RationalPolynomialIdeal.model_validate(ideal_payload)
        variables = ideal.variables
        wire_polys = polynomials.groebner_basis(
            tuple(
                rational_polynomial_to_sympy(generator)
                for generator in ideal.generators
            ),
            symbols_for_variables(variables),
            monomial_order,
        )
        basis_payloads: list[dict[str, Any]] = []
        for poly in wire_polys:
            converted = rational_polynomial_from_sympy(
                poly,
                variables,
                maximum_terms=_MAX_OUTPUT_TERMS,
            )
            basis_payloads.append(converted.model_dump(mode="json"))
        queue.put(("ok", basis_payloads))
    except Exception as exc:
        queue.put(("error", str(exc)))


def _run_groebner_isolated(
    ideal: RationalPolynomialIdeal,
    monomial_order: str,
    wall_seconds: int,
) -> tuple[str, Any]:
    """Run the SymPy Groebner worker with a killable wall-time limit."""

    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    queue: Any = ctx.Queue()
    payload = ideal.model_dump(mode="json")
    process = ctx.Process(
        target=_groebner_worker,
        args=(payload, monomial_order, queue),
    )
    process.start()
    process.join(timeout=float(wall_seconds))
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            with contextlib.suppress(AttributeError):
                process.kill()
            process.join(timeout=1.0)
        return "timeout", None
    if process.exitcode not in (0, None):
        detail = "Gröbner basis worker exited unexpectedly."
        try:
            if not queue.empty():
                status, data = queue.get_nowait()
                if status == "error":
                    detail = str(data)
        except Exception:
            pass
        return "error", detail
    try:
        status, data = queue.get_nowait()
    except Exception:
        return "error", "Gröbner basis worker produced no result."
    if status == "error":
        return "error", str(data)
    return "ok", data


def _check_groebner_output_budget(
    wire_basis: tuple[RationalPolynomial, ...],
    maximum_basis_polynomials: int,
    maximum_output_terms: int,
) -> str | None:
    if len(wire_basis) > maximum_basis_polynomials:
        return "Gröbner basis exceeds the requested polynomial-count limit."
    if (
        sum(len(polynomial.polynomial.terms) for polynomial in wire_basis)
        > maximum_output_terms
    ):
        return "Gröbner basis exceeds the requested aggregate term limit."
    return None


def polynomial_groebner_basis(
    request: PolynomialGroebnerBasisRequest,
) -> PolynomialGroebnerBasisResult:
    """Compute one bounded Gröbner basis with typed wall-time and output limits."""

    ideal = request.ideal
    budget = request.resource_budget
    variables = ideal.variables
    try:
        status, data = _run_groebner_isolated(
            ideal, request.monomial_order, budget.wall_seconds
        )
    except Exception as exc:
        return PolynomialGroebnerBasisResult(
            outcome="LIMIT_EXCEEDED",
            ideal=ideal,
            monomial_order=request.monomial_order,
            detail=str(exc),
        )
    if status == "timeout":
        return PolynomialGroebnerBasisResult(
            outcome="TIMEOUT",
            ideal=ideal,
            monomial_order=request.monomial_order,
            detail="Gröbner basis computation exceeded the wall-time budget.",
        )
    if status == "error":
        return PolynomialGroebnerBasisResult(
            outcome="LIMIT_EXCEEDED",
            ideal=ideal,
            monomial_order=request.monomial_order,
            detail=str(data),
        )
    wire_basis_dicts: list[dict[str, Any]] = data
    wire_basis = tuple(
        RationalPolynomial.model_validate(item) for item in wire_basis_dicts
    )
    budget_error = _check_groebner_output_budget(
        wire_basis,
        budget.maximum_basis_polynomials,
        budget.maximum_output_terms,
    )
    if budget_error is not None:
        return PolynomialGroebnerBasisResult(
            outcome="LIMIT_EXCEEDED",
            ideal=ideal,
            monomial_order=request.monomial_order,
            detail=budget_error,
        )
    if not wire_basis:
        zero = RationalPolynomial(
            variables=variables,
            polynomial=SparseRationalPolynomial(terms=()),
        )
        basis_ideal = RationalPolynomialIdeal(variables=variables, generators=(zero,))
    else:
        basis_ideal = RationalPolynomialIdeal(
            variables=variables, generators=wire_basis
        )
    return PolynomialGroebnerBasisResult(
        outcome="COMPUTED",
        ideal=ideal,
        monomial_order=request.monomial_order,
        basis=basis_ideal,
    )
