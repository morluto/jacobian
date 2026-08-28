"""Domain-owned polynomial map operations backed by SymPy."""

from __future__ import annotations

from collections.abc import Callable

import sympy
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._conversions import (
    rational_from_sympy,
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.maps._models import (
    _MAX_COMPOSITION_DEGREE,
    MAX_GENERIC_DEGREE_AGGREGATE_TERMS,
    MAX_GENERIC_DEGREE_BEZOUT_BOUND,
    MAX_GENERIC_DEGREE_COEFFICIENT_DIGITS,
    MAX_GENERIC_DEGREE_COMPONENT_TERMS,
    MAX_GENERIC_DEGREE_ENCODED_MAP_BYTES,
    MAX_GENERIC_DEGREE_SOURCE_VARIABLES,
    MAX_GENERIC_DEGREE_TARGET_VARIABLES,
    MAX_GENERIC_DEGREE_TOTAL_DEGREE,
    CompositionResult,
    EvalResult,
    GenericDegreeComputationBudget,
    GenericDegreeOutcome,
    GenericDegreeResult,
    JacobianResult,
    VariablePoint,
    _total_degree,
    _validation_error,
)
from jacobian.math.polynomials.maps._singular import run_singular_generic_fiber
from jacobian.math.polynomials.maps.values import (
    MAX_MAP_POLYNOMIAL_TERMS,
    RationalPolynomialMap,
    require_map_polynomial,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    rational_evaluation_component_digit_bounds,
)


def _run_admission(admission: Callable[[], object]) -> None:
    try:
        admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=(), code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=(), code="polynomial.map_admission", message=str(exc)
        ) from exc


def _admit_evaluation(polynomial: RationalPolynomial, point: VariablePoint) -> None:
    require_map_polynomial(polynomial, label="evaluation polynomial")
    if point.variables != polynomial.variables:
        raise _validation_error(
            "evaluation point must use the polynomial's complete ordered axis"
        )
    numerator_digits, denominator_digits = rational_evaluation_component_digit_bounds(
        polynomial,
        point.values,
    )
    if max(numerator_digits, denominator_digits) > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("polynomial", "point"),
            code="polynomial.evaluation_result_exceeds_component_bound",
            message=(
                "exact evaluation exceeds the "
                f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit rational component bound"
            ),
        )


def _admit_composition(
    outer: RationalPolynomial,
    inner: RationalPolynomial,
    outer_variable: str,
    inner_variable: str,
) -> None:
    require_map_polynomial(outer, label="outer polynomial")
    require_map_polynomial(inner, label="inner polynomial")
    if outer.variables != (outer_variable,):
        raise _validation_error("outer polynomial must use exactly outer_variable")
    if inner.variables != (inner_variable,):
        raise _validation_error("inner polynomial must use exactly inner_variable")
    outer_degree = max(
        (term.exponents[0] for term in outer.polynomial.terms), default=0
    )
    inner_degree = max(
        (term.exponents[0] for term in inner.polynomial.terms), default=0
    )
    if outer_degree * inner_degree > _MAX_COMPOSITION_DEGREE:
        raise _validation_error(f"composition exceeds degree {_MAX_COMPOSITION_DEGREE}")


def _admit_generic_degree(polynomial_map: RationalPolynomialMap) -> None:
    source_count = len(polynomial_map.input_variables)
    target_count = len(polynomial_map.output_polynomials)
    if source_count > MAX_GENERIC_DEGREE_SOURCE_VARIABLES:
        raise _validation_error(
            "generic-degree source exceeds the "
            f"{MAX_GENERIC_DEGREE_SOURCE_VARIABLES}-variable operation budget"
        )
    if target_count > MAX_GENERIC_DEGREE_TARGET_VARIABLES:
        raise _validation_error(
            "generic-degree target exceeds the "
            f"{MAX_GENERIC_DEGREE_TARGET_VARIABLES}-component operation budget"
        )
    aggregate_terms = sum(
        len(polynomial.polynomial.terms)
        for polynomial in polynomial_map.output_polynomials
    )
    if aggregate_terms > MAX_GENERIC_DEGREE_AGGREGATE_TERMS:
        raise _validation_error(
            "generic-degree map exceeds the "
            f"{MAX_GENERIC_DEGREE_AGGREGATE_TERMS}-term aggregate input budget"
        )
    if (
        len(polynomial_map.model_dump_json().encode("utf-8"))
        > MAX_GENERIC_DEGREE_ENCODED_MAP_BYTES
    ):
        raise _validation_error(
            "generic-degree map exceeds the "
            f"{MAX_GENERIC_DEGREE_ENCODED_MAP_BYTES}-byte input budget"
        )
    degrees: list[int] = []
    for polynomial in polynomial_map.output_polynomials:
        if len(polynomial.polynomial.terms) > MAX_GENERIC_DEGREE_COMPONENT_TERMS:
            raise _validation_error(
                "generic-degree component exceeds the "
                f"{MAX_GENERIC_DEGREE_COMPONENT_TERMS}-term input budget"
            )
        degree = _total_degree(polynomial)
        degrees.append(degree)
        if degree > MAX_GENERIC_DEGREE_TOTAL_DEGREE:
            raise _validation_error(
                "generic-degree component exceeds total degree "
                f"{MAX_GENERIC_DEGREE_TOTAL_DEGREE}"
            )
        for term in polynomial.polynomial.terms:
            if (
                len(term.coefficient.num.lstrip("-"))
                > MAX_GENERIC_DEGREE_COEFFICIENT_DIGITS
                or len(term.coefficient.den) > MAX_GENERIC_DEGREE_COEFFICIENT_DIGITS
            ):
                raise _validation_error(
                    "generic-degree coefficient exceeds the "
                    f"{MAX_GENERIC_DEGREE_COEFFICIENT_DIGITS}-digit input budget"
                )
    if target_count >= source_count:
        bezout_bound = 1
        for degree in sorted(degrees)[:source_count]:
            bezout_bound *= max(1, degree)
        if bezout_bound > MAX_GENERIC_DEGREE_BEZOUT_BOUND:
            raise _validation_error(
                "generic-degree finite-fiber Bezout bound exceeds "
                f"{MAX_GENERIC_DEGREE_BEZOUT_BOUND}"
            )


def generic_degree(
    polynomial_map: RationalPolynomialMap,
    resource_budget: GenericDegreeComputationBudget,
) -> GenericDegreeResult:
    """Compute the exact degree of the map's generic scheme-theoretic fiber."""

    _run_admission(lambda: _admit_generic_degree(polynomial_map))
    backend = run_singular_generic_fiber(
        polynomial_map,
        resource_budget,
    )
    if backend.outcome != "COMPUTED":
        operational_outcome = (
            "BOUND_EXCEEDED" if backend.outcome == "LIMIT_EXCEEDED" else backend.outcome
        )
        return GenericDegreeResult(
            outcome=operational_outcome,
            source=polynomial_map,
            detail=backend.detail,
        )
    if backend.certificate is None or backend.dimension is None:
        return GenericDegreeResult(
            outcome="ERROR",
            source=polynomial_map,
            detail="Singular returned incomplete generic-fiber evidence.",
        )
    mathematical_outcome: GenericDegreeOutcome
    if backend.dimension == -1:
        mathematical_outcome = "NOT_DOMINANT"
        degree = None
    elif backend.dimension == 0:
        if backend.vector_dimension is None:
            return GenericDegreeResult(
                outcome="ERROR",
                source=polynomial_map,
                detail="Singular returned a finite fiber without its exact degree.",
            )
        mathematical_outcome = "GENERICALLY_FINITE"
        degree = backend.vector_dimension
    else:
        mathematical_outcome = "DOMINANT_NOT_GENERICALLY_FINITE"
        degree = None
    return GenericDegreeResult._from_kernel(
        outcome=mathematical_outcome,
        source=polynomial_map,
        degree=degree,
        evidence=backend.certificate,
        detail=None,
    )


def evaluate_polynomial(
    polynomial: RationalPolynomial, point: VariablePoint
) -> EvalResult:
    """Evaluate one exact polynomial at its complete ordered rational point."""

    _run_admission(lambda: _admit_evaluation(polynomial, point))
    backend_polynomial = rational_polynomial_to_sympy(polynomial)
    substitutions = dict(
        zip(
            symbols_for_variables(point.variables),
            (value.as_fraction() for value in point.values),
            strict=True,
        )
    )
    value = backend_polynomial.as_expr().subs(substitutions)
    return EvalResult(value=rational_from_sympy(value))


def jacobian_matrix(polynomial_map: RationalPolynomialMap) -> JacobianResult:
    """Compute a row-major Jacobian over the map's source ring."""

    variables = symbols_for_variables(polynomial_map.input_variables)
    outputs = [
        rational_polynomial_to_sympy(polynomial).as_expr()
        for polynomial in polynomial_map.output_polynomials
    ]
    entries = tuple(
        rational_polynomial_from_sympy(
            sympy.Poly(sympy.diff(output, variable), *variables, domain=sympy.QQ),
            polynomial_map.input_variables,
            maximum_terms=MAX_MAP_POLYNOMIAL_TERMS,
        )
        for output in outputs
        for variable in variables
    )
    return JacobianResult(
        n_inputs=len(variables),
        n_outputs=len(outputs),
        entries=entries,
    )


def compose_polynomials(
    outer: RationalPolynomial,
    inner: RationalPolynomial,
    *,
    outer_variable: str,
    inner_variable: str,
) -> CompositionResult:
    """Substitute the inner univariate polynomial into the outer polynomial."""

    _run_admission(
        lambda: _admit_composition(outer, inner, outer_variable, inner_variable)
    )
    outer_expression = rational_polynomial_to_sympy(outer).as_expr()
    inner_expression = rational_polynomial_to_sympy(inner).as_expr()
    outer_symbol = symbols_for_variables(outer.variables)[0]
    inner_symbol = symbols_for_variables(inner.variables)[0]
    composition = sympy.Poly(
        sympy.expand(outer_expression.subs(outer_symbol, inner_expression)),
        inner_symbol,
        domain=sympy.QQ,
    )
    return CompositionResult(
        polynomial=rational_polynomial_from_sympy(
            composition,
            inner.variables,
            maximum_terms=MAX_MAP_POLYNOMIAL_TERMS,
        )
    )


__all__ = [
    "compose_polynomials",
    "evaluate_polynomial",
    "generic_degree",
    "jacobian_matrix",
]
