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
    CompositionRequest,
    CompositionResult,
    EvalRequest,
    EvalResult,
    GenericDegreeOutcome,
    GenericDegreeRequest,
    GenericDegreeResult,
    JacobianResult,
    _total_degree,
    _validation_error,
)
from jacobian.math.polynomials.maps._singular import run_singular_generic_fiber
from jacobian.math.polynomials.maps.values import (
    MAX_MAP_POLYNOMIAL_TERMS,
    RationalPolynomialMap,
    require_map_polynomial,
)
from jacobian.math.polynomials.values import rational_evaluation_component_digit_bounds


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


def _admit_evaluation(request: EvalRequest) -> None:
    require_map_polynomial(request.polynomial, label="evaluation polynomial")
    if request.point.variables != request.polynomial.variables:
        raise _validation_error(
            "evaluation point must use the polynomial's complete ordered axis"
        )
    numerator_digits, denominator_digits = rational_evaluation_component_digit_bounds(
        request.polynomial,
        request.point.values,
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


def _admit_composition(request: CompositionRequest) -> None:
    require_map_polynomial(request.outer, label="outer polynomial")
    require_map_polynomial(request.inner, label="inner polynomial")
    if request.outer.variables != (request.outer_variable,):
        raise _validation_error("outer polynomial must use exactly outer_variable")
    if request.inner.variables != (request.inner_variable,):
        raise _validation_error("inner polynomial must use exactly inner_variable")
    outer_degree = max(
        (term.exponents[0] for term in request.outer.polynomial.terms), default=0
    )
    inner_degree = max(
        (term.exponents[0] for term in request.inner.polynomial.terms), default=0
    )
    if outer_degree * inner_degree > _MAX_COMPOSITION_DEGREE:
        raise _validation_error(f"composition exceeds degree {_MAX_COMPOSITION_DEGREE}")


def _admit_generic_degree(request: GenericDegreeRequest) -> None:
    polynomial_map = request.polynomial_map
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


def compute_generic_degree(request: GenericDegreeRequest) -> GenericDegreeResult:
    """Compute the exact degree of the map's generic scheme-theoretic fiber."""

    _run_admission(lambda: _admit_generic_degree(request))
    backend = run_singular_generic_fiber(
        request.polynomial_map,
        request.resource_budget,
    )
    if backend.outcome != "COMPUTED":
        operational_outcome = (
            "BOUND_EXCEEDED" if backend.outcome == "LIMIT_EXCEEDED" else backend.outcome
        )
        return GenericDegreeResult(
            outcome=operational_outcome,
            source=request.polynomial_map,
            detail=backend.detail,
        )
    if backend.certificate is None or backend.dimension is None:
        return GenericDegreeResult(
            outcome="ERROR",
            source=request.polynomial_map,
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
                source=request.polynomial_map,
                detail="Singular returned a finite fiber without its exact degree.",
            )
        mathematical_outcome = "GENERICALLY_FINITE"
        degree = backend.vector_dimension
    else:
        mathematical_outcome = "DOMINANT_NOT_GENERICALLY_FINITE"
        degree = None
    return GenericDegreeResult._from_kernel(
        outcome=mathematical_outcome,
        source=request.polynomial_map,
        degree=degree,
        evidence=backend.certificate,
        detail=None,
    )


def evaluate_polynomial(request: EvalRequest) -> EvalResult:
    """Evaluate one exact polynomial at its complete ordered rational point."""

    _run_admission(lambda: _admit_evaluation(request))
    polynomial = rational_polynomial_to_sympy(request.polynomial)
    substitutions = dict(
        zip(
            symbols_for_variables(request.point.variables),
            (value.as_fraction() for value in request.point.values),
            strict=True,
        )
    )
    value = polynomial.as_expr().subs(substitutions)
    return EvalResult(value=rational_from_sympy(value))


def compute_jacobian(request: RationalPolynomialMap) -> JacobianResult:
    """Compute a row-major Jacobian over the map's source ring."""

    variables = symbols_for_variables(request.input_variables)
    outputs = [
        rational_polynomial_to_sympy(polynomial).as_expr()
        for polynomial in request.output_polynomials
    ]
    entries = tuple(
        rational_polynomial_from_sympy(
            sympy.Poly(sympy.diff(output, variable), *variables, domain=sympy.QQ),
            request.input_variables,
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


def compose_polynomials(request: CompositionRequest) -> CompositionResult:
    """Substitute the inner univariate polynomial into the outer polynomial."""

    _run_admission(lambda: _admit_composition(request))
    outer = rational_polynomial_to_sympy(request.outer).as_expr()
    inner = rational_polynomial_to_sympy(request.inner).as_expr()
    outer_variable = symbols_for_variables(request.outer.variables)[0]
    inner_variable = symbols_for_variables(request.inner.variables)[0]
    composition = sympy.Poly(
        sympy.expand(outer.subs(outer_variable, inner)),
        inner_variable,
        domain=sympy.QQ,
    )
    return CompositionResult(
        polynomial=rational_polynomial_from_sympy(
            composition,
            request.inner.variables,
            maximum_terms=MAX_MAP_POLYNOMIAL_TERMS,
        )
    )


__all__ = [
    "compose_polynomials",
    "compute_generic_degree",
    "compute_jacobian",
    "evaluate_polynomial",
]
