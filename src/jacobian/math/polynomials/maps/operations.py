"""Domain-owned polynomial map operations backed by SymPy."""

from __future__ import annotations

from collections.abc import Callable

import sympy
from pydantic import ValidationError
from pydantic_core import PydanticCustomError, PydanticSerializationError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.backends import BackendUnavailableError
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._conversions import (
    rational_from_sympy,
    rational_function_to_sympy,
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials._elementary_kernel import rational_polynomial_evaluate
from jacobian.math.polynomials.maps._models import (
    _MAX_COMPOSITION_DEGREE,
    MAX_GENERIC_DEGREE_AGGREGATE_TERMS,
    MAX_GENERIC_DEGREE_BEZOUT_BOUND,
    MAX_GENERIC_DEGREE_COEFFICIENT_DIGITS,
    MAX_GENERIC_DEGREE_COMPONENT_TERMS,
    MAX_GENERIC_DEGREE_SOURCE_VARIABLES,
    MAX_GENERIC_DEGREE_TARGET_VARIABLES,
    MAX_GENERIC_DEGREE_TOTAL_DEGREE,
    CompositionResult,
    EvalResult,
    GenericDegreeComputationBudget,
    GenericDegreeOutcome,
    GenericDegreeResult,
    GenericFiberPolynomial,
    JacobianResult,
    VariablePoint,
    _is_unit_generic_fiber_basis,
    _total_degree,
    _validation_error,
)
from jacobian.math.polynomials.maps._singular import run_singular_generic_fiber
from jacobian.math.polynomials.maps.values import (
    MAX_MAP_POLYNOMIAL_TERMS,
    PolynomialJacobianMatrix,
    RationalPolynomialMap,
    require_map_polynomial,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    MAX_POLYNOMIAL_VARIABLES,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
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
    if (
        not polynomial.variables
        or point.variables != polynomial.variables
        or len(point.values) != len(polynomial.variables)
    ):
        raise _validation_error(
            "evaluation point must use the polynomial's complete ordered axis"
        )
    if len(polynomial.variables) == 1:
        # The existing scalar evaluator owns this regime's source and output
        # admission. Its degree-127/256-digit envelope contains every old
        # one-axis map case (degree <=64, hence <=65 distinct terms, and
        # 128-digit coefficients). No map-admission cap is needed here.
        return
    require_map_polynomial(polynomial, label="evaluation polynomial")
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
    if outer.variables != (outer_variable,):
        raise _validation_error("outer polynomial must use exactly outer_variable")
    if inner.variables != (inner_variable,):
        raise _validation_error("inner polynomial must use exactly inner_variable")
    if _is_identity_polynomial(outer) or _is_identity_polynomial(inner):
        # Identity substitution retains exactly the other canonical operand:
        # no coefficient arithmetic, support expansion, or backend allocation.
        # Only bounded canonical recognition and delivery remain; the general
        # substitution's smaller support/height/degree caps do not constrain
        # this regime.
        return
    require_map_polynomial(outer, label="outer polynomial")
    require_map_polynomial(inner, label="inner polynomial")
    outer_degree = max(
        (term.exponents[0] for term in outer.polynomial.terms), default=0
    )
    inner_degree = max(
        (term.exponents[0] for term in inner.polynomial.terms), default=0
    )
    if outer_degree * inner_degree > _MAX_COMPOSITION_DEGREE:
        raise _validation_error(f"composition exceeds degree {_MAX_COMPOSITION_DEGREE}")


def _is_identity_polynomial(polynomial: RationalPolynomial) -> bool:
    terms = polynomial.polynomial.terms
    return (
        len(terms) == 1
        and terms[0].exponents == (1,)
        and terms[0].coefficient.num == terms[0].coefficient.den == 1
    )


def _composition_operand(
    polynomial: RationalPolynomial, *, label: str
) -> RationalPolynomial:
    """Recognize native canonical values before the no-arithmetic path."""
    try:
        # Bound every traversed native container before model_dump materializes
        # it. Forged models need not satisfy any constructor/schema limits.
        if not isinstance(polynomial, RationalPolynomial):
            raise ValueError("expected a rational polynomial")
        domain = getattr(polynomial, "domain", None)
        variables = getattr(polynomial, "variables", None)
        sparse = getattr(polynomial, "polynomial", None)
        if (
            type(domain) is not str
            or domain != "QQ"
            or not isinstance(variables, tuple)
            or len(variables) > MAX_POLYNOMIAL_VARIABLES
            or any(type(name) is not str or len(name) > 32 for name in variables)
            or not isinstance(sparse, SparseRationalPolynomial)
        ):
            raise ValueError("invalid polynomial shape")
        terms = getattr(sparse, "terms", None)
        if not isinstance(terms, tuple) or len(terms) > MAX_POLYNOMIAL_TERMS:
            raise ValueError("invalid polynomial support")
        for term in terms:
            if not isinstance(term, RationalPolynomialTerm):
                raise ValueError("invalid polynomial term")
            exponents = getattr(term, "exponents", None)
            coefficient = getattr(term, "coefficient", None)
            if (
                not isinstance(exponents, tuple)
                or len(exponents) > MAX_POLYNOMIAL_VARIABLES
                or any(
                    type(exponent) is not int
                    or not 0 <= exponent <= MAX_POLYNOMIAL_EXPONENT
                    for exponent in exponents
                )
                or not isinstance(coefficient, CanonicalRational)
            ):
                raise ValueError("invalid monomial shape")
            for component in (
                getattr(coefficient, "num", None),
                getattr(coefficient, "den", None),
            ):
                # 10**d < 2**(4*d): a cheap preflight before exact scalar
                # validation and fraction reduction, which enforce the tight cap.
                if (
                    type(component) is not int
                    or component.bit_length() > 4 * MAX_CANONICAL_RATIONAL_DIGITS
                ):
                    raise ValueError("invalid rational component")
        # model_copy/model_construct values are caller-authored at a native
        # boundary. Reparse numeric leaves; never normalize malformed input by
        # returning it unchanged or letting a symbolic conversion repair it.
        return RationalPolynomial.model_validate(polynomial.model_dump())
    except (ValidationError, PydanticSerializationError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=(label,),
            code="polynomial.composition_operand",
            message="composition requires a canonical rational polynomial",
        ) from exc


def _admit_generic_degree(polynomial_map: RationalPolynomialMap) -> None:
    source_count = len(polynomial_map.input_variables)
    target_count = len(polynomial_map.output_polynomials)
    if target_count == 0:
        raise _validation_error("generic degree requires at least one target component")
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
                len(format_canonical_integer(abs(term.coefficient.num)))
                > MAX_GENERIC_DEGREE_COEFFICIENT_DIGITS
                or len(format_canonical_integer(term.coefficient.den))
                > MAX_GENERIC_DEGREE_COEFFICIENT_DIGITS
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
        detail = backend.detail or "generic-degree backend did not produce a result"
        if backend.outcome == "UNAVAILABLE":
            raise BackendUnavailableError("singular", detail=detail)
        if backend.outcome == "TIMEOUT":
            raise OperationExecutionTimeoutError(detail)
        if backend.outcome == "CANCELLED":
            raise OperationExecutionCancelledError(detail)
        raise RuntimeError(detail)
    if backend.certificate is None or backend.dimension is None:
        raise RuntimeError("Singular returned incomplete generic-fiber evidence")
    mathematical_outcome: GenericDegreeOutcome
    if backend.dimension == -1:
        mathematical_outcome = "NOT_DOMINANT"
        degree = None
    elif backend.dimension == 0:
        if backend.vector_dimension is None:
            raise RuntimeError(
                "Singular returned a finite fiber without its exact degree"
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
    )


def _generic_fiber_polynomial_expression(
    polynomial: GenericFiberPolynomial,
    source_symbols: tuple[sympy.Symbol, ...],
    parameter_symbols: tuple[sympy.Symbol, ...],
) -> sympy.Expr:
    """Convert bounded generic-fiber evidence without parsing caller text."""

    expression = sympy.Integer(0)
    for term in polynomial.terms:
        coefficient = rational_function_to_sympy(term.coefficient)
        substitutions = dict(
            zip(
                symbols_for_variables(term.coefficient.variables),
                parameter_symbols,
                strict=True,
            )
        )
        coefficient = coefficient.xreplace(substitutions)
        monomial = sympy.prod(
            symbol**exponent
            for symbol, exponent in zip(
                source_symbols, term.source_exponents, strict=True
            )
        )
        expression += coefficient * monomial
    return sympy.cancel(expression)


def _generic_fiber_source_relations_hold(claim: GenericDegreeResult) -> bool:
    evidence = claim.evidence
    if evidence is None:
        return False
    if (
        evidence.source_variable_order != claim.source.input_variables
        or len(evidence.target_parameters) != len(claim.source.output_polynomials)
        or len(evidence.basis_from_source) != len(claim.source.output_polynomials)
    ):
        return False
    source_symbols = tuple(
        sympy.Dummy(f"source_{index}")
        for index in range(len(claim.source.input_variables))
    )
    parameter_symbols = tuple(
        sympy.Dummy(f"target_{index}")
        for index in range(len(evidence.target_parameters))
    )
    source_name_symbols = symbols_for_variables(claim.source.input_variables)
    source_generators = []
    for polynomial, parameter in zip(
        claim.source.output_polynomials, parameter_symbols, strict=True
    ):
        expression = (
            rational_polynomial_to_sympy(polynomial)
            .as_expr()
            .xreplace(dict(zip(source_name_symbols, source_symbols, strict=True)))
        )
        source_generators.append(expression - parameter)
    for basis_index, basis_polynomial in enumerate(evidence.basis):
        expected = sum(
            source_generator
            * _generic_fiber_polynomial_expression(
                evidence.basis_from_source[source_index][basis_index],
                source_symbols,
                parameter_symbols,
            )
            for source_index, source_generator in enumerate(source_generators)
        )
        actual = _generic_fiber_polynomial_expression(
            basis_polynomial, source_symbols, parameter_symbols
        )
        if sympy.cancel(actual - expected) != 0:
            return False
    return True


def verify_generic_degree(claim: GenericDegreeResult) -> bool:
    """Verify the retained generic-fiber evidence and declared outcome."""
    evidence = claim.evidence
    if evidence is None or not _generic_fiber_source_relations_hold(claim):
        return False
    unit_basis = _is_unit_generic_fiber_basis(evidence)
    if claim.outcome == "GENERICALLY_FINITE":
        return (
            not unit_basis
            and bool(evidence.standard_monomials)
            and claim.degree == len(evidence.standard_monomials)
        )
    if claim.degree is not None:
        return False
    if claim.outcome == "NOT_DOMINANT":
        return unit_basis and not evidence.standard_monomials
    return not unit_basis and bool(evidence.standard_monomials)


def evaluate_polynomial(
    polynomial: RationalPolynomial, point: VariablePoint
) -> EvalResult:
    """Evaluate one exact polynomial at its complete ordered rational point."""

    _run_admission(lambda: _admit_evaluation(polynomial, point))
    if len(polynomial.variables) == 1:
        return EvalResult(
            value=rational_polynomial_evaluate(polynomial, point.values[0]).value
        )
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
        tuple(
            rational_polynomial_from_sympy(
                sympy.Poly(sympy.diff(output, variable), *variables, domain=sympy.QQ),
                polynomial_map.input_variables,
                maximum_terms=MAX_MAP_POLYNOMIAL_TERMS,
            )
            for variable in variables
        )
        for output in outputs
    )
    return JacobianResult(
        source=polynomial_map,
        matrix=PolynomialJacobianMatrix(
            input_variables=polynomial_map.input_variables,
            entries=entries,
        ),
    )


def verify_jacobian(claim: JacobianResult) -> bool:
    """Verify Jacobian entries against the retained polynomial map source."""

    if not isinstance(claim, JacobianResult):
        return False
    try:
        return jacobian_matrix(claim.source) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def compose_polynomials(
    outer: RationalPolynomial,
    inner: RationalPolynomial,
    *,
    outer_variable: str,
    inner_variable: str,
) -> CompositionResult:
    """Substitute the inner univariate polynomial into the outer polynomial."""

    outer = _composition_operand(outer, label="outer")
    inner = _composition_operand(inner, label="inner")
    _run_admission(
        lambda: _admit_composition(outer, inner, outer_variable, inner_variable)
    )
    if _is_identity_polynomial(outer):
        return CompositionResult(polynomial=inner)
    if _is_identity_polynomial(inner):
        return CompositionResult(
            polynomial=RationalPolynomial(
                variables=inner.variables, polynomial=outer.polynomial
            )
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
    "verify_generic_degree",
    "verify_jacobian",
]
