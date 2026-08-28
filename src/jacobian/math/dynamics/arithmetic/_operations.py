"""Wire adapters for exact bounded arithmetic dynamics."""

from __future__ import annotations

from typing import Any, NoReturn

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math._rational_height import RationalHeight
from jacobian.math.dynamics.arithmetic._models import (
    MAX_DYNATOMIC_DEGREE,
    MAX_ITERATE_DEGREE,
    CoefficientHeight,
    CycleMultiplierRequest,
    CycleMultiplierResult,
    DynatomicPolynomialRequest,
    DynatomicPolynomialResult,
    FiniteFieldMapRequest,
    FiniteFieldMapResult,
    MapIterateRequest,
    MapIterateResult,
    OrbitPrefixRequest,
    OrbitPrefixResult,
    OrbitRepeatEvidence,
    PolynomialCoefficientRequest,
    _add_heights,
    _divide_height_polynomials,
    _fraction_height,
    _iterate_heights,
    _mobius,
    _multiply_height_polynomials,
    _require_polynomial_height,
    _validation_code,
)
from jacobian.math.dynamics.arithmetic.operations import (
    cycle_multiplier,
    dynatomic_polynomial,
    finite_field_functional_graph,
    iterate_polynomial,
    orbit_prefix,
    polynomial_coefficients,
    polynomial_from_coefficients,
)


def _domain_error(location: tuple[str | int, ...], code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=location, code=f"arithmetic_dynamics.{code}", message=message
    )


def _translate_value_error(exc: ValueError, location: tuple[str | int, ...]) -> None:
    message = str(exc)
    _domain_error(location, _validation_code(message), message)


def _polynomial(request: PolynomialCoefficientRequest) -> Any:
    try:
        values = request.coefficient_values()
        return polynomial_from_coefficients(values)
    except ValueError as exc:
        _translate_value_error(exc, ("coefficients",))


def _admit_iterate(request: MapIterateRequest) -> Any:
    polynomial = _polynomial(request)
    degree = 0 if polynomial.is_zero else int(polynomial.degree())
    output_degree = 1 if request.n == 0 else degree**request.n
    if output_degree > MAX_ITERATE_DEGREE:
        _domain_error(
            ("n",),
            "iterate_degree_exceeds_bound",
            "iterate output degree exceeds bound",
        )
    try:
        _iterate_heights(
            tuple(_fraction_height(value) for value in request.coefficient_values()),
            request.n,
        )
    except ValueError as exc:
        _translate_value_error(exc, ("coefficients",))
    return polynomial


def _admit_dynatomic(request: DynatomicPolynomialRequest) -> Any:
    polynomial = _polynomial(request)
    degree = 0 if polynomial.is_zero else int(polynomial.degree())
    if degree < 2:
        _domain_error(
            ("coefficients",),
            "dynatomic_degree_too_small",
            "dynatomic polynomial requires map degree at least two",
        )
    if degree**request.n > MAX_DYNATOMIC_DEGREE:
        _domain_error(
            ("n",),
            "dynatomic_degree_exceeds_bound",
            "dynatomic output degree exceeds bound",
        )
    source = tuple(_fraction_height(value) for value in request.coefficient_values())
    numerator: tuple[CoefficientHeight, ...] = (RationalHeight(1, 1),)
    denominator: tuple[CoefficientHeight, ...] = (RationalHeight(1, 1),)
    try:
        for divisor in range(1, request.n + 1):
            if request.n % divisor:
                continue
            term = list(_iterate_heights(source, divisor))
            if len(term) < 2:
                term.extend([None] * (2 - len(term)))
            term[1] = _add_heights(term[1], RationalHeight(1, 1))
            mobius = _mobius(request.n // divisor)
            if mobius == 1:
                numerator = _multiply_height_polynomials(numerator, tuple(term))
                _require_polynomial_height(numerator, "dynatomic numerator")
            elif mobius == -1:
                denominator = _multiply_height_polynomials(denominator, tuple(term))
                _require_polynomial_height(denominator, "dynatomic denominator")
        quotient = _divide_height_polynomials(numerator, denominator)
        _require_polynomial_height(quotient, "dynatomic quotient")
    except ValueError as exc:
        _translate_value_error(exc, ("coefficients",))
    return polynomial


def _admit_cycle(request: CycleMultiplierRequest) -> Any:
    polynomial = _polynomial(request)
    for value in request.cycle:
        try:
            value.as_fraction()
        except ValueError as exc:
            _translate_value_error(exc, ("cycle",))
    return polynomial


def _admit_finite_field(request: FiniteFieldMapRequest) -> tuple[int, ...]:
    values: list[int] = []
    for index, value in enumerate(request.coefficients):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            _domain_error(
                ("coefficients", index),
                "coefficient_not_canonical",
                "coefficient must be a canonical integer",
            )
        if str(parsed) != value:
            _domain_error(
                ("coefficients", index),
                "coefficient_not_canonical",
                "coefficient must be a canonical integer",
            )
        values.append(parsed)
    return tuple(values)


def _format_coefficients(polynomial: Any) -> tuple[CanonicalRational, ...]:
    return tuple(
        CanonicalRational.from_fraction(value)
        for value in polynomial_coefficients(polynomial)
    )


def compute_map_iterate(request: MapIterateRequest) -> MapIterateResult:
    polynomial = _admit_iterate(request)
    try:
        result = iterate_polynomial(polynomial, request.n)
    except ValueError as exc:
        _translate_value_error(exc, ("n",))
    return MapIterateResult._from_kernel(
        request,
        coefficients=_format_coefficients(result),
        degree=0 if result.is_zero else int(result.degree()),
    )


def compute_orbit_prefix(request: OrbitPrefixRequest) -> OrbitPrefixResult:
    polynomial = _polynomial(request)
    try:
        result = orbit_prefix(
            polynomial, request.start.as_fraction(), request.max_steps
        )
    except ValueError as exc:
        _translate_value_error(exc, ("start",))
    repeat = (
        None
        if result.repeat is None
        else OrbitRepeatEvidence(
            first_seen_index=result.repeat.first_seen_index,
            repeated_at_index=result.repeat.repeated_at_index,
            preperiod=result.repeat.preperiod,
            period=result.repeat.period,
        )
    )
    return OrbitPrefixResult._from_kernel(
        request,
        orbit=tuple(CanonicalRational.from_fraction(value) for value in result.orbit),
        termination=result.termination,
        repeat=repeat,
    )


def compute_dynatomic_polynomial(
    request: DynatomicPolynomialRequest,
) -> DynatomicPolynomialResult:
    polynomial = _admit_dynatomic(request)
    try:
        result = dynatomic_polynomial(polynomial, request.n)
    except ValueError as exc:
        _translate_value_error(exc, ("n",))
    return DynatomicPolynomialResult._from_kernel(
        request,
        coefficients=_format_coefficients(result),
        degree=0 if result.is_zero else int(result.degree()),
    )


def compute_cycle_multiplier(
    request: CycleMultiplierRequest,
) -> CycleMultiplierResult:
    polynomial = _admit_cycle(request)
    points = tuple(value.as_fraction() for value in request.cycle)
    try:
        multiplier = cycle_multiplier(polynomial, points)
    except ValueError as exc:
        _translate_value_error(exc, ("cycle",))
    return CycleMultiplierResult._from_kernel(
        request, multiplier=CanonicalRational.from_fraction(multiplier)
    )


def compute_finite_field_map(request: FiniteFieldMapRequest) -> FiniteFieldMapResult:
    coefficients = _admit_finite_field(request)
    try:
        graph = finite_field_functional_graph(coefficients, request.prime)
    except ValueError as exc:
        _translate_value_error(exc, ("prime",))
    return FiniteFieldMapResult._from_kernel(
        request,
        edges=graph.edges,
        cycles=graph.cycles,
        tail_lengths=graph.tail_lengths,
    )


__all__ = [
    "compute_cycle_multiplier",
    "compute_dynatomic_polynomial",
    "compute_finite_field_map",
    "compute_map_iterate",
    "compute_orbit_prefix",
]
