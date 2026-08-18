"""Wire adapters for exact bounded arithmetic dynamics."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from jacobian.math.arithmetic_dynamics._models import (
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
)
from jacobian.math.arithmetic_dynamics.operations import (
    cycle_multiplier,
    dynatomic_polynomial,
    finite_field_functional_graph,
    iterate_polynomial,
    orbit_prefix,
    polynomial_coefficients,
    polynomial_from_coefficients,
)


def _polynomial(request: PolynomialCoefficientRequest) -> Any:
    return polynomial_from_coefficients(request.coefficient_values())


def _format_coefficients(polynomial: Any) -> tuple[str, ...]:
    values = tuple(str(value) for value in polynomial_coefficients(polynomial))
    if any(len(value) > 65_538 for value in values):
        raise ValueError("polynomial coefficient output exceeds digit bound")
    return values


def compute_map_iterate(request: MapIterateRequest) -> MapIterateResult:
    result = iterate_polynomial(_polynomial(request), request.n)
    return MapIterateResult(
        source_coefficients=request.coefficients,
        n=request.n,
        coefficients=_format_coefficients(result),
        degree=0 if result.is_zero else int(result.degree()),
    )


def compute_orbit_prefix(request: OrbitPrefixRequest) -> OrbitPrefixResult:
    result = orbit_prefix(
        _polynomial(request),
        Fraction(request.start),
        request.max_steps,
    )
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
    found_repeat = result.termination == "REPEAT_FOUND"
    return OrbitPrefixResult(
        source_coefficients=request.coefficients,
        start=request.start,
        orbit=tuple(str(value) for value in result.orbit),
        requested_steps=request.max_steps,
        computed_steps=len(result.orbit) - 1,
        termination=result.termination,
        repeat=repeat,
        eventual_behavior_complete=found_repeat,
        truncated=not found_repeat,
    )


def compute_dynatomic_polynomial(
    request: DynatomicPolynomialRequest,
) -> DynatomicPolynomialResult:
    result = dynatomic_polynomial(_polynomial(request), request.n)
    return DynatomicPolynomialResult(
        source_coefficients=request.coefficients,
        coefficients=_format_coefficients(result),
        degree=0 if result.is_zero else int(result.degree()),
        n=request.n,
    )


def compute_cycle_multiplier(
    request: CycleMultiplierRequest,
) -> CycleMultiplierResult:
    points = tuple(Fraction(value) for value in request.cycle)
    return CycleMultiplierResult(
        source_coefficients=request.coefficients,
        multiplier=str(cycle_multiplier(_polynomial(request), points)),
        cycle=request.cycle,
        period=len(points),
    )


def compute_finite_field_map(request: FiniteFieldMapRequest) -> FiniteFieldMapResult:
    graph = finite_field_functional_graph(
        tuple(int(value) for value in request.coefficients),
        request.prime,
    )
    return FiniteFieldMapResult(
        prime=request.prime,
        coefficients=request.coefficients,
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
