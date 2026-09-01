"""Arithmetic dynamics operation declarations."""

from typing import Any, NoReturn

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.dynamics.arithmetic._models import (
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
from jacobian.math.polynomials.values import RationalPolynomial


def _translate_value_error(
    exc: ValueError, location: tuple[str | int, ...]
) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code=f"arithmetic_dynamics.{_validation_code(str(exc))}",
        message=str(exc),
    ) from exc


def _request_polynomial(request: PolynomialCoefficientRequest) -> RationalPolynomial:
    try:
        return polynomial_from_coefficients(request.coefficient_values())
    except ValueError as exc:
        _translate_value_error(exc, ("coefficients",))


def compute_map_iterate(request: MapIterateRequest) -> MapIterateResult:
    polynomial = _request_polynomial(request)
    try:
        result = iterate_polynomial(polynomial, request.n)
        coefficients = tuple(value for value in polynomial_coefficients(result))
    except ValueError as exc:
        location = ("coefficients",) if "coefficient" in str(exc) else ("n",)
        _translate_value_error(exc, location)
    return MapIterateResult._from_kernel(
        source_coefficients=request.coefficients,
        n=request.n,
        coefficients=tuple(
            CanonicalRational.from_fraction(value) for value in coefficients
        ),
        degree=0
        if not result.polynomial.terms
        else max(term.exponents[0] for term in result.polynomial.terms),
    )


def compute_orbit_prefix(request: OrbitPrefixRequest) -> OrbitPrefixResult:
    polynomial = _request_polynomial(request)
    try:
        result = orbit_prefix(polynomial, request.start, request.max_steps)
    except ValueError as exc:
        location = ("start",) if "orbit start" in str(exc) else ("coefficients",)
        _translate_value_error(exc, location)
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
        source_coefficients=request.coefficients,
        start=request.start,
        requested_steps=request.max_steps,
        orbit=tuple(CanonicalRational.from_fraction(value) for value in result.orbit),
        termination=result.termination,
        repeat=repeat,
    )


def compute_dynatomic_polynomial(
    request: DynatomicPolynomialRequest,
) -> DynatomicPolynomialResult:
    polynomial = _request_polynomial(request)
    try:
        result = dynatomic_polynomial(polynomial, request.n)
        coefficients = polynomial_coefficients(result)
    except ValueError as exc:
        location = ("coefficients",) if "coefficient" in str(exc) else ("n",)
        _translate_value_error(exc, location)
    return DynatomicPolynomialResult._from_kernel(
        source_coefficients=request.coefficients,
        n=request.n,
        coefficients=tuple(
            CanonicalRational.from_fraction(value) for value in coefficients
        ),
        degree=0
        if not result.polynomial.terms
        else max(term.exponents[0] for term in result.polynomial.terms),
    )


def compute_cycle_multiplier(
    request: CycleMultiplierRequest,
) -> CycleMultiplierResult:
    polynomial = _request_polynomial(request)
    try:
        multiplier = cycle_multiplier(polynomial, request.cycle)
    except ValueError as exc:
        _translate_value_error(exc, ("cycle",))
    return CycleMultiplierResult._from_kernel(
        source_coefficients=request.coefficients,
        cycle=request.cycle,
        multiplier=multiplier,
    )


def compute_finite_field_map(request: FiniteFieldMapRequest) -> FiniteFieldMapResult:
    values: list[int] = []
    for index, value in enumerate(request.coefficients):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            _translate_value_error(
                ValueError("coefficient must be a canonical integer"),
                ("coefficients", index),
            )
        if str(parsed) != value:
            _translate_value_error(
                ValueError("coefficient must be a canonical integer"),
                ("coefficients", index),
            )
        values.append(parsed)
    try:
        graph = finite_field_functional_graph(tuple(values), request.prime)
    except ValueError as exc:
        location = ("coefficients",) if "coefficient" in str(exc) else ("prime",)
        _translate_value_error(exc, location)
    return FiniteFieldMapResult._from_kernel(
        prime=request.prime,
        coefficients=request.coefficients,
        edges=graph.edges,
        cycles=graph.cycles,
        tail_lengths=graph.tail_lengths,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="arithmetic_dynamics.map.iterate.compute",
        title="Compute the n-th iterate of a polynomial map",
        description="Compute phi^n by exact polynomial composition. "
        "Phi^0 is the identity; coefficients are low-to-high.",
        request_type=MapIterateRequest,
        result_type=MapIterateResult,
        run=compute_map_iterate,
        tags=("arithmetic-dynamics", "polynomial", "exact"),
        examples=(
            OperationExample(
                name="f_x_squared_plus_1_iterate_2",
                description="Compute f^2 for f(x)=x^2+1; n must be non-negative.",
                input={
                    "coefficients": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                    "n": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="arithmetic_dynamics.point.orbit.compute",
        title="Compute orbit prefix of a point",
        description="Compute P, f(P), ..., f^N(P) for a polynomial map and detect "
        "the first repeat if one occurs within the prefix. A repeat includes "
        "typed preperiod/period evidence; exhausting a step or output bound is "
        "explicitly truncated and makes no eventual-behavior claim.",
        request_type=OrbitPrefixRequest,
        result_type=OrbitPrefixResult,
        run=compute_orbit_prefix,
        tags=("arithmetic-dynamics", "orbit", "exact"),
        examples=(
            OperationExample(
                name="orbit_of_0_under_x2",
                description="Orbit of 0 under f(x)=x^2 for 5 steps; "
                "start must be a rational number.",
                input={
                    "coefficients": [
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                    "start": {"num": "0", "den": "1"},
                    "max_steps": 5,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="arithmetic_dynamics.dynatomic_polynomial.compute",
        title="Compute the n-th dynatomic polynomial",
        description="Compute the Mobius-normalized formal-period polynomial "
        "Phi*_n(x) = product_{d|n} (f^d(x)-x)^{mu(n/d)} for a "
        "degree-at-least-two map, using exact polynomial division.",
        request_type=DynatomicPolynomialRequest,
        result_type=DynatomicPolynomialResult,
        run=compute_dynatomic_polynomial,
        tags=("arithmetic-dynamics", "dynatomic", "exact"),
        examples=(
            OperationExample(
                name="dynatomic_n1_x2",
                description="Dynatomic polynomial for n=1 of f(x)=x^2; n must be at least 1.",
                input={
                    "coefficients": [
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                    "n": 1,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="arithmetic_dynamics.cycle.multiplier.compute",
        title="Compute the multiplier of a periodic cycle",
        description="Compute the product of f'(P_i) over all cycle points, "
        "giving the exact multiplier of a periodic cycle. The request is "
        "accepted only when the distinct points follow the map in order.",
        request_type=CycleMultiplierRequest,
        result_type=CycleMultiplierResult,
        run=compute_cycle_multiplier,
        tags=("arithmetic-dynamics", "cycle", "multiplier", "exact"),
        examples=(
            OperationExample(
                name="multiplier_fixed_0_x2",
                description="Multiplier of the fixed point 0 under f(x)=x^2; "
                "cycle points must be rational.",
                input={
                    "coefficients": [
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                    "cycle": [{"num": "0", "den": "1"}],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="arithmetic_dynamics.finite_field.functional_graph.compute",
        title="Compute functional graph of a polynomial map over GF(p)",
        description="Compute the complete functional graph of a polynomial map "
        "over a finite field, including cycles, tail lengths, and "
        "all edges.",
        request_type=FiniteFieldMapRequest,
        result_type=FiniteFieldMapResult,
        run=compute_finite_field_map,
        tags=("arithmetic-dynamics", "finite-field", "exact"),
        examples=(
            OperationExample(
                name="x2_mod_5",
                description="Functional graph of x^2 over GF(5); prime must be a prime number.",
                input={"prime": 5, "coefficients": ["0", "0", "1"]},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
