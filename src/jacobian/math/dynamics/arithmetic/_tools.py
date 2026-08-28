"""Arithmetic dynamics operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
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
)
from jacobian.math.dynamics.arithmetic._operations import (
    compute_cycle_multiplier,
    compute_dynatomic_polynomial,
    compute_finite_field_map,
    compute_map_iterate,
    compute_orbit_prefix,
)


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "arithmetic_dynamics.map.iterate.compute",
        "Compute the n-th iterate of a polynomial map",
        "Compute phi^n by exact polynomial composition. "
        "Phi^0 is the identity; coefficients are low-to-high.",
        MapIterateRequest,
        MapIterateResult,
        compute_map_iterate,
        "arithmetic-dynamics",
        "polynomial",
        "exact",
        examples=(
            example(
                "f_x_squared_plus_1_iterate_2",
                "Compute f^2 for f(x)=x^2+1; n must be non-negative.",
                {
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
    _op(
        "arithmetic_dynamics.point.orbit.compute",
        "Compute orbit prefix of a point",
        "Compute P, f(P), ..., f^N(P) for a polynomial map and detect "
        "the first repeat if one occurs within the prefix. A repeat includes "
        "typed preperiod/period evidence; exhausting a step or output bound is "
        "explicitly truncated and makes no eventual-behavior claim.",
        OrbitPrefixRequest,
        OrbitPrefixResult,
        compute_orbit_prefix,
        "arithmetic-dynamics",
        "orbit",
        "exact",
        examples=(
            example(
                "orbit_of_0_under_x2",
                "Orbit of 0 under f(x)=x^2 for 5 steps; "
                "start must be a rational number.",
                {
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
    _op(
        "arithmetic_dynamics.dynatomic_polynomial.compute",
        "Compute the n-th dynatomic polynomial",
        "Compute the Mobius-normalized formal-period polynomial "
        "Phi*_n(x) = product_{d|n} (f^d(x)-x)^{mu(n/d)} for a "
        "degree-at-least-two map, using exact polynomial division.",
        DynatomicPolynomialRequest,
        DynatomicPolynomialResult,
        compute_dynatomic_polynomial,
        "arithmetic-dynamics",
        "dynatomic",
        "exact",
        examples=(
            example(
                "dynatomic_n1_x2",
                "Dynatomic polynomial for n=1 of f(x)=x^2; n must be at least 1.",
                {
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
    _op(
        "arithmetic_dynamics.cycle.multiplier.compute",
        "Compute the multiplier of a periodic cycle",
        "Compute the product of f'(P_i) over all cycle points, "
        "giving the exact multiplier of a periodic cycle. The request is "
        "accepted only when the distinct points follow the map in order.",
        CycleMultiplierRequest,
        CycleMultiplierResult,
        compute_cycle_multiplier,
        "arithmetic-dynamics",
        "cycle",
        "multiplier",
        "exact",
        examples=(
            example(
                "multiplier_fixed_0_x2",
                "Multiplier of the fixed point 0 under f(x)=x^2; "
                "cycle points must be rational.",
                {
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
    _op(
        "arithmetic_dynamics.finite_field.functional_graph.compute",
        "Compute functional graph of a polynomial map over GF(p)",
        "Compute the complete functional graph of a polynomial map "
        "over a finite field, including cycles, tail lengths, and "
        "all edges.",
        FiniteFieldMapRequest,
        FiniteFieldMapResult,
        compute_finite_field_map,
        "arithmetic-dynamics",
        "finite-field",
        "exact",
        examples=(
            example(
                "x2_mod_5",
                "Functional graph of x^2 over GF(5); prime must be a prime number.",
                {"prime": 5, "coefficients": ["0", "0", "1"]},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
