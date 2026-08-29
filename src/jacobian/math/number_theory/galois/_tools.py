"""Galois theory operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.galois._models import (
    FrobeniusCycleRequest,
    FrobeniusCycleResult,
    GaloisFactorRequest,
    GaloisFactorResult,
    GaloisGroupRequest,
    GaloisGroupResult,
    SolvableRequest,
    SolvableResult,
)
from jacobian.math.number_theory.galois.operations import (
    frobenius_cycle,
    galois_factor,
    galois_group,
    solvable,
)


def _galois_factor(request: GaloisFactorRequest) -> GaloisFactorResult:
    return galois_factor(request.field_order, request.coefficients)


def _frobenius_cycle(request: FrobeniusCycleRequest) -> FrobeniusCycleResult:
    return frobenius_cycle(
        request.field_order,
        request.polynomial_degree,
        request.factorization_degrees,
    )


def _galois_group(request: GaloisGroupRequest) -> GaloisGroupResult:
    return galois_group(request.coefficients)


def _solvable(request: SolvableRequest) -> SolvableResult:
    return solvable(request.coefficients)


def _op[RequestT: StrictModel, ResultT: StrictModel](
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
        "polynomial.galois.factor_mod_p.compute",
        "Factor a polynomial over GF(p)",
        "Factor a polynomial over a prime finite field GF(p) using SymPy, "
        "retaining the unit, monic factors, positive multiplicities, and a "
        "reconstruction-checked irreducibility result.",
        GaloisFactorRequest,
        GaloisFactorResult,
        _galois_factor,
        "galois-theory",
        "factorization",
        "exact",
        examples=(
            example(
                "factor_x2_plus_1_over_f5",
                "Factor x^2 + 1 over F_5.",
                {"field_order": 5, "coefficients": [1, 0, 1]},
            ),
        ),
    ),
    _op(
        "polynomial.galois.frobenius_cycle.compute",
        "Compute the Frobenius cycle type",
        "Compute the Frobenius cycle type from a factorization pattern over "
        "GF(p), returning the cycle type and irreducibility.",
        FrobeniusCycleRequest,
        FrobeniusCycleResult,
        _frobenius_cycle,
        "galois-theory",
        "frobenius",
        "exact",
        examples=(
            example(
                "irreducible_quadratic",
                "Frobenius cycle of an irreducible quadratic.",
                {
                    "field_order": 3,
                    "polynomial_degree": 2,
                    "factorization_degrees": [2],
                },
            ),
        ),
    ),
    _op(
        "polynomial.galois_group.compute",
        "Compute the Galois group of a polynomial over Q",
        "Compute the Galois group of a polynomial with rational coefficients "
        "using SymPy's galois_group function. The request must be irreducible "
        "and degree at most six; the result includes explicit generators.",
        GaloisGroupRequest,
        GaloisGroupResult,
        _galois_group,
        "galois-theory",
        "galois-group",
        "exact",
        examples=(
            example(
                "galois_group_of_x2_minus_2",
                "Galois group of x^2 - 2 over Q.",
                {"coefficients": [-2, 0, 1]},
            ),
        ),
    ),
    _op(
        "polynomial.solvable_by_radicals.decide",
        "Decide if a polynomial is solvable by radicals",
        "Check whether a polynomial is solvable by radicals based on its "
        "Galois group, within SymPy's irreducible degree-at-most-six domain.",
        SolvableRequest,
        SolvableResult,
        _solvable,
        "galois-theory",
        "solvable",
        "exact",
        examples=(
            example(
                "x3_solvable",
                "Check x^3 - 2 is solvable by radicals.",
                {"coefficients": [-2, 0, 0, 1]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
