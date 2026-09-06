"""Galois theory operation declarations."""

from typing import Any

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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial.galois.factor_mod_p.compute",
        title="Factor a polynomial over GF(p)",
        description="Factor a polynomial over a prime finite field GF(p) using SymPy, "
        "retaining the unit, monic factors, positive multiplicities, and a "
        "reconstruction-checked irreducibility result.",
        request_type=GaloisFactorRequest,
        result_type=GaloisFactorResult,
        run=_galois_factor,
        tags=("galois-theory", "factorization", "exact"),
        examples=(
            OperationExample(
                name="factor_x2_plus_1_over_f5",
                description="Factor x^2 + 1 over F_5.",
                input={"field_order": 5, "coefficients": [1, 0, 1]},
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.galois.frobenius_cycle.compute",
        title="Compute the Frobenius cycle type",
        description="Compute the Frobenius cycle type from a factorization pattern over "
        "GF(p), returning the cycle type and irreducibility.",
        request_type=FrobeniusCycleRequest,
        result_type=FrobeniusCycleResult,
        run=_frobenius_cycle,
        tags=("galois-theory", "frobenius", "exact"),
        examples=(
            OperationExample(
                name="irreducible_quadratic",
                description="Frobenius cycle of an irreducible quadratic.",
                input={
                    "field_order": 3,
                    "polynomial_degree": 2,
                    "factorization_degrees": [2],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.galois_group.compute",
        title="Compute the Galois group of a polynomial over Q",
        description="Compute the Galois group of a polynomial with rational coefficients "
        "using SymPy's galois_group function. The request must be irreducible "
        "and degree at most six; the result includes explicit generators.",
        request_type=GaloisGroupRequest,
        result_type=GaloisGroupResult,
        run=_galois_group,
        tags=("galois-theory", "galois-group", "exact"),
        examples=(
            OperationExample(
                name="galois_group_of_x2_minus_2",
                description="Galois group of x^2 - 2 over Q.",
                input={
                    "polynomial": {
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2],
                                },
                                {
                                    "coefficient": {"num": "-2", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.solvable_by_radicals.decide",
        title="Decide if a polynomial is solvable by radicals",
        description="Check whether a polynomial is solvable by radicals based on its "
        "Galois group, within SymPy's irreducible degree-at-most-six domain.",
        request_type=SolvableRequest,
        result_type=SolvableResult,
        run=_solvable,
        tags=("galois-theory", "solvable", "exact"),
        examples=(
            OperationExample(
                name="x3_solvable",
                description="Check x^3 - 2 is solvable by radicals.",
                input={
                    "polynomial": {
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [3],
                                },
                                {
                                    "coefficient": {"num": "-2", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
