"""Polynomial Ga-orbit operation declaration."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.derivations.orbits._models import (
    GaPolynomialOrbitRequest,
)
from jacobian.math.polynomials.derivations.orbits._operations import (
    compute_ga_polynomial_orbit,
)
from jacobian.math.polynomials.values import RationalPolynomial


def _rational(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


def _poly(terms: list[tuple[int, list[int]]]) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": ["x", "y", "t"],
        "polynomial": {
            "terms": [
                {"coefficient": _rational(coefficient), "exponents": exponents}
                for coefficient, exponents in terms
            ]
        },
    }


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="algebraic_group.ga.polynomial_orbit.compute",
        title="Apply an additive-group action to a polynomial",
        description=(
            "Apply a checked polynomial Ga coaction to one polynomial and return "
            "its exact orbit polynomial in QQ[x_1,...,x_n,t]. The source polynomial "
            "must use the action's exact ordered variable axis. The operation "
            "rechecks the counit and additive composition law, then admits source "
            "degree, expansion count, intermediate work, coefficient height, and "
            "serialized output before substitution."
        ),
        request_type=GaPolynomialOrbitRequest,
        result_type=RationalPolynomial,
        run=compute_ga_polynomial_orbit,
        tags=("algebraic-group", "ga", "orbit", "polynomial", "exact"),
        discovery_terms=(
            "Ga action on a polynomial",
            "additive group polynomial orbit",
            "polynomial coaction evaluation",
            "locally nilpotent derivation orbit polynomial",
        ),
        examples=(
            OperationExample(
                name="translation_action_on_square",
                description=(
                    "Apply x->x+t*y, y->y to x^2 and return x^2+2*x*y*t+y^2*t^2; "
                    "the input polynomial must use the action's ordered QQ ring."
                ),
                input={
                    "action": {
                        "source_variables": ["x", "y"],
                        "parameter": "t",
                        "generator_images": [
                            _poly([(1, [1, 0, 0]), (1, [0, 1, 1])]),
                            _poly([(1, [0, 1, 0])]),
                        ],
                    },
                    "polynomial": {
                        "domain": "QQ",
                        "variables": ["x", "y"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": _rational(1),
                                    "exponents": [2, 0],
                                }
                            ]
                        },
                    },
                },
            ),
        ),
    ),
)
