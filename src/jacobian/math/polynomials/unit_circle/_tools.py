"""Public declarations for exact unit-circle polynomial operations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.unit_circle._models import (
    FejerRieszFactorResult,
    HermitianLaurentPolynomial,
    UnitCircleArcEnergyRequest,
    UnitCircleArcEnergyResult,
)
from jacobian.math.polynomials.unit_circle.operations import (
    real_symmetric_degree_one_fejer_riesz_factor,
    unit_circle_arc_energy,
)


def _run_arc_energy(request: UnitCircleArcEnergyRequest) -> UnitCircleArcEnergyResult:
    return unit_circle_arc_energy(
        request.polynomial, request.start_turn, request.end_turn
    )


def _run_fejer(source: HermitianLaurentPolynomial) -> FejerRieszFactorResult:
    return real_symmetric_degree_one_fejer_riesz_factor(source)


POLYNOMIAL_ONE_PLUS_Z = {
    "polynomial": {
        "domain": "QQ",
        "variables": ["z"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [1]},
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
            ]
        },
    },
    "start_turn": {"num": "-1", "den": "4"},
    "end_turn": {"num": "1", "den": "4"},
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial.unit_circle.arc_energy.compute",
        title="Compute exact unit-circle arc energy",
        description=(
            "Compute the exact normalized energy integral of a bounded rational "
            "polynomial on an oriented arc with unwrapped rational turns. The "
            "result is A+B/pi, with B in the standard real cyclotomic field "
            "fixed by the rational endpoint conductor."
        ),
        request_type=UnitCircleArcEnergyRequest,
        result_type=UnitCircleArcEnergyResult,
        run=_run_arc_energy,
        tags=("polynomial", "unit-circle", "integral", "exact"),
        examples=(
            OperationExample(
                name="one_plus_z_right_semicircle",
                description="Energy of 1+z on the right semicircle.",
                input=POLYNOMIAL_ONE_PLUS_Z,
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.unit_circle.real_symmetric_degree_one_fejer_riesz_factor.compute",
        title="Decide exact degree-one scalar Fejer-Riesz factorization",
        description=(
            "For a real-symmetric rational Laurent polynomial supported on "
            "{-1,0,1}, return its normalized exact outer factor, the zero "
            "conclusion, or an exact cosine witness of negativity."
        ),
        request_type=HermitianLaurentPolynomial,
        result_type=FejerRieszFactorResult,
        run=_run_fejer,
        tags=(
            "polynomial",
            "unit-circle",
            "fejer-riesz",
            "degree-one",
            "exact",
        ),
        examples=(
            OperationExample(
                name="boundary_zero",
                description="The factor of 2-z-z^-1 is 1-z.",
                input={
                    "terms": [
                        {"exponent": -1, "coefficient": {"num": "-1", "den": "1"}},
                        {"exponent": 0, "coefficient": {"num": "2", "den": "1"}},
                        {"exponent": 1, "coefficient": {"num": "-1", "den": "1"}},
                    ]
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
