"""Exact optimization operation declarations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.optimization._general_linear_program import general_linear_program
from jacobian.math.optimization._general_models import (
    GeneralRationalLinearProgramRequest,
    GeneralRationalLinearProgramResult,
)
from jacobian.math.optimization._models import (
    RationalLinearProgramRequest,
    RationalLinearProgramResult,
)
from jacobian.math.optimization.operations import linear_program


def _linear_program_request(
    request: RationalLinearProgramRequest,
) -> RationalLinearProgramResult:
    return linear_program(request.program)


def _general_linear_program_request(
    request: GeneralRationalLinearProgramRequest,
) -> GeneralRationalLinearProgramResult:
    return general_linear_program(request.program)


RATIONAL_LINEAR_OPERATIONS: MathTools = (
    MathTool(
        operation_id="optimization.linear.rational_optimum.compute",
        title="Solve a rational linear program",
        description=(
            "Use exact SymPy simplex calls to return a source-bound standard-form "
            "rational LP outcome. Optimal and feasible outcomes retain replayed "
            "points; infeasible outcomes carry a Farkas witness; unbounded outcomes "
            "carry a feasible point and recession direction; UNKNOWN makes no claim."
        ),
        request_type=RationalLinearProgramRequest,
        result_type=RationalLinearProgramResult,
        run=_linear_program_request,
        tags=(
            "optimization",
            "linear-program",
            "rational",
            "optimum",
            "bounded",
        ),
        examples=(
            example(
                "one_variable_unit_lp",
                "Optimize x subject to x=1 and x>=0.",
                {
                    "program": {
                        "variables": ["x"],
                        "objective": [{"num": "1", "den": "1"}],
                        "coefficients": [[{"num": "1", "den": "1"}]],
                        "rhs": [{"num": "1", "den": "1"}],
                    },
                },
            ),
        ),
    ),
)

RATIONAL_LINEAR_OPERATIONS = (
    *RATIONAL_LINEAR_OPERATIONS,
    MathTool(
        operation_id="optimization.linear.rational_general_optimum.compute",
        title="Solve a general-form rational linear program",
        description=(
            "Solve a bounded exact rational LP with labeled LE, EQ, and GE rows, "
            "finite bounds, and free variables. Results retain the original source "
            "coordinates and replayed optimality, Farkas, or recession evidence; "
            "private slack and sign-split columns never appear on the wire."
        ),
        request_type=GeneralRationalLinearProgramRequest,
        result_type=GeneralRationalLinearProgramResult,
        run=_general_linear_program_request,
        tags=(
            "optimization",
            "linear-program",
            "rational",
            "inequality",
            "bounded",
        ),
        examples=(
            example(
                "bounded_inequality_lp",
                "Minimize x subject to x>=1; all rows and bounds are closed exact rationals.",
                {
                    "program": {
                        "variables": [
                            {
                                "name": "x",
                                "lower_bound": {"num": "0", "den": "1"},
                                "upper_bound": None,
                            }
                        ],
                        "objective": {
                            "sense": "MINIMIZE",
                            "coefficients": [{"num": "1", "den": "1"}],
                        },
                        "constraints": [
                            {
                                "label": "minimum",
                                "coefficients": [{"num": "1", "den": "1"}],
                                "relation": "GE",
                                "rhs": {"num": "1", "den": "1"},
                            }
                        ],
                    }
                },
            ),
        ),
    ),
)

TOOLS: MathTools = RATIONAL_LINEAR_OPERATIONS
__all__ = ["TOOLS"]
