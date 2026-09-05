"""Exact optimization operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.optimization._general_linear_program import general_linear_program
from jacobian.math.optimization._general_models import (
    GeneralRationalLinearProgramRequest,
    GeneralRationalLinearProgramResult,
)
from jacobian.math.optimization._models import (
    MAX_LINEAR_PROGRAM_BASES,
    MAX_LINEAR_PROGRAM_CONSTRAINTS,
    MAX_LINEAR_PROGRAM_SCALAR_UPDATES,
    MAX_LINEAR_PROGRAM_VARIABLES,
    RationalLinearProgramRequest,
    RationalLinearProgramResult,
)
from jacobian.math.optimization.operations import linear_program

_BASIS_ENVELOPE = (
    "Resource admission removes zero columns and zero equalities for work estimates. "
    "For n remaining columns and m remaining rows, the basis estimate is "
    "max C(n+1,r) over 0<=r<=min(n,m); the work estimate is "
    "8(m+1)^2(n+m+2) + max C(n+1,r)[4r^3+2r^2(n+2)+4r(n+2)]. "
    f"Limits are {MAX_LINEAR_PROGRAM_BASES} bases and "
    f"{MAX_LINEAR_PROGRAM_SCALAR_UPDATES} scalar updates, plus source-derived "
    "rational-minor height within the canonical rational digit limit. "
    "Rejections report derived counts, estimates and limits; shape bounds alone "
    "do not guarantee admission. Execution has one 600-second cooperative safety "
    "deadline; expiration yields an execution error, never a mathematical conclusion."
)


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
            "Return a source-bound standard-form rational LP outcome using exact "
            "basis linear algebra. Optimal and feasible outcomes retain checked "
            "points; infeasible outcomes carry a Farkas witness; unbounded outcomes "
            "carry a feasible point and recession direction. Operational failure "
            "uses the execution-error path. " + _BASIS_ENVELOPE
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
            OperationExample(
                name="one_variable_unit_lp",
                description="Optimize x subject to x=1 and x>=0.",
                input={
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
            "coordinates and checked optimality, Farkas, or recession evidence; "
            "private slack and sign-split columns never appear on the wire. "
            "Bound-only boxes and one-variable intervals have direct exact paths. "
            "Other programs normalize each free variable to two columns and each "
            "one-sided variable to one, plus one slack per inequality and one "
            "row/slack per two-sided variable bound. Normalized limits are "
            f"{MAX_LINEAR_PROGRAM_VARIABLES} columns and {MAX_LINEAR_PROGRAM_CONSTRAINTS} rows. "
            + _BASIS_ENVELOPE
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
            OperationExample(
                name="bounded_inequality_lp",
                description="Minimize x subject to x>=1; all rows and bounds are closed exact rationals.",
                input={
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
