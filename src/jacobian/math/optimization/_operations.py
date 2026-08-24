"""Bounded rational optimization operations backed by SymPy."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.optimization._models import (
    RationalLinearProgramRequest,
    RationalLinearProgramResult,
    StandardFormRationalLinearProgram,
    _dual_replay,
    _primal_replay,
)


def _rational(value: CanonicalRational) -> Any:
    import sympy

    return sympy.Rational(value.as_fraction())


def _wire(value: Any) -> CanonicalRational:
    import sympy

    rational = sympy.Rational(value)
    return CanonicalRational(
        num=format_canonical_integer(int(rational.p)),
        den=format_canonical_integer(int(rational.q)),
    )


def _incomplete_result(
    program: StandardFormRationalLinearProgram,
) -> RationalLinearProgramResult:
    """Return the typed non-mathematical outcome for missing evidence.

    A backend branch that detects a negative status but cannot produce the
    required certificate must not become ``INFEASIBLE`` or ``UNBOUNDED``;
    the honest result is this explicitly operational incomplete outcome.
    """

    return RationalLinearProgramResult(status="UNKNOWN", program=program)


def _infeasible_result(
    program: StandardFormRationalLinearProgram,
    coefficients: Any,
    rhs: Any,
) -> RationalLinearProgramResult:
    """Certify detected infeasibility with an exact Farkas witness.

    Solves ``min 0`` over ``-A^T(u-v) <= 0``, ``b^T(u-v) <= -1`` with
    ``u,v >= 0``, so any solution ``y=u-v`` is a free-vector witness under
    the public convention ``A^T y >= 0``, ``b^T y < 0``. Result
    construction replays both inequalities against the retained source,
    so a backend that cannot evidence its own detection degrades to the
    typed unknown outcome.
    """

    import sympy
    from sympy.solvers.simplex import InfeasibleLPError, UnboundedLPError, linprog

    rows = len(program.rhs)
    try:
        _, solution = linprog(
            sympy.zeros(1, 2 * rows),
            A=(-coefficients.T)
            .row_join(coefficients.T)
            .col_join(rhs.T.row_join(-rhs.T)),
            b=sympy.zeros(len(program.variables), 1).col_join(sympy.Matrix([[-1]])),
        )
    except (InfeasibleLPError, UnboundedLPError):
        return _incomplete_result(program)
    values = [solution[i] - solution[rows + i] for i in range(rows)]
    witness = tuple(_wire(value) for value in values)
    try:
        return RationalLinearProgramResult(
            status="INFEASIBLE",
            program=program,
            farkas_witness=witness,
        )
    except ValidationError:
        return _incomplete_result(program)


def _unbounded_result(
    program: StandardFormRationalLinearProgram,
    objective: Any,
    coefficients: Any,
    rhs: Any,
) -> RationalLinearProgramResult:
    """Certify detected unboundedness with a feasible point and a ray.

    The point solves ``min 0`` over ``Ax=b, x>=0``. The ray solves
    ``min c^T d`` over ``Ad=0``, ``d>=0``, and ``sum(d)<=1``, whose optimum
    is negative exactly when an improving recession direction exists.
    Result construction replays every condition against the retained
    source, so unevidenced backend detections degrade to the typed
    unknown outcome.
    """

    import sympy
    from sympy.solvers.simplex import InfeasibleLPError, UnboundedLPError, linprog

    rows = len(program.rhs)
    width = len(program.variables)
    ray_matrix = sympy.ones(1, width).col_join(coefficients).col_join(-coefficients)
    ray_vector = (
        sympy.Matrix([[1]])
        .col_join(sympy.zeros(rows, 1))
        .col_join(sympy.zeros(rows, 1))
    )
    try:
        _, point_values = linprog(
            sympy.zeros(1, width),
            A=coefficients.col_join(-coefficients),
            b=rhs.col_join(-rhs),
        )
        _, ray_values = linprog(objective, A=ray_matrix, b=ray_vector)
    except (InfeasibleLPError, UnboundedLPError):
        return _incomplete_result(program)
    point = tuple(_wire(value) for value in point_values)
    direction = tuple(_wire(value) for value in ray_values)
    try:
        return RationalLinearProgramResult(
            status="UNBOUNDED",
            program=program,
            feasible_point=point,
            recession_direction=direction,
        )
    except ValidationError:
        return _incomplete_result(program)


def _linear_program(
    request: RationalLinearProgramRequest,
) -> RationalLinearProgramResult:
    import sympy
    from sympy.solvers.simplex import (
        InfeasibleLPError,
        UnboundedLPError,
        linprog,
        lpmax,
    )

    program = request.program
    objective = sympy.Matrix([[_rational(value) for value in program.objective]])
    coefficients = sympy.Matrix(
        [[_rational(value) for value in row] for row in program.coefficients]
    )
    rhs = sympy.Matrix([_rational(value) for value in program.rhs])
    try:
        _, primal_values = linprog(
            objective,
            A=coefficients.col_join(-coefficients),
            b=rhs.col_join(-rhs),
        )
    except InfeasibleLPError:
        return _infeasible_result(program, coefficients, rhs)
    except UnboundedLPError:
        return _unbounded_result(program, objective, coefficients, rhs)

    if len(primal_values) != len(program.variables):
        raise RuntimeError("SymPy returned a primal candidate with the wrong dimension")
    primal_candidate = tuple(_wire(value) for value in primal_values)
    # The shared replay is authoritative: derived diagnostics are computed
    # from the source program and candidate, never from SymPy's summary.
    primal_replay = _primal_replay(program, primal_candidate)
    if not primal_replay.feasible:
        raise RuntimeError("SymPy returned an infeasible linear-program candidate")
    try:
        dual_symbols = sympy.symbols(f"_dual0:{rhs.rows}")
        dual_column = sympy.Matrix(dual_symbols)
        _, dual_solution = lpmax(
            (rhs.transpose() * dual_column)[0],
            [
                sympy.Le(left, right, evaluate=False)
                for left, right in zip(
                    coefficients.transpose() * dual_column,
                    objective.transpose(),
                    strict=True,
                )
            ],
        )
    except (InfeasibleLPError, UnboundedLPError):
        return RationalLinearProgramResult(
            status="PRIMAL_FEASIBLE",
            program=program,
            primal_candidate=primal_candidate,
            primal_objective=primal_replay.objective,
            primal_residuals=primal_replay.residuals,
        )

    dual_values = [dual_solution.get(symbol, sympy.S.Zero) for symbol in dual_symbols]
    if len(dual_values) != rhs.rows:
        raise RuntimeError("SymPy returned a dual candidate with the wrong dimension")
    dual_candidate = tuple(_wire(value) for value in dual_values)
    dual_replay = _dual_replay(program, dual_candidate)
    if dual_replay.feasible and dual_replay.objective == primal_replay.objective:
        return RationalLinearProgramResult(
            status="OPTIMAL",
            program=program,
            primal_candidate=primal_candidate,
            primal_objective=primal_replay.objective,
            primal_residuals=primal_replay.residuals,
            dual_candidate=dual_candidate,
            dual_objective=dual_replay.objective,
            dual_slacks=dual_replay.slacks,
        )
    # A usable dual certificate could not be established; feasibility of
    # the retained primal point remains the strongest supportable claim.
    return RationalLinearProgramResult(
        status="PRIMAL_FEASIBLE",
        program=program,
        primal_candidate=primal_candidate,
        primal_objective=primal_replay.objective,
        primal_residuals=primal_replay.residuals,
    )


RATIONAL_LINEAR_OPERATIONS: MathTools = (
    MathTool(
        operation_id="optimization.linear.rational_optimum.compute",
        version="2",
        title="Solve a rational linear program",
        description=(
            "Use exact SymPy simplex calls to return a source-bound optimum "
            "with primal and dual certificates, a feasible point only, an "
            "exact Farkas-certified infeasibility, certified unboundedness, "
            "or an explicit unknown outcome when required evidence cannot "
            "be produced for a standard-form rational LP."
        ),
        request_type=RationalLinearProgramRequest,
        result_type=RationalLinearProgramResult,
        run=_linear_program,
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

__all__ = ["RATIONAL_LINEAR_OPERATIONS"]
