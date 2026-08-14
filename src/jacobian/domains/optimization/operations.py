"""Bounded rational optimization operations backed by SymPy."""

from __future__ import annotations

from typing import Any

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.validated_analysis import (
    RationalLinearProgramRequest,
    RationalLinearProgramResult,
)
from jacobian.domains._examples import example
from jacobian.math_tools import MathTool, MathTools


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
        primal_value, primal_values = linprog(
            objective,
            A=coefficients.col_join(-coefficients),
            b=rhs.col_join(-rhs),
        )
    except InfeasibleLPError:
        return RationalLinearProgramResult(
            status="INFEASIBLE",
        )
    except UnboundedLPError:
        return RationalLinearProgramResult(
            status="UNBOUNDED",
        )

    if len(primal_values) != objective.cols:
        raise RuntimeError("SymPy returned a primal candidate with the wrong dimension")
    primal = sympy.Matrix(primal_values)
    residuals = coefficients * primal - rhs
    if any(value < 0 for value in primal) or any(value != 0 for value in residuals):
        raise RuntimeError("SymPy returned an infeasible linear-program candidate")
    primal_candidate = tuple(_wire(value) for value in primal)
    primal_objective = _wire(primal_value)
    primal_residuals = tuple(_wire(value) for value in residuals)
    try:
        dual_symbols = sympy.symbols(f"_dual0:{rhs.rows}")
        dual_column = sympy.Matrix(dual_symbols)
        dual_value, dual_solution = lpmax(
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
            primal_candidate=primal_candidate,
            primal_objective=primal_objective,
            primal_residuals=primal_residuals,
        )

    dual_values = [dual_solution.get(symbol, sympy.S.Zero) for symbol in dual_symbols]
    if len(dual_values) != rhs.rows:
        raise RuntimeError("SymPy returned a dual candidate with the wrong dimension")
    dual = sympy.Matrix(dual_values)
    slacks = objective.transpose() - coefficients.transpose() * dual
    if any(value < 0 for value in slacks) or primal_value != dual_value:
        raise RuntimeError("SymPy returned inconsistent linear-program candidates")
    return RationalLinearProgramResult(
        status="OPTIMAL",
        primal_candidate=primal_candidate,
        primal_objective=primal_objective,
        primal_residuals=primal_residuals,
        dual_candidate=tuple(_wire(value) for value in dual),
        dual_objective=_wire(dual_value),
        dual_slacks=tuple(_wire(value) for value in slacks),
    )


RATIONAL_LINEAR_OPERATIONS: MathTools = (
    MathTool(
        operation_id="optimization.linear.rational_optimum.compute",
        version="1",
        title="Solve a rational linear program",
        description=(
            "Use exact SymPy simplex calls to return an optimum, feasible point, "
            "infeasibility, or unboundedness for a standard-form rational LP."
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
