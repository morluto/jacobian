"""Bounded constraint-satisfaction object construction backed by Z3."""

from __future__ import annotations

from jacobian.contracts.candidate_construction import (
    IntegerFeasibilityCheckRequest,
    IntegerFeasibilityCheckResult,
    IntegerFeasibilityRequest,
    IntegerFeasibilityResult,
    IntegerLinearConstraint,
)


def construct_integer_feasibility(
    request: IntegerFeasibilityRequest,
) -> IntegerFeasibilityResult:
    """Find one feasible integer point satisfying all declared constraints using Z3."""
    import z3

    variables = [z3.Int(f"x{i}") for i in range(request.variable_count)]
    solver = z3.Solver()
    solver.set("timeout", request.timeout_ms)

    for constraint in request.constraints:
        terms = []
        for coeff, var in zip(constraint.coefficients, variables):
            terms.append(coeff * var)
        expr = sum(terms)
        if constraint.relation == "LE":
            solver.add(expr <= constraint.rhs)
        elif constraint.relation == "EQ":
            solver.add(expr == constraint.rhs)
        else:  # GE
            solver.add(expr >= constraint.rhs)

    result = solver.check()
    if result == z3.sat:
        model = solver.model()
        assignment = tuple(int(model.eval(var).as_long()) for var in variables)
        return IntegerFeasibilityResult(
            status="FEASIBLE",
            assignment=assignment,
            detail="Z3 found a feasible integer assignment.",
        )
    elif result == z3.unsat:
        return IntegerFeasibilityResult(
            status="INFEASIBLE",
            detail="Z3 proved infeasibility for the declared constraints.",
        )
    else:
        return IntegerFeasibilityResult(
            status="UNKNOWN",
            detail="Z3 returned unknown (timeout or resource limit).",
        )


def verify_integer_feasibility(
    request: IntegerFeasibilityCheckRequest,
) -> IntegerFeasibilityCheckResult:
    """Independently verify that an assignment satisfies all constraints."""

    for i, constraint in enumerate(request.constraints):
        total = sum(
            coeff * request.assignment[j]
            for j, coeff in enumerate(constraint.coefficients)
        )
        if constraint.relation == "LE" and not (total <= constraint.rhs):
            return IntegerFeasibilityCheckResult(
                satisfies=False,
                first_violated_constraint=i,
                detail=f"Constraint {i} violated: {total} > {constraint.rhs}.",
            )
        elif constraint.relation == "EQ" and not (total == constraint.rhs):
            return IntegerFeasibilityCheckResult(
                satisfies=False,
                first_violated_constraint=i,
                detail=f"Constraint {i} violated: {total} != {constraint.rhs}.",
            )
        elif constraint.relation == "GE" and not (total >= constraint.rhs):
            return IntegerFeasibilityCheckResult(
                satisfies=False,
                first_violated_constraint=i,
                detail=f"Constraint {i} violated: {total} < {constraint.rhs}.",
            )
    return IntegerFeasibilityCheckResult(
        satisfies=True,
        detail="All constraints satisfied by the given assignment.",
    )
