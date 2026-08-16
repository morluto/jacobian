"""Exact optimality verification backed by direct rational arithmetic."""

from __future__ import annotations

from fractions import Fraction

from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.optimality_verification import (
    RationalOptimalityVerifyRequest,
    RationalOptimalityVerifyResult,
)


def _check_primal_feasibility(
    request: RationalOptimalityVerifyRequest,
) -> tuple[bool, str]:
    """Check that the primal candidate satisfies Ax = b and x >= 0."""
    program = request.program
    primal = request.primal_candidate

    for var, val in zip(program.variables, primal):
        if val.as_fraction() < 0:
            return False, f"primal variable {var} is negative"

    for i, (row, rhs) in enumerate(zip(program.coefficients, program.rhs)):
        total = sum(
            coeff.as_fraction() * val.as_fraction()
            for coeff, val in zip(row, primal)
        )
        if total != rhs.as_fraction():
            return False, f"primal constraint {i} violated: {total} != {rhs.as_fraction()}"
    return True, ""


def _check_dual_feasibility(
    request: RationalOptimalityVerifyRequest,
) -> tuple[bool, str]:
    """Check that the dual candidate satisfies the dual feasibility conditions."""
    program = request.program
    dual = request.dual_candidate

    # For standard form min cTx s.t. Ax=b, x>=0:
    # The dual is max bTy s.t. ATy <= c (where y is the dual)
    # We check that ATy <= c for each variable
    n_vars = len(program.variables)
    n_constraints = len(program.coefficients)

    # AT is n_vars x n_constraints (transpose of coefficients)
    for j, var in enumerate(program.variables):
        dual_dot_col = sum(
            program.coefficients[i][j].as_fraction() * dual[i].as_fraction()
            for i in range(n_constraints)
        )
        objective_j = program.objective[j].as_fraction()
        if dual_dot_col > objective_j:
            return False, (
                f"dual constraint for variable {var} violated: "
                f"{dual_dot_col} > {objective_j}"
            )
    return True, ""


def verify_rational_optimality(
    request: RationalOptimalityVerifyRequest,
) -> RationalOptimalityVerifyResult:
    """Verify an exact rational LP optimum from primal and dual evidence."""
    from jacobian.contracts.exact import CanonicalRational

    # Check primal feasibility
    primal_ok, primal_msg = _check_primal_feasibility(request)
    if not primal_ok:
        return RationalOptimalityVerifyResult(
            status="REJECTED",
            detail=f"Primal infeasible: {primal_msg}",
        )

    # Check dual feasibility
    dual_ok, dual_msg = _check_dual_feasibility(request)
    if not dual_ok:
        return RationalOptimalityVerifyResult(
            status="REJECTED",
            detail=f"Dual infeasible: {dual_msg}",
        )

    # Compute primal objective
    primal_obj = sum(
        coeff.as_fraction() * val.as_fraction()
        for coeff, val in zip(request.program.objective, request.primal_candidate)
    )

    # Compute dual objective (b^T y)
    dual_obj = sum(
        rhs.as_fraction() * dual_val.as_fraction()
        for rhs, dual_val in zip(request.program.rhs, request.dual_candidate)
    )

    primal_obj_rational = CanonicalRational.from_fraction(primal_obj)
    dual_obj_rational = CanonicalRational.from_fraction(dual_obj)

    # Check that the claimed objective matches the primal objective
    if request.claimed_objective.as_fraction() != primal_obj:
        return RationalOptimalityVerifyResult(
            status="REJECTED",
            primal_objective=primal_obj_rational,
            dual_objective=dual_obj_rational,
            detail=(
                f"Claimed objective {request.claimed_objective.as_fraction()} does not "
                f"match primal objective {primal_obj}"
            ),
        )

    # Check that primal and dual objectives agree (strong duality)
    if primal_obj != dual_obj:
        return RationalOptimalityVerifyResult(
            status="REJECTED",
            primal_objective=primal_obj_rational,
            dual_objective=dual_obj_rational,
            detail=(
                f"Primal objective {primal_obj} does not match dual objective {dual_obj}; "
                f"strong duality fails."
            ),
        )

    return RationalOptimalityVerifyResult(
        status="VERIFIED",
        primal_objective=primal_obj_rational,
        dual_objective=dual_obj_rational,
        detail="Primal and dual evidence agree on the exact objective value.",
    )
