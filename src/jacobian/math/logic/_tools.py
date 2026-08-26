"""Catalog declarations for bounded logic operations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.logic._cnf import (
    CnfCanonicalizeRequest,
    CnfCanonicalizeResult,
    SatAssignmentCheckRequest,
    SatAssignmentCheckResult,
    canonicalize_cnf,
    check_sat_assignment,
)
from jacobian.math.logic._sat import (
    SatRefutationCheckRequest,
    SatRefutationCheckResult,
    SatSolveRequest,
    SatSolveResult,
    check_sat_refutation,
    solve_sat,
)
from jacobian.math.logic._smt import SmtSolveRequest, SmtSolveResult, solve_smt
from jacobian.math.logic._unsat_core import SMT_UNSAT_CORE_OPERATION

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    MathTool(
        operation_id="sat.cnf.canonicalize",
        title="Canonicalize a bounded named CNF",
        description="Return one canonical CNF; no source, identifier, or artifact is retained.",
        request_type=CnfCanonicalizeRequest,
        result_type=CnfCanonicalizeResult,
        run=canonicalize_cnf,
        tags=("sat", "cnf", "canonical"),
        examples=(
            example(
                "two_variables",
                "Normalize a small named CNF.",
                {"variable_names": ["b", "a"], "clauses": [[1, -2], [2]]},
            ),
        ),
    ),
    MathTool(
        operation_id="sat.assignment.check",
        title="Check a total SAT assignment",
        description="Evaluate one complete Boolean assignment against one canonical CNF.",
        request_type=SatAssignmentCheckRequest,
        result_type=SatAssignmentCheckResult,
        run=check_sat_assignment,
        tags=("sat", "cnf", "assignment", "predicate"),
        examples=(
            example(
                "satisfying_assignment",
                "Check a total assignment against a canonical CNF.",
                {
                    "cnf": {"variables": ["a", "b"], "clauses": [[-1, 2], [1]]},
                    "assignment": [True, True],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="sat.solve",
        title="Solve a bounded CNF",
        description="Run the maintained Z3 Python binding on one canonical CNF.",
        request_type=SatSolveRequest,
        result_type=SatSolveResult,
        run=solve_sat,
        tags=("sat", "cnf", "solve", "z3"),
        examples=(
            example(
                "two_variable_cnf",
                "Solve a small canonical CNF.",
                {"cnf": {"variables": ["a", "b"], "clauses": [[-1, 2], [1]]}},
            ),
        ),
    ),
    MathTool(
        operation_id="sat.refutation.check",
        title="Check a bounded LPR SAT refutation",
        description=(
            "Replay one typed LPR/ASCII-v1 refutation against its exact canonical "
            "CNF through the source-pinned CakeML checker. Only VALID_REFUTATION "
            "establishes UNSAT; unavailable or failed replay is a non-conclusion."
        ),
        request_type=SatRefutationCheckRequest,
        result_type=SatRefutationCheckResult,
        run=check_sat_refutation,
        tags=("sat", "cnf", "lpr", "refutation", "certificate"),
        examples=(
            example(
                "unit_contradiction",
                "Check an LPR empty-clause derivation from two contradictory units.",
                {
                    "cnf": {"variables": ["x"], "clauses": [[-1], [1]]},
                    "refutation": {
                        "profile": "LPR_ASCII_V1",
                        "steps": [
                            {
                                "kind": "addition",
                                "clause_id": 3,
                                "clause": [],
                                "at_hint_clause_ids": [1, 2],
                                "propagation_hints": [],
                            }
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="smt.solve",
        title="Solve a bounded SMT-LIB query",
        description="Run the maintained Z3 Python binding on one QF SMT-LIB query.",
        request_type=SmtSolveRequest,
        result_type=SmtSolveResult,
        run=solve_smt,
        tags=("smt", "solve", "smtlib", "z3"),
        examples=(
            example(
                "positive_integer",
                "Solve a bounded quantifier-free linear-integer query.",
                {
                    "logic": "QF_LIA",
                    "smtlib": "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
                },
            ),
        ),
    ),
    SMT_UNSAT_CORE_OPERATION,
)
