"""Catalog declarations for bounded logic operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.logic._cnf import (
    CnfCanonicalizeRequest,
    CnfCanonicalizeResult,
    SatAssignmentCheckRequest,
    SatAssignmentCheckResult,
    canonicalize_cnf,
    check_sat_assignment,
)
from jacobian.math.logic._sat import SatSolveRequest, SatSolveResult, solve_sat
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
            OperationExample(
                name="two_variables",
                description="Normalize a small named CNF.",
                input={"variable_names": ["b", "a"], "clauses": [[1, -2], [2]]},
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
            OperationExample(
                name="satisfying_assignment",
                description="Check a total assignment against a canonical CNF.",
                input={
                    "cnf": {"variables": ["a", "b"], "clauses": [[-1, 2], [1]]},
                    "assignment": [True, True],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="sat.solve",
        title="Solve a bounded CNF",
        description=(
            "Run the maintained Z3 Python binding on one canonical CNF. "
            "timeout_ms provides up to 120 seconds for the full lifecycle without "
            "increasing the fixed solver work limit. Resource exhaustion and backend "
            "failures raise execution errors; UNKNOWN denotes a healthy inconclusive "
            "solver answer."
        ),
        request_type=SatSolveRequest,
        result_type=SatSolveResult,
        run=solve_sat,
        tags=("sat", "cnf", "solve", "z3"),
        examples=(
            OperationExample(
                name="two_variable_cnf",
                description="Solve a small canonical CNF.",
                input={"cnf": {"variables": ["a", "b"], "clauses": [[-1, 2], [1]]}},
            ),
        ),
    ),
    MathTool(
        operation_id="smt.solve",
        title="Find a model or decide a bounded SMT-LIB query",
        description=(
            "Decide one bounded quantifier-free SMT-LIB query and return SAT, "
            "UNSAT, or UNKNOWN. The admitted fragments are QF_UF, QF_LIA, QF_LRA, "
            "the bounded QF_NRA fragment, and the bounded QF_BV fragment: "
            "quantifier-free Boolean combinations "
            "of rational-polynomial equalities, inequalities, and strict "
            "inequalities over declared real variables, with no division, integer "
            "variables, or transcendental operators; and fixed-width bit-vector "
            "arithmetic, shifts, bitwise logic, concat, extract, and signed or "
            "unsigned comparisons over declared bit-vector constants, with "
            "uninterpreted functions, conversions, and quantifiers rejected. "
            "A QF_NRA SAT result carries an "
            "exact source-bound model whose components are canonical rationals or "
            "exact real algebraic roots (never floats); QF_NRA and QF_BV time, work, or "
            "model-materialization limits return UNKNOWN with a bounded reason. "
            "Other fragments return a bounded display model. timeout_ms provides "
            "up to 120 seconds for the full lifecycle without increasing the fixed "
            "solver work limit. Resource exhaustion and backend failures raise "
            "execution errors; UNKNOWN denotes a healthy inconclusive solver answer."
        ),
        request_type=SmtSolveRequest,
        result_type=SmtSolveResult,
        run=solve_smt,
        tags=("smt", "solve", "smtlib", "z3"),
        discovery_terms=(
            "satisfying model",
            "quantifier-free linear integer constraints",
            "bounded nonlinear real arithmetic satisfiability",
            "polynomial equality and inequality feasibility",
        ),
        examples=(
            OperationExample(
                name="positive_integer",
                description="Solve a bounded quantifier-free linear-integer query.",
                input={
                    "logic": "QF_LIA",
                    "smtlib": "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
                },
            ),
        ),
    ),
    SMT_UNSAT_CORE_OPERATION,
)
