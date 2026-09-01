"""First-order term rewriting operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.logic.term_rewriting._models import (
    CriticalPairsRequest,
    CriticalPairsResult,
    MatchingRequest,
    MatchingResult,
    NormalFormRequest,
    NormalFormResult,
    RewriteStepRequest,
    RewriteStepResult,
    SubstitutionRequest,
    SubstitutionResult,
    UnificationRequest,
    UnificationResult,
)
from jacobian.math.logic.term_rewriting.operations import (
    critical_pairs_result,
    matching_result,
    normal_form_result,
    rewrite_step_result,
    substitution_result,
    unification_result,
)


def compute_substitution(request: SubstitutionRequest) -> SubstitutionResult:
    """Unpack a wire request for the native substitution operation."""

    return substitution_result(request.signature, request.term, request.substitution)


def compute_matching(request: MatchingRequest) -> MatchingResult:
    """Unpack a wire request for the native matching operation."""

    return matching_result(request.signature, request.pattern, request.subject)


def compute_unification(request: UnificationRequest) -> UnificationResult:
    """Unpack a wire request for the native unification operation."""

    return unification_result(request.signature, request.left, request.right)


def compute_rewrite_step(request: RewriteStepRequest) -> RewriteStepResult:
    """Unpack a wire request for the native rewrite-step operation."""

    return rewrite_step_result(
        request.signature, request.term, request.rules, request.selection
    )


def compute_normal_form(request: NormalFormRequest) -> NormalFormResult:
    """Unpack a wire request for the native normal-form operation."""

    return normal_form_result(
        request.signature,
        request.term,
        request.rules,
        request.strategy,
        request.max_steps,
    )


def compute_critical_pairs(request: CriticalPairsRequest) -> CriticalPairsResult:
    """Unpack a wire request for the native critical-pair operation."""

    return critical_pairs_result(request.signature, request.rules)


_MATCH_EXAMPLE = {
    "signature": {"arities": [2, 0, 0]},
    "pattern": {
        "is_variable": False,
        "symbol": 0,
        "children": [
            {"is_variable": True, "symbol": 0, "children": []},
            {"is_variable": True, "symbol": 1, "children": []},
        ],
    },
    "subject": {
        "is_variable": False,
        "symbol": 0,
        "children": [
            {"is_variable": False, "symbol": 1, "children": []},
            {"is_variable": False, "symbol": 2, "children": []},
        ],
    },
}

_UNIFY_EXAMPLE = {
    "signature": {"arities": [2, 0, 0]},
    "left": {
        "is_variable": False,
        "symbol": 0,
        "children": [
            {"is_variable": True, "symbol": 0, "children": []},
            {"is_variable": False, "symbol": 1, "children": []},
        ],
    },
    "right": {
        "is_variable": False,
        "symbol": 0,
        "children": [
            {"is_variable": False, "symbol": 2, "children": []},
            {"is_variable": False, "symbol": 1, "children": []},
        ],
    },
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="term_rewriting.critical_pairs.compute",
        title="Compute first-order critical pairs",
        description="Enumerate every unifiable nonvariable overlap of a bounded finite "
        "term-rewrite system. Each source-indexed pair records its overlap "
        "position, deterministic rename-apart MGU, and both peak reducts.",
        request_type=CriticalPairsRequest,
        result_type=CriticalPairsResult,
        run=compute_critical_pairs,
        tags=("term-rewriting", "critical-pairs", "exact"),
        examples=(
            OperationExample(
                name="overlap_at_nested_function",
                description="Overlap g(y) -> y into f(g(x)) -> x.",
                input={
                    "signature": {"arities": [1, 1, 0]},
                    "rules": [
                        {
                            "lhs": {
                                "symbol": 0,
                                "children": [
                                    {
                                        "symbol": 1,
                                        "children": [
                                            {"is_variable": True, "symbol": 0}
                                        ],
                                    }
                                ],
                            },
                            "rhs": {"is_variable": True, "symbol": 0},
                        },
                        {
                            "lhs": {
                                "symbol": 1,
                                "children": [{"is_variable": True, "symbol": 1}],
                            },
                            "rhs": {"is_variable": True, "symbol": 1},
                        },
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="term_rewriting.matching.compute",
        title="Match a pattern against a subject term",
        description="One-way matching: find a substitution that makes a pattern "
        "(with variables) structurally equal to a ground subject term.",
        request_type=MatchingRequest,
        result_type=MatchingResult,
        run=compute_matching,
        tags=("term-rewriting", "matching", "exact"),
        examples=(
            OperationExample(
                name="match_pattern",
                description="Match f(x, y) against f(g, h).",
                input=_MATCH_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="term_rewriting.unification.compute",
        title="Unify two terms",
        description="Compute the most general unifier (MGU) of two first-order terms.",
        request_type=UnificationRequest,
        result_type=UnificationResult,
        run=compute_unification,
        tags=("term-rewriting", "unification", "exact"),
        examples=(
            OperationExample(
                name="unify_two_terms",
                description="Unify f(x, c) with f(d, c).",
                input=_UNIFY_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="term_rewriting.rewrite_step.compute",
        title="Enumerate or select one-step term rewrites",
        description="Return every applicable one-step derivation, or apply one agent-selected "
        "rule at one agent-selected position. Each result includes its position, "
        "rule index, matching substitution, and rewritten term.",
        request_type=RewriteStepRequest,
        result_type=RewriteStepResult,
        run=compute_rewrite_step,
        tags=("term-rewriting", "rewrite-step", "exact"),
        examples=(
            OperationExample(
                name="rewrite_f_to_g",
                description="Rewrite f(x) to g(x) in a simple term.",
                input={
                    "signature": {"arities": [1, 1, 0]},
                    "term": {
                        "is_variable": False,
                        "symbol": 0,
                        "children": [
                            {"is_variable": False, "symbol": 2, "children": []},
                        ],
                    },
                    "rules": [
                        {
                            "lhs": {
                                "is_variable": False,
                                "symbol": 0,
                                "children": [
                                    {"is_variable": True, "symbol": 0, "children": []},
                                ],
                            },
                            "rhs": {
                                "is_variable": False,
                                "symbol": 1,
                                "children": [
                                    {"is_variable": True, "symbol": 0, "children": []},
                                ],
                            },
                        }
                    ],
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
