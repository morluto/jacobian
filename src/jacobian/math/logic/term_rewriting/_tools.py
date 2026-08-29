"""First-order term rewriting operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


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
    _op(
        "term_rewriting.critical_pairs.compute",
        "Compute first-order critical pairs",
        "Enumerate every unifiable nonvariable overlap of a bounded finite "
        "term-rewrite system. Each source-indexed pair records its overlap "
        "position, deterministic rename-apart MGU, and both peak reducts.",
        CriticalPairsRequest,
        CriticalPairsResult,
        compute_critical_pairs,
        "term-rewriting",
        "critical-pairs",
        "exact",
        examples=(
            example(
                "overlap_at_nested_function",
                "Overlap g(y) -> y into f(g(x)) -> x.",
                {
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
    _op(
        "term_rewriting.matching.compute",
        "Match a pattern against a subject term",
        "One-way matching: find a substitution that makes a pattern "
        "(with variables) structurally equal to a ground subject term.",
        MatchingRequest,
        MatchingResult,
        compute_matching,
        "term-rewriting",
        "matching",
        "exact",
        examples=(
            example(
                "match_pattern",
                "Match f(x, y) against f(g, h).",
                _MATCH_EXAMPLE,
            ),
        ),
    ),
    _op(
        "term_rewriting.unification.compute",
        "Unify two terms",
        "Compute the most general unifier (MGU) of two first-order terms.",
        UnificationRequest,
        UnificationResult,
        compute_unification,
        "term-rewriting",
        "unification",
        "exact",
        examples=(
            example(
                "unify_two_terms",
                "Unify f(x, c) with f(d, c).",
                _UNIFY_EXAMPLE,
            ),
        ),
    ),
    _op(
        "term_rewriting.rewrite_step.compute",
        "Enumerate or select one-step term rewrites",
        "Return every applicable one-step derivation, or apply one agent-selected "
        "rule at one agent-selected position. Each result includes its position, "
        "rule index, matching substitution, and rewritten term.",
        RewriteStepRequest,
        RewriteStepResult,
        compute_rewrite_step,
        "term-rewriting",
        "rewrite-step",
        "exact",
        examples=(
            example(
                "rewrite_f_to_g",
                "Rewrite f(x) to g(x) in a simple term.",
                {
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
