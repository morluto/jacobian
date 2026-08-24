"""First-order term rewriting operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.term_rewriting._models import (
    CriticalPairsRequest,
    CriticalPairsResult,
    MatchingRequest,
    MatchingResult,
    RewriteStepRequest,
    RewriteStepResult,
    UnificationRequest,
    UnificationResult,
)
from jacobian.math.term_rewriting._operations import (
    compute_critical_pairs,
    compute_matching,
    compute_rewrite_step,
    compute_unification,
)


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
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
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

_TERM_REWRITING_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
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

TOOLS = _TERM_REWRITING_OPERATIONS

__all__ = ["TOOLS"]
