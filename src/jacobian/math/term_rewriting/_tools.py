"""First-order term rewriting operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.term_rewriting._models import (
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
from jacobian.math.term_rewriting._operations import (
    compute_matching,
    compute_normal_form,
    compute_rewrite_step,
    compute_substitution,
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


_SUBST_EXAMPLE = {
    "term": {
        "is_variable": False,
        "symbol": 0,
        "children": [
            {"is_variable": True, "symbol": 0, "children": []},
            {"is_variable": False, "symbol": 1, "children": []},
        ],
    },
    "substitution": {
        "mapping": {
            "0": {
                "is_variable": False,
                "symbol": 2,
                "children": [],
            },
        },
    },
}

_MATCH_EXAMPLE = {
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
        "term_rewriting.substitution.compute",
        "Apply a substitution to a term",
        "Apply a variable-to-term substitution to a first-order term, "
        "replacing each variable with its binding.",
        SubstitutionRequest,
        SubstitutionResult,
        compute_substitution,
        "term-rewriting",
        "substitution",
        "exact",
        examples=(
            example(
                "simple_substitution",
                "Substitute a function symbol for a variable.",
                _SUBST_EXAMPLE,
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
                    "term": {
                        "is_variable": False,
                        "symbol": 0,
                        "children": [
                            {"is_variable": False, "symbol": 1, "children": []},
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
    _op(
        "term_rewriting.normal_form.compute",
        "Run an explicit bounded term-normalization strategy",
        "Apply the explicitly declared leftmost-outermost, rule-order strategy. "
        "Return NORMAL_FORM only when no rewrite applies; otherwise return "
        "STEP_LIMIT with the exact next-step witness.",
        NormalFormRequest,
        NormalFormResult,
        compute_normal_form,
        "term-rewriting",
        "normal-form",
        "exact",
        examples=(
            example(
                "strip_f_layers",
                "Strip f layers from f(f(a)) using rule f(x) -> x.",
                {
                    "term": {
                        "is_variable": False,
                        "symbol": 0,
                        "children": [
                            {
                                "is_variable": False,
                                "symbol": 0,
                                "children": [
                                    {"is_variable": False, "symbol": 1, "children": []},
                                ],
                            },
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
                                "is_variable": True,
                                "symbol": 0,
                                "children": [],
                            },
                        }
                    ],
                    "strategy": "LEFTMOST_OUTERMOST_RULE_ORDER",
                    "max_steps": 100,
                },
            ),
        ),
    ),
)

TOOLS = _TERM_REWRITING_OPERATIONS

__all__ = ["TOOLS"]
