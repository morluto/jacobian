"""Context-free language operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.logic.languages.context_free.context_free._models import (
    DependencyGraphRequest,
    DependencyGraphResult,
    FirstSetsRequest,
    FirstSetsResult,
    SymbolProfilesRequest,
    SymbolProfilesResult,
)
from jacobian.math.logic.languages.context_free.context_free._operations import (
    compute_dependency_graph,
    compute_first_sets,
    compute_symbol_profiles,
)


def _op[RequestT: StrictModel, ResultT: StrictModel](
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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "grammar.symbol_profiles.compute",
        "Compute nullable nonterminals of a CFG",
        "Compute which nonterminals are nullable (can derive epsilon) via "
        "fixed-point iteration.",
        SymbolProfilesRequest,
        SymbolProfilesResult,
        compute_symbol_profiles,
        "grammar",
        "nullable",
        "exact",
        examples=(
            example(
                "simple_grammar",
                "Compute nullable symbols in S -> aS | epsilon.",
                {
                    "grammar": {
                        "nonterminals": ["S"],
                        "terminals": ["a"],
                        "rules": [
                            {"head": "S", "body": ["a", "S"]},
                            {"head": "S", "body": []},
                        ],
                        "start_symbol": "S",
                    },
                },
            ),
        ),
    ),
    _op(
        "grammar.dependency_graph.compute",
        "Compute the dependency graph of a CFG",
        "Compute the dependency graph: A depends on B if A has a rule "
        "containing B in its body.",
        DependencyGraphRequest,
        DependencyGraphResult,
        compute_dependency_graph,
        "grammar",
        "dependency-graph",
        "exact",
        examples=(
            example(
                "simple_grammar",
                "Dependency graph of S -> aS | epsilon.",
                {
                    "grammar": {
                        "nonterminals": ["S"],
                        "terminals": ["a"],
                        "rules": [
                            {"head": "S", "body": ["a", "S"]},
                            {"head": "S", "body": []},
                        ],
                        "start_symbol": "S",
                    },
                },
            ),
        ),
    ),
    _op(
        "grammar.first_sets.compute",
        "Compute FIRST sets of a CFG",
        "Compute the FIRST set for each nonterminal via fixed-point iteration.",
        FirstSetsRequest,
        FirstSetsResult,
        compute_first_sets,
        "grammar",
        "first-sets",
        "exact",
        examples=(
            example(
                "simple_grammar",
                "FIRST sets of S -> aS | epsilon.",
                {
                    "grammar": {
                        "nonterminals": ["S"],
                        "terminals": ["a"],
                        "rules": [
                            {"head": "S", "body": ["a", "S"]},
                            {"head": "S", "body": []},
                        ],
                        "start_symbol": "S",
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
