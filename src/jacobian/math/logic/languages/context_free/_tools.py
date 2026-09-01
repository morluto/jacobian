"""Context-free language operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.logic.languages.context_free._models import (
    DependencyGraphRequest,
    DependencyGraphResult,
    FirstSetsRequest,
    FirstSetsResult,
    SymbolProfilesRequest,
    SymbolProfilesResult,
)
from jacobian.math.logic.languages.context_free.operations import (
    dependency_edges,
    first_sets,
    nullable_nonterminals,
)


def compute_symbol_profiles(request: SymbolProfilesRequest) -> SymbolProfilesResult:
    return SymbolProfilesResult(nullable=nullable_nonterminals(request.grammar))


def compute_dependency_graph(request: DependencyGraphRequest) -> DependencyGraphResult:
    return DependencyGraphResult(edges=dependency_edges(request.grammar))


def compute_first_sets(request: FirstSetsRequest) -> FirstSetsResult:
    return FirstSetsResult(first_sets=first_sets(request.grammar))


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="grammar.symbol_profiles.compute",
        title="Compute nullable nonterminals of a CFG",
        description="Compute which nonterminals are nullable (can derive epsilon) via "
        "fixed-point iteration.",
        request_type=SymbolProfilesRequest,
        result_type=SymbolProfilesResult,
        run=compute_symbol_profiles,
        tags=("grammar", "nullable", "exact"),
        examples=(
            OperationExample(
                name="simple_grammar",
                description="Compute nullable symbols in S -> aS | epsilon.",
                input={
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
    MathTool(
        operation_id="grammar.dependency_graph.compute",
        title="Compute the dependency graph of a CFG",
        description="Compute the dependency graph: A depends on B if A has a rule "
        "containing B in its body.",
        request_type=DependencyGraphRequest,
        result_type=DependencyGraphResult,
        run=compute_dependency_graph,
        tags=("grammar", "dependency-graph", "exact"),
        examples=(
            OperationExample(
                name="simple_grammar",
                description="Dependency graph of S -> aS | epsilon.",
                input={
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
    MathTool(
        operation_id="grammar.first_sets.compute",
        title="Compute FIRST sets of a CFG",
        description="Compute the FIRST set for each nonterminal via fixed-point iteration.",
        request_type=FirstSetsRequest,
        result_type=FirstSetsResult,
        run=compute_first_sets,
        tags=("grammar", "first-sets", "exact"),
        examples=(
            OperationExample(
                name="simple_grammar",
                description="FIRST sets of S -> aS | epsilon.",
                input={
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
