from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.lean import (
    LeanDeclarationInspectRequest,
    LeanDeclarationKind,
    LeanDeclarationSearchOutput,
    LeanDeclarationSearchRequest,
    LeanDeclarationSource,
    LeanDeclarationTypePattern,
    LeanDependencyGraphArtifact,
    LeanDependencyGraphRequest,
    LeanEnvironment,
)


def test_declaration_search_requires_a_name_or_structured_type_pattern() -> None:
    with pytest.raises(ValidationError, match="name_contains or type_pattern"):
        LeanDeclarationSearchRequest(environment=LeanEnvironment.MATHLIB)


def test_declaration_search_contract_normalizes_no_hidden_query_semantics() -> None:
    request = LeanDeclarationSearchRequest(
        environment=LeanEnvironment.MATHLIB,
        name_contains="irrational_sqrt",
        type_pattern=LeanDeclarationTypePattern(
            constants=("Irrational", "Real.sqrt"),
        ),
        namespace_prefixes=("Mathlib",),
        kinds=(LeanDeclarationKind.THEOREM,),
        result_limit=7,
    )

    assert request.type_pattern is not None
    assert request.type_pattern.constants == ("Irrational", "Real.sqrt")
    assert request.result_limit == 7


@pytest.mark.parametrize(
    "constants",
    [
        (),
        ("Irrational", "Irrational"),
    ],
)
def test_declaration_type_pattern_rejects_empty_or_duplicate_constants(
    constants: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError):
        LeanDeclarationTypePattern(constants=constants)


def test_declaration_inspect_requires_one_bounded_exact_name() -> None:
    request = LeanDeclarationInspectRequest(
        environment=LeanEnvironment.CORE,
        declaration_name="Nat.add",
    )
    assert request.declaration_name == "Nat.add"

    with pytest.raises(ValidationError):
        LeanDeclarationInspectRequest(
            environment=LeanEnvironment.CORE,
            declaration_name="",
        )


def test_dependency_graph_contract_rejects_inconsistent_completeness() -> None:
    query = LeanDependencyGraphRequest(
        root_declaration="Nat.add",
        max_depth=1,
        max_nodes=4,
    )

    with pytest.raises(ValidationError, match=r"complete.*frontier"):
        LeanDependencyGraphArtifact(
            environment="CORE",
            environment_digest="sha256:" + "a" * 64,
            query=query,
            nodes=({"name": "Nat.add", "kind": "DEFINITION", "depth": 0},),
            edges=(),
            frontier=("Nat.add",),
            node_budget_exhausted=False,
            closure_complete=True,
        )


def test_dependency_graph_request_has_hard_traversal_budgets() -> None:
    with pytest.raises(ValidationError):
        LeanDependencyGraphRequest(
            root_declaration="Nat.add",
            max_depth=9,
        )
    with pytest.raises(ValidationError):
        LeanDependencyGraphRequest(
            root_declaration="Nat.add",
            max_nodes=501,
        )


def test_declaration_source_range_cannot_run_backwards() -> None:
    with pytest.raises(ValidationError, match="run backwards"):
        LeanDeclarationSource(
            module="Init.Prelude",
            line=10,
            column=8,
            end_line=10,
            end_column=7,
        )


def test_search_stop_reason_and_match_reasons_bind_the_query() -> None:
    query = LeanDeclarationSearchRequest(name_contains="add", result_limit=2)
    with pytest.raises(ValidationError, match="match reasons"):
        LeanDeclarationSearchOutput(
            environment="CORE",
            environment_digest="sha256:" + "a" * 64,
            lean_version="4.31.0",
            lean_commit="lean-commit",
            query=query,
            declarations=(
                {
                    "name": "Nat.add",
                    "type": "Nat → Nat → Nat",
                    "kind": "DEFINITION",
                    "match_reasons": [],
                },
            ),
            scanned_declarations=10,
            stop_reason="EXHAUSTED",
        )

    with pytest.raises(ValidationError, match="result-limit"):
        LeanDeclarationSearchOutput(
            environment="CORE",
            environment_digest="sha256:" + "a" * 64,
            lean_version="4.31.0",
            lean_commit="lean-commit",
            query=query,
            declarations=(
                {
                    "name": "Nat.add",
                    "type": "Nat → Nat → Nat",
                    "kind": "DEFINITION",
                    "match_reasons": ["NAME_SUBSTRING"],
                },
            ),
            scanned_declarations=10,
            stop_reason="RESULT_LIMIT",
        )


def test_dependency_graph_nodes_cannot_exceed_requested_depth() -> None:
    with pytest.raises(ValidationError, match="depth budget"):
        LeanDependencyGraphArtifact(
            environment="CORE",
            environment_digest="sha256:" + "a" * 64,
            query=LeanDependencyGraphRequest(
                root_declaration="Nat.add",
                max_depth=0,
                max_nodes=2,
            ),
            nodes=(
                {"name": "Nat.add", "kind": "DEFINITION", "depth": 0},
                {"name": "Nat", "kind": "INDUCTIVE", "depth": 1},
            ),
            edges=(),
            frontier=("Nat.add",),
            node_budget_exhausted=False,
            closure_complete=False,
        )
