"""Lean declaration search, inspection, dependency, and failure adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from jacobian.artifacts import ArtifactService
from jacobian.builtin_capabilities import (
    LeanDeclarationInspectAdapter,
    LeanDeclarationSearchAdapter,
    LeanDependencyGraphAdapter,
)
from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityInstallTier,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
    CapabilityRequest,
)
from jacobian.contracts.lean import LeanDependencyGraphArtifact, LeanEnvironment
from jacobian.lean_frontend.declaration_protocol import (
    LeanDeclarationBackendResult,
    LeanDeclarationDependenciesQuery,
    LeanDeclarationQuery,
    LeanDeclarationResultEnvelope,
    LeanDeclarationSearchQuery,
)
from jacobian.lean_frontend.declarations import (
    LeanDeclarationBackend,
    LeanDeclarationBackendError,
    LeanDeclarationService,
)
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository

_DIGEST = "sha256:" + "a" * 64
_RUNTIME = CapabilityProviderRuntime(
    provider="jacobian.lean4",
    availability=CapabilityProviderAvailability.AVAILABLE,
    version="4.31.0",
    digest="sha256:" + "b" * 64,
    digest_kind=CapabilityProviderDigestKind.EXECUTABLE,
    platform="test",
    install_tier=CapabilityInstallTier.T3,
    license_id="Apache-2.0",
    features=("CORE", "MATHLIB"),
)


@dataclass
class FakeBackend(LeanDeclarationBackend):
    response: dict[str, Any]
    calls: list[tuple[LeanEnvironment, LeanDeclarationQuery]] = field(
        default_factory=list
    )

    def environment_digest(self, _environment: LeanEnvironment) -> str:
        return _DIGEST

    def query(
        self, environment: LeanEnvironment, query: LeanDeclarationQuery
    ) -> LeanDeclarationBackendResult:
        self.calls.append((environment, query))
        payload = LeanDeclarationResultEnvelope.model_validate(
            {"request_id": "test", "payload": self.response}
        ).payload
        return LeanDeclarationBackendResult(
            environment_digest=_DIGEST,
            payload=payload,
        )


class MissingDeclarationBackend:
    def environment_digest(self, _environment: LeanEnvironment) -> str:
        return _DIGEST

    def query(
        self, _environment: LeanEnvironment, _query: LeanDeclarationQuery
    ) -> LeanDeclarationBackendResult:
        raise LeanDeclarationBackendError(
            "LEAN_DECLARATION_NOT_FOUND",
            "Lean did not find the exact declaration 'Missing.name'.",
        )


def test_search_adapter_exposes_bounded_computed_retrieval() -> None:
    backend = FakeBackend(
        {
            "operation": "search",
            "declarations": [
                {
                    "name": "irrational_sqrt_two",
                    "type": "Irrational √2",
                    "kind": "THEOREM",
                    "namespace": None,
                    "docstring": None,
                    "source": {
                        "module": "Mathlib.NumberTheory.Real.Irrational",
                        "line": 143,
                        "column": 8,
                        "end_line": 143,
                        "end_column": 27,
                    },
                    "match_reasons": ["NAME_SUBSTRING"],
                }
            ],
            "scanned_declarations": 20_001,
            "stop_reason": "RESULT_LIMIT",
        }
    )
    adapter = LeanDeclarationSearchAdapter(LeanDeclarationService(backend), _RUNTIME)
    result = adapter.invoke(
        CapabilityRequest(
            capability_id="lean.declaration.search",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "MATHLIB",
                "name_contains": "irrational_sqrt_two",
                "result_limit": 1,
            },
        )
    )
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.completeness.status is CapabilityCompletenessStatus.PARTIAL
    assert result.output["environment_digest"] == _DIGEST
    assert result.output["declarations"][0]["name"] == "irrational_sqrt_two"
    assert result.scope is not None
    assert (
        result.scope.parameters["matching"]
        == "case-sensitive name substring and exact constants occurring in the elaborated type"
    )
    assert backend.calls[0][1] == LeanDeclarationSearchQuery(
        name_contains="irrational_sqrt_two",
        type_constants=(),
        namespace_prefixes=(),
        target_module_prefixes=(),
        kinds=(),
        limit=1,
    )


def test_exhausted_search_reports_computed_complete_coverage() -> None:
    backend = FakeBackend(
        {
            "operation": "search",
            "declarations": [],
            "scanned_declarations": 626_944,
            "stop_reason": "EXHAUSTED",
        }
    )
    adapter = LeanDeclarationSearchAdapter(LeanDeclarationService(backend), _RUNTIME)
    result = adapter.invoke(
        CapabilityRequest(
            capability_id="lean.declaration.search",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "MATHLIB",
                "type_pattern": {"constants": ["Jacobian.DoesNotExist"]},
            },
        )
    )
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE
    assert result.completeness.assurance_level is CapabilityAssuranceLevel.COMPUTED
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_inspect_adapter_returns_docs_without_promoting_the_theorem() -> None:
    backend = FakeBackend(
        {
            "operation": "inspect",
            "declaration": {
                "name": "Nat.add",
                "type": "Nat → Nat → Nat",
                "kind": "DEFINITION",
                "namespace": "Nat",
                "docstring": "Addition of natural numbers.",
                "source": None,
                "match_reasons": [],
            },
        }
    )
    adapter = LeanDeclarationInspectAdapter(LeanDeclarationService(backend), _RUNTIME)
    result = adapter.invoke(
        CapabilityRequest(
            capability_id="lean.declaration.inspect",
            mode=CapabilityMode.EXPLORE,
            input={"environment": "CORE", "declaration_name": "Nat.add"},
        )
    )
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.assurance.verification_record_uri is None
    assert result.output["declaration"]["docstring"].startswith("Addition")
    assert result.output["environment_digest"] == _DIGEST


def test_dependency_adapter_exposes_partial_typed_subgraph(tmp_path: Path) -> None:
    backend = FakeBackend(
        {
            "operation": "dependencies",
            "nodes": [
                {"name": "Nat.add_assoc", "kind": "THEOREM", "depth": 0},
                {"name": "Nat.add", "kind": "DEFINITION", "depth": 1},
            ],
            "edges": [
                {
                    "source": "Nat.add_assoc",
                    "target": "Nat.add",
                    "kinds": ["TYPE", "VALUE"],
                }
            ],
            "frontier": ["Nat.add"],
            "node_budget_exhausted": False,
            "closure_complete": False,
        }
    )
    store = ArtifactRepository(tmp_path)
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="test.lean-dependencies",
        version="1",
        definition={"dependency_api": "Lean.Expr.getUsedConstantsAsSet"},
    )
    schema_uri = schemas.register(
        name="test.lean-dependency-graph",
        version="1",
        schema=LeanDependencyGraphArtifact.model_json_schema(),
    )
    adapter = LeanDependencyGraphAdapter(
        LeanDeclarationService(backend),
        _RUNTIME,
        artifacts,
        semantics_uri=semantics_uri,
        dependency_graph_schema_uri=schema_uri,
    )
    result = adapter.invoke(
        CapabilityRequest(
            capability_id="lean.declaration.dependencies",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "CORE",
                "root_declaration": "Nat.add_assoc",
                "max_depth": 1,
                "max_nodes": 20,
            },
        )
    )
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.completeness.status is CapabilityCompletenessStatus.PARTIAL
    assert result.output["edges"][0]["kinds"] == ["TYPE", "VALUE"]
    assert result.output["closure_complete"] is False
    assert result.output["dependency_graph_uri"] in result.artifact_uris
    sent = backend.calls[0][1]
    assert isinstance(sent, LeanDeclarationDependenciesQuery)
    assert sent.max_depth == 1
    assert sent.max_nodes == 20


def test_missing_declaration_is_an_explicit_failed_operation() -> None:
    adapter = LeanDeclarationInspectAdapter(
        LeanDeclarationService(MissingDeclarationBackend()), _RUNTIME
    )
    with pytest.raises(CapabilityInvocationError) as raised:
        adapter.invoke(
            CapabilityRequest(
                capability_id="lean.declaration.inspect",
                mode=CapabilityMode.EXPLORE,
                input={"environment": "CORE", "declaration_name": "Missing.name"},
            )
        )
    assert raised.value.diagnostic.code == "LEAN_DECLARATION_NOT_FOUND"
