"""Lean declaration search, inspection, dependency, and failure adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jacobian.artifacts import ArtifactService
from jacobian.capability_service import CapabilityAdapter
from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
    CapabilityRequest,
)
from jacobian.contracts.lean import LeanEnvironment
from jacobian.lean_frontend.declaration_operations import (
    build_lean_declaration_query_bundle,
)
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
from jacobian.operation_installation import OperationInstaller
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
    configuration={
        "profiles": {
            "CORE": {
                "lean_version": "4.31.0",
                "lean_commit": "lean-commit",
                "mathlib_commit": None,
            },
            "MATHLIB": {
                "lean_version": "4.31.0",
                "lean_commit": "lean-commit",
                "mathlib_commit": "mathlib-commit",
            },
        }
    },
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
            lean_version="4.31.0",
            lean_commit="lean-commit",
            mathlib_commit=(
                "mathlib-commit" if environment is LeanEnvironment.MATHLIB else None
            ),
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


def _query_adapter(
    tmp_path: Path,
    backend: LeanDeclarationBackend,
    operation_id: str,
) -> CapabilityAdapter:
    store = ArtifactRepository(tmp_path)
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    installation = OperationInstaller(store, schemas, artifacts).install(
        build_lean_declaration_query_bundle(
            LeanDeclarationService(backend),
            _RUNTIME,
        )
    )
    return next(
        adapter
        for adapter in installation.adapters
        if adapter.descriptor.capability_id == operation_id
    )


def test_search_adapter_exposes_bounded_computed_retrieval(tmp_path: Path) -> None:
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
    adapter = _query_adapter(tmp_path, backend, "lean.declaration.search")
    assert adapter.descriptor.invocation_examples[0].input["name_contains"] == (
        "irrational_sqrt"
    )
    assert "shell-searching" in adapter.descriptor.description
    result = adapter.invoke(
        CapabilityRequest(
            capability_id="lean.declaration.search",
            input={
                "environment": "MATHLIB",
                "name_contains": "irrational_sqrt_two",
                "result_limit": 1,
            },
        )
    )
    assert result.output["result"]["environment_digest"] == _DIGEST
    assert result.output["result"]["lean_version"] == "4.31.0"
    assert result.output["result"]["lean_commit"] == "lean-commit"
    assert result.output["result"]["mathlib_commit"] == "mathlib-commit"
    assert result.output["result"]["declarations"][0]["name"] == ("irrational_sqrt_two")
    assert backend.calls[0][1] == LeanDeclarationSearchQuery(
        name_contains="irrational_sqrt_two",
        type_constants=(),
        namespace_prefixes=(),
        target_module_prefixes=(),
        kinds=(),
        limit=1,
    )


def test_inspect_adapter_returns_docs_without_promoting_the_theorem(
    tmp_path: Path,
) -> None:
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
    adapter = _query_adapter(tmp_path, backend, "lean.declaration.inspect")
    assert adapter.descriptor.invocation_examples[0].input["declaration_name"] == (
        "irrational_sqrt_two"
    )
    result = adapter.invoke(
        CapabilityRequest(
            capability_id="lean.declaration.inspect",
            input={"environment": "CORE", "declaration_name": "Nat.add"},
        )
    )
    assert result.output["result"]["declaration"]["docstring"].startswith("Addition")
    assert result.output["result"]["environment_digest"] == _DIGEST
    assert result.output["result"]["lean_version"] == "4.31.0"
    assert result.output["result"]["lean_commit"] == "lean-commit"
    assert result.output["result"]["mathlib_commit"] is None


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
    adapter = _query_adapter(tmp_path, backend, "lean.declaration.dependencies")
    result = adapter.invoke(
        CapabilityRequest(
            capability_id="lean.declaration.dependencies",
            input={
                "environment": "CORE",
                "root_declaration": "Nat.add_assoc",
                "max_depth": 1,
                "max_nodes": 20,
            },
        )
    )
    assert result.output["preview"]["edges"][0]["kinds"] == ["TYPE", "VALUE"]
    assert result.output["preview"]["closure_complete"] is False
    assert result.output["result_uri"] in result.artifact_uris
    sent = backend.calls[0][1]
    assert isinstance(sent, LeanDeclarationDependenciesQuery)
    assert sent.max_depth == 1
    assert sent.max_nodes == 20


def test_missing_declaration_is_an_explicit_failed_operation(tmp_path: Path) -> None:
    adapter = _query_adapter(
        tmp_path,
        MissingDeclarationBackend(),
        "lean.declaration.inspect",
    )
    result = adapter.invoke(
        CapabilityRequest(
            capability_id="lean.declaration.inspect",
            input={"environment": "CORE", "declaration_name": "Missing.name"},
        )
    )
    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == "LEAN_DECLARATION_NOT_FOUND"
    assert result.output["error"]["code"] == "LEAN_DECLARATION_NOT_FOUND"
