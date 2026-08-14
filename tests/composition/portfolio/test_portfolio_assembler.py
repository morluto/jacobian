"""Behavioral tests for explicit built-in domain installation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

from jacobian.artifacts import ArtifactService
from jacobian.contracts.operations import OperationDiagnostic
from jacobian.contracts.results import ContractModel
from jacobian.domain_bundles import DomainBundle
from jacobian.installation.context import InstallationContext
from jacobian.operation_binding import OperationBinder
from jacobian.operation_bindings import inline_operation
from jacobian.operation_declarations import OperationDeclaration
from jacobian.operations import (
    DomainDiagnostics,
    DomainSemantics,
)
from jacobian.portfolio.domain_binding import DomainBundleBinder
from jacobian.portfolio.model import PortfolioPlan
from jacobian.registry import CheckerRegistry
from jacobian.runtime.config import CheckerAuthorityMode
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService


class _SyntheticRequest(ContractModel):
    value: int = Field(ge=0, le=100)


class _SyntheticResult(ContractModel):
    doubled: int


def _synthetic_bundle(
    *,
    domain_id: str = "synthetic",
    operations: tuple[OperationDeclaration[Any, Any], ...] | None = None,
) -> DomainBundle:
    def compute(request: _SyntheticRequest) -> _SyntheticResult:
        return _SyntheticResult(doubled=request.value * 2)

    return DomainBundle(
        domain_id=domain_id,
        schema_namespace=f"jacobian.{domain_id}",
        semantics=DomainSemantics(
            name=f"jacobian.{domain_id}",
            version="1",
            definition={"description": f"synthetic {domain_id} semantics"},
        ),
        operations=operations
        if operations is not None
        else (
            inline_operation(
                OperationDeclaration(
                    operation_id=f"{domain_id}.compute.double",
                    version="2",
                    title="Double a bounded integer",
                    description="Double one bounded nonnegative integer.",
                    request_type=_SyntheticRequest,
                    result_type=_SyntheticResult,
                    execute=compute,
                    tags=("synthetic",),
                )
            ),
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=OperationDiagnostic(
                code="INVALID_SYNTHETIC_REQUEST",
                stage="synthetic_input_validation",
                message="Input does not satisfy the synthetic contract.",
            )
        ),
    )


@dataclass
class _RecordingContext:
    """A self-contained InstallationContext plus its owned resources."""

    context: InstallationContext
    store: ArtifactRepository
    registered: list[str]


@pytest.fixture
def assembly(tmp_path: Path) -> Iterator[_RecordingContext]:
    """Build a real, narrow InstallationContext without the full runtime."""

    store = ArtifactRepository(tmp_path)
    try:
        schemas = SchemaRegistry(store)
        artifacts = ArtifactService(store, schemas)
        operations = OperationBinder(store, schemas, artifacts)
        checkers = CheckerRegistry(store)
        verification = VerificationService(store, checkers, schemas)
        registered: list[str] = []

        def register_operation(adapter: Any) -> None:
            registered.append(adapter.descriptor.operation_id)

        context = InstallationContext(
            store=store,
            schemas=schemas,
            artifacts=artifacts,
            values=operations.values,
            checkers=checkers,
            verification=verification,
            binder=operations,
            checker_authority=CheckerAuthorityMode.NONE,
            register_operation=register_operation,
        )
        yield _RecordingContext(context=context, store=store, registered=registered)
    finally:
        store.close()


def test_install_domains_installs_every_available_bundle_and_registers_adapters(
    assembly: _RecordingContext,
) -> None:
    assembler = DomainBundleBinder(assembly.context)
    plan = PortfolioPlan(components=(_synthetic_bundle(domain_id="alpha"),))

    result = assembler.bind(plan)

    assert set(result) == {"alpha"}
    assert assembly.registered == ["alpha.compute.double"]
    assert tuple(
        adapter.descriptor.operation_id for adapter in result["alpha"].adapters
    ) == ("alpha.compute.double",)


def test_install_domains_preserves_declaration_order_across_bundles(
    assembly: _RecordingContext,
) -> None:
    assembler = DomainBundleBinder(assembly.context)
    plan = PortfolioPlan(
        components=(
            _synthetic_bundle(domain_id="alpha"),
            _synthetic_bundle(domain_id="beta"),
        )
    )

    result = assembler.bind(plan)

    assert tuple(result) == ("alpha", "beta")
    assert assembly.registered == ["alpha.compute.double", "beta.compute.double"]


def test_install_domains_validates_the_plan_before_installing(
    assembly: _RecordingContext,
) -> None:
    assembler = DomainBundleBinder(assembly.context)
    bundle = _synthetic_bundle(domain_id="alpha")
    plan = PortfolioPlan(components=(bundle, bundle))

    with pytest.raises(ValueError, match="duplicate domain bundles"):
        assembler.bind(plan)

    # Nothing was installed because validation failed before the loop.
    assert assembly.registered == []


def test_empty_plan_yields_complete_empty_result(
    assembly: _RecordingContext,
) -> None:
    assembler = DomainBundleBinder(assembly.context)
    plan = PortfolioPlan(components=())

    result = assembler.bind(plan)

    assert result == {}


def test_install_failure_propagates_without_silent_partial_portfolio(
    assembly: _RecordingContext,
) -> None:
    """A bundle installation defect must propagate, not be absorbed.

    OperationBinder rejects an empty-operation bundle. The assembler must
    not normalize that into a diagnostic; it must raise so the caller's
    enclosing transaction rolls back the partial portfolio atomically.
    """

    assembler = DomainBundleBinder(assembly.context)
    broken = _synthetic_bundle(domain_id="broken", operations=())
    plan = PortfolioPlan(
        components=(
            _synthetic_bundle(domain_id="alpha"),
            broken,
            _synthetic_bundle(domain_id="gamma"),
        )
    )

    with pytest.raises(ValueError, match="must not be empty"):
        assembler.bind(plan)

    # The earlier bundle's adapter registration happened in-memory only and the
    # caller is expected to roll back its enclosing transaction; the assembler
    # itself never returns a silently-degraded result.
    assert "broken" not in assembly.registered


def test_duplicate_operation_id_within_a_bundle_propagates(
    assembly: _RecordingContext,
) -> None:
    """OperationBinder rejects duplicate operation IDs within a bundle.

    The assembler must propagate that defect rather than recording a skip.
    """

    assembler = DomainBundleBinder(assembly.context)
    base = _synthetic_bundle(domain_id="alpha")
    duplicate = base.operations[0]
    bundle = replace(base, operations=(base.operations[0], duplicate))

    with pytest.raises(ValueError, match="duplicate operation ID"):
        assembler.bind(PortfolioPlan(components=(bundle,)))

    assert assembly.registered == []
