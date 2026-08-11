"""Behavioral tests for the portfolio assembler's installation absorption.

The assembler records declared provider unavailability and bundles affected by
an unavailable declared dependency. Every other installation failure
(programming, schema, store, or configuration defects) must propagate so the
caller's enclosing transaction rolls back atomically.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

from jacobian.artifacts import ArtifactService
from jacobian.capability_service import CapabilityService
from jacobian.contracts.capabilities import (
    CapabilityDiagnostic,
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
)
from jacobian.contracts.results import ContractModel
from jacobian.installation.context import InstallationContext
from jacobian.operation_bindings import InstalledOperation, inline_operation
from jacobian.operation_installation import InstalledDomainBundle, OperationInstaller
from jacobian.operations import (
    DomainBundle,
    DomainDiagnostics,
    DomainSemantics,
    OperationSpec,
)
from jacobian.portfolio import (
    DEPENDENCY_UNAVAILABLE,
    PROVIDER_UNAVAILABLE,
    PortfolioPlan,
)
from jacobian.portfolio.domain_installation import DomainBundleInstaller
from jacobian.portfolio.result import (
    BundleInstallationStatus,
    PortfolioInstallationResult,
)
from jacobian.provider_runtime import known_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.runtime.config import CheckerAuthorityMode
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService


class _SyntheticRequest(ContractModel):
    value: int = Field(ge=0, le=100)


class _SyntheticResult(ContractModel):
    doubled: int


def _available_runtime() -> CapabilityProviderRuntime:
    return known_provider_runtime("jacobian.synthetic", features=("deterministic",))


def _unavailable_runtime(
    message: str = "synthetic provider is unavailable",
) -> CapabilityProviderRuntime:
    return CapabilityProviderRuntime(
        provider="jacobian.synthetic.unavailable",
        availability=CapabilityProviderAvailability.UNAVAILABLE,
        platform="test-platform",
        install_tier=CapabilityInstallTier.T0,
        license_id="MIT",
        diagnostic=message,
    )


def _synthetic_bundle(
    *,
    domain_id: str = "synthetic",
    runtime: CapabilityProviderRuntime | None = None,
    capabilities: tuple[InstalledOperation[Any, Any], ...] | None = None,
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
        provider_runtime=runtime or _available_runtime(),
        backend_version="synthetic-1",
        capabilities=capabilities
        if capabilities is not None
        else (
            inline_operation(
                OperationSpec(
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
            invalid_request=CapabilityDiagnostic(
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
        operations = OperationInstaller(store, schemas, artifacts)
        capabilities = CapabilityService(store)
        checkers = CheckerRegistry(store)
        verification = VerificationService(store, checkers)
        registered: list[str] = []

        def register_capability(adapter: Any) -> None:
            capabilities.register(adapter)
            registered.append(adapter.descriptor.capability_id)

        context = InstallationContext(
            store=store,
            schemas=schemas,
            artifacts=artifacts,
            capabilities=capabilities,
            checkers=checkers,
            verification=verification,
            operations=operations,
            checker_authority=CheckerAuthorityMode.NONE,
            register_capability=register_capability,
            register_checker_relationship=capabilities._register_catalog_relationship,
        )
        yield _RecordingContext(context=context, store=store, registered=registered)
    finally:
        store.close()


def test_install_domains_installs_every_available_bundle_and_registers_adapters(
    assembly: _RecordingContext,
) -> None:
    assembler = DomainBundleInstaller(assembly.context)
    plan = PortfolioPlan(domain_bundles=(_synthetic_bundle(domain_id="alpha"),))

    result = assembler.install(plan)

    assert isinstance(result, PortfolioInstallationResult)
    assert result.is_complete
    assert result.diagnostics == ()
    assert set(result.installed) == {"alpha"}
    assert result.installed_domain_ids == ("alpha",)
    assert result.skipped_domain_ids == ()
    assert assembly.registered == ["alpha.compute.double"]

    outcome = result.outcome_for("alpha")
    assert outcome is not None
    assert outcome.status is BundleInstallationStatus.INSTALLED
    assert outcome.installed is result.installed["alpha"]
    assert outcome.capability_ids == ("alpha.compute.double",)
    assert outcome.diagnostic is None
    assert result.diagnostic_for("alpha") is None


def test_install_domains_preserves_declaration_order_across_bundles(
    assembly: _RecordingContext,
) -> None:
    assembler = DomainBundleInstaller(assembly.context)
    plan = PortfolioPlan(
        domain_bundles=(
            _synthetic_bundle(domain_id="alpha"),
            _synthetic_bundle(domain_id="beta"),
        )
    )

    result = assembler.install(plan)

    assert result.installed_domain_ids == ("alpha", "beta")
    assert [outcome.domain_id for outcome in result.outcomes] == ["alpha", "beta"]
    assert assembly.registered == ["alpha.compute.double", "beta.compute.double"]


def test_unavailable_provider_is_skipped_with_typed_diagnostic(
    assembly: _RecordingContext,
) -> None:
    assembler = DomainBundleInstaller(assembly.context)
    plan = PortfolioPlan(
        domain_bundles=(
            _synthetic_bundle(domain_id="alpha"),
            _synthetic_bundle(
                domain_id="beta",
                runtime=_unavailable_runtime("beta provider is missing"),
            ),
        )
    )

    result = assembler.install(plan)

    assert not result.is_complete
    assert set(result.installed) == {"alpha"}
    assert result.installed_domain_ids == ("alpha",)
    assert result.skipped_domain_ids == ("beta",)
    # The unavailable bundle's adapters are never registered.
    assert assembly.registered == ["alpha.compute.double"]

    diagnostic = result.diagnostic_for("beta")
    assert diagnostic is not None
    assert diagnostic.code == PROVIDER_UNAVAILABLE
    assert diagnostic.component_id == "beta"
    assert diagnostic.message == "beta provider is missing"

    outcome = result.outcome_for("beta")
    assert outcome is not None
    assert outcome.status is BundleInstallationStatus.SKIPPED_PROVIDER_UNAVAILABLE
    assert outcome.installed is None
    # Capability IDs are still exposed so callers can see what is missing.
    assert outcome.capability_ids == ("beta.compute.double",)


def test_unavailable_provider_does_not_block_subsequent_bundles(
    assembly: _RecordingContext,
) -> None:
    """An unavailable optional provider removes only its own bundle."""

    assembler = DomainBundleInstaller(assembly.context)
    plan = PortfolioPlan(
        domain_bundles=(
            _synthetic_bundle(domain_id="alpha"),
            _synthetic_bundle(
                domain_id="beta",
                runtime=_unavailable_runtime("beta provider is missing"),
            ),
            _synthetic_bundle(domain_id="gamma"),
        )
    )

    result = assembler.install(plan)

    assert set(result.installed) == {"alpha", "gamma"}
    assert result.installed_domain_ids == ("alpha", "gamma")
    assert result.skipped_domain_ids == ("beta",)
    assert assembly.registered == [
        "alpha.compute.double",
        "gamma.compute.double",
    ]


def test_unavailable_dependency_skips_affected_bundle_and_continues(
    assembly: _RecordingContext,
) -> None:
    def dependent_installer(*_args: object, **_kwargs: object) -> InstalledDomainBundle:
        raise AssertionError("an installer with unavailable dependencies must not run")

    dependent = replace(
        _synthetic_bundle(domain_id="dependent"),
        capabilities=(),
        managed_capability_ids=("dependent.compute.double",),
        managed_installer=dependent_installer,
        dependency_ids=("optional",),
    )
    plan = PortfolioPlan(
        domain_bundles=(
            _synthetic_bundle(
                domain_id="optional",
                runtime=_unavailable_runtime("optional provider is missing"),
            ),
            dependent,
            _synthetic_bundle(domain_id="unrelated"),
        )
    )

    result = DomainBundleInstaller(assembly.context).install(plan)

    assert result.installed_domain_ids == ("unrelated",)
    assert result.skipped_domain_ids == ("optional", "dependent")
    assert assembly.registered == ["unrelated.compute.double"]

    diagnostic = result.diagnostic_for("dependent")
    assert diagnostic is not None
    assert diagnostic.code == DEPENDENCY_UNAVAILABLE
    assert diagnostic.stage == "dependency_availability"
    assert "optional" in diagnostic.message

    outcome = result.outcome_for("dependent")
    assert outcome is not None
    assert outcome.status is BundleInstallationStatus.SKIPPED_DEPENDENCY_UNAVAILABLE
    assert outcome.capability_ids == ("dependent.compute.double",)
    assert outcome.installed is None


def test_install_domains_validates_the_plan_before_installing(
    assembly: _RecordingContext,
) -> None:
    assembler = DomainBundleInstaller(assembly.context)
    bundle = _synthetic_bundle(domain_id="alpha")
    plan = PortfolioPlan(domain_bundles=(bundle, bundle))

    with pytest.raises(ValueError, match="duplicate domain bundles"):
        assembler.install(plan)

    # Nothing was installed because validation failed before the loop.
    assert assembly.registered == []


def test_empty_plan_yields_complete_empty_result(
    assembly: _RecordingContext,
) -> None:
    assembler = DomainBundleInstaller(assembly.context)
    plan = PortfolioPlan(domain_bundles=())

    result = assembler.install(plan)

    assert result.is_complete
    assert result.installed == {}
    assert result.diagnostics == ()
    assert result.outcomes == ()
    assert result.installed_domain_ids == ()
    assert result.skipped_domain_ids == ()


def test_outcome_for_and_diagnostic_for_return_none_for_unknown_domains(
    assembly: _RecordingContext,
) -> None:
    assembler = DomainBundleInstaller(assembly.context)
    plan = PortfolioPlan(domain_bundles=(_synthetic_bundle(domain_id="alpha"),))

    result = assembler.install(plan)

    assert result.outcome_for("absent") is None
    assert result.diagnostic_for("absent") is None


def test_install_failure_propagates_without_silent_partial_portfolio(
    assembly: _RecordingContext,
) -> None:
    """A bundle installation defect must propagate, not be absorbed.

    OperationInstaller rejects an empty-capability bundle. The assembler must
    not normalize that into a diagnostic; it must raise so the caller's
    enclosing transaction rolls back the partial portfolio atomically.
    """

    assembler = DomainBundleInstaller(assembly.context)
    broken = _synthetic_bundle(domain_id="broken", capabilities=())
    plan = PortfolioPlan(
        domain_bundles=(
            _synthetic_bundle(domain_id="alpha"),
            broken,
            _synthetic_bundle(domain_id="gamma"),
        )
    )

    with pytest.raises(ValueError, match="must not be empty"):
        assembler.install(plan)

    # The earlier bundle's adapter registration happened in-memory only and the
    # caller is expected to roll back its enclosing transaction; the assembler
    # itself never returns a silently-degraded result.
    assert "broken" not in assembly.registered


def test_duplicate_capability_id_within_a_bundle_propagates(
    assembly: _RecordingContext,
) -> None:
    """OperationInstaller rejects duplicate capability IDs within a bundle.

    The assembler must propagate that defect rather than recording a skip.
    """

    assembler = DomainBundleInstaller(assembly.context)
    base = _synthetic_bundle(domain_id="alpha")
    duplicate = base.capabilities[0]
    bundle = replace(base, capabilities=(base.capabilities[0], duplicate))

    with pytest.raises(ValueError, match="duplicate capability ID"):
        assembler.install(PortfolioPlan(domain_bundles=(bundle,)))

    assert assembly.registered == []
