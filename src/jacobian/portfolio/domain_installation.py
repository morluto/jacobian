"""Installation policy for explicit domain-bundle portfolio plans."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.contracts.capabilities import CapabilityProviderAvailability
from jacobian.installation.context import InstallationContext
from jacobian.operation_installation import InstalledDomainBundle
from jacobian.operations import DomainBundle
from jacobian.portfolio.model import ManagedPortfolioComponent, PortfolioPlan
from jacobian.portfolio.result import (
    DEPENDENCY_UNAVAILABLE,
    PROVIDER_UNAVAILABLE,
    BundleInstallation,
    BundleInstallationStatus,
    PortfolioDiagnostic,
    PortfolioInstallationResult,
)


@dataclass(frozen=True, slots=True)
class DomainBundleInstaller:
    """Install a validated plan while preserving fail-closed omission policy."""

    context: InstallationContext

    def install(self, plan: PortfolioPlan) -> PortfolioInstallationResult:
        """Install available bundles and record only declared unavailability."""

        plan.validate()
        installed: dict[str, InstalledDomainBundle] = {}
        diagnostics: list[PortfolioDiagnostic] = []
        outcomes: list[BundleInstallation] = []
        for bundle in plan.components:
            capability_ids = bundle.capability_ids
            runtime = bundle.provider_runtime
            if runtime.availability is not CapabilityProviderAvailability.AVAILABLE:
                diagnostic = PortfolioDiagnostic(
                    code=PROVIDER_UNAVAILABLE,
                    component_id=bundle.domain_id,
                    stage="provider_availability",
                    message=runtime.diagnostic or f"{runtime.provider} is unavailable",
                )
                diagnostics.append(diagnostic)
                outcomes.append(
                    BundleInstallation(
                        domain_id=bundle.domain_id,
                        status=BundleInstallationStatus.SKIPPED_PROVIDER_UNAVAILABLE,
                        capability_ids=capability_ids,
                        installed=None,
                        diagnostic=diagnostic,
                    )
                )
                continue

            dependency_ids = (
                bundle.dependency_ids
                if isinstance(bundle, ManagedPortfolioComponent)
                else ()
            )
            unavailable_dependencies = tuple(
                dependency_id
                for dependency_id in dependency_ids
                if dependency_id not in installed
            )
            if unavailable_dependencies:
                diagnostic = PortfolioDiagnostic(
                    code=DEPENDENCY_UNAVAILABLE,
                    component_id=bundle.domain_id,
                    stage="dependency_availability",
                    message=(
                        "required domain bundle dependencies are unavailable: "
                        + ", ".join(unavailable_dependencies)
                    ),
                )
                diagnostics.append(diagnostic)
                outcomes.append(
                    BundleInstallation(
                        domain_id=bundle.domain_id,
                        status=(
                            BundleInstallationStatus.SKIPPED_DEPENDENCY_UNAVAILABLE
                        ),
                        capability_ids=capability_ids,
                        installed=None,
                        diagnostic=diagnostic,
                    )
                )
                continue

            if isinstance(bundle, DomainBundle):
                installation = self.context.operations.install(bundle)
            else:
                dependencies = {
                    dependency_id: installed[dependency_id]
                    for dependency_id in dependency_ids
                }
                installation = bundle.install(self.context, dependencies)
                _validate_managed_installation(bundle, installation)
            installed[bundle.domain_id] = installation
            for adapter in installation.adapters:
                self.context.register_capability(adapter)
            outcomes.append(
                BundleInstallation(
                    domain_id=bundle.domain_id,
                    status=BundleInstallationStatus.INSTALLED,
                    capability_ids=capability_ids,
                    installed=installation,
                    diagnostic=None,
                )
            )

        return PortfolioInstallationResult(
            installed=installed,
            diagnostics=tuple(diagnostics),
            outcomes=tuple(outcomes),
        )


def _validate_managed_installation(
    bundle: ManagedPortfolioComponent,
    installation: InstalledDomainBundle,
) -> None:
    installed_ids = tuple(
        adapter.descriptor.capability_id for adapter in installation.adapters
    )
    if installed_ids != bundle.capability_ids:
        raise ValueError(
            f"managed component {bundle.domain_id} installed capability IDs "
            f"{installed_ids!r}, expected {bundle.capability_ids!r}"
        )
    mismatched_providers = tuple(
        adapter.descriptor.capability_id
        for adapter in installation.adapters
        if adapter.descriptor.provider_runtime != bundle.provider_runtime
    )
    if mismatched_providers:
        raise ValueError(
            f"managed component {bundle.domain_id} installed adapters with provider "
            f"runtimes that differ from the bundle: {mismatched_providers!r}"
        )
