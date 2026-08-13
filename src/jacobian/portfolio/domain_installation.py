"""Installation policy for explicit domain-bundle portfolio plans."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.contracts.operations import ProviderAvailability
from jacobian.installation.context import InstallationContext
from jacobian.operation_installation import InstalledDomainBundle
from jacobian.portfolio.model import PortfolioPlan
from jacobian.portfolio.result import (
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
            operation_ids = bundle.operation_ids
            runtime = bundle.provider_runtime
            if runtime.availability is not ProviderAvailability.AVAILABLE:
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
                        operation_ids=operation_ids,
                        installed=None,
                        diagnostic=diagnostic,
                    )
                )
                continue

            installation = self.context.operations.install(bundle)
            installed[bundle.domain_id] = installation
            for adapter in installation.adapters:
                self.context.register_operation(adapter)
            outcomes.append(
                BundleInstallation(
                    domain_id=bundle.domain_id,
                    status=BundleInstallationStatus.INSTALLED,
                    operation_ids=operation_ids,
                    installed=installation,
                    diagnostic=None,
                )
            )

        return PortfolioInstallationResult(
            installed=installed,
            diagnostics=tuple(diagnostics),
            outcomes=tuple(outcomes),
        )
