"""Bind explicit built-in domain declarations to one execution runtime."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.installation.context import InstallationContext
from jacobian.operation_binding import BoundDomainOperations
from jacobian.portfolio.model import PortfolioPlan


@dataclass(frozen=True, slots=True)
class DomainBundleBinder:
    """Bind every declaration in a validated built-in portfolio."""

    context: InstallationContext

    def bind(self, plan: PortfolioPlan) -> dict[str, BoundDomainOperations]:
        """Return bound operations keyed by their unique domain identity."""

        plan.validate()
        installed: dict[str, BoundDomainOperations] = {}
        for bundle in plan.components:
            bound = self.context.binder.bind(bundle)
            installed[bundle.domain_id] = bound
            for adapter in bound.adapters:
                self.context.register_operation(adapter)
        return installed
