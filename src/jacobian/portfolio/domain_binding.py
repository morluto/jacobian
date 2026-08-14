"""Bind explicit built-in domain declarations to one execution runtime."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.operation_binding import BoundDomainOperations
from jacobian.portfolio.context import PortfolioContext
from jacobian.portfolio.model import PortfolioPlan


@dataclass(frozen=True, slots=True)
class DomainBundleBinder:
    """Bind every declaration in a validated built-in portfolio."""

    context: PortfolioContext

    def bind(self, plan: PortfolioPlan) -> dict[str, BoundDomainOperations]:
        """Return bound operations keyed by their unique domain identity."""

        plan.validate()
        bound_by_domain: dict[str, BoundDomainOperations] = {}
        for bundle in plan.components:
            bound = self.context.binder.bind(bundle)
            bound_by_domain[bundle.domain_id] = bound
            for adapter in bound.adapters:
                self.context.register_operation(adapter)
        return bound_by_domain
