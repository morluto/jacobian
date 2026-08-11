"""Installation of resource-backed portfolio capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field

from jacobian.contracts.capabilities import CapabilityProviderAvailability
from jacobian.graphs.composition import (
    install_graph_composition_capabilities,
)
from jacobian.installation.context import InstallationContext
from jacobian.lean_frontend.statement import install_lean_statement_capabilities
from jacobian.polynomial_interval_capabilities import (
    install_polynomial_interval_capabilities,
)
from jacobian.polynomial_positivity_capabilities import (
    install_polynomial_positivity_capabilities,
)
from jacobian.portfolio.provider_resolution import ProviderAvailabilityResolver
from jacobian.portfolio.result import PortfolioInstallation


@dataclass(frozen=True, slots=True)
class ResourceCapabilityInstaller:
    """Install resources after their core capability dependencies exist."""

    context: InstallationContext
    provider_resolver: ProviderAvailabilityResolver = field(
        default_factory=ProviderAvailabilityResolver
    )

    def install(self, result: PortfolioInstallation) -> None:
        ctx = self.context
        if result.graph is None:
            raise RuntimeError("graph capabilities must precede resource installation")
        graph_adapters, result.graph_composition = (
            install_graph_composition_capabilities(
                ctx.store,
                ctx.schemas,
                ctx.artifacts,
                semantics_uri=result.graph.semantics_uri,
                graph_schema_uri=result.graph.graph_schema_uri,
            )
        )
        for graph_adapter in graph_adapters:
            ctx.register_capability(graph_adapter)

        interval_adapters, result.polynomial_interval = (
            install_polynomial_interval_capabilities(
                ctx.store,
                ctx.schemas,
                ctx.artifacts,
                ctx.verification,
                ctx.checkers,
                authorize_checker=ctx.authorizes_bundled_checkers,
            )
        )
        for interval_adapter in interval_adapters:
            if interval_adapter is not None:
                ctx.register_capability(interval_adapter)
        positivity_adapters, result.polynomial_positivity = (
            install_polynomial_positivity_capabilities(
                ctx.store,
                ctx.schemas,
                ctx.artifacts,
                ctx.verification,
                ctx.checkers,
                authorize_checker=ctx.authorizes_bundled_checkers,
            )
        )
        for positivity_adapter in positivity_adapters:
            if positivity_adapter is not None:
                ctx.register_capability(positivity_adapter)
        for relationship in (
            *result.polynomial_interval.catalog_relationships,
            *result.polynomial_positivity.catalog_relationships,
        ):
            ctx.register_checker_relationship(
                relationship.source_capability_id,
                relationship.related_capability,
            )

        lean_runtime = self.provider_resolver.resolve_lean_frontend()
        result.lean_statement_runtime = lean_runtime
        lean_adapters, lean_statement = install_lean_statement_capabilities(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            provider_runtime=lean_runtime,
        )
        if lean_runtime.availability is CapabilityProviderAvailability.AVAILABLE:
            result.lean_statement = lean_statement
            for lean_adapter in lean_adapters:
                ctx.register_capability(lean_adapter)
