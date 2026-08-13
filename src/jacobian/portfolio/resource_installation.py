"""Installation of resource-backed portfolio operations."""

from __future__ import annotations

from dataclasses import dataclass, field

from jacobian.contracts.operations import ProviderAvailability
from jacobian.graphs.composition import (
    install_graph_composition_capabilities,
)
from jacobian.graphs.installation import GraphInstallation
from jacobian.installation.context import InstallationContext
from jacobian.lean_frontend.statement import install_lean_statement_capabilities
from jacobian.polynomial_interval_operations import (
    install_polynomial_interval_operations,
)
from jacobian.polynomial_positivity_operations import (
    install_polynomial_positivity_operations,
)
from jacobian.portfolio.provider_resolution import ProviderAvailabilityResolver


@dataclass(frozen=True, slots=True)
class ResourceOperationInstaller:
    """Install resources after their core operation dependencies exist."""

    context: InstallationContext
    provider_resolver: ProviderAvailabilityResolver = field(
        default_factory=ProviderAvailabilityResolver
    )

    def install(self, graph: GraphInstallation) -> None:
        ctx = self.context
        graph_adapters, _ = install_graph_composition_capabilities(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            semantics_uri=graph.semantics_uri,
            graph_schema_uri=graph.graph_schema_uri,
        )
        for graph_adapter in graph_adapters:
            ctx.register_operation(graph_adapter)

        interval_adapters, _ = install_polynomial_interval_operations(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        for interval_adapter in interval_adapters:
            if interval_adapter is not None:
                ctx.register_operation(interval_adapter)

        positivity_adapters, _ = install_polynomial_positivity_operations(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        for positivity_adapter in positivity_adapters:
            if positivity_adapter is not None:
                ctx.register_operation(positivity_adapter)

        lean_runtime = self.provider_resolver.resolve_lean_frontend()
        lean_adapters, _ = install_lean_statement_capabilities(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            provider_runtime=lean_runtime,
        )
        if lean_runtime.availability is ProviderAvailability.AVAILABLE:
            for lean_adapter in lean_adapters:
                ctx.register_operation(lean_adapter)
