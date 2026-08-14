"""Compilation of resource-backed operation descriptors."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.catalog_build_context import CatalogBuildContext
from jacobian.contracts.operations import ProviderAvailability
from jacobian.graphs.composition import (
    build_graph_composition_operations,
)
from jacobian.graphs.operation_resources import GraphOperationResources
from jacobian.lean_frontend.statement import install_lean_statement_operations
from jacobian.polynomial_interval_operations import (
    install_polynomial_interval_operations,
)
from jacobian.polynomial_positivity_operations import (
    install_polynomial_positivity_operations,
)
from jacobian.providers.lean_runtime import lean_frontend_provider_runtime


@dataclass(frozen=True, slots=True)
class CatalogResourceBuilder:
    """Build resource-backed descriptors after their dependencies."""

    context: CatalogBuildContext

    def bind(self, graph: GraphOperationResources) -> None:
        ctx = self.context
        graph_adapters = build_graph_composition_operations(
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
            authorize_checker=ctx.authorize_bundled_checkers,
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
            authorize_checker=ctx.authorize_bundled_checkers,
        )
        for positivity_adapter in positivity_adapters:
            if positivity_adapter is not None:
                ctx.register_operation(positivity_adapter)

        lean_runtime = lean_frontend_provider_runtime()
        lean_adapters, _ = install_lean_statement_operations(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            provider_runtime=lean_runtime,
        )
        if lean_runtime.availability is ProviderAvailability.AVAILABLE:
            for lean_adapter in lean_adapters:
                ctx.register_operation(lean_adapter)
