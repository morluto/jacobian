"""Installation of built-in application and domain operations."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.builtin_operation_modules import load_builtin_operation_modules
from jacobian.catalog_build_context import CatalogBuildContext
from jacobian.checker_identity import batch_checker_manifest_measurement
from jacobian.contracts.operations import ProviderAvailability
from jacobian.domains.polynomial_nullstellensatz.core import (
    install_nullstellensatz_core,
)
from jacobian.domains.polynomial_nullstellensatz.singular import (
    install_singular_producer,
)
from jacobian.exact_domain_checkers import (
    ExactDomainCheckerInstallation,
    ExactOperationGroup,
    install_exact_domain_verification,
)
from jacobian.finite_coverage import install_finite_coverage
from jacobian.graphs.isomorphism import build_graph_isomorphism_operation
from jacobian.graphs.operation_resources import (
    GraphOperationResources,
    build_graph_operations,
)
from jacobian.polynomial_system_operations import (
    install_polynomial_system_operations,
)
from jacobian.polynomial_system_search import PolynomialSystemRationalSearchAdapter
from jacobian.polynomials import build_polynomial_operations
from jacobian.polytope import PolytopeService
from jacobian.polytope_operations import PolytopeSeparationAdapter
from jacobian.provider_runtime import known_provider_runtime
from jacobian.providers.singular_runtime import singular_provider_runtime
from jacobian.universal_algebra_operations import (
    install_universal_algebra_operations,
)


@dataclass(frozen=True, slots=True)
class CatalogOperationBuilder:
    """Build core operation and domain-checker descriptors for the catalog."""

    context: CatalogBuildContext

    def _bind_graph_operations(
        self,
        ctx: CatalogBuildContext,
    ) -> GraphOperationResources:
        """Build retained graph operation descriptors and resources."""

        graph_adapters, graph = build_graph_operations(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorize_bundled_checkers,
        )
        for graph_adapter in graph_adapters:
            self.context.register_operation(graph_adapter)
        return graph

    def _bind_nullstellensatz(self) -> None:
        """Install the named Nullstellensatz family at the composition root.

        This family has an artifact-producing core and an optional Singular
        producer that depends on that core. It is deliberately not represented
        as a generic catalog callback: ordinary mathematical operations come
        from the fixed built-in module inventory.
        """

        ctx = self.context
        core_runtime = known_provider_runtime(
            "jacobian.nullstellensatz-core",
            features=(
                "normalized-jacobian-degree-slice",
                "rabinowitsch-chart-cover",
                "independent-exact-replay",
            ),
        )
        core = install_nullstellensatz_core(ctx, core_runtime)
        for adapter in core.adapters:
            ctx.register_operation(adapter)
        singular_runtime = singular_provider_runtime()
        if singular_runtime.availability is not ProviderAvailability.AVAILABLE:
            return

        singular = install_singular_producer(ctx, core, singular_runtime)
        for adapter in singular.adapters:
            ctx.register_operation(adapter)

    def bind(
        self,
        polytope: PolytopeService,
    ) -> GraphOperationResources:
        """Build core catalog entries in explicit dependency order."""

        ctx = self.context

        self.context.register_operation(PolytopeSeparationAdapter(polytope))
        finite_coverage_adapter, _ = install_finite_coverage(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorize_bundled_checkers,
        )
        if finite_coverage_adapter is not None:
            self.context.register_operation(finite_coverage_adapter)

        graph = self._bind_graph_operations(ctx)

        exact_groups = {}
        for (
            module_name,
            operations,
            checker_declarations,
        ) in load_builtin_operation_modules():
            bound = ctx.binder.bind(operations)
            for adapter in bound.adapters:
                ctx.register_operation(adapter)
            if checker_declarations:
                exact_groups[module_name] = (
                    operations,
                    bound,
                    checker_declarations,
                )
        self.bind_domain_verification(exact_groups)
        self._bind_nullstellensatz()

        graph_isomorphism_adapter, _ = build_graph_isomorphism_operation(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            graph,
            authorize_checker=ctx.authorize_bundled_checkers,
        )
        if graph_isomorphism_adapter is not None:
            self.context.register_operation(graph_isomorphism_adapter)

        polynomial_adapters, _ = build_polynomial_operations(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorize_bundled_checkers,
        )
        for polynomial_adapter in polynomial_adapters:
            self.context.register_operation(polynomial_adapter)

        polynomial_system_adapter, polynomial_system = (
            install_polynomial_system_operations(
                ctx.store,
                ctx.schemas,
                ctx.artifacts,
                ctx.verification,
                ctx.checkers,
                authorize_checker=ctx.authorize_bundled_checkers,
            )
        )
        if polynomial_system_adapter is not None:
            self.context.register_operation(polynomial_system_adapter)
        self.context.register_operation(
            PolynomialSystemRationalSearchAdapter(
                ctx.artifacts,
                polynomial_system,
            )
        )

        universal_adapters, _ = install_universal_algebra_operations(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorize_bundled_checkers,
        )
        for universal_adapter in universal_adapters:
            self.context.register_operation(universal_adapter)
        return graph

    def bind_domain_verification(
        self,
        operation_groups: dict[str, ExactOperationGroup],
    ) -> ExactDomainCheckerInstallation | None:
        ctx = self.context
        if not operation_groups:
            return None
        # Batch identity material across the complete declaration set while the
        # exact-domain installer resolves both legacy and declaration-owned
        # provider runtimes. Nested measurement remains safe for direct callers.
        with batch_checker_manifest_measurement():
            adapters, installation = install_exact_domain_verification(
                ctx.store,
                ctx.schemas,
                ctx.artifacts,
                ctx.values,
                ctx.verification,
                ctx.checkers,
                groups=operation_groups,
                authorize=ctx.authorize_bundled_checkers,
            )
        for adapter in adapters:
            self.context.register_operation(adapter)
        return installation
