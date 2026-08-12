"""Installation of built-in application and domain capabilities."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.contracts.capabilities import CapabilityProviderAvailability
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.polynomial_nullstellensatz.core import (
    install_nullstellensatz_core,
)
from jacobian.domains.polynomial_nullstellensatz.singular import (
    install_singular_producer,
)
from jacobian.exact_domain_checkers import (
    ExactDomainCheckerInstallation,
    install_exact_domain_verification,
)
from jacobian.finite_coverage import install_finite_coverage
from jacobian.graphs.coloring import (
    GraphColoringInstallation,
    install_graph_coloring_capabilities,
)
from jacobian.graphs.installation import GraphInstallation, install_graph_capabilities
from jacobian.graphs.isomorphism import install_graph_isomorphism
from jacobian.installation.context import InstallationContext
from jacobian.polynomial_system_capabilities import (
    install_polynomial_system_capabilities,
)
from jacobian.polynomial_system_search import PolynomialSystemRationalSearchAdapter
from jacobian.polynomials import install_polynomial_capabilities
from jacobian.polytope_capabilities import PolytopeSeparationAdapter
from jacobian.portfolio.builtin import build_builtin_portfolio
from jacobian.portfolio.domain_installation import DomainBundleInstaller
from jacobian.portfolio.model import PortfolioPlan
from jacobian.portfolio.result import (
    PortfolioInstallationResult,
)
from jacobian.provider_runtime import known_provider_runtime
from jacobian.providers.singular_runtime import singular_provider_runtime
from jacobian.runtime.services import RuntimeServices
from jacobian.universal_algebra_capabilities import (
    install_universal_algebra_capabilities,
)


@dataclass(frozen=True, slots=True)
class CoreApplicationInstaller:
    """Install core application adapters and domain-dependent checkers."""

    context: InstallationContext

    def _install_graph_capabilities(
        self,
        ctx: InstallationContext,
        services: RuntimeServices,
    ) -> tuple[GraphInstallation, GraphColoringInstallation]:
        """Install retained graph and coloring capabilities."""

        graph_adapters, graph = install_graph_capabilities(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        for graph_adapter in graph_adapters:
            self.context.register_capability(graph_adapter)
        coloring_adapters, graph_coloring = install_graph_coloring_capabilities(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            services.core.sat,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        for coloring_adapter in coloring_adapters:
            self.context.register_capability(coloring_adapter)
        return graph, graph_coloring

    def _install_nullstellensatz(self) -> None:
        """Install the named Nullstellensatz family at the composition root.

        This family has an artifact-producing core and an optional Singular
        producer that depends on that core.  It is deliberately not represented
        as a generic portfolio callback: ordinary portfolio plans contain only
        :class:`DomainBundle` declarations.
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
            ctx.register_capability(adapter)
        singular_runtime = singular_provider_runtime()
        if (
            singular_runtime.availability
            is not CapabilityProviderAvailability.AVAILABLE
        ):
            return

        singular = install_singular_producer(ctx, core, singular_runtime)
        for adapter in singular.adapters:
            ctx.register_capability(adapter)

    def install(
        self,
        services: RuntimeServices,
    ) -> GraphInstallation:
        """Install the core portfolio in its explicit dependency order."""

        ctx = self.context

        self.context.register_capability(PolytopeSeparationAdapter(services.polytope))
        finite_coverage_adapter, _ = install_finite_coverage(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        if finite_coverage_adapter is not None:
            self.context.register_capability(finite_coverage_adapter)

        graph, _ = self._install_graph_capabilities(ctx, services)

        portfolio = build_builtin_portfolio()
        bundle_result = DomainBundleInstaller(ctx).install(portfolio)
        self.install_domain_verification(bundle_result, portfolio)
        self._install_nullstellensatz()

        graph_isomorphism_adapter, _ = install_graph_isomorphism(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            graph,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        if graph_isomorphism_adapter is not None:
            self.context.register_capability(graph_isomorphism_adapter)

        polynomial_adapters, _ = install_polynomial_capabilities(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        for polynomial_adapter in polynomial_adapters:
            self.context.register_capability(polynomial_adapter)

        polynomial_system_adapter, polynomial_system = (
            install_polynomial_system_capabilities(
                ctx.store,
                ctx.schemas,
                ctx.artifacts,
                ctx.verification,
                ctx.checkers,
                authorize_checker=ctx.authorizes_bundled_checkers,
            )
        )
        if polynomial_system_adapter is not None:
            self.context.register_capability(polynomial_system_adapter)
        self.context.register_capability(
            PolynomialSystemRationalSearchAdapter(
                ctx.artifacts,
                polynomial_system,
            )
        )

        universal_adapters, _ = install_universal_algebra_capabilities(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        for universal_adapter in universal_adapters:
            self.context.register_capability(universal_adapter)
        return graph

    def install_domain_verification(
        self,
        bundles: PortfolioInstallationResult,
        plan: PortfolioPlan,
    ) -> ExactDomainCheckerInstallation | None:
        ctx = self.context
        exact_bundles = {
            bundle.domain_id: (bundle, bundles.installed[bundle.domain_id])
            for bundle in plan.components
            if isinstance(bundle, DomainBundle)
            and bundle.checker_declarations
            and bundle.domain_id in bundles.installed
        }
        if not exact_bundles:
            return None
        adapters, installation = install_exact_domain_verification(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            bundles=exact_bundles,
            authorize=ctx.authorizes_bundled_checkers,
        )
        for adapter in adapters:
            self.context.register_capability(adapter)
        return installation
