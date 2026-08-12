"""Installation of built-in application and domain capabilities."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.exact_domain_checkers import install_exact_domain_verification
from jacobian.finite_coverage import install_finite_coverage
from jacobian.graphs.coloring import install_graph_coloring_capabilities
from jacobian.graphs.installation import install_graph_capabilities
from jacobian.graphs.isomorphism import install_graph_isomorphism
from jacobian.installation.context import InstallationContext
from jacobian.operations import DomainBundle
from jacobian.polynomial_system_capabilities import (
    install_polynomial_system_capabilities,
)
from jacobian.polynomial_system_search import PolynomialSystemRationalSearchAdapter
from jacobian.polynomials import install_polynomial_capabilities
from jacobian.polytope_capabilities import PolytopeSeparationAdapter
from jacobian.portfolio.builtin import build_builtin_portfolio
from jacobian.portfolio.domain_installation import DomainBundleInstaller
from jacobian.portfolio.model import PortfolioPlan
from jacobian.portfolio.result import PortfolioInstallation
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
        result: PortfolioInstallation,
    ) -> None:
        """Install retained graph and coloring capabilities."""

        graph_adapters, result.graph = install_graph_capabilities(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        for graph_adapter in graph_adapters:
            self.context.register_capability(graph_adapter)
        coloring_adapters, result.graph_coloring = install_graph_coloring_capabilities(
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

    def install(
        self,
        services: RuntimeServices,
        result: PortfolioInstallation,
    ) -> None:
        """Install the core portfolio in its explicit dependency order."""

        ctx = self.context

        self.context.register_capability(PolytopeSeparationAdapter(services.polytope))
        finite_coverage_adapter, result.finite_coverage = install_finite_coverage(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        if finite_coverage_adapter is not None:
            self.context.register_capability(finite_coverage_adapter)

        self._install_graph_capabilities(ctx, services, result)

        portfolio = build_builtin_portfolio()
        bundle_result = DomainBundleInstaller(ctx).install(portfolio)
        result.domain_bundles = dict(bundle_result.installed)
        result.portfolio_diagnostics = bundle_result.diagnostics
        result.portfolio_outcomes = bundle_result.outcomes
        self.install_domain_verification(result, portfolio)

        if result.graph is not None:
            graph_isomorphism_adapter, result.graph_isomorphism = (
                install_graph_isomorphism(
                    ctx.store,
                    ctx.schemas,
                    ctx.artifacts,
                    ctx.verification,
                    ctx.checkers,
                    result.graph,
                    authorize_checker=ctx.authorizes_bundled_checkers,
                )
            )
            if graph_isomorphism_adapter is not None:
                self.context.register_capability(graph_isomorphism_adapter)

        polynomial_adapters, result.polynomial = install_polynomial_capabilities(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        for polynomial_adapter in polynomial_adapters:
            self.context.register_capability(polynomial_adapter)

        polynomial_system_adapter, result.polynomial_system = (
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
                result.polynomial_system,
            )
        )

        universal_adapters, result.universal_algebra = (
            install_universal_algebra_capabilities(
                ctx.store,
                ctx.schemas,
                ctx.artifacts,
                ctx.verification,
                ctx.checkers,
                authorize_checker=ctx.authorizes_bundled_checkers,
            )
        )
        for universal_adapter in universal_adapters:
            self.context.register_capability(universal_adapter)

    def install_domain_verification(
        self,
        result: PortfolioInstallation,
        plan: PortfolioPlan,
    ) -> None:
        ctx = self.context
        exact_bundles = {
            bundle.domain_id: (bundle, result.domain_bundles[bundle.domain_id])
            for bundle in plan.components
            if isinstance(bundle, DomainBundle)
            and bundle.checker_declarations
            and bundle.domain_id in result.domain_bundles
        }
        if not exact_bundles:
            return
        adapters, result.exact_domain_checkers = install_exact_domain_verification(
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
