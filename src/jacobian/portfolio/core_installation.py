"""Installation of built-in application and domain capabilities."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.atomic_capabilities import install_atomic_capabilities
from jacobian.builtin_capabilities import KnowledgeSearchAdapter
from jacobian.exact_domain_checkers import install_exact_domain_verification
from jacobian.finite_coverage import install_finite_coverage
from jacobian.finite_partition import install_finite_partition
from jacobian.geometry_verification import install_geometry_checker
from jacobian.graph_capabilities import install_graph_capabilities
from jacobian.graph_coloring_capabilities import install_graph_coloring_capabilities
from jacobian.graph_isomorphism import install_graph_isomorphism
from jacobian.graph_shrinking import install_graph_shrinking
from jacobian.installation.context import InstallationContext
from jacobian.matrix_capabilities import install_matrix_capabilities
from jacobian.matrix_determinant_capabilities import install_matrix_determinant_checker
from jacobian.matrix_rank_capabilities import install_matrix_rank_checker
from jacobian.polynomial_capabilities import install_polynomial_capabilities
from jacobian.polynomial_system_capabilities import (
    install_polynomial_system_capabilities,
)
from jacobian.polynomial_system_search import PolynomialSystemRationalSearchAdapter
from jacobian.portfolio.builtin import BUILTIN_PORTFOLIO
from jacobian.portfolio.domain_installation import DomainBundleInstaller
from jacobian.portfolio.result import PortfolioInstallation
from jacobian.runtime.services import ApplicationServices
from jacobian.universal_algebra_capabilities import (
    install_universal_algebra_capabilities,
)


@dataclass(frozen=True, slots=True)
class CoreApplicationInstaller:
    """Install core application adapters and domain-dependent checkers."""

    context: InstallationContext

    def install(
        self,
        application: ApplicationServices,
        result: PortfolioInstallation,
    ) -> None:
        """Install the core portfolio in its explicit dependency order."""

        ctx = self.context
        core = application.core

        for atomic_adapter in install_atomic_capabilities(ctx, application):
            self.context.register_capability(atomic_adapter)
        for claim_adapter in application.claim_decomposition_adapters:
            self.context.register_capability(claim_adapter)
        self.context.register_capability(KnowledgeSearchAdapter(core.memory))

        finite_partition_adapter, result.finite_partition = install_finite_partition(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        self.context.register_capability(finite_partition_adapter)
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

        graph_adapters, result.graph = install_graph_capabilities(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        for graph_adapter in graph_adapters:
            self.context.register_capability(graph_adapter)
        graph_shrinking_adapter, result.graph_shrinking = install_graph_shrinking(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            core.plugins,
            ctx.checkers,
            application.shrinking,
            result.graph,
            application.reference_installer,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        self.context.register_capability(graph_shrinking_adapter)

        coloring_adapters, result.graph_coloring = install_graph_coloring_capabilities(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            core.sat,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        for coloring_adapter in coloring_adapters:
            self.context.register_capability(coloring_adapter)

        bundle_result = DomainBundleInstaller(ctx).install(BUILTIN_PORTFOLIO)
        result.domain_bundles = dict(bundle_result.installed)
        result.portfolio_diagnostics = bundle_result.diagnostics
        result.portfolio_outcomes = bundle_result.outcomes
        self.install_domain_verification(result)

        graph_isomorphism_adapter, result.graph_isomorphism = install_graph_isomorphism(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            result.graph,
            authorize_checker=ctx.authorizes_bundled_checkers,
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

        matrix_adapters, result.matrix = install_matrix_capabilities(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
        )
        for matrix_adapter in matrix_adapters:
            self.context.register_capability(matrix_adapter)
        determinant_adapter, result.matrix_determinant_checker = (
            install_matrix_determinant_checker(
                ctx.store,
                ctx.schemas,
                ctx.artifacts,
                result.matrix,
                ctx.verification,
                ctx.checkers,
                authorize_checker=ctx.authorizes_bundled_checkers,
            )
        )
        if determinant_adapter is not None:
            self.context.register_capability(determinant_adapter)
        rank_adapter, result.matrix_rank_checker = install_matrix_rank_checker(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            result.matrix,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        if rank_adapter is not None:
            self.context.register_capability(rank_adapter)

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
                ctx.checkers,
                authorize_checker=ctx.authorizes_bundled_checkers,
            )
        )
        for universal_adapter in universal_adapters:
            self.context.register_capability(universal_adapter)

    def install_domain_verification(
        self,
        result: PortfolioInstallation,
    ) -> None:
        ctx = self.context
        geometry = result.domain_bundles.get("geometry")
        if geometry is not None:
            geometry_adapter, result.geometry_checker = install_geometry_checker(
                ctx.store,
                ctx.schemas,
                ctx.artifacts,
                geometry,
                ctx.verification,
                ctx.checkers,
                authorize_checker=ctx.authorizes_bundled_checkers,
            )
            if geometry_adapter is not None:
                self.context.register_capability(geometry_adapter)

        polynomial = result.domain_bundles.get("polynomial")
        matrix = result.domain_bundles.get("matrix")
        exact_bundles = {
            "polynomial": polynomial,
            "matrix": matrix,
            "certified_snf": result.domain_bundles.get("certified_snf"),
            "graph": result.domain_bundles.get("graph_optimization"),
            "graph_invariants": result.domain_bundles.get("graph_invariants"),
            "graph_symmetry": result.domain_bundles.get("graph_symmetry"),
            "combinatorics": result.domain_bundles.get("combinatorics"),
            "number_theory": result.domain_bundles.get("number_theory"),
            "probability": result.domain_bundles.get("probability"),
            "poset": result.domain_bundles.get("poset"),
            "projective_geometry": result.domain_bundles.get("projective_geometry"),
            "topology": result.domain_bundles.get("topology"),
        }
        if not any(exact_bundles.values()):
            return
        adapters, result.exact_domain_checkers = install_exact_domain_verification(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            ctx.verification,
            ctx.checkers,
            polynomial=polynomial,
            matrix=matrix,
            certified_snf=exact_bundles["certified_snf"],
            graph=exact_bundles["graph"],
            graph_invariants=exact_bundles["graph_invariants"],
            graph_symmetry=exact_bundles["graph_symmetry"],
            combinatorics=exact_bundles["combinatorics"],
            number_theory=exact_bundles["number_theory"],
            probability=exact_bundles["probability"],
            poset=exact_bundles["poset"],
            projective_geometry=exact_bundles["projective_geometry"],
            topology=exact_bundles["topology"],
            authorize=ctx.authorizes_bundled_checkers,
        )
        for adapter in adapters:
            self.context.register_capability(adapter)
