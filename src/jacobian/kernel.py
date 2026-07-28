"""Application composition root for the v0.2 research kernel."""

from __future__ import annotations

import logging
from pathlib import Path

from jacobian.artifacts import ArtifactService
from jacobian.atomic_capabilities import install_atomic_capabilities
from jacobian.builtin_capabilities import (
    KnowledgeSearchAdapter,
    LeanCheckAdapter,
    LeanDeclarationInspectAdapter,
    LeanDeclarationSearchAdapter,
    LeanDependencyGraphAdapter,
)
from jacobian.cadical import install_cadical_capabilities
from jacobian.capabilities import (
    CapabilityAdapter,
    CapabilityPolicy,
    CapabilityService,
    load_capability_adapter,
)
from jacobian.claim_decomposition_capabilities import (
    install_claim_decomposition_capabilities,
)
from jacobian.claims import ClaimValidationService
from jacobian.conjectures import ConjectureService
from jacobian.contracts.capabilities import (
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
)
from jacobian.contracts.lean import LeanDependencyGraphArtifact, LeanEnvironment
from jacobian.cvc5 import install_cvc5_capability
from jacobian.domains.builtins import BUILTIN_DOMAIN_BUNDLES
from jacobian.evaluation import EvaluationService
from jacobian.exact_domain_checkers import (
    ExactDomainCheckerInstallation,
    install_exact_domain_verification,
)
from jacobian.experiment_router import ExperimentRouter
from jacobian.experiments import ExperimentService
from jacobian.finite_coverage import (
    FiniteCoverageInstallation,
    install_finite_coverage,
)
from jacobian.finite_partition import (
    FinitePartitionInstallation,
    install_finite_partition,
)
from jacobian.flint_hnf import install_python_flint_hnf_capability
from jacobian.flint_linear import (
    install_python_flint_inconsistency_capability,
    install_python_flint_linear_capability,
)
from jacobian.geometry_verification import (
    GeometryCheckerInstallation,
    install_geometry_checker,
)
from jacobian.graph_capabilities import GraphInstallation, install_graph_capabilities
from jacobian.graph_coloring_capabilities import (
    GraphColoringInstallation,
    install_graph_coloring_capabilities,
)
from jacobian.graph_composition_capabilities import (
    GraphCompositionInstallation,
    install_graph_composition_capabilities,
)
from jacobian.graph_isomorphism import (
    GraphIsomorphismInstallation,
    install_graph_isomorphism,
)
from jacobian.graph_shrinking import (
    GraphShrinkingInstallation,
    install_graph_shrinking,
)
from jacobian.implementation import cached_package_digests
from jacobian.lean import LeanService
from jacobian.lean_declarations import (
    LeanDeclarationService,
    installed_lean_declaration_service,
)
from jacobian.lean_exploration import (
    LeanExplorationInstallation,
    install_lean_exploration_capabilities,
)
from jacobian.lean_proof_edit import (
    LeanProofEditInstallation,
    install_lean_proof_edit_capability,
)
from jacobian.lean_statement_capabilities import (
    LeanStatementInstallation,
    install_lean_statement_capabilities,
)
from jacobian.linear import LinearArtifactService, install_linear_artifacts
from jacobian.linear_capabilities import (
    LinearRationalInconsistencyCheckerInstallation,
    LinearRationalSolutionCheckerInstallation,
    install_linear_rational_inconsistency_checker,
    install_linear_rational_solution_checker,
)
from jacobian.matrix_capabilities import (
    MatrixInstallation,
    install_matrix_capabilities,
)
from jacobian.matrix_determinant_capabilities import (
    MatrixDeterminantCheckerInstallation,
    install_matrix_determinant_checker,
)
from jacobian.matrix_normal_form_capabilities import (
    MatrixNormalFormCheckerInstallation,
    install_matrix_normal_form_checker,
)
from jacobian.matrix_normal_forms import (
    MatrixNormalFormArtifactService,
    install_matrix_normal_form_artifacts,
)
from jacobian.matrix_rank_capabilities import (
    MatrixRankCheckerInstallation,
    install_matrix_rank_checker,
)
from jacobian.memory import ResearchMemory
from jacobian.operation_installation import (
    InstalledDomainBundle,
    OperationInstaller,
)
from jacobian.operations import DomainBundle
from jacobian.plugin_execution import PluginExecutor
from jacobian.plugins.registry import PluginRegistry
from jacobian.polynomial_capabilities import (
    PolynomialInstallation,
    install_polynomial_capabilities,
)
from jacobian.polynomial_expression_capabilities import (
    PolynomialExpressionCheckerInstallation,
    install_polynomial_expression_checker,
)
from jacobian.polynomial_expressions import (
    PolynomialExpressionArtifactService,
    install_polynomial_expression_artifacts,
)
from jacobian.polynomial_interval_capabilities import (
    PolynomialIntervalInstallation,
    install_polynomial_interval_capabilities,
)
from jacobian.polynomial_positivity_capabilities import (
    PolynomialPositivityInstallation,
    install_polynomial_positivity_capabilities,
)
from jacobian.polynomial_system_capabilities import (
    PolynomialSystemInstallation,
    install_polynomial_system_capabilities,
)
from jacobian.polynomial_system_search import PolynomialSystemRationalSearchAdapter
from jacobian.polytope import PolytopeService
from jacobian.provider_runtime import (
    cadical_provider_runtime,
    carcara_provider_runtime,
    cvc5_provider_runtime,
    drat_trim_provider_runtime,
    lean_provider_runtime,
    python_flint_hnf_provider_runtime,
    python_flint_provider_runtime,
    sympy_polynomial_normalization_provider_runtime,
)
from jacobian.references import (
    REFERENCE_INSTALLATION_DOMAINS,
    LeanCheckerInstallation,
    PolytopeCheckerInstallation,
    ReferenceInstallation,
    ReferenceInstaller,
)
from jacobian.registry import CheckerRegistry
from jacobian.sat import SatArtifactService, install_sat_artifacts
from jacobian.sat_capabilities import (
    SatAssignmentCheckerInstallation,
    SatCnfMaterializationAdapter,
    SatUnsatProofCheckerInstallation,
    install_sat_assignment_checker,
    install_sat_unsat_proof_checker,
)
from jacobian.sat_lrat_capabilities import (
    install_sat_lrat_verifier,
)
from jacobian.schema_registry import SchemaRegistry
from jacobian.search import SearchService
from jacobian.shrinking import ShrinkService
from jacobian.smt import SmtArtifactService, install_smt_artifacts
from jacobian.smt_capabilities import (
    SmtUnsatProofCheckerInstallation,
    install_smt_unsat_proof_checker,
)
from jacobian.store import ArtifactStore
from jacobian.structures import StructureService
from jacobian.sympy_polynomial_normalization import (
    install_sympy_polynomial_normalization_capability,
)
from jacobian.transformations import TransformationService
from jacobian.universal_algebra_capabilities import (
    UniversalAlgebraInstallation,
    install_universal_algebra_capabilities,
)
from jacobian.verification import VerificationService
from jacobian.witnesses import WitnessSearchService
from jacobian.workspaces import WorkspaceService

_LOGGER = logging.getLogger(__name__)


class JacobianKernel:
    """Local v0.2 services over one content-addressed store."""

    def __init__(
        self,
        root: str | Path,
        *,
        install_references: bool = False,
        hydrate_authorized: bool = False,
        capability_adapter_entrypoints: tuple[str, ...] = (),
        capability_exclusions: frozenset[str] = frozenset(),
        capability_policy: CapabilityPolicy | None = None,
    ) -> None:
        """Compose services over ``root``.

        ``install_references`` authorizes bundled reference checkers and
        installs reference plugins. ``hydrate_authorized`` reconstitutes
        process-local verify adapters from checkers already authorized in the
        store without writing new authorization. The two flags are mutually
        exclusive.
        """

        if install_references and hydrate_authorized:
            raise ValueError(
                "install_references and hydrate_authorized are mutually exclusive"
            )
        self.store = ArtifactStore(root)
        self._initialize(
            install_references=install_references,
            hydrate_authorized=hydrate_authorized,
            capability_adapter_entrypoints=capability_adapter_entrypoints,
            capability_exclusions=capability_exclusions,
            capability_policy=capability_policy,
        )

    def _initialize(
        self,
        *,
        install_references: bool,
        hydrate_authorized: bool,
        capability_adapter_entrypoints: tuple[str, ...],
        capability_exclusions: frozenset[str],
        capability_policy: CapabilityPolicy | None,
    ) -> None:
        """Assemble foundational services, then install the portfolio atomically."""

        # Construction-time exclusions support controlled portfolio ablations.
        # They are not a runtime authorization or access-control mechanism.
        self._capability_exclusions = capability_exclusions
        self.schemas = SchemaRegistry(self.store)
        self.artifacts = ArtifactService(self.store, self.schemas)
        self.operation_installer = OperationInstaller(
            self.store,
            self.schemas,
            self.artifacts,
        )
        self.domain_bundles: dict[str, InstalledDomainBundle] = {}
        self.sat: SatArtifactService = install_sat_artifacts(
            self.store,
            self.schemas,
            self.artifacts,
        )
        self.smt: SmtArtifactService = install_smt_artifacts(
            self.store,
            self.schemas,
            self.artifacts,
        )
        self.linear: LinearArtifactService = install_linear_artifacts(
            self.store,
            self.schemas,
            self.artifacts,
        )
        self.matrix_normal_forms: MatrixNormalFormArtifactService = (
            install_matrix_normal_form_artifacts(
                self.store,
                self.schemas,
                self.artifacts,
            )
        )
        self.polynomial_expressions: PolynomialExpressionArtifactService = (
            install_polynomial_expression_artifacts(
                self.store,
                self.schemas,
                self.artifacts,
            )
        )
        self.memory = ResearchMemory(self.store, self.schemas)
        self.workspaces = WorkspaceService(self.store, self.schemas)
        self.plugins = PluginRegistry(self.store)
        self.checkers = CheckerRegistry(self.store)
        self.checkers.bind_existing_when_omitted = hydrate_authorized
        self.claims = ClaimValidationService(
            self.store,
            self.schemas,
            self.plugins,
        )
        claim_decomposition_adapters, self.claim_decomposition = (
            install_claim_decomposition_capabilities(
                self.store,
                self.schemas,
                self.artifacts,
            )
        )
        self.plugin_executor = PluginExecutor()
        self.structures = StructureService(
            self.store,
            self.schemas,
            self.plugins,
            self.plugin_executor,
        )
        self.transformations = TransformationService(
            self.store,
            self.schemas,
            self.plugins,
            self.plugin_executor,
        )
        self.polytope = PolytopeService(self.store, self.schemas)
        self.evaluation = EvaluationService(
            self.store,
            self.schemas,
            self.plugins,
            self.claims,
            self.plugin_executor,
        )
        self.experiments = ExperimentService(
            self.store,
            self.schemas,
            self.plugins,
            self.claims,
            self.plugin_executor,
            self.evaluation,
            self.structures,
        )
        self.verification = VerificationService(
            self.store,
            self.checkers,
            checker_timeout_seconds=105,
        )
        self.witnesses = WitnessSearchService(
            self.store,
            self.schemas,
            self.plugins,
            self.claims,
            self.plugin_executor,
            self.verification,
        )
        self.search = SearchService(
            self.store,
            self.schemas,
            self.plugins,
            self.claims,
            self.plugin_executor,
            self.evaluation,
            self.witnesses,
            self.verification,
        )
        self.experiment_router = ExperimentRouter(self.experiments, self.search)
        self.conjectures = ConjectureService(
            self.store,
            self.schemas,
            self.plugins,
            self.claims,
            self.plugin_executor,
            self.search,
            self.verification,
        )
        self.shrinking = ShrinkService(
            self.store,
            self.schemas,
            self.plugins,
            self.claims,
            self.plugin_executor,
            self.verification,
        )
        self.reference_installer = ReferenceInstaller(
            self.store,
            self.schemas,
            self.artifacts,
            self.plugins,
            self.checkers,
            transformation_claim_schema_uri=(self.transformations.claim_schema_uri),
        )
        self.references: dict[str, ReferenceInstallation] = {}
        self.polytope_checkers: PolytopeCheckerInstallation | None = None
        self.lean_checkers: dict[LeanEnvironment, LeanCheckerInstallation] = {}
        self.lean: LeanService | None = None
        self.lean_runtime: CapabilityProviderRuntime | None = None
        self.lean_proof_edit: LeanProofEditInstallation | None = None
        self.lean_declarations: LeanDeclarationService | None = None
        self.lean_exploration: LeanExplorationInstallation | None = None
        self.geometry_checker: GeometryCheckerInstallation | None = None
        self.exact_domain_checkers: ExactDomainCheckerInstallation | None = None
        self.capabilities = CapabilityService(
            self.store,
            self.memory,
            policy=capability_policy,
        )
        with (
            self.checkers.policy_transaction(),
            self.store.transaction(),
            cached_package_digests(),
        ):
            self._install_capability_portfolio(
                install_references=install_references,
                capability_adapter_entrypoints=capability_adapter_entrypoints,
                claim_decomposition_adapters=claim_decomposition_adapters,
            )

    def _install_capability_portfolio(
        self,
        *,
        install_references: bool,
        capability_adapter_entrypoints: tuple[str, ...],
        claim_decomposition_adapters: tuple[CapabilityAdapter, ...],
    ) -> None:
        """Install capability descriptors and optional checker authority."""

        self.register_capability(SatCnfMaterializationAdapter(self.sat))
        self.sat_assignment_checker: SatAssignmentCheckerInstallation
        sat_assignment_adapter, self.sat_assignment_checker = (
            install_sat_assignment_checker(
                self.store,
                self.schemas,
                self.artifacts,
                self.sat,
                self.verification,
                self.checkers,
                authorize_checker=install_references,
            )
        )
        if sat_assignment_adapter is not None:
            self.register_capability(sat_assignment_adapter)
        self.drat_trim_runtime: CapabilityProviderRuntime = drat_trim_provider_runtime()
        self.sat_unsat_proof_checker: SatUnsatProofCheckerInstallation
        proof_adapter, self.sat_unsat_proof_checker = install_sat_unsat_proof_checker(
            self.store,
            self.schemas,
            self.artifacts,
            self.sat,
            self.verification,
            self.checkers,
            self.drat_trim_runtime,
            authorize_checker=install_references,
        )
        if proof_adapter is not None:
            self.register_capability(proof_adapter)
        lrat_adapter, self.sat_lrat = install_sat_lrat_verifier(
            self.store,
            self.schemas,
            self.artifacts,
            self.sat,
            self.verification,
            self.checkers,
            authorize_checker=install_references,
        )
        if lrat_adapter is not None:
            self.register_capability(lrat_adapter)
        self.carcara_runtime: CapabilityProviderRuntime = carcara_provider_runtime()
        self.smt_unsat_proof_checker: SmtUnsatProofCheckerInstallation
        smt_proof_adapter, self.smt_unsat_proof_checker = (
            install_smt_unsat_proof_checker(
                self.store,
                self.schemas,
                self.artifacts,
                self.smt,
                self.verification,
                self.checkers,
                self.carcara_runtime,
                authorize_checker=install_references,
            )
        )
        if smt_proof_adapter is not None:
            self.register_capability(smt_proof_adapter)
        self.linear_solution_checker: LinearRationalSolutionCheckerInstallation
        linear_verification_adapter, self.linear_solution_checker = (
            install_linear_rational_solution_checker(
                self.store,
                self.schemas,
                self.artifacts,
                self.linear,
                self.verification,
                self.checkers,
                authorize_checker=install_references,
            )
        )
        if linear_verification_adapter is not None:
            self.register_capability(linear_verification_adapter)
        self.linear_inconsistency_checker: (
            LinearRationalInconsistencyCheckerInstallation
        )
        (
            linear_inconsistency_verification_adapter,
            self.linear_inconsistency_checker,
        ) = install_linear_rational_inconsistency_checker(
            self.store,
            self.schemas,
            self.artifacts,
            self.linear,
            self.verification,
            self.checkers,
            authorize_checker=install_references,
        )
        if linear_inconsistency_verification_adapter is not None:
            self.register_capability(linear_inconsistency_verification_adapter)
        self._install_python_flint_capabilities()
        self._install_matrix_normal_form_capabilities(
            authorize_checker=install_references
        )
        self._install_polynomial_expression_capabilities(
            authorize_checker=install_references
        )
        self.cadical_runtime: CapabilityProviderRuntime = cadical_provider_runtime()
        if (
            self.cadical_runtime.availability
            is CapabilityProviderAvailability.AVAILABLE
        ):
            try:
                cadical_adapters = install_cadical_capabilities(
                    self.sat,
                    self.cadical_runtime,
                )
            except (OSError, ValueError) as exc:
                _LOGGER.warning(
                    "CaDiCaL SAT exploration is not installed: %s",
                    exc,
                )
            else:
                for cadical_adapter in cadical_adapters:
                    self.register_capability(cadical_adapter)
        self.cvc5_runtime: CapabilityProviderRuntime = cvc5_provider_runtime()
        if self.cvc5_runtime.availability is CapabilityProviderAvailability.AVAILABLE:
            try:
                cvc5_adapter = install_cvc5_capability(
                    self.smt,
                    self.cvc5_runtime,
                )
            except (OSError, ValueError) as exc:
                _LOGGER.warning(
                    "cvc5 SMT proof exploration is not installed: %s",
                    exc,
                )
            else:
                self.register_capability(cvc5_adapter)
        for atomic_adapter in install_atomic_capabilities(self):
            self.register_capability(atomic_adapter)
        for claim_decomposition_adapter in claim_decomposition_adapters:
            self.register_capability(claim_decomposition_adapter)
        self.register_capability(KnowledgeSearchAdapter(self.memory))
        self.finite_partition: FinitePartitionInstallation
        finite_partition, self.finite_partition = install_finite_partition(
            self.store,
            self.schemas,
            self.artifacts,
            self.verification,
            self.checkers,
            authorize_checker=install_references,
        )
        self.register_capability(finite_partition)
        self.finite_coverage: FiniteCoverageInstallation
        finite_coverage, self.finite_coverage = install_finite_coverage(
            self.store,
            self.schemas,
            self.artifacts,
            self.verification,
            self.checkers,
            authorize_checker=install_references,
        )
        if finite_coverage is not None:
            self.register_capability(finite_coverage)
        self.graph: GraphInstallation
        graph_adapters, self.graph = install_graph_capabilities(
            self.store,
            self.schemas,
            self.artifacts,
            self.checkers,
            authorize_checker=install_references,
        )
        for graph_adapter in graph_adapters:
            self.register_capability(graph_adapter)
        self.graph_shrinking: GraphShrinkingInstallation
        graph_shrinking, self.graph_shrinking = install_graph_shrinking(
            self.store,
            self.schemas,
            self.artifacts,
            self.plugins,
            self.checkers,
            self.shrinking,
            self.graph,
            self.reference_installer,
            authorize_checker=install_references,
        )
        self.register_capability(graph_shrinking)
        self._install_graph_coloring_capabilities(install_references)
        self._install_builtin_domain_bundles()
        self._install_builtin_domain_verification(install_references)
        self.graph_isomorphism: GraphIsomorphismInstallation
        graph_isomorphism, self.graph_isomorphism = install_graph_isomorphism(
            self.store,
            self.schemas,
            self.artifacts,
            self.verification,
            self.checkers,
            self.graph,
            authorize_checker=install_references,
        )
        if graph_isomorphism is not None:
            self.register_capability(graph_isomorphism)
        self.polynomial: PolynomialInstallation
        polynomial_adapters, self.polynomial = install_polynomial_capabilities(
            self.store,
            self.schemas,
            self.artifacts,
            self.verification,
            self.checkers,
            authorize_checker=install_references,
        )
        for polynomial_adapter in polynomial_adapters:
            self.register_capability(polynomial_adapter)
        self.matrix: MatrixInstallation
        matrix_adapters, self.matrix = install_matrix_capabilities(
            self.store,
            self.schemas,
            self.artifacts,
        )
        for matrix_adapter in matrix_adapters:
            self.register_capability(matrix_adapter)
        self.matrix_determinant_checker: MatrixDeterminantCheckerInstallation
        determinant_verification, self.matrix_determinant_checker = (
            install_matrix_determinant_checker(
                self.store,
                self.schemas,
                self.artifacts,
                self.matrix,
                self.verification,
                self.checkers,
                authorize_checker=install_references,
            )
        )
        if determinant_verification is not None:
            self.register_capability(determinant_verification)
        self.matrix_rank_checker: MatrixRankCheckerInstallation
        rank_verification, self.matrix_rank_checker = install_matrix_rank_checker(
            self.store,
            self.schemas,
            self.artifacts,
            self.matrix,
            self.verification,
            self.checkers,
            authorize_checker=install_references,
        )
        if rank_verification is not None:
            self.register_capability(rank_verification)
        self.polynomial_system: PolynomialSystemInstallation
        polynomial_system_adapter, self.polynomial_system = (
            install_polynomial_system_capabilities(
                self.store,
                self.schemas,
                self.artifacts,
                self.verification,
                self.checkers,
                authorize_checker=install_references,
            )
        )
        if polynomial_system_adapter is not None:
            self.register_capability(polynomial_system_adapter)
        self.register_capability(
            PolynomialSystemRationalSearchAdapter(
                self.artifacts, self.polynomial_system
            )
        )
        self.universal_algebra: UniversalAlgebraInstallation
        universal_algebra_adapters, self.universal_algebra = (
            install_universal_algebra_capabilities(
                self.store,
                self.schemas,
                self.artifacts,
                self.checkers,
                authorize_checker=install_references,
            )
        )
        for universal_algebra_adapter in universal_algebra_adapters:
            self.register_capability(universal_algebra_adapter)
        self._install_resource_capabilities(install_references)
        if install_references or (
            self.checkers.bind_existing_when_omitted
            and self.plugins.has_any_domain(REFERENCE_INSTALLATION_DOMAINS)
        ):
            self._install_authorized_references()
        for entrypoint in capability_adapter_entrypoints:
            self.register_capability(load_capability_adapter(entrypoint, self))

    def _install_authorized_references(self) -> None:
        self.references = self.reference_installer.install_all()
        self.polytope_checkers = self.reference_installer.install_polytope_checkers(
            claim_schema_uri=self.polytope.claim_schema_uri,
            semantics_uri=self.polytope.semantics_uri,
            point_schema_uri=self.polytope.point_schema_uri,
        )
        self.lean_checkers = self.reference_installer.install_lean_checkers()
        profiles = {
            environment.value: {
                "semantics_uri": installation.semantics_uri,
                "import_name": installation.import_name,
                "mathlib_commit": installation.mathlib_commit,
                "allowed_axioms": list(installation.allowed_axioms),
                "checker_timeout_seconds": installation.checker_timeout_seconds,
            }
            for environment, installation in sorted(
                self.lean_checkers.items(),
                key=lambda item: item[0].value,
            )
        }
        runtime = lean_provider_runtime(
            profiles=profiles,
            checker_ids=tuple(
                installation.checker_id
                for _, installation in sorted(
                    self.lean_checkers.items(),
                    key=lambda item: item[0].value,
                )
            ),
        )
        self.lean_runtime = runtime
        if runtime.availability is not CapabilityProviderAvailability.AVAILABLE:
            _LOGGER.warning(
                "lean.check is not installed: %s",
                runtime.diagnostic,
            )
            return
        try:
            self.lean_declarations = installed_lean_declaration_service(runtime)
        except (OSError, RuntimeError) as exc:
            _LOGGER.warning(
                "Lean declaration discovery is not installed: %s",
                exc,
            )
        self._install_lean_declaration_adapters(runtime)
        self.lean = LeanService(
            self.store,
            self.artifacts,
            self.verification,
            self.lean_checkers,
        )
        self.register_capability(LeanCheckAdapter(self.lean, runtime))
        lean_exploration_adapters, self.lean_exploration = (
            install_lean_exploration_capabilities(
                self.store,
                self.schemas,
                self.artifacts,
                self.lean_checkers,
                runtime,
            )
        )
        for lean_exploration_adapter in lean_exploration_adapters:
            self.register_capability(lean_exploration_adapter)
        self._install_lean_proof_edit()

    def _install_resource_capabilities(self, install_references: bool) -> None:
        """Install resource-mined domain atomics after core services exist."""
        self.graph_composition: GraphCompositionInstallation
        graph_adapters, self.graph_composition = install_graph_composition_capabilities(
            self.store,
            self.schemas,
            self.artifacts,
            semantics_uri=self.graph.semantics_uri,
            graph_schema_uri=self.graph.graph_schema_uri,
        )
        for adapter in graph_adapters:
            self.register_capability(adapter)

        self.polynomial_interval: PolynomialIntervalInstallation
        interval_adapters, self.polynomial_interval = (
            install_polynomial_interval_capabilities(
                self.store,
                self.schemas,
                self.artifacts,
                self.verification,
                self.checkers,
                authorize_checker=install_references,
            )
        )
        for interval_adapter in interval_adapters:
            if interval_adapter is not None:
                self.register_capability(interval_adapter)

        self.polynomial_positivity: PolynomialPositivityInstallation
        positivity_adapters, self.polynomial_positivity = (
            install_polynomial_positivity_capabilities(
                self.store,
                self.schemas,
                self.artifacts,
                self.verification,
                self.checkers,
                authorize_checker=install_references,
            )
        )
        for positivity_adapter in positivity_adapters:
            if positivity_adapter is not None:
                self.register_capability(positivity_adapter)

        self.lean_statement: LeanStatementInstallation
        lean_adapters, self.lean_statement = install_lean_statement_capabilities(
            self.store,
            self.schemas,
            self.artifacts,
        )
        for lean_statement_adapter in lean_adapters:
            self.register_capability(lean_statement_adapter)

    def _install_lean_proof_edit(self) -> None:
        if self.lean is None or self.lean_runtime is None:
            return
        adapter, self.lean_proof_edit = install_lean_proof_edit_capability(
            self.store,
            self.schemas,
            self.artifacts,
            self.lean,
            self.lean_runtime,
        )
        self.register_capability(adapter)

    def _install_lean_declaration_adapters(
        self,
        runtime: CapabilityProviderRuntime,
    ) -> None:
        if self.lean_declarations is None:
            return
        self.register_capability(
            LeanDeclarationSearchAdapter(self.lean_declarations, runtime)
        )
        self.register_capability(
            LeanDependencyGraphAdapter(
                self.lean_declarations,
                runtime,
                self.artifacts,
                semantics_uri=self.store.register_descriptor(
                    kind="semantics",
                    name="jacobian.lean4-declaration-dependencies",
                    version="1",
                    definition={
                        "description": (
                            "bounded constant dependencies extracted from elaborated "
                            "Lean declaration types and values"
                        ),
                        "provider_digest": runtime.digest,
                        "dependency_api": "Lean.Expr.getUsedConstantsAsSet",
                        "verification": "computed metadata; no theorem verification",
                    },
                ),
                dependency_graph_schema_uri=self.schemas.register(
                    name="jacobian.lean4-dependency-graph",
                    version="1",
                    schema=LeanDependencyGraphArtifact.model_json_schema(),
                ),
            )
        )
        self.register_capability(
            LeanDeclarationInspectAdapter(self.lean_declarations, runtime)
        )

    def _install_graph_coloring_capabilities(self, authorize_checker: bool) -> None:
        self.graph_coloring: GraphColoringInstallation
        graph_coloring_adapters, self.graph_coloring = (
            install_graph_coloring_capabilities(
                self.store,
                self.schemas,
                self.artifacts,
                self.sat,
                self.checkers,
                authorize_checker=authorize_checker,
            )
        )
        for graph_coloring_adapter in graph_coloring_adapters:
            self.register_capability(graph_coloring_adapter)

    def _install_matrix_normal_form_capabilities(
        self,
        *,
        authorize_checker: bool,
    ) -> None:
        self.matrix_normal_form_checker: MatrixNormalFormCheckerInstallation
        verification_adapter, self.matrix_normal_form_checker = (
            install_matrix_normal_form_checker(
                self.store,
                self.schemas,
                self.artifacts,
                self.matrix_normal_forms,
                self.verification,
                self.checkers,
                authorize_checker=authorize_checker,
            )
        )
        if verification_adapter is not None:
            self.register_capability(verification_adapter)

        self.python_flint_hnf_runtime: CapabilityProviderRuntime = (
            python_flint_hnf_provider_runtime()
        )
        if (
            self.python_flint_hnf_runtime.availability
            is not CapabilityProviderAvailability.AVAILABLE
        ):
            return
        try:
            adapter = install_python_flint_hnf_capability(
                self.matrix_normal_forms,
                self.python_flint_hnf_runtime,
            )
        except (OSError, ValueError) as exc:
            _LOGGER.warning(
                "Python-FLINT Hermite normal form is not installed: %s",
                exc,
            )
        else:
            self.register_capability(adapter)

    def _install_builtin_domain_bundles(self) -> None:
        for bundle in BUILTIN_DOMAIN_BUNDLES:
            self._install_capability_bundle(bundle)

    def _install_builtin_domain_verification(self, authorize: bool) -> None:
        geometry = self.domain_bundles.get("geometry")
        if geometry is not None:
            geometry_adapter, self.geometry_checker = install_geometry_checker(
                self.store,
                self.schemas,
                self.artifacts,
                geometry,
                self.verification,
                self.checkers,
                authorize_checker=authorize,
            )
            if geometry_adapter is not None:
                self.register_capability(geometry_adapter)

        polynomial = self.domain_bundles.get("polynomial")
        matrix = self.domain_bundles.get("matrix")
        graph = self.domain_bundles.get("graph_optimization")
        graph_invariants = self.domain_bundles.get("graph_invariants")
        projective_geometry = self.domain_bundles.get("projective_geometry")
        if polynomial is None or matrix is None:
            return
        adapters, self.exact_domain_checkers = install_exact_domain_verification(
            self.store,
            self.schemas,
            self.artifacts,
            self.verification,
            self.checkers,
            polynomial=polynomial,
            matrix=matrix,
            graph=graph,
            graph_invariants=graph_invariants,
            projective_geometry=projective_geometry,
            authorize=authorize,
        )
        for adapter in adapters:
            self.register_capability(adapter)

    def _install_capability_bundle(self, bundle: DomainBundle) -> None:
        if (
            bundle.provider_runtime.availability
            is not CapabilityProviderAvailability.AVAILABLE
        ):
            return
        installation = self.operation_installer.install(bundle)
        if bundle.domain_id in self.domain_bundles:
            raise ValueError(f"duplicate capability bundle: {bundle.domain_id}")
        self.domain_bundles[bundle.domain_id] = installation
        for adapter in installation.adapters:
            self.register_capability(adapter)

    def _install_python_flint_capabilities(self) -> None:
        """Install exact rational linear producers when the pin is available."""

        self.python_flint_runtime = python_flint_provider_runtime()
        if (
            self.python_flint_runtime.availability
            is not CapabilityProviderAvailability.AVAILABLE
        ):
            return
        try:
            solution_adapter = install_python_flint_linear_capability(
                self.linear,
                self.python_flint_runtime,
            )
        except (OSError, ValueError) as exc:
            _LOGGER.warning(
                "Python-FLINT rational solution exploration is not installed: %s",
                exc,
            )
        else:
            self.register_capability(solution_adapter)
        try:
            inconsistency_adapter = install_python_flint_inconsistency_capability(
                self.linear,
                self.python_flint_runtime,
            )
        except (OSError, ValueError) as exc:
            _LOGGER.warning(
                "Python-FLINT rational inconsistency exploration is not installed: %s",
                exc,
            )
        else:
            self.register_capability(inconsistency_adapter)

    def _install_polynomial_expression_capabilities(
        self,
        *,
        authorize_checker: bool,
    ) -> None:
        self.polynomial_expression_checker: PolynomialExpressionCheckerInstallation
        verification_adapter, self.polynomial_expression_checker = (
            install_polynomial_expression_checker(
                self.store,
                self.schemas,
                self.artifacts,
                self.polynomial_expressions,
                self.verification,
                self.checkers,
                authorize_checker=authorize_checker,
            )
        )
        if verification_adapter is not None:
            self.register_capability(verification_adapter)

        self.sympy_polynomial_normalization_runtime: CapabilityProviderRuntime = (
            sympy_polynomial_normalization_provider_runtime()
        )
        if (
            self.sympy_polynomial_normalization_runtime.availability
            is not CapabilityProviderAvailability.AVAILABLE
        ):
            return
        try:
            adapter = install_sympy_polynomial_normalization_capability(
                self.polynomial_expressions,
                self.sympy_polynomial_normalization_runtime,
            )
        except (OSError, ValueError) as exc:
            _LOGGER.warning(
                "SymPy typed polynomial normalization is not installed: %s",
                exc,
            )
        else:
            self.register_capability(adapter)

    def register_capability(self, adapter: CapabilityAdapter) -> None:
        """Install an operator-owned adapter without changing the kernel or MCP."""

        if adapter.descriptor.capability_id in self._capability_exclusions:
            _LOGGER.debug(
                "Capability %s excluded by operator configuration",
                adapter.descriptor.capability_id,
            )
            return
        self.capabilities.register(adapter)
