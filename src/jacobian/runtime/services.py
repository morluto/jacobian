"""Foundational services owned by one Jacobian runtime."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.artifacts import ArtifactService
from jacobian.capabilities import CapabilityAdapter, CapabilityService
from jacobian.claim_decomposition_capabilities import (
    ClaimDecompositionInstallation,
    install_claim_decomposition_capabilities,
)
from jacobian.claims import ClaimValidationService
from jacobian.conjectures import ConjectureService
from jacobian.evaluation import EvaluationService
from jacobian.experiment_router import ExperimentRouter
from jacobian.experiments import ExperimentService
from jacobian.matrices.linear import LinearArtifactService
from jacobian.matrices.normal_forms import MatrixNormalFormArtifactService
from jacobian.memory import ResearchMemory
from jacobian.operation_installation import OperationInstaller
from jacobian.plugin_execution import PluginExecutor
from jacobian.plugins.registry import PluginRegistry
from jacobian.polynomial_expressions import PolynomialExpressionArtifactService
from jacobian.polytope import PolytopeService
from jacobian.references import ReferenceInstaller
from jacobian.registry import CheckerRegistry
from jacobian.sat_smt.sat import SatArtifactService
from jacobian.sat_smt.smt import SmtArtifactService
from jacobian.schema_registry import SchemaRegistry
from jacobian.search import SearchService
from jacobian.shrinking import ShrinkService
from jacobian.store import ArtifactStore
from jacobian.structures import StructureService
from jacobian.transformations import TransformationService
from jacobian.verification import VerificationService
from jacobian.witnesses import WitnessSearchService
from jacobian.workspaces import WorkspaceService


@dataclass(slots=True)
class CoreServices:
    """Foundational persistence and registries shared across the portfolio."""

    store: ArtifactStore
    schemas: SchemaRegistry
    artifacts: ArtifactService
    operations: OperationInstaller
    sat: SatArtifactService
    smt: SmtArtifactService
    linear: LinearArtifactService
    matrix_normal_forms: MatrixNormalFormArtifactService
    polynomial_expressions: PolynomialExpressionArtifactService
    memory: ResearchMemory
    workspaces: WorkspaceService
    plugins: PluginRegistry
    checkers: CheckerRegistry
    capabilities: CapabilityService

    def close(self) -> None:
        self.store.close()


@dataclass(frozen=True, slots=True)
class ApplicationServices:
    """Application services assembled over one foundational service graph."""

    core: CoreServices
    claims: ClaimValidationService
    claim_decomposition: ClaimDecompositionInstallation
    claim_decomposition_adapters: tuple[CapabilityAdapter, ...]
    plugin_executor: PluginExecutor
    structures: StructureService
    transformations: TransformationService
    polytope: PolytopeService
    evaluation: EvaluationService
    experiments: ExperimentService
    verification: VerificationService
    witnesses: WitnessSearchService
    search: SearchService
    experiment_router: ExperimentRouter
    conjectures: ConjectureService
    shrinking: ShrinkService
    reference_installer: ReferenceInstaller

    def close(self) -> None:
        """Quiesce application-owned workers before foundational teardown."""

        failures: list[Exception] = []
        for close in (self.search.close, self.experiments.close):
            try:
                close()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise ExceptionGroup("application services did not quiesce", failures)


def build_application_services(core: CoreServices) -> ApplicationServices:
    """Build the capability-independent application service graph."""

    claims = ClaimValidationService(core.store, core.schemas, core.plugins)
    decomposition_adapters, claim_decomposition = (
        install_claim_decomposition_capabilities(
            core.store,
            core.schemas,
            core.artifacts,
        )
    )
    plugin_executor = PluginExecutor()
    structures = StructureService(
        core.store,
        core.schemas,
        core.plugins,
        plugin_executor,
    )
    transformations = TransformationService(
        core.store,
        core.schemas,
        core.plugins,
        plugin_executor,
    )
    polytope = PolytopeService(core.store, core.schemas)
    evaluation = EvaluationService(
        core.store,
        core.schemas,
        core.plugins,
        claims,
        plugin_executor,
    )
    experiments = ExperimentService(
        core.store,
        core.schemas,
        core.plugins,
        claims,
        plugin_executor,
        evaluation,
        structures,
    )
    verification = VerificationService(
        core.store,
        core.checkers,
        checker_timeout_seconds=105,
    )
    witnesses = WitnessSearchService(
        core.store,
        core.schemas,
        core.plugins,
        claims,
        plugin_executor,
        verification,
    )
    search = SearchService(
        core.store,
        core.schemas,
        core.plugins,
        claims,
        plugin_executor,
        evaluation,
        witnesses,
        verification,
    )
    experiment_router = ExperimentRouter(experiments, search)
    conjectures = ConjectureService(
        core.store,
        core.schemas,
        core.plugins,
        claims,
        plugin_executor,
        search,
        verification,
    )
    shrinking = ShrinkService(
        core.store,
        core.schemas,
        core.plugins,
        claims,
        plugin_executor,
        verification,
    )
    reference_installer = ReferenceInstaller(
        core.store,
        core.schemas,
        core.artifacts,
        core.plugins,
        core.checkers,
        transformation_claim_schema_uri=transformations.claim_schema_uri,
    )
    return ApplicationServices(
        core=core,
        claims=claims,
        claim_decomposition=claim_decomposition,
        claim_decomposition_adapters=decomposition_adapters,
        plugin_executor=plugin_executor,
        structures=structures,
        transformations=transformations,
        polytope=polytope,
        evaluation=evaluation,
        experiments=experiments,
        verification=verification,
        witnesses=witnesses,
        search=search,
        experiment_router=experiment_router,
        conjectures=conjectures,
        shrinking=shrinking,
        reference_installer=reference_installer,
    )
