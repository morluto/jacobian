"""Foundational services owned by one Jacobian runtime."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.artifacts import ArtifactService
from jacobian.capability_service import CapabilityAdapter, CapabilityService
from jacobian.claim_decomposition_capabilities import (
    ClaimDecompositionInstallation,
    install_claim_decomposition_capabilities,
)
from jacobian.claims import ClaimValidationService
from jacobian.conjectures import ConjectureService
from jacobian.evaluation import EvaluationService
from jacobian.experiment_router import ExperimentRouter
from jacobian.experiments import ExperimentService
from jacobian.operation_installation import OperationInstaller
from jacobian.plugin_execution import PluginExecutor
from jacobian.plugins.registry import PluginRegistry
from jacobian.polynomial_expressions import PolynomialExpressionArtifactService
from jacobian.polytope import PolytopeService
from jacobian.reasoning_log import ReasoningLogService
from jacobian.references import ReferenceInstaller
from jacobian.registry import CheckerRegistry
from jacobian.sat_smt.sat import SatArtifactService
from jacobian.sat_smt.smt import SmtArtifactService
from jacobian.schema_registry import SchemaRegistry
from jacobian.search import SearchService
from jacobian.shrinking import ShrinkService
from jacobian.storage.repository import ArtifactRepository
from jacobian.structures import StructureService
from jacobian.transformations import TransformationService
from jacobian.verification import VerificationService
from jacobian.witnesses import WitnessSearchService


@dataclass(slots=True)
class CoreServices:
    """Foundational persistence and registries shared across the portfolio."""

    store: ArtifactRepository
    schemas: SchemaRegistry
    artifacts: ArtifactService
    operations: OperationInstaller
    sat: SatArtifactService
    smt: SmtArtifactService
    polynomial_expressions: PolynomialExpressionArtifactService
    plugins: PluginRegistry
    checkers: CheckerRegistry
    capabilities: CapabilityService
    reasoning_log: ReasoningLogService

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

        failures: list[BaseException] = []
        for close in (self.search.close, self.experiments.close):
            try:
                close()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            exception_failures = [
                failure for failure in failures if isinstance(failure, Exception)
            ]
            if len(exception_failures) == len(failures):
                raise ExceptionGroup(
                    "application services did not quiesce", exception_failures
                )
            raise BaseExceptionGroup("application services did not quiesce", failures)


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
