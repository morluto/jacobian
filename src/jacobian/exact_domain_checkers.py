"""Operator-controlled declarations for independent exact-operation replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.capabilities import CapabilityAdapter, CapabilityInvocationError
from jacobian.checker_artifacts import put_witness_envelope
from jacobian.checker_installation import CheckerInstaller
from jacobian.checker_operations import CheckerOperation, ExactReplayCheckerDeclaration
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityMode,
    CapabilityProviderRuntime,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import WitnessEnvelope
from jacobian.contracts.exact_domain_verification import (
    ExactDomainResultVerificationOutput,
    ExactDomainResultVerificationRequest,
)
from jacobian.contracts.results import (
    Conclusion,
    Execution,
    ExecutionStatus,
    Verification,
)
from jacobian.domains.graph_optimization.checkers import (
    GRAPH_OPTIMIZATION_EXACT_REPLAY_CHECKERS,
)
from jacobian.domains.matrix_lattice.checkers import MATRIX_EXACT_REPLAY_CHECKERS
from jacobian.domains.number_theory.checkers import (
    NUMBER_THEORY_EXACT_REPLAY_CHECKERS,
)
from jacobian.domains.polynomial.checkers import POLYNOMIAL_EXACT_REPLAY_CHECKERS
from jacobian.domains.posets.checkers import FINITE_POSET_EXACT_REPLAY_CHECKERS
from jacobian.domains.probability.checkers import PROBABILITY_EXACT_REPLAY_CHECKERS
from jacobian.domains.projective_geometry.checkers import (
    PROJECTIVE_GEOMETRY_EXACT_REPLAY_CHECKERS,
)
from jacobian.domains.topology.checkers import TOPOLOGY_EXACT_REPLAY_CHECKERS
from jacobian.operation_installation import InstalledDomainBundle
from jacobian.provider_runtime import (
    exact_domain_checker_provider_runtime,
    graded_syzygy_checker_provider_runtime,
    graph_exact_checker_provider_runtime,
    poset_exact_checker_provider_runtime,
    probability_exact_checker_provider_runtime,
    projective_arrangement_checker_provider_runtime,
    topology_exact_checker_provider_runtime,
)
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError, model_schema
from jacobian.store import ArtifactStore, StoredArtifact, StoreError
from jacobian.verification import VerificationService


@dataclass(frozen=True, slots=True)
class ExactDomainCheckerInstallation:
    """Authorized checker identities keyed by producer capability ID."""

    checker_ids: dict[str, str | None]
    provider_runtimes: dict[str, CapabilityProviderRuntime]
    witness_schema_uri: str | None = None


@dataclass(frozen=True, slots=True)
class _InstalledDeclaration:
    declaration: ExactReplayCheckerDeclaration
    input_schema_uri: str
    result_schema_uri: str
    semantics_uri: str
    checker_id: str


def _provider_runtime_key(declaration: ExactReplayCheckerDeclaration) -> str:
    if declaration.entrypoint_module == "jacobian_checkers.exact_domain_operations":
        return "python-flint"
    if declaration.entrypoint_module == "jacobian_checkers.graph_exact_operations":
        return "finite-graph"
    if (
        declaration.entrypoint_module
        == "jacobian_checkers.exact_probability_operations"
    ):
        return "finite-probability"
    if declaration.entrypoint_module == "jacobian_checkers.jacobian_syzygy":
        return "graded-syzygy"
    if declaration.entrypoint_module == "jacobian_checkers.projective_arrangements":
        return "projective-arrangement"
    if declaration.entrypoint_module == "jacobian_checkers.simplicial_topology":
        return "topology"
    if declaration.entrypoint_module == "jacobian_checkers.finite_posets":
        return "poset"
    raise ValueError(
        "exact replay checker declaration uses an unsupported provider runtime"
    )


def install_exact_domain_checkers(
    checkers: CheckerRegistry,
    *,
    polynomial: InstalledDomainBundle,
    matrix: InstalledDomainBundle,
    graph: InstalledDomainBundle | None = None,
    graph_invariants: InstalledDomainBundle | None = None,
    number_theory: InstalledDomainBundle | None = None,
    probability: InstalledDomainBundle | None = None,
    poset: InstalledDomainBundle | None = None,
    projective_geometry: InstalledDomainBundle | None = None,
    topology: InstalledDomainBundle | None = None,
    authorize: bool,
) -> ExactDomainCheckerInstallation:
    """Install independent exact replay against dynamically registered schemas."""

    installer = CheckerInstaller(checkers)
    provider_runtimes = {
        "python-flint": exact_domain_checker_provider_runtime(),
        "finite-graph": graph_exact_checker_provider_runtime(),
        "finite-probability": probability_exact_checker_provider_runtime(),
        "poset": poset_exact_checker_provider_runtime(),
        "graded-syzygy": graded_syzygy_checker_provider_runtime(),
        "projective-arrangement": (projective_arrangement_checker_provider_runtime()),
        "topology": topology_exact_checker_provider_runtime(),
    }
    checker_ids: dict[str, str | None] = {}
    declarations_by_id: dict[str, ExactReplayCheckerDeclaration] = {}
    for installed, declaration in (
        *((polynomial, item) for item in POLYNOMIAL_EXACT_REPLAY_CHECKERS),
        *((matrix, item) for item in MATRIX_EXACT_REPLAY_CHECKERS),
        *_available_graph_declaration_bundles(
            graph=graph,
            graph_invariants=graph_invariants,
        ),
        *(
            (number_theory, item)
            for item in NUMBER_THEORY_EXACT_REPLAY_CHECKERS
            if number_theory is not None
        ),
        *(
            (probability, item)
            for item in PROBABILITY_EXACT_REPLAY_CHECKERS
            if probability is not None
        ),
        *(
            (projective_geometry, item)
            for item in PROJECTIVE_GEOMETRY_EXACT_REPLAY_CHECKERS
            if projective_geometry is not None
        ),
        *(
            (poset, item)
            for item in FINITE_POSET_EXACT_REPLAY_CHECKERS
            if poset is not None
        ),
        *(
            (topology, item)
            for item in TOPOLOGY_EXACT_REPLAY_CHECKERS
            if topology is not None
        ),
    ):
        declarations_by_id[declaration.capability_id] = declaration
        runtime_key = _provider_runtime_key(declaration)
        operation = CheckerOperation(
            name=f"{declaration.capability_id} independent {declaration.replay_method}",
            entrypoint=(f"{declaration.entrypoint_module}:{declaration.function}"),
            evidence_kind=EvidenceKind.WITNESS,
            format_id=declaration.format_id,
            format_version="1",
            claim_schema_uris=(installed.input_schema_uris[declaration.request_model],),
            semantics_uris=(installed.semantics_uri,),
            candidate_schema_uris=(
                installed.result_schema_uris[declaration.capability_id],
            ),
            reason=declaration.reason,
            provider_runtime=provider_runtimes[runtime_key],
        )
        checker_ids[declaration.capability_id] = installer.install(
            operation,
            authorize=authorize,
        ).checker_id
    authorized_ids = {
        runtime_key: tuple(
            checker_id
            for capability_id, checker_id in checker_ids.items()
            if checker_id is not None
            and _provider_runtime_key(declarations_by_id[capability_id]) == runtime_key
        )
        for runtime_key in provider_runtimes
    }
    return ExactDomainCheckerInstallation(
        checker_ids=checker_ids,
        provider_runtimes={
            "python-flint": exact_domain_checker_provider_runtime(
                checker_ids=authorized_ids["python-flint"]
            ),
            "finite-graph": graph_exact_checker_provider_runtime(
                checker_ids=authorized_ids["finite-graph"]
            ),
            "finite-probability": probability_exact_checker_provider_runtime(
                checker_ids=authorized_ids["finite-probability"]
            ),
            "poset": poset_exact_checker_provider_runtime(
                checker_ids=authorized_ids["poset"]
            ),
            "graded-syzygy": graded_syzygy_checker_provider_runtime(
                checker_ids=authorized_ids["graded-syzygy"]
            ),
            "projective-arrangement": (
                projective_arrangement_checker_provider_runtime(
                    checker_ids=authorized_ids["projective-arrangement"]
                )
            ),
            "topology": topology_exact_checker_provider_runtime(
                checker_ids=authorized_ids["topology"]
            ),
        },
    )


def install_exact_domain_verification(
    store: ArtifactStore,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    polynomial: InstalledDomainBundle,
    matrix: InstalledDomainBundle,
    graph: InstalledDomainBundle | None = None,
    graph_invariants: InstalledDomainBundle | None = None,
    number_theory: InstalledDomainBundle | None = None,
    probability: InstalledDomainBundle | None = None,
    poset: InstalledDomainBundle | None = None,
    projective_geometry: InstalledDomainBundle | None = None,
    topology: InstalledDomainBundle | None = None,
    authorize: bool,
) -> tuple[tuple[CapabilityAdapter, ...], ExactDomainCheckerInstallation]:
    """Authorize exact replay and expose domain-owned verification capabilities."""

    installed = install_exact_domain_checkers(
        checkers,
        polynomial=polynomial,
        matrix=matrix,
        graph=graph,
        graph_invariants=graph_invariants,
        number_theory=number_theory,
        probability=probability,
        poset=poset,
        projective_geometry=projective_geometry,
        topology=topology,
        authorize=authorize,
    )
    witness_schema_uri = schemas.register_model(
        name="jacobian.witness-envelope",
        version="1",
        model=WitnessEnvelope,
    )
    installation = ExactDomainCheckerInstallation(
        checker_ids=installed.checker_ids,
        witness_schema_uri=witness_schema_uri,
        provider_runtimes=installed.provider_runtimes,
    )
    if not any(
        checker_id is not None for checker_id in installation.checker_ids.values()
    ):
        return (), installation
    all_polynomial_declarations = tuple(
        _installed_declaration(polynomial, declaration, installation)
        for declaration in POLYNOMIAL_EXACT_REPLAY_CHECKERS
        if installation.checker_ids.get(declaration.capability_id) is not None
    )
    polynomial_declarations = tuple(
        declaration
        for declaration in all_polynomial_declarations
        if declaration.declaration.verification_capability_id is None
    )
    matrix_declarations = tuple(
        _installed_declaration(matrix, declaration, installation)
        for declaration in MATRIX_EXACT_REPLAY_CHECKERS
        if installation.checker_ids.get(declaration.capability_id) is not None
    )
    graph_declarations = tuple(
        _installed_declaration(
            bundle,
            declaration,
            installation,
        )
        for bundle, declaration in _available_graph_declaration_bundles(
            graph=graph,
            graph_invariants=graph_invariants,
        )
        if installation.checker_ids.get(declaration.capability_id) is not None
    )
    number_theory_declarations = (
        tuple(
            _installed_declaration(
                number_theory,
                declaration,
                installation,
            )
            for declaration in NUMBER_THEORY_EXACT_REPLAY_CHECKERS
            if installation.checker_ids.get(declaration.capability_id) is not None
        )
        if number_theory is not None
        else ()
    )
    projective_declarations = (
        tuple(
            _installed_declaration(
                projective_geometry,
                declaration,
                installation,
            )
            for declaration in PROJECTIVE_GEOMETRY_EXACT_REPLAY_CHECKERS
            if installation.checker_ids.get(declaration.capability_id) is not None
        )
        if projective_geometry is not None
        else ()
    )
    probability_declarations = (
        tuple(
            _installed_declaration(
                probability,
                declaration,
                installation,
            )
            for declaration in PROBABILITY_EXACT_REPLAY_CHECKERS
            if installation.checker_ids.get(declaration.capability_id) is not None
        )
        if probability is not None
        else ()
    )
    topology_declarations = (
        tuple(
            _installed_declaration(
                topology,
                declaration,
                installation,
            )
            for declaration in TOPOLOGY_EXACT_REPLAY_CHECKERS
            if installation.checker_ids.get(declaration.capability_id) is not None
        )
        if topology is not None
        else ()
    )
    poset_declarations = (
        tuple(
            _installed_declaration(
                poset,
                declaration,
                installation,
            )
            for declaration in FINITE_POSET_EXACT_REPLAY_CHECKERS
            if installation.checker_ids.get(declaration.capability_id) is not None
        )
        if poset is not None
        else ()
    )
    dedicated_declarations = (
        *(
            declaration
            for declaration in all_polynomial_declarations
            if declaration.declaration.verification_capability_id is not None
        ),
        *graph_declarations,
        *number_theory_declarations,
        *projective_declarations,
    )
    dedicated_adapters: tuple[CapabilityAdapter, ...] = tuple(
        ExactDomainResultVerificationAdapter(
            capability_id=_verification_metadata(declaration)[0],
            title=_verification_metadata(declaration)[1],
            description=_verification_metadata(declaration)[2],
            tags=declaration.declaration.verification_tags,
            store=store,
            schemas=schemas,
            artifacts=artifacts,
            verification=verification,
            declarations=(declaration,),
            witness_schema_uri=witness_schema_uri,
            provider_runtime=installation.provider_runtimes[
                _provider_runtime_key(declaration.declaration)
            ],
        )
        for declaration in dedicated_declarations
    )
    adapters: list[CapabilityAdapter] = []
    if polynomial_declarations:
        adapters.append(
            ExactDomainResultVerificationAdapter(
                capability_id="polynomial.result.verify",
                title="Verify an exact polynomial result",
                description=(
                    "Independently replay one supported stored polynomial result "
                    "against its exact input lineage."
                ),
                tags=("verification", "exact", "polynomial"),
                store=store,
                schemas=schemas,
                artifacts=artifacts,
                verification=verification,
                declarations=polynomial_declarations,
                witness_schema_uri=witness_schema_uri,
                provider_runtime=installation.provider_runtimes["python-flint"],
            )
        )
    if matrix_declarations:
        adapters.append(
            ExactDomainResultVerificationAdapter(
                capability_id="matrix.result.verify",
                title="Verify an exact matrix result",
                description=(
                    "Independently replay one supported stored matrix result "
                    "against its exact input lineage."
                ),
                tags=("verification", "exact", "matrix"),
                store=store,
                schemas=schemas,
                artifacts=artifacts,
                verification=verification,
                declarations=matrix_declarations,
                witness_schema_uri=witness_schema_uri,
                provider_runtime=installation.provider_runtimes["python-flint"],
            )
        )
    if probability_declarations:
        adapters.append(
            ExactDomainResultVerificationAdapter(
                capability_id="probability.result.verify",
                title="Verify an exact finite-probability result",
                description=(
                    "Independently replay one supported stored finite-probability "
                    "result against its exact input lineage."
                ),
                tags=("verification", "exact", "probability", "finite"),
                store=store,
                schemas=schemas,
                artifacts=artifacts,
                verification=verification,
                declarations=probability_declarations,
                witness_schema_uri=witness_schema_uri,
                provider_runtime=installation.provider_runtimes["finite-probability"],
            )
        )
    if topology_declarations:
        adapters.append(
            ExactDomainResultVerificationAdapter(
                capability_id="topology.result.verify",
                title="Verify an exact simplicial-topology result",
                description=(
                    "Independently reconstruct one finite complex and replay its "
                    "oriented boundaries or prime-field homology quotient evidence."
                ),
                tags=(
                    "verification",
                    "exact",
                    "topology",
                    "simplicial-homology",
                ),
                store=store,
                schemas=schemas,
                artifacts=artifacts,
                verification=verification,
                declarations=topology_declarations,
                witness_schema_uri=witness_schema_uri,
                provider_runtime=installation.provider_runtimes["topology"],
            )
        )
    if poset_declarations:
        adapters.append(
            ExactDomainResultVerificationAdapter(
                capability_id="poset.result.verify",
                title="Verify an exact finite-poset result",
                description=(
                    "Independently reconstruct one finite poset and replay its "
                    "closure, Dilworth witnesses, ideal recurrence, or Möbius values."
                ),
                tags=(
                    "verification",
                    "exact",
                    "poset",
                    "partial-order",
                ),
                store=store,
                schemas=schemas,
                artifacts=artifacts,
                verification=verification,
                declarations=poset_declarations,
                witness_schema_uri=witness_schema_uri,
                provider_runtime=installation.provider_runtimes["poset"],
            )
        )
    adapters.extend(dedicated_adapters)
    return tuple(adapters), installation


def _verification_metadata(
    installed: _InstalledDeclaration,
) -> tuple[str, str, str]:
    declaration = installed.declaration
    if (
        declaration.verification_capability_id is None
        or declaration.verification_title is None
        or declaration.verification_description is None
    ):
        raise ValueError(
            "separately exposed exact replay requires verification metadata"
        )
    return (
        declaration.verification_capability_id,
        declaration.verification_title,
        declaration.verification_description,
    )


def _available_graph_declaration_bundles(
    *,
    graph: InstalledDomainBundle | None,
    graph_invariants: InstalledDomainBundle | None,
) -> tuple[tuple[InstalledDomainBundle, ExactReplayCheckerDeclaration], ...]:
    available: list[tuple[InstalledDomainBundle, ExactReplayCheckerDeclaration]] = []
    for declaration in GRAPH_OPTIMIZATION_EXACT_REPLAY_CHECKERS:
        installed = tuple(
            bundle
            for bundle in (graph, graph_invariants)
            if bundle is not None
            and declaration.capability_id in bundle.result_schema_uris
        )
        if len(installed) > 1:
            raise ValueError(
                "graph exact replay declaration is owned by multiple bundles"
            )
        if installed:
            available.append((installed[0], declaration))
    return tuple(available)


def _installed_declaration(
    bundle: InstalledDomainBundle,
    declaration: ExactReplayCheckerDeclaration,
    installation: ExactDomainCheckerInstallation,
) -> _InstalledDeclaration:
    checker_id = installation.checker_ids[declaration.capability_id]
    if checker_id is None:
        raise ValueError("exact-domain checker is not authorized")
    return _InstalledDeclaration(
        declaration=declaration,
        input_schema_uri=bundle.input_schema_uris[declaration.request_model],
        result_schema_uri=bundle.result_schema_uris[declaration.capability_id],
        semantics_uri=bundle.semantics_uri,
        checker_id=checker_id,
    )


class ExactDomainArtifactError(ValueError):
    """A result is not one of the exactly bound supported producer artifacts."""


class ExactDomainResultVerificationAdapter:
    """Verify one stored exact producer result using independent bounded replay."""

    def __init__(
        self,
        *,
        capability_id: str,
        title: str,
        description: str,
        tags: tuple[str, ...],
        store: ArtifactStore,
        schemas: SchemaRegistry,
        artifacts: ArtifactService,
        verification: VerificationService,
        declarations: tuple[_InstalledDeclaration, ...],
        witness_schema_uri: str,
        provider_runtime: CapabilityProviderRuntime,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.artifacts = artifacts
        self.verification = verification
        self.declarations_by_schema = {
            declaration.result_schema_uri: declaration for declaration in declarations
        }
        self.witness_schema_uri = witness_schema_uri
        self._descriptor = CapabilityDescriptor(
            capability_id=capability_id,
            version="1",
            title=title,
            description=description,
            provider=provider_runtime.provider,
            provider_runtime=provider_runtime,
            modes=(CapabilityMode.VERIFY,),
            input_schema=model_schema(ExactDomainResultVerificationRequest),
            output_schema=model_schema(ExactDomainResultVerificationOutput),
            tags=tags,
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = ExactDomainResultVerificationRequest.model_validate(request.input)
        try:
            declaration, input_artifact, result_artifact, semantics_artifact = (
                self._resolve(validated.result_uri)
            )
        except (ExactDomainArtifactError, StoreError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_EXACT_DOMAIN_RESULT",
                    stage="artifact_resolution",
                    message=str(exc),
                    path="result_uri",
                    hint=(
                        "Pass a result_uri returned by one supported exact "
                        "polynomial, matrix, graph, probability, projective-"
                        "geometry, topology, or poset producer."
                    ),
                )
            ) from exc

        if not _checker_supports(
            declaration.declaration.capability_id,
            input_artifact.payload,
        ):
            output = ExactDomainResultVerificationOutput(
                status="UNSUPPORTED",
                conclusion="UNKNOWN",
                operation_id=declaration.declaration.capability_id,
                input_uri=input_artifact.artifact_uri,
                result_uri=result_artifact.artifact_uri,
                checker_id=declaration.checker_id,
                detail=(
                    "The authorized checker does not support this result's bounded "
                    "input scope; no mathematical conclusion follows."
                ),
            )
            return CapabilityResult(
                capability_id=self.descriptor.capability_id,
                capability_version=self.descriptor.version,
                mode=request.mode,
                execution=Execution(status=ExecutionStatus.COMPLETED),
                output=output.model_dump(mode="json"),
                scope=CapabilityScope(
                    description="the authorized independent checker's bounded scope",
                    parameters={
                        "operation_id": declaration.declaration.capability_id,
                        "scope_supported": False,
                    },
                    artifact_uri=input_artifact.artifact_uri,
                ),
                completeness=CapabilityCompleteness(
                    status=CapabilityCompletenessStatus.NOT_APPLICABLE,
                    basis="the result lies outside this checker's declared scope",
                    assurance_level=CapabilityAssuranceLevel.COMPUTED,
                ),
                assurance=CapabilityAssurance(
                    level=CapabilityAssuranceLevel.COMPUTED,
                    basis=(
                        "checker scope was evaluated without making a "
                        "mathematical conclusion"
                    ),
                ),
                artifact_uris=(
                    input_artifact.artifact_uri,
                    result_artifact.artifact_uri,
                ),
            )

        witness = put_witness_envelope(
            self.artifacts,
            witness_schema_uri=self.witness_schema_uri,
            witness_format=declaration.declaration.format_id,
            claim_artifact=input_artifact,
            semantics_artifact=semantics_artifact,
            candidate_artifact=result_artifact,
            payload={
                "operation_id": declaration.declaration.capability_id,
                "input_uri": input_artifact.artifact_uri,
                "result_uri": result_artifact.artifact_uri,
            },
            summary=(
                f"{declaration.declaration.capability_id} independent replay witness"
            ),
        )
        checked = self.verification.verify_witness(
            claim_uri=input_artifact.artifact_uri,
            candidate_uri=result_artifact.artifact_uri,
            witness_uri=witness.artifact_uri,
            checker_id=declaration.checker_id,
            include_artifact_metadata=True,
            include_semantics_artifact=True,
        )
        verified = (
            checked.execution.status is ExecutionStatus.COMPLETED
            and checked.conclusion is Conclusion.TRUE
            and checked.assurance.verification is Verification.VERIFIED
            and checked.verification_record_uri is not None
        )
        status: Literal["VERIFIED", "REJECTED", "TIMEOUT", "CANCELLED", "ERROR"]
        if verified:
            status = "VERIFIED"
        elif checked.execution.status is ExecutionStatus.COMPLETED:
            status = "REJECTED"
        elif checked.execution.status is ExecutionStatus.TIMEOUT:
            status = "TIMEOUT"
        elif checked.execution.status is ExecutionStatus.CANCELLED:
            status = "CANCELLED"
        else:
            status = "ERROR"
        detail = checked.execution.detail
        if detail is None and checked.input.errors:
            detail = checked.input.errors[0]
        if detail is None:
            detail = (
                "the authorized independent checker accepted the exact result"
                if verified
                else "the exact result was not independently accepted"
            )
        output = ExactDomainResultVerificationOutput(
            status=status,
            conclusion="TRUE" if verified else "UNKNOWN",
            operation_id=declaration.declaration.capability_id,
            input_uri=input_artifact.artifact_uri,
            result_uri=result_artifact.artifact_uri,
            witness_uri=witness.artifact_uri,
            checker_id=declaration.checker_id,
            verification_record_uri=(
                checked.verification_record_uri if verified else None
            ),
            detail=detail,
        )
        record_uri = checked.verification_record_uri if verified else None
        artifact_uris = [
            input_artifact.artifact_uri,
            result_artifact.artifact_uri,
            witness.artifact_uri,
        ]
        if record_uri is not None:
            artifact_uris.append(record_uri)
        assurance_level = (
            CapabilityAssuranceLevel.VERIFIED
            if verified
            else (
                CapabilityAssuranceLevel.COMPUTED
                if checked.execution.status is ExecutionStatus.COMPLETED
                else CapabilityAssuranceLevel.HEURISTIC
            )
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=checked.execution,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="the complete stored exact operation input and result",
                parameters={
                    "operation_id": declaration.declaration.capability_id,
                    "input_uri": input_artifact.artifact_uri,
                    "result_uri": result_artifact.artifact_uri,
                },
                artifact_uri=input_artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.NOT_APPLICABLE,
                basis="direct exact replay makes no search-completeness claim",
                assurance_level=(
                    CapabilityAssuranceLevel.COMPUTED
                    if checked.execution.status is ExecutionStatus.COMPLETED
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
            ),
            assurance=CapabilityAssurance(
                level=assurance_level,
                basis=(
                    "accepted in a clean process by the operator-authorized "
                    "independent exact replay checker"
                    if verified
                    else (
                        "checker replay completed without accepting the candidate; "
                        "no opposite conclusion follows"
                        if checked.execution.status is ExecutionStatus.COMPLETED
                        else "checker replay did not complete; no conclusion follows"
                    )
                ),
                verification_record_uri=record_uri,
            ),
            artifact_uris=tuple(artifact_uris),
        )

    def _resolve(
        self, result_uri: str
    ) -> tuple[
        _InstalledDeclaration,
        StoredArtifact,
        StoredArtifact,
        StoredArtifact,
    ]:
        result = self.store.get(result_uri)
        declaration = self.declarations_by_schema.get(result.manifest.schema_uri)
        if declaration is None:
            raise ExactDomainArtifactError(
                "artifact schema is not a supported exact-domain result"
            )
        if result.manifest.semantics_uri != declaration.semantics_uri:
            raise ExactDomainArtifactError(
                "result semantics do not match the declared producer"
            )
        if len(result.manifest.parents) != 1:
            raise ExactDomainArtifactError(
                "result lineage must identify exactly one producer input"
            )
        input_artifact = self.store.get(result.manifest.parents[0])
        if (
            input_artifact.manifest.schema_uri != declaration.input_schema_uri
            or input_artifact.manifest.semantics_uri != declaration.semantics_uri
        ):
            raise ExactDomainArtifactError(
                "result parent is not the exact declared producer input"
            )
        try:
            self.schemas.validate(declaration.result_schema_uri, result.payload)
            self.schemas.validate(
                declaration.input_schema_uri,
                input_artifact.payload,
            )
        except (SchemaRegistryError, ValidationError, ValueError) as exc:
            raise ExactDomainArtifactError(
                "stored operation artifacts do not satisfy their contracts"
            ) from exc
        semantics = self.store.get(declaration.semantics_uri)
        return declaration, input_artifact, result, semantics


def _checker_supports(operation_id: str, payload: object) -> bool:
    if operation_id in {
        "graph.hamiltonian_path.decide",
        "graph.induced_tree.maximum.compute",
    }:
        maximum_order = 18 if operation_id == "graph.hamiltonian_path.decide" else 16
        return (
            isinstance(payload, dict)
            and isinstance(payload.get("graph"), dict)
            and isinstance(payload["graph"].get("vertices"), list)
            and len(payload["graph"]["vertices"]) <= maximum_order
        )
    if operation_id == "geometry.projective_line_arrangement.flats.materialize":
        return (
            isinstance(payload, dict)
            and isinstance(payload.get("lines"), list)
            and len(payload["lines"]) <= 64
        )
    if not operation_id.startswith("polynomial."):
        return True
    if not isinstance(payload, dict):
        return False
    if operation_id == "polynomial.jacobian_syzygy.minimum_degree.compute":
        return True
    polynomial_fields = {
        "polynomial.compute.gcd": ("left", "right"),
        "polynomial.compute.resultant": ("left", "right"),
        "polynomial.compute.discriminant": ("polynomial",),
        "polynomial.compute.square_free_decomposition": ("polynomial",),
    }[operation_id]
    return all(
        isinstance(payload.get(field), dict)
        and payload[field].get("variables")
        and len(payload[field]["variables"]) == 1
        for field in polynomial_fields
    )


__all__ = [
    "ExactDomainArtifactError",
    "ExactDomainCheckerInstallation",
    "ExactDomainResultVerificationAdapter",
    "install_exact_domain_checkers",
    "install_exact_domain_verification",
]
