"""Model-facing construction and verification for exact SAT artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.checker_artifacts import put_witness_envelope
from jacobian.checker_authorization import authorize_checker_operation
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import (
    CertificateEnvelope,
    EvidenceBindings,
    WitnessEnvelope,
)
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiagnostic,
    OperationExample,
    OperationInputKind,
    OperationRequest,
    ProviderAvailability,
    ProviderObservation,
)
from jacobian.contracts.results import (
    Conclusion,
    ContractModel,
    Execution,
    ExecutionStatus,
)
from jacobian.contracts.sat import (
    CanonicalCnf,
    SatAssignmentVerificationOutput,
    SatAssignmentVerificationRequest,
    SatCnfMaterializationOutput,
    SatCnfMaterializationRequest,
    SatUnsatProofVerificationOutput,
    SatUnsatProofVerificationRequest,
    canonicalize_cnf,
)
from jacobian.operation_adapters import OperationAdapter, parse_operation_input
from jacobian.operation_catalog import OperationCatalog, OperationCatalogError
from jacobian.operation_declarations import OperationDeclaration
from jacobian.operation_errors import OperationInvocationError
from jacobian.operation_execution import execute_operation
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed, Failed
from jacobian.provider_runtime import known_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.sat_smt.sat import SatArtifactError, SatArtifactService
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService


@dataclass(frozen=True, slots=True)
class SatAssignmentCheckerInstallation:
    witness_schema_uri: str
    checker_id: str | None


@dataclass(frozen=True, slots=True)
class SatUnsatProofCheckerInstallation:
    certificate_schema_uri: str
    checker_id: str | None


class SatCnfMaterializationAdapter:
    """Create one canonical CNF artifact without making a SAT conclusion."""

    def __init__(self, sat: SatArtifactService) -> None:
        self.sat = sat
        self.spec = OperationDeclaration(
            operation_id="sat.cnf.materialize",
            version="1",
            request_type=SatCnfMaterializationRequest,
            result_type=CanonicalCnf,
            execute=_canonical_cnf,
            title="Materialize a canonical SAT CNF",
            description=(
                "Encode exact finite existence problems, including finite colorings "
                "and forbidden configurations, as named Boolean variables and "
                "clauses. Canonicalize and store the exact CNF consumed by SAT model "
                "and certified exhaustive UNSAT search operations."
            ),
            tags=(
                "sat",
                "cnf",
                "canonical-cnf",
                "materialization",
                "model",
                "unsat",
                "proof",
                "boolean-encoding",
                "finite-coloring",
                "forbidden-configurations",
                "exact-finite-existence",
                "certified-exhaustive-search",
            ),
            examples=(
                OperationExample(
                    name="finite-coloring-cnf",
                    description=(
                        "Encode two items with exactly one of two colors and forbid "
                        "them from sharing a color."
                    ),
                    input={
                        "variable_names": [
                            "item1_red",
                            "item1_blue",
                            "item2_red",
                            "item2_blue",
                        ],
                        "clauses": [
                            [1, 2],
                            [-1, -2],
                            [3, 4],
                            [-3, -4],
                            [-1, -3],
                            [-2, -4],
                        ],
                    },
                ),
            ),
        )
        self._descriptor = OperationDescriptor(
            operation_id=self.spec.operation_id,
            version=self.spec.version,
            title=self.spec.title,
            description=self.spec.description,
            provider="jacobian.sat",
            provider_runtime=known_provider_runtime(
                "jacobian.sat",
                features=("canonical-cnf", "cnf-materialization"),
            ),
            input_schema=model_schema(SatCnfMaterializationRequest),
            output_schema=model_schema(SatCnfMaterializationOutput),
            tags=self.spec.tags,
            examples=self.spec.examples,
        )

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> SatCnfMaterializationRequest:
        try:
            return parse_operation_input(SatCnfMaterializationRequest, request.input)
        except ValidationError as exc:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="INVALID_CNF",
                    stage="cnf_validation",
                    message="The named-variable CNF is not valid.",
                    expected=(
                        "unique valid variable names and bounded nonzero integer "
                        "literals referring only to declared variables"
                    ),
                    hint=(
                        "Inspect sat.cnf.materialize with math.find and execute it with math.run to correct "
                        "the variable_names or clauses."
                    ),
                )
            ) from exc

    def invoke(self, validated: SatCnfMaterializationRequest) -> OperationProjection:
        terminal = execute_operation(self.spec, validated)
        if not isinstance(terminal, Completed):
            return OperationProjection(
                operation_id=self.spec.operation_id,
                version=self.spec.version,
                terminal=terminal,
            )

        stored = self.sat.put_canonical_cnf(terminal.value)
        resolved = self.sat.resolve_cnf(stored.artifact_uri)
        binding = resolved.binding
        canonical_bindings = resolved.cnf.variables
        bindings_inline = (
            canonical_bindings if len(canonical_bindings) <= 4096 else None
        )
        caller_names = tuple(validated.variable_names)
        canonical_names = tuple(variable.name for variable in canonical_bindings)
        caller_order_changed = caller_names != canonical_names
        output = SatCnfMaterializationOutput(
            cnf_uri=binding.cnf_artifact_uri,
            schema_uri=self.sat.installation.cnf_schema_uri,
            semantics_uri=self.sat.installation.semantics_uri,
            cnf_object_digest=binding.cnf_object_digest,
            cnf_payload_digest=binding.cnf_payload_digest,
            variable_map_digest=binding.variable_map_digest,
            dimacs_digest=binding.dimacs_digest,
            projection_format=binding.projection_format,
            projection_version=binding.projection_version,
            variable_count=binding.variable_count,
            clause_count=binding.clause_count,
            variable_bindings=bindings_inline,
            variable_bindings_complete=bindings_inline is not None,
            caller_order_changed=caller_order_changed,
            variable_order_note=(
                "Caller order differs from canonical DIMACS order. Input literals "
                "were remapped soundly; interpret solver results only through the "
                "named assignment map or the canonical variable bindings."
                if caller_order_changed
                else (
                    "Caller order already matches canonical DIMACS order. Continue "
                    "to interpret solver results through named assignments."
                )
            ),
        )
        return OperationProjection(
            operation_id=self.spec.operation_id,
            version=self.spec.version,
            terminal=terminal,
            publication=PublishedOperation(
                output=output,
                artifact_uris=(binding.cnf_artifact_uri,),
            ),
        )


def _canonical_cnf(request: SatCnfMaterializationRequest) -> CanonicalCnf:
    return canonicalize_cnf(
        variable_names=request.variable_names,
        clauses=request.clauses,
    )


def install_sat_assignment_checker(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    sat: SatArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[
    OperationAdapter[SatAssignmentVerificationRequest] | None,
    SatAssignmentCheckerInstallation,
]:
    """Install the assignment evidence schema and optionally authorize replay."""

    witness_schema_uri = schemas.register_model(
        name="jacobian.witness-envelope",
        version="1",
        model=WitnessEnvelope,
    )
    checker_id = authorize_checker_operation(
        checkers,
        CheckerOperation(
            name="exact total SAT assignment replay checker",
            entrypoint="jacobian_checkers.sat:check_assignment",
            evidence_kind=EvidenceKind.WITNESS,
            format_id="sat.assignment",
            format_version="1",
            claim_schema_uris=(sat.installation.cnf_schema_uri,),
            semantics_uris=(sat.installation.semantics_uri,),
            candidate_schema_uris=(sat.installation.assignment_schema_uri,),
            reason="bundled independent SAT assignment checker",
        ),
        authorize=authorize_checker,
    ).checker_id
    installation = SatAssignmentCheckerInstallation(
        witness_schema_uri=witness_schema_uri,
        checker_id=checker_id,
    )
    adapter: OperationAdapter[SatAssignmentVerificationRequest] | None = None
    if checker_id is not None:
        adapter = SatAssignmentVerificationAdapter(
            store=store,
            artifacts=artifacts,
            sat=sat,
            verification=verification,
            installation=installation,
        )
    return adapter, installation


def install_sat_unsat_proof_checker(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    sat: SatArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    runtime: ProviderObservation,
    *,
    authorize_checker: bool,
) -> tuple[
    OperationAdapter[SatUnsatProofVerificationRequest] | None,
    SatUnsatProofCheckerInstallation,
]:
    """Install the proof certificate schema and optionally authorize DRAT replay."""

    certificate_schema_uri = schemas.register_model(
        name="jacobian.certificate-envelope",
        version="1",
        model=CertificateEnvelope,
    )
    checker_id = authorize_checker_operation(
        checkers,
        CheckerOperation(
            name="pinned DRAT-trim exact SAT UNSAT proof checker",
            entrypoint="jacobian_checkers.sat:check_unsat_proof",
            evidence_kind=EvidenceKind.CERTIFICATE,
            format_id="sat.unsat-proof",
            format_version="1",
            claim_schema_uris=(sat.installation.cnf_schema_uri,),
            semantics_uris=(sat.installation.semantics_uri,),
            candidate_schema_uris=(sat.installation.proof_schema_uri,),
            provider_runtime=runtime,
            reason="operator-authorized pinned DRAT-trim proof replay",
        ),
        authorize=(
            authorize_checker and runtime.availability is ProviderAvailability.AVAILABLE
        ),
    ).checker_id
    installation = SatUnsatProofCheckerInstallation(
        certificate_schema_uri=certificate_schema_uri,
        checker_id=checker_id,
    )
    adapter: OperationAdapter[SatUnsatProofVerificationRequest] | None = None
    if checker_id is not None:
        adapter = SatUnsatProofVerificationAdapter(
            store=store,
            artifacts=artifacts,
            sat=sat,
            verification=verification,
            installation=installation,
            runtime=runtime,
        )
    return adapter, installation


def bind_selected_sat_verification(
    operation_id: str,
    descriptor: OperationDescriptor,
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    sat: SatArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    catalog: OperationCatalog,
) -> OperationAdapter[Any] | None:
    """Bind one SAT verifier from passive schemas and catalog authority."""

    if operation_id not in {"sat.model.verify", "sat.unsat_proof.verify"}:
        return None
    binding = catalog.checker_binding(operation_id)
    if binding is None:
        raise OperationCatalogError(
            f"checker binding is missing; run `jacobian update`: {operation_id}"
        )
    registration = checkers.require_catalog_binding(
        binding.checker_id,
        implementation_digest=binding.manifest_digest,
    )
    if operation_id == "sat.model.verify":
        installation = SatAssignmentCheckerInstallation(
            witness_schema_uri=schemas.register_model(
                name="jacobian.witness-envelope",
                version="1",
                model=WitnessEnvelope,
            ),
            checker_id=binding.checker_id,
        )
        return SatAssignmentVerificationAdapter(
            store=store,
            artifacts=artifacts,
            sat=sat,
            verification=verification,
            installation=installation,
        )
    runtime = registration.implementation.provider_runtime
    if runtime is None:
        raise OperationCatalogError(
            "SAT proof checker provider observation is missing; run `jacobian update`"
        )
    proof_installation = SatUnsatProofCheckerInstallation(
        certificate_schema_uri=schemas.register_model(
            name="jacobian.certificate-envelope",
            version="1",
            model=CertificateEnvelope,
        ),
        checker_id=binding.checker_id,
    )
    return SatUnsatProofVerificationAdapter(
        store=store,
        artifacts=artifacts,
        sat=sat,
        verification=verification,
        installation=proof_installation,
        runtime=runtime,
    )


class SatAssignmentVerificationAdapter:
    """Verify one assignment; never infer UNSAT from assignment rejection."""

    def __init__(
        self,
        *,
        store: ArtifactRepository,
        artifacts: ArtifactService,
        sat: SatArtifactService,
        verification: VerificationService,
        installation: SatAssignmentCheckerInstallation,
    ) -> None:
        checker_id = installation.checker_id
        if checker_id is None:
            raise ValueError("SAT assignment checker is not authorized")
        self.store = store
        self.artifacts = artifacts
        self.sat = sat
        self.verification = verification
        self.installation = installation
        self._descriptor = OperationDescriptor(
            operation_id="sat.model.verify",
            version="1",
            title="Verify a SAT assignment",
            description=(
                "Independently replay one total assignment against every clause "
                "of its exact bound canonical CNF, establishing a finite Boolean "
                "existence witness when accepted."
            ),
            provider="jacobian.sat",
            provider_runtime=known_provider_runtime(
                "jacobian.sat",
                features=("total-assignment-replay", "canonical-cnf"),
                checker_ids=(checker_id,),
            ),
            input_schema=model_schema(SatAssignmentVerificationRequest),
            output_schema=model_schema(SatAssignmentVerificationOutput),
            tags=(
                "sat",
                "cnf",
                "assignment",
                "verification",
                "finite-coloring",
                "exact-finite-existence",
                "named-assignment",
            ),
            accepted_input_kinds=(
                OperationInputKind.STRUCTURED_REQUEST,
                OperationInputKind.TYPED_ARTIFACT,
            ),
            accepted_artifact_types=(self.sat.installation.assignment_schema_uri,),
        )

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> SatAssignmentVerificationRequest:
        return parse_operation_input(SatAssignmentVerificationRequest, request.input)

    def invoke(
        self, validated: SatAssignmentVerificationRequest
    ) -> OperationProjection:
        try:
            resolved = self.sat.resolve_assignment(validated.assignment_uri)
            semantics = self.store.get(self.sat.installation.semantics_uri)
        except (SatArtifactError, StorageError) as exc:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="INVALID_SAT_ASSIGNMENT",
                    stage="artifact_resolution",
                    message=str(exc),
                    path="assignment_uri",
                    schema_uri=self.sat.installation.assignment_schema_uri,
                    expected=(
                        "a valid SAT assignment artifact bound by payload and lineage "
                        "to one canonical CNF"
                    ),
                    hint=(
                        "Inspect sat.model.find with math.find and execute it with math.run to produce an assignment "
                        "artifact for the intended canonical CNF. If that optional "
                        "producer is unavailable, install the CaDiCaL provider; do "
                        "not invent an assignment URI."
                    ),
                )
            ) from exc

        checker_id = self.installation.checker_id
        if checker_id is None:
            raise RuntimeError("checker is not installed")
        witness_artifact = put_witness_envelope(
            self.artifacts,
            witness_schema_uri=self.installation.witness_schema_uri,
            witness_format="sat.assignment",
            claim_artifact=resolved.cnf_artifact,
            semantics_artifact=semantics,
            candidate_artifact=resolved.artifact,
            payload={
                "cnf_uri": resolved.cnf_artifact.artifact_uri,
                "assignment_uri": resolved.artifact.artifact_uri,
            },
            summary="SAT assignment verification witness",
        )
        checked = self.verification.verify_witness(
            claim_uri=resolved.cnf_artifact.artifact_uri,
            candidate_uri=resolved.artifact.artifact_uri,
            witness_uri=witness_artifact.artifact_uri,
            checker_id=checker_id,
            include_artifact_metadata=True,
        )
        verified = (
            checked.execution.status is ExecutionStatus.COMPLETED
            and checked.conclusion is Conclusion.TRUE
            and checked.verification_record_uri is not None
        )
        status: Literal[
            "VERIFIED_SATISFYING",
            "REJECTED",
            "TIMEOUT",
            "CANCELLED",
            "ERROR",
        ]
        if verified:
            status = "VERIFIED_SATISFYING"
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
                "the authorized checker accepted the total assignment"
                if verified
                else "the assignment was not independently accepted"
            )
        output = SatAssignmentVerificationOutput(
            status=status,
            conclusion="TRUE" if verified else "UNKNOWN",
            cnf_uri=resolved.cnf_artifact.artifact_uri,
            assignment_uri=resolved.artifact.artifact_uri,
            witness_uri=witness_artifact.artifact_uri,
            checker_id=checker_id,
            verification_record_uri=(
                checked.verification_record_uri if verified else None
            ),
            detail=detail,
        )
        record_uri = checked.verification_record_uri if verified else None
        artifact_uris = [
            resolved.cnf_artifact.artifact_uri,
            resolved.artifact.artifact_uri,
            witness_artifact.artifact_uri,
        ]
        if record_uri is not None:
            artifact_uris.append(record_uri)
        return _verification_projection(
            descriptor=self.descriptor,
            execution=checked.execution,
            output=output,
            artifact_uris=tuple(artifact_uris),
            verification_record_uri=record_uri,
            failure_code="SAT_ASSIGNMENT_CHECK_NONCONCLUSIVE",
            detail=detail,
        )


class SatUnsatProofVerificationAdapter:
    """Verify one raw proof; rejection never establishes satisfiability."""

    def __init__(
        self,
        *,
        store: ArtifactRepository,
        artifacts: ArtifactService,
        sat: SatArtifactService,
        verification: VerificationService,
        installation: SatUnsatProofCheckerInstallation,
        runtime: ProviderObservation,
    ) -> None:
        checker_id = installation.checker_id
        if checker_id is None:
            raise ValueError("SAT UNSAT proof checker is not authorized")
        self.store = store
        self.artifacts = artifacts
        self.sat = sat
        self.verification = verification
        self.installation = installation
        descriptor_runtime = runtime.model_copy(update={"checker_ids": (checker_id,)})
        self._descriptor = OperationDescriptor(
            operation_id="sat.unsat_proof.verify",
            version="1",
            title="Verify a SAT UNSAT proof",
            description=(
                "Independently replay exact normalized text DRAT against its bound "
                "canonical CNF in operator-authorized pinned DRAT-trim, establishing "
                "UNSAT for that canonical CNF only when accepted. This verifier does "
                "not certify any graph, coloring, or other domain encoding."
            ),
            provider="drat-trim",
            provider_runtime=descriptor_runtime,
            input_schema=model_schema(SatUnsatProofVerificationRequest),
            output_schema=model_schema(SatUnsatProofVerificationOutput),
            tags=(
                "sat",
                "cnf",
                "unsat",
                "proof",
                "verification",
                "drat",
                "finite-coloring",
                "forbidden-configurations",
                "certified-exhaustive-search",
            ),
            accepted_input_kinds=(
                OperationInputKind.STRUCTURED_REQUEST,
                OperationInputKind.TYPED_ARTIFACT,
            ),
            accepted_artifact_types=(self.sat.installation.proof_schema_uri,),
        )

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> SatUnsatProofVerificationRequest:
        return parse_operation_input(SatUnsatProofVerificationRequest, request.input)

    def invoke(
        self, validated: SatUnsatProofVerificationRequest
    ) -> OperationProjection:
        try:
            resolved = self.sat.resolve_proof(validated.proof_uri)
            semantics = self.store.get(self.sat.installation.semantics_uri)
        except (SatArtifactError, StorageError) as exc:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="INVALID_SAT_UNSAT_PROOF",
                    stage="artifact_resolution",
                    message=str(exc),
                    path="proof_uri",
                    schema_uri=self.sat.installation.proof_schema_uri,
                    expected=(
                        "a valid raw DRAT proof artifact bound by payload and "
                        "lineage to one canonical CNF"
                    ),
                    hint=(
                        "Inspect sat.unsat_proof.find with math.find and execute it with math.run to produce a proof "
                        "artifact for the intended canonical CNF. If that optional "
                        "producer is unavailable, install the CaDiCaL provider; do "
                        "not invent a proof URI."
                    ),
                )
            ) from exc

        checker_id = self.installation.checker_id
        if checker_id is None:
            raise RuntimeError("checker is not installed")
        bindings = EvidenceBindings(
            claim_digest=resolved.cnf_artifact.manifest.object_digest,
            semantics_digest=semantics.manifest.object_digest,
            candidate_digest=resolved.artifact.manifest.object_digest,
        )
        payload = {
            "cnf_uri": resolved.cnf_artifact.artifact_uri,
            "proof_uri": resolved.artifact.artifact_uri,
        }
        certificate = CertificateEnvelope(
            certificate_type="sat.unsat-proof",
            format_version="1",
            bindings=bindings,
            payload_digest=(
                "sha256:" + hashlib.sha256(canonicalize_json(payload)).hexdigest()
            ),
            payload=payload,
        )
        certificate_artifact = self.artifacts.put(
            schema_uri=self.installation.certificate_schema_uri,
            semantics_uri=self.sat.installation.semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=(
                resolved.cnf_artifact.artifact_uri,
                resolved.artifact.artifact_uri,
            ),
            summary="SAT UNSAT proof verification certificate",
        )
        checked = self.verification.verify_certificate(
            certificate_uri=certificate_artifact.artifact_uri,
            checker_id=checker_id,
            include_artifact_metadata=True,
        )
        verified = (
            checked.execution.status is ExecutionStatus.COMPLETED
            and checked.conclusion is Conclusion.TRUE
            and checked.verification_record_uri is not None
        )
        status: Literal[
            "VERIFIED_UNSAT",
            "REJECTED",
            "TIMEOUT",
            "CANCELLED",
            "ERROR",
        ]
        if verified:
            status = "VERIFIED_UNSAT"
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
                "the authorized DRAT checker accepted the exact bound proof"
                if verified
                else "the proof was not independently accepted"
            )
        output = SatUnsatProofVerificationOutput(
            verified_claim_scope="CANONICAL_CNF_ONLY",
            status=status,
            conclusion="TRUE" if verified else "UNKNOWN",
            cnf_uri=resolved.cnf_artifact.artifact_uri,
            proof_uri=resolved.artifact.artifact_uri,
            certificate_uri=certificate_artifact.artifact_uri,
            checker_id=checker_id,
            verification_record_uri=(
                checked.verification_record_uri if verified else None
            ),
            detail=detail,
        )
        record_uri = checked.verification_record_uri if verified else None
        artifact_uris = [
            resolved.cnf_artifact.artifact_uri,
            resolved.artifact.artifact_uri,
            certificate_artifact.artifact_uri,
        ]
        if record_uri is not None:
            artifact_uris.append(record_uri)
        return _verification_projection(
            descriptor=self.descriptor,
            execution=checked.execution,
            output=output,
            artifact_uris=tuple(artifact_uris),
            verification_record_uri=record_uri,
            failure_code="SAT_UNSAT_PROOF_CHECK_NONCONCLUSIVE",
            detail=detail,
        )


def _verification_projection(
    *,
    descriptor: OperationDescriptor,
    execution: Execution,
    output: ContractModel,
    artifact_uris: tuple[str, ...],
    verification_record_uri: str | None,
    failure_code: str,
    detail: str,
) -> OperationProjection:
    """Project one SAT checker outcome without promoting a non-conclusion."""

    publication = PublishedOperation(output=output, artifact_uris=artifact_uris)
    if execution.status is ExecutionStatus.COMPLETED:
        return OperationProjection(
            operation_id=descriptor.operation_id,
            version=descriptor.version,
            terminal=Completed(
                value=output,
                runtime_ms=execution.runtime_ms,
                detail=execution.detail,
            ),
            publication=publication,
            verification_record_uri=verification_record_uri,
        )
    return OperationProjection(
        operation_id=descriptor.operation_id,
        version=descriptor.version,
        terminal=Failed(
            status=execution.status,
            runtime_ms=execution.runtime_ms,
            diagnostic=OperationDiagnostic(
                code=failure_code,
                stage="verification",
                message=detail,
            ),
        ),
        publication=PublishedOperation(artifact_uris=artifact_uris),
    )
