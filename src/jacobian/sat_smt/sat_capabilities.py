"""Model-facing construction and verification for exact SAT artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.capability_service import CapabilityAdapter, CapabilityInvocationError
from jacobian.checker_artifacts import put_witness_envelope
from jacobian.checker_installation import CheckerInstaller
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInputKind,
    CapabilityInvocationExample,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import (
    CertificateEnvelope,
    EvidenceBindings,
    WitnessEnvelope,
)
from jacobian.contracts.results import (
    Conclusion,
    Execution,
    ExecutionStatus,
    Verification,
)
from jacobian.contracts.sat import (
    SatAssignmentVerificationOutput,
    SatAssignmentVerificationRequest,
    SatCnfMaterializationOutput,
    SatCnfMaterializationRequest,
    SatUnsatProofVerificationOutput,
    SatUnsatProofVerificationRequest,
)
from jacobian.provider_runtime import known_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.sat_smt.sat import SatArtifactError, SatArtifactService
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService


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
        self._descriptor = CapabilityDescriptor(
            capability_id="sat.cnf.materialize",
            version="1",
            title="Materialize a canonical SAT CNF",
            description=(
                "Encode exact finite existence problems, including finite colorings "
                "and forbidden configurations, as named Boolean variables and "
                "clauses. Canonicalize and store the exact CNF consumed by SAT model "
                "and certified exhaustive UNSAT search capabilities."
            ),
            provider="jacobian.sat",
            provider_runtime=known_provider_runtime(
                "jacobian.sat",
                features=("canonical-cnf", "cnf-materialization"),
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(SatCnfMaterializationRequest),
            output_schema=model_schema(SatCnfMaterializationOutput),
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
            invocation_examples=(
                CapabilityInvocationExample(
                    name="finite-coloring-cnf",
                    description=(
                        "Encode two items with exactly one of two colors and forbid "
                        "them from sharing a color."
                    ),
                    mode=CapabilityMode.EXPLORE,
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

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = SatCnfMaterializationRequest.model_validate(request.input)
        except ValidationError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
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

        stored = self.sat.put_cnf(
            variable_names=validated.variable_names,
            clauses=validated.clauses,
        )
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
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="the complete canonicalized CNF supplied in this request",
                parameters={
                    "declared_scope": "FULL_CNF",
                    "variable_count": binding.variable_count,
                    "clause_count": binding.clause_count,
                },
                artifact_uri=binding.cnf_artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.NOT_APPLICABLE,
                basis=(
                    "materialization stores one complete input CNF and makes no "
                    "satisfiability or enumeration claim"
                ),
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis=(
                    "deterministic canonicalization and content-addressed artifact "
                    "storage; no SAT or UNSAT conclusion is claimed"
                ),
            ),
            artifact_uris=(binding.cnf_artifact_uri,),
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
) -> tuple[CapabilityAdapter | None, SatAssignmentCheckerInstallation]:
    """Install the assignment evidence schema and optionally authorize replay."""

    witness_schema_uri = schemas.register_model(
        name="jacobian.witness-envelope",
        version="1",
        model=WitnessEnvelope,
    )
    checker_id = (
        CheckerInstaller(checkers)
        .install(
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
        )
        .checker_id
    )
    installation = SatAssignmentCheckerInstallation(
        witness_schema_uri=witness_schema_uri,
        checker_id=checker_id,
    )
    adapter: CapabilityAdapter | None = None
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
    runtime: CapabilityProviderRuntime,
    *,
    authorize_checker: bool,
) -> tuple[CapabilityAdapter | None, SatUnsatProofCheckerInstallation]:
    """Install the proof certificate schema and optionally authorize DRAT replay."""

    certificate_schema_uri = schemas.register_model(
        name="jacobian.certificate-envelope",
        version="1",
        model=CertificateEnvelope,
    )
    checker_id = (
        CheckerInstaller(checkers)
        .install(
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
                authorize_checker
                and runtime.availability is CapabilityProviderAvailability.AVAILABLE
            ),
        )
        .checker_id
    )
    installation = SatUnsatProofCheckerInstallation(
        certificate_schema_uri=certificate_schema_uri,
        checker_id=checker_id,
    )
    adapter: CapabilityAdapter | None = None
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
        self._descriptor = CapabilityDescriptor(
            capability_id="sat.model.verify",
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
            modes=(CapabilityMode.VERIFY,),
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
                CapabilityInputKind.STRUCTURED_REQUEST,
                CapabilityInputKind.TYPED_ARTIFACT,
            ),
            accepted_artifact_types=(self.sat.installation.assignment_schema_uri,),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = SatAssignmentVerificationRequest.model_validate(request.input)
        try:
            resolved = self.sat.resolve_assignment(validated.assignment_uri)
            semantics = self.store.get(self.sat.installation.semantics_uri)
        except (SatArtifactError, StorageError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
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
        assert checker_id is not None
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
            and checked.assurance.verification is Verification.VERIFIED
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
                description="the full exact canonical CNF bound by the assignment",
                parameters={
                    "declared_scope": "FULL_CNF",
                    "variable_count": resolved.assignment.cnf.variable_count,
                    "clause_count": resolved.assignment.cnf.clause_count,
                },
                artifact_uri=resolved.cnf_artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.NOT_APPLICABLE,
                basis=(
                    "direct assignment replay checks one witness and makes no "
                    "enumeration or UNSAT completeness claim"
                ),
                assurance_level=(
                    CapabilityAssuranceLevel.COMPUTED
                    if checked.execution.status is ExecutionStatus.COMPLETED
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
            ),
            assurance=CapabilityAssurance(
                level=assurance_level,
                basis=(
                    "accepted by the operator-authorized independent SAT "
                    "assignment checker"
                    if verified
                    else (
                        "checker replay completed without accepting the assignment; "
                        "no opposite conclusion follows"
                        if checked.execution.status is ExecutionStatus.COMPLETED
                        else "checker execution did not complete; no mathematical "
                        "conclusion follows"
                    )
                ),
                verification_record_uri=record_uri,
            ),
            artifact_uris=tuple(artifact_uris),
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
        runtime: CapabilityProviderRuntime,
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
        self._descriptor = CapabilityDescriptor(
            capability_id="sat.unsat_proof.verify",
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
            modes=(CapabilityMode.VERIFY,),
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
                CapabilityInputKind.STRUCTURED_REQUEST,
                CapabilityInputKind.TYPED_ARTIFACT,
            ),
            accepted_artifact_types=(self.sat.installation.proof_schema_uri,),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = SatUnsatProofVerificationRequest.model_validate(request.input)
        try:
            resolved = self.sat.resolve_proof(validated.proof_uri)
            semantics = self.store.get(self.sat.installation.semantics_uri)
        except (SatArtifactError, StorageError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
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
        assert checker_id is not None
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
            and checked.assurance.verification is Verification.VERIFIED
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
                description="the full exact canonical CNF bound by the raw proof",
                parameters={
                    "declared_scope": "CANONICAL_CNF_ONLY",
                    "domain_encoding_verified": False,
                    "variable_count": resolved.proof.cnf.variable_count,
                    "clause_count": resolved.proof.cnf.clause_count,
                    "proof_format": resolved.proof.proof_format,
                    "proof_format_version": resolved.proof.proof_format_version,
                },
                artifact_uri=resolved.cnf_artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.NOT_APPLICABLE,
                basis=(
                    "certificate replay checks one exact proof and makes no "
                    "enumeration-completeness claim"
                ),
                assurance_level=(
                    CapabilityAssuranceLevel.COMPUTED
                    if checked.execution.status is ExecutionStatus.COMPLETED
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
            ),
            assurance=CapabilityAssurance(
                level=assurance_level,
                basis=(
                    "accepted by the operator-authorized external DRAT-trim "
                    "runtime bound into the checker registration"
                    if verified
                    else (
                        "checker replay completed without accepting the proof; "
                        "no opposite conclusion follows"
                        if checked.execution.status is ExecutionStatus.COMPLETED
                        else "checker execution did not complete; no mathematical "
                        "conclusion follows"
                    )
                ),
                verification_record_uri=record_uri,
            ),
            artifact_uris=tuple(artifact_uris),
        )
