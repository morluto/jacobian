"""Model-facing independent verification for exact SMT Alethe artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.capability_service import CapabilityAdapter, CapabilityInvocationError
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
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.results import Conclusion, ExecutionStatus, Verification
from jacobian.contracts.smt import (
    SmtUnsatProofVerificationOutput,
    SmtUnsatProofVerificationRequest,
)
from jacobian.registry import CheckerRegistry
from jacobian.sat_smt.smt import SmtArtifactError, SmtArtifactService
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService


@dataclass(frozen=True, slots=True)
class SmtUnsatProofCheckerInstallation:
    certificate_schema_uri: str
    checker_id: str | None


def install_smt_unsat_proof_checker(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    smt: SmtArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    runtime: CapabilityProviderRuntime,
    *,
    authorize_checker: bool,
) -> tuple[CapabilityAdapter | None, SmtUnsatProofCheckerInstallation]:
    """Install the certificate schema and optionally authorize strict replay."""

    certificate_schema_uri = schemas.register_model(
        name="jacobian.certificate-envelope",
        version="1",
        model=CertificateEnvelope,
    )
    checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="pinned strict Carcara Alethe QF_UF UNSAT checker",
                entrypoint="jacobian_checkers.smt:check_unsat_proof",
                evidence_kind=EvidenceKind.CERTIFICATE,
                format_id="smt.unsat-proof",
                format_version="1",
                claim_schema_uris=(smt.installation.problem_schema_uri,),
                semantics_uris=(smt.installation.semantics_uri,),
                candidate_schema_uris=(smt.installation.proof_schema_uri,),
                provider_runtime=runtime,
                reason=(
                    "operator-authorized strict Carcara replay of zero-hole cvc5 "
                    "1.3.4 Alethe proofs in the QF_UF compatibility profile"
                ),
            ),
            authorize=(
                authorize_checker
                and runtime.availability is CapabilityProviderAvailability.AVAILABLE
            ),
        )
        .checker_id
    )
    installation = SmtUnsatProofCheckerInstallation(
        certificate_schema_uri=certificate_schema_uri,
        checker_id=checker_id,
    )
    adapter: CapabilityAdapter | None = None
    if checker_id is not None:
        adapter = SmtUnsatProofVerificationAdapter(
            store=store,
            artifacts=artifacts,
            smt=smt,
            verification=verification,
            installation=installation,
            runtime=runtime,
        )
    return adapter, installation


class SmtUnsatProofVerificationAdapter:
    """Verify one compatible Alethe proof; rejection establishes nothing."""

    def __init__(
        self,
        *,
        store: ArtifactRepository,
        artifacts: ArtifactService,
        smt: SmtArtifactService,
        verification: VerificationService,
        installation: SmtUnsatProofCheckerInstallation,
        runtime: CapabilityProviderRuntime,
    ) -> None:
        checker_id = installation.checker_id
        if checker_id is None:
            raise ValueError("SMT UNSAT proof checker is not authorized")
        self.store = store
        self.artifacts = artifacts
        self.smt = smt
        self.verification = verification
        self.installation = installation
        descriptor_runtime = runtime.model_copy(update={"checker_ids": (checker_id,)})
        self._descriptor = CapabilityDescriptor(
            capability_id="smt.unsat_proof.verify",
            version="1",
            title="Verify a compatible SMT UNSAT proof",
            description=(
                "Replay one exact zero-hole cvc5 1.3.4 Alethe proof for its "
                "bound QF_UF query in operator-authorized strict Carcara."
            ),
            provider="carcara",
            provider_runtime=descriptor_runtime,
            modes=(CapabilityMode.VERIFY,),
            input_schema=model_schema(SmtUnsatProofVerificationRequest),
            output_schema=model_schema(SmtUnsatProofVerificationOutput),
            tags=(
                "smt",
                "qf-uf",
                "unsat",
                "proof",
                "verification",
                "alethe",
                "carcara",
            ),
            accepted_input_kinds=(
                CapabilityInputKind.STRUCTURED_REQUEST,
                CapabilityInputKind.TYPED_ARTIFACT,
            ),
            accepted_artifact_types=(self.smt.installation.proof_schema_uri,),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = SmtUnsatProofVerificationRequest.model_validate(request.input)
        try:
            resolved = self.smt.resolve_proof(validated.proof_uri)
            semantics = self.store.get(self.smt.installation.semantics_uri)
        except (SmtArtifactError, StorageError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_SMT_UNSAT_PROOF",
                    stage="artifact_resolution",
                    message=str(exc),
                    path="proof_uri",
                    schema_uri=self.smt.installation.proof_schema_uri,
                    expected=(
                        "a valid raw Alethe proof artifact bound by payload and "
                        "lineage to one exact pinned-profile SMT query"
                    ),
                    hint=(
                        "Use math.find for smt.unsat_proof.find to produce a proof "
                        "artifact for the intended query. If that optional producer "
                        "is unavailable, install the cvc5 provider; do not invent a "
                        "proof URI."
                    ),
                )
            ) from exc

        checker_id = self.installation.checker_id
        if checker_id is None:
            raise RuntimeError("checker is not installed")
        bindings = EvidenceBindings(
            claim_digest=resolved.problem_artifact.manifest.object_digest,
            semantics_digest=semantics.manifest.object_digest,
            candidate_digest=resolved.artifact.manifest.object_digest,
        )
        payload = {
            "problem_uri": resolved.problem_artifact.artifact_uri,
            "proof_uri": resolved.artifact.artifact_uri,
        }
        certificate = CertificateEnvelope(
            certificate_type="smt.unsat-proof",
            format_version="1",
            bindings=bindings,
            payload_digest=(
                "sha256:" + hashlib.sha256(canonicalize_json(payload)).hexdigest()
            ),
            payload=payload,
        )
        certificate_artifact = self.artifacts.put(
            schema_uri=self.installation.certificate_schema_uri,
            semantics_uri=self.smt.installation.semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=(
                resolved.problem_artifact.artifact_uri,
                resolved.artifact.artifact_uri,
            ),
            summary="SMT UNSAT proof verification certificate",
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
                "strict Carcara accepted the exact bound QF_UF proof"
                if verified
                else "the proof was not independently accepted"
            )
        output = SmtUnsatProofVerificationOutput(
            status=status,
            conclusion="TRUE" if verified else "UNKNOWN",
            problem_uri=resolved.problem_artifact.artifact_uri,
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
            resolved.problem_artifact.artifact_uri,
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
                description="the full exact SMT query bound by the Alethe proof",
                parameters={
                    "declared_scope": resolved.proof.declared_scope,
                    "logic": resolved.proof.problem.logic,
                    "profile": resolved.proof.problem.profile,
                    "proof_format": resolved.proof.proof_format,
                    "proof_format_version": resolved.proof.proof_format_version,
                    "contains_holes": resolved.proof.contains_holes,
                    "alethe_hole_count": resolved.proof.alethe_hole_count,
                },
                artifact_uri=resolved.problem_artifact.artifact_uri,
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
                    "accepted by the operator-authorized external strict Carcara "
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
