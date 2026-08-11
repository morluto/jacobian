"""Independent, bounded verification for exact ASCII LRAT artifacts."""

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
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.results import (
    Conclusion,
    Execution,
    ExecutionStatus,
    Verification,
)
from jacobian.contracts.sat import (
    SatLratProofArtifact,
    SatLratVerificationOutput,
    SatLratVerificationRequest,
)
from jacobian.provider_runtime import known_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.sat_smt.sat import SatArtifactError, SatArtifactService
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService


@dataclass(frozen=True, slots=True)
class SatLratInstallation:
    proof_schema_uri: str
    certificate_schema_uri: str
    checker_id: str | None


def install_sat_lrat_verifier(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    sat: SatArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[CapabilityAdapter | None, SatLratInstallation]:
    proof_schema_uri = schemas.register_model(
        name="jacobian.sat-lrat-proof", version="1", model=SatLratProofArtifact
    )
    certificate_schema_uri = schemas.register_model(
        name="jacobian.certificate-envelope", version="1", model=CertificateEnvelope
    )
    checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="bounded independent ASCII LRAT RUP checker",
                entrypoint="jacobian_checkers.sat_lrat:check_lrat",
                evidence_kind=EvidenceKind.CERTIFICATE,
                format_id="sat.lrat-proof",
                format_version="1",
                claim_schema_uris=(sat.installation.cnf_schema_uri,),
                semantics_uris=(sat.installation.semantics_uri,),
                candidate_schema_uris=(proof_schema_uri,),
                reason="operator-authorized standard-library LRAT RUP replay",
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    installation = SatLratInstallation(
        proof_schema_uri=proof_schema_uri,
        certificate_schema_uri=certificate_schema_uri,
        checker_id=checker_id,
    )
    if checker_id is None:
        return None, installation
    return (
        SatLratVerificationAdapter(
            store=store,
            artifacts=artifacts,
            sat=sat,
            verification=verification,
            installation=installation,
        ),
        installation,
    )


class SatLratVerificationAdapter:
    def __init__(
        self,
        *,
        store: ArtifactRepository,
        artifacts: ArtifactService,
        sat: SatArtifactService,
        verification: VerificationService,
        installation: SatLratInstallation,
    ) -> None:
        self.store = store
        self.artifacts = artifacts
        self.sat = sat
        self.verification = verification
        self.installation = installation
        checker_id = installation.checker_id
        if checker_id is None:
            raise RuntimeError("checker is not installed")
        self._descriptor = CapabilityDescriptor(
            capability_id="sat.lrat.verify",
            version="1",
            title="Verify an LRAT UNSAT certificate",
            description=(
                "Independently replay the deterministic ASCII LRAT RUP profile "
                "against an exact canonical CNF; negative RAT hints are unsupported."
            ),
            provider="jacobian.sat-lrat",
            provider_runtime=known_provider_runtime(
                "jacobian.sat-lrat",
                features=("ascii-lrat", "ordered-rup-hints", "bounded-replay"),
                checker_ids=(checker_id,),
            ),
            input_schema=model_schema(SatLratVerificationRequest),
            output_schema=model_schema(SatLratVerificationOutput),
            tags=("sat", "cnf", "lrat", "unsat", "certificate", "verification"),
            accepted_input_kinds=(CapabilityInputKind.STRUCTURED_REQUEST,),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = SatLratVerificationRequest.model_validate(request.input)
        try:
            resolved = self.sat.resolve_cnf(validated.cnf_uri)
            semantics = self.store.get(self.sat.installation.semantics_uri)
        except (SatArtifactError, StorageError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_SAT_CNF",
                    stage="artifact_resolution",
                    message=str(exc),
                    path="cnf_uri",
                    schema_uri=self.sat.installation.cnf_schema_uri,
                    expected="an exact canonical CNF artifact",
                )
            ) from exc
        proof = SatLratProofArtifact.from_bytes(
            cnf=resolved.binding,
            proof=__import__("base64").b64decode(validated.proof_base64, validate=True),
            limits=validated.limits,
        )
        proof_artifact = self.artifacts.put(
            schema_uri=self.installation.proof_schema_uri,
            semantics_uri=self.sat.installation.semantics_uri,
            payload=proof.model_dump(mode="json"),
            parents=(resolved.artifact.artifact_uri,),
            summary="unverified exact ASCII LRAT proof",
        )
        checker_id = self.installation.checker_id
        if checker_id is None:
            raise RuntimeError("checker is not installed")
        bindings = EvidenceBindings(
            claim_digest=resolved.artifact.manifest.object_digest,
            semantics_digest=semantics.manifest.object_digest,
            candidate_digest=proof_artifact.object_digest,
        )
        payload = {
            "cnf_uri": resolved.artifact.artifact_uri,
            "proof_uri": proof_artifact.artifact_uri,
            "proof_digest": proof.proof_digest,
            "limits": validated.limits.model_dump(mode="json"),
        }
        certificate = CertificateEnvelope(
            certificate_type="sat.lrat-proof",
            format_version="1",
            bindings=bindings,
            payload_digest="sha256:"
            + hashlib.sha256(canonicalize_json(payload)).hexdigest(),
            payload=payload,
        )
        certificate_artifact = self.artifacts.put(
            schema_uri=self.installation.certificate_schema_uri,
            semantics_uri=self.sat.installation.semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=(resolved.artifact.artifact_uri, proof_artifact.artifact_uri),
            summary="LRAT verification certificate",
        )
        if validated.cancelled:
            output = SatLratVerificationOutput(
                status="CANCELLED",
                conclusion="UNKNOWN",
                cnf_uri=resolved.artifact.artifact_uri,
                proof_uri=proof_artifact.artifact_uri,
                certificate_uri=certificate_artifact.artifact_uri,
                checker_id=checker_id,
                detail="request was cancelled before replay; no conclusion follows",
            )
            return CapabilityResult(
                capability_id=self.descriptor.capability_id,
                capability_version=self.descriptor.version,
                execution=Execution(
                    status=ExecutionStatus.CANCELLED,
                    detail="cancelled before independent LRAT replay",
                ),
                output=output.model_dump(mode="json"),
                scope=CapabilityScope(
                    description="the exact canonical CNF and exact LRAT proof bytes",
                    parameters={
                        "proof_format": proof.proof_format,
                        "proof_format_version": proof.proof_format_version,
                        "proof_digest": proof.proof_digest,
                        "variable_map_digest": proof.cnf.variable_map_digest,
                        "dimacs_digest": proof.cnf.dimacs_digest,
                        "limits": validated.limits.model_dump(mode="json"),
                    },
                    artifact_uri=resolved.artifact.artifact_uri,
                ),
                completeness=CapabilityCompleteness(
                    status=CapabilityCompletenessStatus.NOT_APPLICABLE,
                    basis="cancelled certificate replay makes no completeness claim",
                    assurance_level=CapabilityAssuranceLevel.HEURISTIC,
                ),
                assurance=CapabilityAssurance(
                    level=CapabilityAssuranceLevel.HEURISTIC,
                    basis="cancelled before checker execution; no conclusion follows",
                ),
                artifact_uris=(
                    resolved.artifact.artifact_uri,
                    proof_artifact.artifact_uri,
                    certificate_artifact.artifact_uri,
                ),
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
        detail = checked.execution.detail or (
            checked.input.errors[0] if checked.input.errors else "LRAT proof rejected"
        )
        status: Literal[
            "VERIFIED_UNSAT",
            "REJECTED",
            "UNSUPPORTED",
            "TIMEOUT",
            "CANCELLED",
            "ERROR",
        ]
        if verified:
            status = "VERIFIED_UNSAT"
        elif "unsupported LRAT feature" in detail:
            status = "UNSUPPORTED"
        elif "timed out" in detail:
            status = "TIMEOUT"
        elif checked.execution.status is ExecutionStatus.COMPLETED:
            status = "REJECTED"
        else:
            status = "ERROR"
        record_uri = checked.verification_record_uri if verified else None
        projected_execution = (
            Execution(status=ExecutionStatus.TIMEOUT, detail=detail)
            if status == "TIMEOUT"
            else checked.execution
        )
        output = SatLratVerificationOutput(
            status=status,
            conclusion="TRUE" if verified else "UNKNOWN",
            cnf_uri=resolved.artifact.artifact_uri,
            proof_uri=proof_artifact.artifact_uri,
            certificate_uri=certificate_artifact.artifact_uri,
            checker_id=checker_id,
            verification_record_uri=record_uri,
            detail=detail,
        )
        uris = [
            resolved.artifact.artifact_uri,
            proof_artifact.artifact_uri,
            certificate_artifact.artifact_uri,
        ]
        if record_uri:
            uris.append(record_uri)
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=projected_execution,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="the exact canonical CNF and exact LRAT proof bytes",
                parameters={
                    "proof_format": proof.proof_format,
                    "proof_format_version": proof.proof_format_version,
                    "proof_digest": proof.proof_digest,
                    "variable_map_digest": proof.cnf.variable_map_digest,
                    "dimacs_digest": proof.cnf.dimacs_digest,
                    "limits": validated.limits.model_dump(mode="json"),
                },
                artifact_uri=resolved.artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.NOT_APPLICABLE,
                basis="certificate replay makes no search-completeness claim",
                assurance_level=(
                    CapabilityAssuranceLevel.HEURISTIC
                    if status == "TIMEOUT"
                    else CapabilityAssuranceLevel.COMPUTED
                ),
            ),
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else (
                        CapabilityAssuranceLevel.HEURISTIC
                        if status == "TIMEOUT"
                        else CapabilityAssuranceLevel.COMPUTED
                    )
                ),
                basis=(
                    "independent checker accepted the exact bound LRAT proof"
                    if verified
                    else "proof was not accepted; this is not evidence of satisfiability"
                ),
                verification_record_uri=record_uri,
            ),
            artifact_uris=tuple(uris),
        )
