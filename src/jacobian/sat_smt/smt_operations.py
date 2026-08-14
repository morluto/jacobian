"""Model-facing independent verification for exact SMT Alethe artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.checker_authorization import authorize_checker_operation
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiagnostic,
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
from jacobian.contracts.smt import (
    SmtUnsatProofVerificationOutput,
    SmtUnsatProofVerificationRequest,
)
from jacobian.operation_adapters import OperationAdapter, parse_operation_input
from jacobian.operation_catalog import OperationCatalog, OperationCatalogError
from jacobian.operation_errors import OperationInvocationError
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed, Failed
from jacobian.registry import CheckerRegistry
from jacobian.sat_smt.smt import SmtArtifactError, SmtArtifactService
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService


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
    runtime: ProviderObservation,
    *,
    authorize_checker: bool,
) -> tuple[
    OperationAdapter[SmtUnsatProofVerificationRequest] | None,
    SmtUnsatProofCheckerInstallation,
]:
    """Install the certificate schema and optionally authorize strict replay."""

    certificate_schema_uri = schemas.register_model(
        name="jacobian.certificate-envelope",
        version="1",
        model=CertificateEnvelope,
    )
    checker_id = authorize_checker_operation(
        checkers,
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
            authorize_checker and runtime.availability is ProviderAvailability.AVAILABLE
        ),
    ).checker_id
    installation = SmtUnsatProofCheckerInstallation(
        certificate_schema_uri=certificate_schema_uri,
        checker_id=checker_id,
    )
    adapter: OperationAdapter[SmtUnsatProofVerificationRequest] | None = None
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


def bind_selected_smt_unsat_proof_checker(
    descriptor: OperationDescriptor,
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    smt: SmtArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    catalog: OperationCatalog,
) -> OperationAdapter[SmtUnsatProofVerificationRequest]:
    """Bind SMT proof replay from persisted provider and checker observations."""

    operation_id = "smt.unsat_proof.verify"
    binding = catalog.checker_binding(operation_id)
    if binding is None:
        raise OperationCatalogError(
            f"checker binding is incomplete; run `jacobian update`: {operation_id}"
        )
    registration = checkers.require_catalog_binding(
        binding.checker_id,
        implementation_digest=binding.manifest_digest,
    )
    runtime = registration.implementation.provider_runtime
    if runtime is None:
        raise OperationCatalogError(
            f"checker binding is incomplete; run `jacobian update`: {operation_id}"
        )
    return SmtUnsatProofVerificationAdapter(
        store=store,
        artifacts=artifacts,
        smt=smt,
        verification=verification,
        installation=SmtUnsatProofCheckerInstallation(
            certificate_schema_uri=schemas.register_model(
                name="jacobian.certificate-envelope",
                version="1",
                model=CertificateEnvelope,
            ),
            checker_id=binding.checker_id,
        ),
        runtime=runtime,
    )


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
        runtime: ProviderObservation,
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
        self._descriptor = OperationDescriptor(
            operation_id="smt.unsat_proof.verify",
            version="1",
            title="Verify a compatible SMT UNSAT proof",
            description=(
                "Replay one exact zero-hole cvc5 1.3.4 Alethe proof for its "
                "bound QF_UF query in operator-authorized strict Carcara."
            ),
            provider="carcara",
            provider_runtime=descriptor_runtime,
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
                OperationInputKind.STRUCTURED_REQUEST,
                OperationInputKind.TYPED_ARTIFACT,
            ),
            accepted_artifact_types=(self.smt.installation.proof_schema_uri,),
        )

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> SmtUnsatProofVerificationRequest:
        return parse_operation_input(SmtUnsatProofVerificationRequest, request.input)

    def invoke(
        self, validated: SmtUnsatProofVerificationRequest
    ) -> OperationProjection:
        try:
            resolved = self.smt.resolve_proof(validated.proof_uri)
            semantics = self.store.get(self.smt.installation.semantics_uri)
        except (SmtArtifactError, StorageError) as exc:
            raise OperationInvocationError(
                OperationDiagnostic(
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
        return _verification_projection(
            descriptor=self.descriptor,
            execution=checked.execution,
            output=output,
            artifact_uris=tuple(artifact_uris),
            verification_record_uri=record_uri,
            detail=detail,
        )


def _verification_projection(
    *,
    descriptor: OperationDescriptor,
    execution: Execution,
    output: ContractModel,
    artifact_uris: tuple[str, ...],
    verification_record_uri: str | None,
    detail: str,
) -> OperationProjection:
    """Project one SMT checker outcome without promoting a non-conclusion."""

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
                code="SMT_UNSAT_PROOF_CHECK_NONCONCLUSIVE",
                stage="verification",
                message=detail,
            ),
        ),
        publication=PublishedOperation(artifact_uris=artifact_uris),
    )
