"""Independent verification capability for integer row Hermite normal forms.

Ownership: ``jacobian.matrices`` (HNF checker).
Installed by ``FoundationInstaller`` to provide ``matrix.normal_form.hermite.verify``
for evidence bound to the HNF artifact service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from jacobian.artifacts import ArtifactService
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
    CapabilityMode,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import (
    EvidenceBindings,
    WitnessEnvelope,
    WitnessRole,
)
from jacobian.contracts.matrices import (
    MatrixHermiteNormalFormVerificationOutput,
    MatrixHermiteNormalFormVerificationRequest,
)
from jacobian.contracts.results import Conclusion, ExecutionStatus, Verification
from jacobian.matrices.normal_forms import (
    MatrixNormalFormArtifactError,
    MatrixNormalFormArtifactService,
)
from jacobian.provider_runtime import known_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService


@dataclass(frozen=True, slots=True)
class MatrixNormalFormCheckerInstallation:
    witness_schema_uri: str
    checker_id: str | None


def install_matrix_normal_form_checker(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    matrices: MatrixNormalFormArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[CapabilityAdapter | None, MatrixNormalFormCheckerInstallation]:
    """Install the witness schema and optionally authorize exact replay."""

    witness_schema_uri = schemas.register_model(
        name="jacobian.witness-envelope",
        version="1",
        model=WitnessEnvelope,
    )
    checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="integer row Hermite-normal-form replay checker",
                entrypoint=(
                    "jacobian_checkers.matrix_normal_forms:check_hermite_normal_form"
                ),
                evidence_kind=EvidenceKind.WITNESS,
                format_id="matrix.normal_form.hermite",
                format_version="1",
                claim_schema_uris=(matrices.installation.matrix_schema_uri,),
                semantics_uris=(matrices.installation.semantics_uri,),
                candidate_schema_uris=(matrices.installation.normal_form_schema_uri,),
                reason=(
                    "bundled independent exact H=U*A, unimodularity, and row-HNF checker"
                ),
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    installation = MatrixNormalFormCheckerInstallation(
        witness_schema_uri=witness_schema_uri,
        checker_id=checker_id,
    )
    adapter: CapabilityAdapter | None = None
    if checker_id is not None:
        adapter = MatrixHermiteNormalFormVerificationAdapter(
            store=store,
            artifacts=artifacts,
            matrices=matrices,
            verification=verification,
            installation=installation,
        )
    return adapter, installation


class MatrixHermiteNormalFormVerificationAdapter:
    """Verify exact row equivalence and the full FLINT row-HNF convention."""

    def __init__(
        self,
        *,
        store: ArtifactRepository,
        artifacts: ArtifactService,
        matrices: MatrixNormalFormArtifactService,
        verification: VerificationService,
        installation: MatrixNormalFormCheckerInstallation,
    ) -> None:
        checker_id = installation.checker_id
        if checker_id is None:
            raise ValueError("matrix normal-form checker is not authorized")
        self.store = store
        self.artifacts = artifacts
        self.matrices = matrices
        self.verification = verification
        self.installation = installation
        self._descriptor = CapabilityDescriptor(
            capability_id="matrix.normal_form.hermite.verify",
            version="1",
            title="Verify an exact row Hermite normal form",
            description=(
                "Independently check H = U A over ZZ, det(U) = +/-1, and every "
                "FLINT row-HNF condition for one bound stored candidate."
            ),
            provider="jacobian.integer-matrix",
            provider_runtime=known_provider_runtime(
                "jacobian.integer-matrix",
                features=(
                    "exact-integer-replay",
                    "unimodular-left-transformation",
                    "row-hermite-normal-form",
                    "clean-process-checker",
                ),
                checker_ids=(checker_id,),
            ),
            modes=(CapabilityMode.VERIFY,),
            input_schema=model_schema(MatrixHermiteNormalFormVerificationRequest),
            output_schema=model_schema(MatrixHermiteNormalFormVerificationOutput),
            tags=(
                "linear-algebra",
                "integer",
                "matrix",
                "hermite-normal-form",
                "verification",
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = MatrixHermiteNormalFormVerificationRequest.model_validate(
            request.input
        )
        try:
            resolved = self.matrices.resolve_hermite_normal_form(
                validated.normal_form_uri
            )
            semantics = self.store.get(self.matrices.installation.semantics_uri)
        except (MatrixNormalFormArtifactError, StorageError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_HERMITE_NORMAL_FORM",
                    stage="artifact_resolution",
                    message=str(exc),
                    path="normal_form_uri",
                    schema_uri=self.matrices.installation.normal_form_schema_uri,
                    expected=(
                        "a valid H and U bound by payload and lineage to one exact "
                        "integer matrix"
                    ),
                    hint=(
                        "Use matrix.normal_form.hermite or materialize a candidate "
                        "with the registered normal-form schema."
                    ),
                )
            ) from exc

        checker_id = self.installation.checker_id
        if checker_id is None:
            raise RuntimeError("checker is not installed")
        bindings = EvidenceBindings(
            claim_digest=resolved.matrix_artifact.manifest.object_digest,
            semantics_digest=semantics.manifest.object_digest,
            candidate_digest=resolved.artifact.manifest.object_digest,
        )
        witness = WitnessEnvelope(
            witness_format="matrix.normal_form.hermite",
            format_version="1",
            role=WitnessRole.SUPPORTS_CLAIM,
            bindings=bindings,
            payload={
                "matrix_uri": resolved.matrix_artifact.artifact_uri,
                "normal_form_uri": resolved.artifact.artifact_uri,
            },
        )
        witness_artifact = self.artifacts.put(
            schema_uri=self.installation.witness_schema_uri,
            semantics_uri=self.matrices.installation.semantics_uri,
            payload=witness.model_dump(mode="json"),
            parents=(
                resolved.matrix_artifact.artifact_uri,
                resolved.artifact.artifact_uri,
            ),
            summary="integer row-HNF verification witness",
        )
        checked = self.verification.verify_witness(
            claim_uri=resolved.matrix_artifact.artifact_uri,
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
            "VERIFIED_HERMITE_NORMAL_FORM",
            "REJECTED",
            "TIMEOUT",
            "CANCELLED",
            "ERROR",
        ]
        if verified:
            status = "VERIFIED_HERMITE_NORMAL_FORM"
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
                "the authorized checker accepted the exact row-HNF relation"
                if verified
                else "the candidate was not independently accepted"
            )
        output = MatrixHermiteNormalFormVerificationOutput(
            status=status,
            conclusion="TRUE" if verified else "UNKNOWN",
            matrix_uri=resolved.matrix_artifact.artifact_uri,
            normal_form_uri=resolved.artifact.artifact_uri,
            witness_uri=witness_artifact.artifact_uri,
            checker_id=checker_id,
            verification_record_uri=(
                checked.verification_record_uri if verified else None
            ),
            detail=detail,
        )
        record_uri = checked.verification_record_uri if verified else None
        artifact_uris = [
            resolved.matrix_artifact.artifact_uri,
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
                description=(
                    "the full exact source matrix, proposed H, and left transform U"
                ),
                parameters={
                    "declared_scope": "FULL_MATRIX",
                    "row_count": len(resolved.matrix.entries),
                    "column_count": len(resolved.matrix.entries[0]),
                    "normal_form_convention": "FLINT_ROW_HNF",
                    "checked_relation": "H=U*A",
                    "checked_unimodularity": True,
                },
                artifact_uri=resolved.matrix_artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.NOT_APPLICABLE,
                basis=(
                    "direct exact replay checks the full finite relation and form; "
                    "no search or enumeration claim is made"
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
                    "accepted in a clean process by the operator-authorized "
                    "independent exact integer row-HNF checker"
                    if verified
                    else (
                        "checker replay completed without accepting the candidate; "
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
