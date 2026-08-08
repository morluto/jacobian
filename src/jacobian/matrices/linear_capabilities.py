"""Independent verification of exact rational linear-system evidence.

Ownership: ``jacobian.matrices`` (linear checkers).
Installed by ``FoundationInstaller`` to provide
``linear.rational_solution.verify`` and ``linear.rational_inconsistency.verify``
for evidence bound to the linear artifact service.
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
    CapabilityRelationship,
    CapabilityRelationshipStatus,
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
from jacobian.contracts.linear import (
    LinearRationalInconsistencyVerificationOutput,
    LinearRationalInconsistencyVerificationRequest,
    LinearRationalSolutionVerificationOutput,
    LinearRationalSolutionVerificationRequest,
)
from jacobian.contracts.results import Conclusion, ExecutionStatus, Verification
from jacobian.matrices.linear import LinearArtifactError, LinearArtifactService
from jacobian.provider_runtime import known_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService


@dataclass(frozen=True, slots=True)
class LinearRationalSolutionCheckerInstallation:
    witness_schema_uri: str
    checker_id: str | None


@dataclass(frozen=True, slots=True)
class LinearRationalInconsistencyCheckerInstallation:
    witness_schema_uri: str
    checker_id: str | None


def install_linear_rational_inconsistency_checker(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    linear: LinearArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[
    CapabilityAdapter | None,
    LinearRationalInconsistencyCheckerInstallation,
]:
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
                name="exact rational linear-inconsistency replay checker",
                entrypoint="jacobian_checkers.linear:check_rational_inconsistency",
                evidence_kind=EvidenceKind.WITNESS,
                format_id="linear.rational_inconsistency",
                format_version="1",
                claim_schema_uris=(linear.installation.system_schema_uri,),
                semantics_uris=(linear.installation.semantics_uri,),
                candidate_schema_uris=(linear.installation.inconsistency_schema_uri,),
                reason="bundled independent exact rational left-witness checker",
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    installation = LinearRationalInconsistencyCheckerInstallation(
        witness_schema_uri=witness_schema_uri,
        checker_id=checker_id,
    )
    adapter: CapabilityAdapter | None = None
    if checker_id is not None:
        adapter = LinearRationalInconsistencyVerificationAdapter(
            store=store,
            artifacts=artifacts,
            linear=linear,
            verification=verification,
            installation=installation,
        )
    return adapter, installation


def install_linear_rational_solution_checker(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    linear: LinearArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[
    CapabilityAdapter | None,
    LinearRationalSolutionCheckerInstallation,
]:
    """Install evidence schema and optionally authorize independent replay."""

    witness_schema_uri = schemas.register_model(
        name="jacobian.witness-envelope",
        version="1",
        model=WitnessEnvelope,
    )
    checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="exact rational linear-solution replay checker",
                entrypoint="jacobian_checkers.linear:check_rational_solution",
                evidence_kind=EvidenceKind.WITNESS,
                format_id="linear.rational_solution",
                format_version="1",
                claim_schema_uris=(linear.installation.system_schema_uri,),
                semantics_uris=(linear.installation.semantics_uri,),
                candidate_schema_uris=(linear.installation.solution_schema_uri,),
                reason="bundled independent exact rational equation checker",
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    installation = LinearRationalSolutionCheckerInstallation(
        witness_schema_uri=witness_schema_uri,
        checker_id=checker_id,
    )
    adapter: CapabilityAdapter | None = None
    if checker_id is not None:
        adapter = LinearRationalSolutionVerificationAdapter(
            store=store,
            artifacts=artifacts,
            linear=linear,
            verification=verification,
            installation=installation,
        )
    return adapter, installation


class LinearRationalSolutionVerificationAdapter:
    """Verify ``A x = b``; rejection never proves inconsistency."""

    def __init__(
        self,
        *,
        store: ArtifactRepository,
        artifacts: ArtifactService,
        linear: LinearArtifactService,
        verification: VerificationService,
        installation: LinearRationalSolutionCheckerInstallation,
    ) -> None:
        checker_id = installation.checker_id
        if checker_id is None:
            raise ValueError("rational solution checker is not authorized")
        self.store = store
        self.artifacts = artifacts
        self.linear = linear
        self.verification = verification
        self.installation = installation
        self._descriptor = CapabilityDescriptor(
            capability_id="linear.rational_solution.verify",
            version="1",
            title="Verify one exact rational solution",
            description=(
                "Independently replay every equation of one exact stored A x = b "
                "system against its bound total rational vector. Rejection does not "
                "establish inconsistency."
            ),
            provider="jacobian.linear",
            provider_runtime=known_provider_runtime(
                "jacobian.linear",
                features=(
                    "exact-rational-equation-replay",
                    "total-vector",
                    "clean-process-checker",
                ),
                checker_ids=(checker_id,),
            ),
            modes=(CapabilityMode.VERIFY,),
            input_schema=model_schema(LinearRationalSolutionVerificationRequest),
            output_schema=model_schema(LinearRationalSolutionVerificationOutput),
            tags=(
                "linear-algebra",
                "rational",
                "solution",
                "witness",
                "verification",
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = LinearRationalSolutionVerificationRequest.model_validate(
            request.input
        )
        try:
            resolved = self.linear.resolve_solution(validated.solution_uri)
            semantics = self.store.get(self.linear.installation.semantics_uri)
        except (LinearArtifactError, StorageError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_RATIONAL_SOLUTION",
                    stage="artifact_resolution",
                    message=str(exc),
                    path="solution_uri",
                    schema_uri=self.linear.installation.solution_schema_uri,
                    expected=(
                        "a valid total exact rational vector bound by payload and "
                        "lineage to one rational linear-system artifact"
                    ),
                    hint=(
                        "Use linear.rational_solution.find or materialize a candidate "
                        "with the registered solution schema against the intended "
                        "system."
                    ),
                )
            ) from exc

        checker_id = self.installation.checker_id
        if checker_id is None:
            raise RuntimeError("checker is not installed")
        bindings = EvidenceBindings(
            claim_digest=resolved.system_artifact.manifest.object_digest,
            semantics_digest=semantics.manifest.object_digest,
            candidate_digest=resolved.artifact.manifest.object_digest,
        )
        witness = WitnessEnvelope(
            witness_format="linear.rational_solution",
            format_version="1",
            role=WitnessRole.SUPPORTS_CLAIM,
            bindings=bindings,
            payload={
                "system_uri": resolved.system_artifact.artifact_uri,
                "solution_uri": resolved.artifact.artifact_uri,
            },
        )
        witness_artifact = self.artifacts.put(
            schema_uri=self.installation.witness_schema_uri,
            semantics_uri=self.linear.installation.semantics_uri,
            payload=witness.model_dump(mode="json"),
            parents=(
                resolved.system_artifact.artifact_uri,
                resolved.artifact.artifact_uri,
            ),
            summary="rational linear-solution verification witness",
        )
        checked = self.verification.verify_witness(
            claim_uri=resolved.system_artifact.artifact_uri,
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
            "VERIFIED_SOLUTION",
            "REJECTED",
            "TIMEOUT",
            "CANCELLED",
            "ERROR",
        ]
        if verified:
            status = "VERIFIED_SOLUTION"
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
                "the authorized checker accepted every exact bound equation"
                if verified
                else "the candidate was not independently accepted"
            )
        output = LinearRationalSolutionVerificationOutput(
            status=status,
            conclusion="TRUE" if verified else "UNKNOWN",
            system_uri=resolved.system_artifact.artifact_uri,
            solution_uri=resolved.artifact.artifact_uri,
            witness_uri=witness_artifact.artifact_uri,
            checker_id=checker_id,
            verification_record_uri=(
                checked.verification_record_uri if verified else None
            ),
            detail=detail,
        )
        record_uri = checked.verification_record_uri if verified else None
        artifact_uris = [
            resolved.system_artifact.artifact_uri,
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
                description="the full exact rational A x = b system bound by the vector",
                parameters={
                    "declared_scope": "FULL_SYSTEM",
                    "row_count": len(resolved.system.coefficients.entries),
                    "column_count": len(resolved.system.variables),
                    "variable_order_digest": (
                        resolved.solution.system.variable_order_digest
                    ),
                },
                artifact_uri=resolved.system_artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.NOT_APPLICABLE,
                basis=(
                    "direct equation replay checks one vector and makes no uniqueness "
                    "or inconsistency completeness claim"
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
                    "independent exact rational equation checker"
                    if verified
                    else (
                        "checker replay completed without accepting the vector; no "
                        "opposite conclusion follows"
                        if checked.execution.status is ExecutionStatus.COMPLETED
                        else "checker execution did not complete; no mathematical "
                        "conclusion follows"
                    )
                ),
                verification_record_uri=record_uri,
            ),
            artifact_uris=tuple(artifact_uris),
        )


class LinearRationalInconsistencyVerificationAdapter:
    """Independently replay a normalized left inconsistency witness."""

    def __init__(
        self,
        *,
        store: ArtifactRepository,
        artifacts: ArtifactService,
        linear: LinearArtifactService,
        verification: VerificationService,
        installation: LinearRationalInconsistencyCheckerInstallation,
    ) -> None:
        checker_id = installation.checker_id
        if checker_id is None:
            raise ValueError("rational inconsistency checker is not authorized")
        self.store = store
        self.artifacts = artifacts
        self.linear = linear
        self.verification = verification
        self.installation = installation
        self._descriptor = CapabilityDescriptor(
            capability_id="linear.rational_inconsistency.verify",
            version="1",
            title="Verify an exact rational inconsistency certificate",
            description=(
                "Independently check every column of y^T A = 0 and the normalized "
                "pairing y^T b = 1 for one stored exact rational system."
            ),
            provider="jacobian.linear",
            provider_runtime=known_provider_runtime(
                "jacobian.linear",
                features=(
                    "exact-rational-left-witness-replay",
                    "normalized-nonzero-pairing",
                    "clean-process-checker",
                ),
                checker_ids=(checker_id,),
            ),
            modes=(CapabilityMode.VERIFY,),
            input_schema=model_schema(LinearRationalInconsistencyVerificationRequest),
            output_schema=model_schema(LinearRationalInconsistencyVerificationOutput),
            tags=(
                "linear-algebra",
                "rational",
                "inconsistency",
                "certificate",
                "verification",
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = LinearRationalInconsistencyVerificationRequest.model_validate(
            request.input
        )
        try:
            resolved = self.linear.resolve_inconsistency(validated.certificate_uri)
            semantics = self.store.get(self.linear.installation.semantics_uri)
        except (LinearArtifactError, StorageError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_RATIONAL_INCONSISTENCY_CERTIFICATE",
                    stage="artifact_resolution",
                    message=str(exc),
                    path="certificate_uri",
                    schema_uri=self.linear.installation.inconsistency_schema_uri,
                    expected=(
                        "a normalized exact rational left witness bound by payload "
                        "and lineage to one rational linear-system artifact"
                    ),
                    hint=(
                        "Use linear.rational_inconsistency.find or materialize a "
                        "candidate against the intended system."
                    ),
                )
            ) from exc

        checker_id = self.installation.checker_id
        if checker_id is None:
            raise RuntimeError("checker is not installed")
        bindings = EvidenceBindings(
            claim_digest=resolved.system_artifact.manifest.object_digest,
            semantics_digest=semantics.manifest.object_digest,
            candidate_digest=resolved.artifact.manifest.object_digest,
        )
        witness = WitnessEnvelope(
            witness_format="linear.rational_inconsistency",
            format_version="1",
            role=WitnessRole.SUPPORTS_CLAIM,
            bindings=bindings,
            payload={
                "system_uri": resolved.system_artifact.artifact_uri,
                "certificate_uri": resolved.artifact.artifact_uri,
            },
        )
        witness_artifact = self.artifacts.put(
            schema_uri=self.installation.witness_schema_uri,
            semantics_uri=self.linear.installation.semantics_uri,
            payload=witness.model_dump(mode="json"),
            parents=(
                resolved.system_artifact.artifact_uri,
                resolved.artifact.artifact_uri,
            ),
            summary="rational linear-inconsistency verification witness",
        )
        checked = self.verification.verify_witness(
            claim_uri=resolved.system_artifact.artifact_uri,
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
            "VERIFIED_INCONSISTENT",
            "REJECTED",
            "TIMEOUT",
            "CANCELLED",
            "ERROR",
        ]
        if verified:
            status = "VERIFIED_INCONSISTENT"
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
                "the authorized checker accepted every exact left-witness equation"
                if verified
                else "the candidate was not independently accepted"
            )
        output = LinearRationalInconsistencyVerificationOutput(
            status=status,
            conclusion="TRUE" if verified else "UNKNOWN",
            system_uri=resolved.system_artifact.artifact_uri,
            certificate_uri=resolved.artifact.artifact_uri,
            witness_uri=witness_artifact.artifact_uri,
            checker_id=checker_id,
            verification_record_uri=(
                checked.verification_record_uri if verified else None
            ),
            detail=detail,
        )
        record_uri = checked.verification_record_uri if verified else None
        artifact_uris = [
            resolved.system_artifact.artifact_uri,
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
                    "the full exact rational A x = b system bound by the left witness"
                ),
                parameters={
                    "declared_scope": "FULL_SYSTEM",
                    "row_count": len(resolved.system.coefficients.entries),
                    "column_count": len(resolved.system.variables),
                    "variable_order_digest": (
                        resolved.certificate.system.variable_order_digest
                    ),
                    "normalization": "Y_TRANSPOSE_B_EQUALS_ONE",
                },
                artifact_uri=resolved.system_artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.NOT_APPLICABLE,
                basis=(
                    "direct exact replay of one left witness establishes "
                    "inconsistency when accepted"
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
                    "independent exact rational left-witness checker"
                    if verified
                    else (
                        "checker replay completed without accepting the witness; no "
                        "opposite conclusion follows"
                        if checked.execution.status is ExecutionStatus.COMPLETED
                        else "checker execution did not complete; no mathematical "
                        "conclusion follows"
                    )
                ),
                verification_record_uri=record_uri,
            ),
            artifact_uris=tuple(artifact_uris),
            relationships=(
                (
                    CapabilityRelationship(
                        relation_id=("linear.relation.inconsistency-certificate-of"),
                        source_artifact_uris=(resolved.artifact.artifact_uri,),
                        target_artifact_uris=(resolved.system_artifact.artifact_uri,),
                        status=CapabilityRelationshipStatus.VERIFIED,
                        verification_record_uri=record_uri,
                    ),
                )
                if verified
                else ()
            ),
        )
