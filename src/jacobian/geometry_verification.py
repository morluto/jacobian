"""Independent verification capability for selected exact geometry results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from jacobian.artifacts import ArtifactService
from jacobian.capabilities import CapabilityAdapter, CapabilityInvocationError
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
    CapabilityMode,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import WitnessEnvelope
from jacobian.contracts.geometry import (
    GeometryVerificationOutput,
    GeometryVerificationRequest,
    PointPairRequest,
    PointSetRequest,
    PointTripleRequest,
    PolygonRequest,
    SegmentIntersectionRequest,
    SimplePolygonPointRequest,
)
from jacobian.contracts.results import Conclusion, ExecutionStatus, Verification
from jacobian.operation_installation import InstalledDomainBundle
from jacobian.provider_runtime import known_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError, model_schema
from jacobian.store import ArtifactStore, StoredArtifact, StoreError
from jacobian.verification import VerificationService

_OPERATION_MODELS = {
    "geometry.points.compute.convex_hull": PointSetRequest,
    "geometry.points.compute.squared_distance": PointPairRequest,
    "geometry.segment.compute.midpoint": PointPairRequest,
    "geometry.segments.intersection.compute": SegmentIntersectionRequest,
    "geometry.polygon.simple.decide": PolygonRequest,
    "geometry.polygon.point.classify": SimplePolygonPointRequest,
    "geometry.triangle.compute.orientation": PointTripleRequest,
    "geometry.triangle.compute.centroid": PointTripleRequest,
}


class GeometryResultArtifactError(ValueError):
    """A geometry result is invalid or not bound to its exact input."""


@dataclass(frozen=True, slots=True)
class GeometryCheckerInstallation:
    witness_schema_uri: str
    checker_id: str | None


@dataclass(frozen=True, slots=True)
class _ResolvedResult:
    operation_id: str
    input_artifact: StoredArtifact
    result_artifact: StoredArtifact


def install_geometry_checker(
    store: ArtifactStore,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    geometry: InstalledDomainBundle,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[CapabilityAdapter | None, GeometryCheckerInstallation]:
    """Install schemas and optionally authorize exact geometry replay."""

    missing = set(_OPERATION_MODELS) - geometry.result_schema_uris.keys()
    if missing:
        raise ValueError(
            f"geometry bundle lacks verifiable operations: {sorted(missing)}"
        )
    witness_schema_uri = schemas.register_model(
        name="jacobian.witness-envelope",
        version="1",
        model=WitnessEnvelope,
    )
    result_schema_uris = tuple(
        geometry.result_schema_uris[operation_id] for operation_id in _OPERATION_MODELS
    )
    claim_schema_uris = tuple(
        dict.fromkeys(
            geometry.input_schema_uris[cast(type[Any], model)]
            for model in _OPERATION_MODELS.values()
        )
    )
    checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="selected exact rational geometry replay checker",
                entrypoint="jacobian_checkers.exact_geometry:check_exact_geometry",
                evidence_kind=EvidenceKind.WITNESS,
                format_id="geometry.exact_rational_result",
                format_version="1",
                claim_schema_uris=claim_schema_uris,
                semantics_uris=(geometry.semantics_uri,),
                candidate_schema_uris=result_schema_uris,
                reason=(
                    "bundled independent standard-library rational replay of "
                    "selected geometry identities"
                ),
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    installation = GeometryCheckerInstallation(
        witness_schema_uri=witness_schema_uri,
        checker_id=checker_id,
    )
    if checker_id is None:
        return None, installation
    return (
        GeometryResultVerificationAdapter(
            store=store,
            schemas=schemas,
            artifacts=artifacts,
            geometry=geometry,
            verification=verification,
            installation=installation,
        ),
        installation,
    )


class GeometryResultVerificationAdapter:
    """Verify one stored geometry result by independent exact replay."""

    def __init__(
        self,
        *,
        store: ArtifactStore,
        schemas: SchemaRegistry,
        artifacts: ArtifactService,
        geometry: InstalledDomainBundle,
        verification: VerificationService,
        installation: GeometryCheckerInstallation,
    ) -> None:
        checker_id = installation.checker_id
        if checker_id is None:
            raise ValueError("geometry checker is not authorized")
        self.store = store
        self.schemas = schemas
        self.artifacts = artifacts
        self.geometry = geometry
        self.verification = verification
        self.installation = installation
        self._operation_by_schema = {
            geometry.result_schema_uris[operation_id]: operation_id
            for operation_id in _OPERATION_MODELS
        }
        self._descriptor = CapabilityDescriptor(
            capability_id="geometry.result.verify",
            version="1",
            title="Verify an exact rational geometry result",
            description=(
                "Independently replay selected point, segment, polygon, hull, "
                "and triangle outcomes over exact rational coordinates."
            ),
            provider="jacobian.exact-geometry-checker",
            provider_runtime=known_provider_runtime(
                "jacobian.exact-geometry-checker",
                features=("standard-library-rational-replay", "clean-process-checker"),
                checker_ids=(checker_id,),
            ),
            modes=(CapabilityMode.VERIFY,),
            input_schema=model_schema(GeometryVerificationRequest),
            output_schema=model_schema(GeometryVerificationOutput),
            tags=("geometry", "rational", "verification"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = GeometryVerificationRequest.model_validate(request.input)
        try:
            resolved = self._resolve(validated.result_uri)
            semantics = self.store.get(self.geometry.semantics_uri)
        except (GeometryResultArtifactError, StoreError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_GEOMETRY_RESULT",
                    stage="artifact_resolution",
                    message=str(exc),
                    path="result_uri",
                    expected="a supported exact geometry result with exact input lineage",
                    hint="Invoke one of the selected geometry producers first.",
                )
            ) from exc

        checker_id = self.installation.checker_id
        assert checker_id is not None
        witness = put_witness_envelope(
            self.artifacts,
            witness_schema_uri=self.installation.witness_schema_uri,
            witness_format="geometry.exact_rational_result",
            claim_artifact=resolved.input_artifact,
            semantics_artifact=semantics,
            candidate_artifact=resolved.result_artifact,
            payload={
                "operation_id": resolved.operation_id,
                "input_uri": resolved.input_artifact.artifact_uri,
                "result_uri": resolved.result_artifact.artifact_uri,
            },
            summary=f"{resolved.operation_id} independent verification witness",
        )
        checked = self.verification.verify_witness(
            claim_uri=resolved.input_artifact.artifact_uri,
            candidate_uri=resolved.result_artifact.artifact_uri,
            witness_uri=witness.artifact_uri,
            checker_id=checker_id,
            include_artifact_metadata=True,
            include_semantics_artifact=True,
        )
        verified = (
            checked.execution.status is ExecutionStatus.COMPLETED
            and checked.conclusion is Conclusion.TRUE
            and checked.assurance.verification is Verification.VERIFIED
            and checked.verification_record_uri is not None
        )
        status: Literal["VERIFIED_RESULT", "REJECTED", "TIMEOUT", "CANCELLED", "ERROR"]
        if verified:
            status = "VERIFIED_RESULT"
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
                "the authorized checker accepted the exact geometry result"
                if verified
                else "the geometry result was not independently accepted"
            )
        output = GeometryVerificationOutput(
            status=status,
            conclusion="TRUE" if verified else "UNKNOWN",
            operation_id=cast(Any, resolved.operation_id),
            input_uri=resolved.input_artifact.artifact_uri,
            result_uri=resolved.result_artifact.artifact_uri,
            witness_uri=witness.artifact_uri,
            checker_id=checker_id,
            verification_record_uri=(
                checked.verification_record_uri if verified else None
            ),
            detail=detail,
        )
        artifact_uris = [
            resolved.input_artifact.artifact_uri,
            resolved.result_artifact.artifact_uri,
            witness.artifact_uri,
        ]
        if checked.verification_record_uri is not None:
            artifact_uris.append(checked.verification_record_uri)
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=checked.execution,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="the full supplied exact rational geometry input",
                parameters={
                    "operation_id": resolved.operation_id,
                    "arithmetic": "EXACT_RATIONAL",
                    "method": "DIRECT_IDENTITY_REPLAY",
                },
                artifact_uri=resolved.input_artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.NOT_APPLICABLE,
                basis="direct identities cover the complete finite input",
                assurance_level=(
                    CapabilityAssuranceLevel.COMPUTED
                    if checked.execution.status is ExecutionStatus.COMPLETED
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
            ),
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else (
                        CapabilityAssuranceLevel.COMPUTED
                        if checked.execution.status is ExecutionStatus.COMPLETED
                        else CapabilityAssuranceLevel.HEURISTIC
                    )
                ),
                basis=(
                    "accepted in a clean process by the operator-authorized "
                    "independent exact geometry checker"
                    if verified
                    else "checker did not accept the candidate; no opposite conclusion follows"
                ),
                verification_record_uri=(
                    checked.verification_record_uri if verified else None
                ),
            ),
            artifact_uris=tuple(artifact_uris),
        )

    def _resolve(self, result_uri: str) -> _ResolvedResult:
        try:
            result = self.store.get(result_uri)
        except StoreError as exc:
            raise GeometryResultArtifactError(
                "candidate is not an available geometry result"
            ) from exc
        operation_id = self._operation_by_schema.get(result.manifest.schema_uri)
        if (
            operation_id is None
            or result.manifest.semantics_uri != self.geometry.semantics_uri
        ):
            raise GeometryResultArtifactError(
                "candidate is not a supported exact geometry result"
            )
        if len(result.manifest.parents) != 1:
            raise GeometryResultArtifactError(
                "candidate lineage must identify exactly one geometry input"
            )
        input_artifact = self.store.get(result.manifest.parents[0])
        expected_input_schema = self.geometry.input_schema_uris[
            cast(type[Any], _OPERATION_MODELS[operation_id])
        ]
        if (
            input_artifact.manifest.schema_uri != expected_input_schema
            or input_artifact.manifest.semantics_uri != self.geometry.semantics_uri
            or input_artifact.manifest.parents
        ):
            raise GeometryResultArtifactError(
                "candidate source is not the expected exact geometry input"
            )
        try:
            self.schemas.validate(expected_input_schema, input_artifact.payload)
            self.schemas.validate(
                self.geometry.result_schema_uris[operation_id], result.payload
            )
        except SchemaRegistryError as exc:
            raise GeometryResultArtifactError(
                "candidate or source does not satisfy its registered geometry schema"
            ) from exc
        return _ResolvedResult(
            operation_id=operation_id,
            input_artifact=input_artifact,
            result_artifact=result,
        )
