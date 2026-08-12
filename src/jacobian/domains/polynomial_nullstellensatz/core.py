"""Managed materialization and independent verification adapters."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
from jacobian.capability_errors import CapabilityInvocationError
from jacobian.checker_installation import CheckerInstaller
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInputKind,
    CapabilityProviderRuntime,
    CapabilityRequest,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.nullstellensatz import (
    JacobianDegreeSliceMaterializeOutput,
    JacobianDegreeSliceMaterializeRequest,
    NormalizedJacobianDegreeSliceSystem,
    NullstellensatzCertificateBundle,
    NullstellensatzVerificationOutput,
    NullstellensatzVerificationRequest,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.domains.polynomial_nullstellensatz.system import (
    materialize_degree_23_system,
)
from jacobian.installation.context import InstallationContext
from jacobian.operation_installation import InstalledDomainBundle
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed, Failed
from jacobian.schema_registry import model_schema
from jacobian.storage.errors import ArtifactNotFoundError, StorageError

MATERIALIZE_CAPABILITY_ID = "polynomial.jacobian_degree_slice.system.materialize"
VERIFY_CAPABILITY_ID = "polynomial.nullstellensatz.infeasibility_certificate.verify"
DOMAIN_ID = "polynomial_nullstellensatz"
CERTIFICATE_FORMAT = "polynomial.nullstellensatz.chart-cover"
_MAX_DIAGNOSTIC_REASON_CHARS = 512


@dataclass(frozen=True, slots=True)
class NullstellensatzCoreInstallation:
    semantics_uri: str
    system_schema_uri: str
    certificate_bundle_schema_uri: str
    certificate_envelope_schema_uri: str
    checker_id: str | None


def _diagnostic(
    code: str,
    stage: str,
    message: str,
    hint: str,
    *,
    expected: str | None = None,
    actual_type: str | None = None,
    details: dict[str, str] | None = None,
) -> CapabilityDiagnostic:
    return CapabilityDiagnostic(
        code=code,
        stage=stage,
        message=message,
        hint=hint,
        expected=expected,
        actual_type=actual_type,
        details=details or {},
    )


def _failure_details(exc: Exception) -> dict[str, str]:
    """Return bounded failure metadata without rendering an entire error tree."""
    details = {"exception_type": type(exc).__name__}
    if isinstance(exc, ValidationError):
        error_count = exc.error_count()
        details["validation_error_count"] = str(error_count)
        reason = f"validation_error: {error_count} invalid field(s)"
    else:
        reason = str(exc)
    details["reason"] = reason[:_MAX_DIAGNOSTIC_REASON_CHARS]
    return details


def _request_value_summary(value: object) -> str:
    """Return a URI when short, otherwise describe it without echoing contents."""
    if isinstance(value, str):
        if len(value) <= _MAX_DIAGNOSTIC_REASON_CHARS:
            return value
        return f"string(length={len(value)})"
    if isinstance(value, (list, tuple, dict, set)):
        return f"{type(value).__name__}(length={len(value)})"
    return type(value).__name__


class JacobianDegreeSliceMaterializeAdapter:
    def __init__(
        self,
        context: InstallationContext,
        installation: NullstellensatzCoreInstallation,
        provider_runtime: CapabilityProviderRuntime,
    ) -> None:
        self.context = context
        self.installation = installation
        self._descriptor = CapabilityDescriptor(
            capability_id=MATERIALIZE_CAPABILITY_ID,
            version="1",
            title="Materialize the normalized Jacobian degree-(2,3) slice",
            description=(
                "Materialize the exact 12-chart QQ polynomial system for normalized "
                "bivariate constant-Jacobian maps of exact component degrees (2,3)."
            ),
            provider=provider_runtime.provider,
            provider_runtime=provider_runtime,
            input_schema=model_schema(JacobianDegreeSliceMaterializeRequest),
            output_schema=model_schema(JacobianDegreeSliceMaterializeOutput),
            tags=("polynomial", "jacobian", "degree-slice", "rabinowitsch", "exact"),
            produced_artifact_types=(self.installation.system_schema_uri,),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> OperationProjection:
        try:
            JacobianDegreeSliceMaterializeRequest.model_validate(request.input)
        except ValidationError as exc:
            raise CapabilityInvocationError(
                _diagnostic(
                    "INVALID_JACOBIAN_DEGREE_SLICE_REQUEST",
                    "request_validation",
                    "The normalized degree-(2,3) materialization request is invalid.",
                    "Use the fixed statement ID and coefficient_domain=QQ.",
                )
            ) from exc
        started = time.monotonic()
        system = materialize_degree_23_system()
        stored = self.context.artifacts.put(
            schema_uri=self.installation.system_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=system.model_dump(mode="json"),
            summary="normalized bivariate Jacobian degree-(2,3) chart system",
            producer_write=True,
        )
        output = JacobianDegreeSliceMaterializeOutput(
            system_uri=stored.artifact_uri,
            system_digest=stored.object_digest,
        )
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=Completed(
                value=output,
                runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
            ),
            publication=PublishedOperation(
                output=output,
                artifact_uris=(stored.artifact_uri,),
            ),
        )


class NullstellensatzVerificationAdapter:
    def __init__(
        self,
        context: InstallationContext,
        installation: NullstellensatzCoreInstallation,
        provider_runtime: CapabilityProviderRuntime,
    ) -> None:
        self.context = context
        self.installation = installation
        self._descriptor = CapabilityDescriptor(
            capability_id=VERIFY_CAPABILITY_ID,
            version="1",
            title="Verify a chart-cover Nullstellensatz certificate",
            description=(
                "Independently multiply and sum every bounded QQ certificate identity "
                "for the exact 12-chart normalized Jacobian degree slice."
            ),
            provider=provider_runtime.provider,
            provider_runtime=provider_runtime,
            input_schema=model_schema(NullstellensatzVerificationRequest),
            output_schema=model_schema(NullstellensatzVerificationOutput),
            tags=(
                "polynomial",
                "nullstellensatz",
                "certificate",
                "verification",
                "exact",
            ),
            accepted_input_kinds=(CapabilityInputKind.TYPED_ARTIFACT,),
            accepted_artifact_types=(
                installation.system_schema_uri,
                installation.certificate_bundle_schema_uri,
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> OperationProjection:
        actual_artifact_type: str | None = None
        try:
            validated = NullstellensatzVerificationRequest.model_validate(request.input)
            system_artifact = self.context.store.get(validated.system_uri)
            bundle_artifact = self.context.store.get(validated.certificate_bundle_uri)
            if (
                system_artifact.manifest.schema_uri
                != self.installation.system_schema_uri
            ):
                actual_artifact_type = system_artifact.manifest.schema_uri
                raise ValueError(
                    "system_uri does not reference the registered system schema"
                )
            if (
                bundle_artifact.manifest.schema_uri
                != self.installation.certificate_bundle_schema_uri
            ):
                actual_artifact_type = bundle_artifact.manifest.schema_uri
                raise ValueError("certificate_bundle_uri has the wrong schema")
            if (
                system_artifact.manifest.semantics_uri
                != self.installation.semantics_uri
                or bundle_artifact.manifest.semantics_uri
                != self.installation.semantics_uri
            ):
                raise ValueError("artifacts use incompatible semantics")
            system = NormalizedJacobianDegreeSliceSystem.model_validate(
                system_artifact.payload
            )
            if system != materialize_degree_23_system():
                raise ValueError("system artifact differs from the frozen degree slice")
            bundle = NullstellensatzCertificateBundle.model_validate(
                bundle_artifact.payload
            )
            if (
                bundle.system_uri != system_artifact.artifact_uri
                or bundle.system_digest != system_artifact.manifest.object_digest
                or system_artifact.artifact_uri not in bundle_artifact.manifest.parents
            ):
                raise ValueError("certificate bundle has a stale system binding")
        except (
            ValidationError,
            ValueError,
            ArtifactNotFoundError,
            StorageError,
        ) as exc:
            requested_system_uri = (
                _request_value_summary(validated.system_uri)
                if "validated" in locals()
                else _request_value_summary(request.input.get("system_uri"))
            )
            requested_bundle_uri = (
                _request_value_summary(validated.certificate_bundle_uri)
                if "validated" in locals()
                else _request_value_summary(request.input.get("certificate_bundle_uri"))
            )
            raise CapabilityInvocationError(
                _diagnostic(
                    "INVALID_NULLSTELLENSATZ_VERIFICATION_REQUEST",
                    "artifact_resolution",
                    "The system and certificate bundle are not a compatible bound pair.",
                    (
                        "Use the exact materialized system URI and a certificate bundle "
                        "created for that system by a compatible installed producer. Do not "
                        "substitute unrelated artifacts. Without a compatible producer, this "
                        "verifier cannot establish infeasibility."
                    ),
                    expected=(
                        f"system schema {self.installation.system_schema_uri}; "
                        "certificate bundle schema "
                        f"{self.installation.certificate_bundle_schema_uri}"
                    ),
                    actual_type=actual_artifact_type,
                    details={
                        **_failure_details(exc),
                        "system_uri": requested_system_uri,
                        "certificate_bundle_uri": requested_bundle_uri,
                    },
                )
            ) from exc

        replay = {
            "system_uri": system_artifact.artifact_uri,
            "certificate_bundle_uri": bundle_artifact.artifact_uri,
        }
        semantics = self.context.store.get(self.installation.semantics_uri)
        envelope = CertificateEnvelope(
            certificate_type=CERTIFICATE_FORMAT,
            format_version="1",
            bindings=EvidenceBindings(
                claim_digest=system_artifact.manifest.object_digest,
                semantics_digest=semantics.manifest.object_digest,
                candidate_digest=bundle_artifact.manifest.object_digest,
            ),
            payload_digest="sha256:"
            + hashlib.sha256(canonicalize_json(replay)).hexdigest(),
            payload=replay,
        )
        evidence = self.context.artifacts.put(
            schema_uri=self.installation.certificate_envelope_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=envelope.model_dump(mode="json"),
            parents=(system_artifact.artifact_uri, bundle_artifact.artifact_uri),
            summary="bound Nullstellensatz chart-cover replay envelope",
        )
        checker_id = self.installation.checker_id
        checked = (
            self.context.verification.verify_certificate(
                certificate_uri=evidence.artifact_uri,
                checker_id=checker_id,
                timeout_seconds=float(validated.timeout_seconds),
            )
            if checker_id is not None
            else None
        )
        verified = bool(
            checked is not None
            and checked.execution.status is ExecutionStatus.COMPLETED
            and checked.verification_record_uri is not None
        )
        record_uri = checked.verification_record_uri if verified and checked else None
        output = NullstellensatzVerificationOutput(
            system_uri=system_artifact.artifact_uri,
            certificate_bundle_uri=bundle_artifact.artifact_uri,
            evidence_uri=evidence.artifact_uri,
            verification_record_uri=record_uri,
            checker_id=checker_id,
            conclusion="TRUE" if verified else "UNKNOWN",
            checked_chart_count=12 if verified else 0,
        )
        artifact_uris = [
            system_artifact.artifact_uri,
            bundle_artifact.artifact_uri,
            evidence.artifact_uri,
        ]
        if record_uri is not None:
            artifact_uris.append(record_uri)
        execution = (
            checked.execution
            if checked is not None
            else Execution(
                status=ExecutionStatus.COMPLETED,
                detail="no operator-authorized compatible checker is installed",
            )
        )
        terminal = (
            Completed(
                value=output,
                runtime_ms=execution.runtime_ms,
                detail=execution.detail,
            )
            if execution.status is ExecutionStatus.COMPLETED
            else Failed(
                status=execution.status,
                runtime_ms=execution.runtime_ms,
                diagnostic=CapabilityDiagnostic(
                    code="NULLSTELLENSATZ_CHECKER_NOT_COMPLETED",
                    stage="checker_replay",
                    message=(
                        execution.detail
                        or "The authorized Nullstellensatz checker did not complete."
                    ),
                ),
            )
        )
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=terminal,
            publication=PublishedOperation(
                output=output,
                artifact_uris=tuple(artifact_uris),
            ),
            verification_record_uri=record_uri if verified else None,
        )


def install_nullstellensatz_core(
    context: InstallationContext,
    provider_runtime: CapabilityProviderRuntime,
) -> InstalledDomainBundle:
    semantics_uri = context.store.register_descriptor(
        kind="semantics",
        name="jacobian.normalized-bivariate-jacobian-degree-2-3",
        version="1",
        definition={
            "coefficient_domain": "QQ",
            "source_characteristic": 0,
            "normalization": "F(0)=0; JF(0)=I; det(JF)=1",
            "component_degrees": [2, 3],
            "exact_degree_encoding": (
                "12 charts t*a_i*b_j-1, equivalent to nonzero quadratic and cubic top vectors"
            ),
            "certificate_identity": "sum(h_i*f_i)=1 in QQ[a20,...,b03,t]",
        },
    )
    system_schema_uri = context.schemas.register_model(
        name="jacobian.normalized-jacobian-degree-2-3-system",
        version="1",
        model=NormalizedJacobianDegreeSliceSystem,
        producer_only=True,
    )
    bundle_schema_uri = context.schemas.register_model(
        name="jacobian.nullstellensatz-chart-cover-certificate",
        version="1",
        model=NullstellensatzCertificateBundle,
        producer_only=True,
    )
    envelope_schema_uri = context.schemas.register_model(
        name="jacobian.certificate-envelope",
        version="1",
        model=CertificateEnvelope,
    )
    installed_checker = CheckerInstaller(context.checkers).install(
        CheckerOperation(
            name="exact bounded Nullstellensatz chart-cover checker",
            entrypoint="jacobian_checkers.nullstellensatz:check_chart_cover",
            evidence_kind=EvidenceKind.CERTIFICATE,
            format_id=CERTIFICATE_FORMAT,
            format_version="1",
            claim_schema_uris=(system_schema_uri,),
            semantics_uris=(semantics_uri,),
            candidate_schema_uris=(bundle_schema_uri,),
            reason=(
                "bundled standard-library sparse QQ replay independent of Singular and Groebner generation"
            ),
        ),
        authorize=context.authorizes_bundled_checkers,
    )
    installation = NullstellensatzCoreInstallation(
        semantics_uri=semantics_uri,
        system_schema_uri=system_schema_uri,
        certificate_bundle_schema_uri=bundle_schema_uri,
        certificate_envelope_schema_uri=envelope_schema_uri,
        checker_id=installed_checker.checker_id,
    )
    adapters = (
        JacobianDegreeSliceMaterializeAdapter(context, installation, provider_runtime),
        NullstellensatzVerificationAdapter(context, installation, provider_runtime),
    )
    return InstalledDomainBundle(
        adapters=adapters,
        semantics_uri=semantics_uri,
        input_schema_uris={
            JacobianDegreeSliceMaterializeRequest: system_schema_uri,
            NullstellensatzVerificationRequest: envelope_schema_uri,
        },
        result_schema_uris={MATERIALIZE_CAPABILITY_ID: system_schema_uri},
        named_schema_uris={
            "nullstellensatz_certificate_bundle": bundle_schema_uri,
            "certificate_envelope": envelope_schema_uri,
        },
    )


__all__ = [
    "CERTIFICATE_FORMAT",
    "DOMAIN_ID",
    "MATERIALIZE_CAPABILITY_ID",
    "VERIFY_CAPABILITY_ID",
    "NullstellensatzCoreInstallation",
    "install_nullstellensatz_core",
]
