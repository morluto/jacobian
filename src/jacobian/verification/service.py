"""Authorized witness and certificate replay services."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from jacobian.bounded_process import ProcessResourceLimits
from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    loads_strict_json,
)
from jacobian.contracts.artifacts import ArtifactPutResult
from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.contracts.checkers import CheckerDecision, EvidenceKind
from jacobian.contracts.evidence import (
    CertificateEnvelope,
    EvidenceBindings,
    PreservationEnvelope,
    WitnessEnvelope,
)
from jacobian.contracts.results import (
    Arithmetic,
    Assurance,
    Conclusion,
    Coverage,
    Execution,
    ExecutionStatus,
    InputStatus,
    InputValidation,
    Method,
    ResultEnvelope,
    Verification,
)
from jacobian.contracts.transformations import (
    TransformationClaim,
    TransformationEnvelope,
    TransformationVerificationRecord,
)
from jacobian.contracts.verification import VerificationRecord
from jacobian.process_policy import (
    ProcessRequest,
    ProcessTermination,
    execute_process,
)
from jacobian.registry import (
    CheckerExecutableChangedError,
    CheckerRegistry,
    CheckerRegistryError,
)
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError, model_schema
from jacobian.storage.errors import ArtifactNotFoundError, StorageError
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification._helpers import (
    _CHECKER_CANCELLED,
    _CHECKER_CHANGED,
    _CHECKER_DIAGNOSTICS_TOO_LARGE,
    _CHECKER_INVALID_DECISION,
    _CHECKER_OUTPUT_TOO_LARGE,
    _CHECKER_TIMEOUT,
    _CHECKER_UNREADABLE_RESPONSE,
    _checker_failure_detail,
    _digest_bytes,
    _environment_digest,
    _verification_input_failure_detail,
    _verification_storage_failure_detail,
)
from jacobian.worker_environment import worker_environment

_LOGGER = logging.getLogger(__name__)


class CheckerExecutionError(RuntimeError):
    """An authorized checker failed operationally."""


class CheckerExecutionCancelledError(CheckerExecutionError):
    """An authorized checker was cancelled by its caller."""


class VerificationService:
    """The only application service allowed to persist verified records."""

    def __init__(
        self,
        store: ArtifactRepository,
        checker_registry: CheckerRegistry,
        *,
        checker_timeout_seconds: float = 30,
        max_checker_output_bytes: int = 1024 * 1024,
        max_checker_diagnostic_bytes: int = 1024 * 1024,
    ) -> None:
        self.store = store
        self.schemas = SchemaRegistry(store)
        self.checker_registry = checker_registry
        self.checker_timeout_seconds = checker_timeout_seconds
        self.max_checker_output_bytes = max_checker_output_bytes
        self.max_checker_diagnostic_bytes = max_checker_diagnostic_bytes
        self.record_schema_uri = store.register_descriptor(
            kind="schema",
            name="jacobian.verification-record",
            version="1",
            definition=model_schema(VerificationRecord),
        )
        self.record_semantics_uri = store.register_descriptor(
            kind="semantics",
            name="jacobian.verification-record",
            version="1",
            definition={
                "description": "authorized checker result bound to exact evidence"
            },
        )
        self.preservation_schema_uri = store.register_descriptor(
            kind="schema",
            name="jacobian.preservation-envelope",
            version="1",
            definition=model_schema(PreservationEnvelope),
        )
        self.transformation_record_schema_uri = store.register_descriptor(
            kind="schema",
            name="jacobian.transformation-verification-record",
            version="1",
            definition=model_schema(TransformationVerificationRecord),
        )

    def _semantics_digest(self, artifact: StoredArtifact) -> str:
        semantics = self.store.get(artifact.manifest.semantics_uri)
        return semantics.manifest.object_digest

    def _validate_artifact(self, artifact: StoredArtifact) -> None:
        self.schemas.validate(artifact.manifest.schema_uri, artifact.payload)

    def _run_checker(
        self,
        *,
        entrypoint: str,
        expected_digest: str,
        request: dict[str, Any],
        provider_runtime: CapabilityProviderRuntime | None = None,
        timeout_seconds: float | None = None,
    ) -> CheckerDecision:
        environment = worker_environment(
            extra_variables=(
                "ELAN_HOME",
                "JACOBIAN_CHECKER_EXECUTABLE",
                "JACOBIAN_CHECKER_RUNTIME_DIGEST",
                "JACOBIAN_CHECKER_LAKE_DIGEST",
                "JACOBIAN_LEAN_RUNTIME",
            )
        )
        effective_timeout = min(
            30 if timeout_seconds is None else timeout_seconds,
            self.checker_timeout_seconds,
        )
        arguments: list[str] = [
            "-m",
            "jacobian.checker_worker",
            entrypoint,
            expected_digest,
        ]
        if provider_runtime is not None:
            arguments.append(
                canonicalize_json(provider_runtime.model_dump(mode="json")).decode(
                    "utf-8"
                )
            )
        completed = execute_process(
            ProcessRequest(
                executable=sys.executable,
                arguments=tuple(arguments),
                stdin_bytes=canonicalize_json(request),
                timeout_seconds=effective_timeout,
                environment=environment,
                cwd=str(Path.cwd()),
                stdout_limit_bytes=self.max_checker_output_bytes,
                stderr_limit_bytes=self.max_checker_diagnostic_bytes,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, int(effective_timeout) + 1),
                    # Lean/Mathlib reserves a large virtual heap even for small
                    # proofs; keep it bounded without applying the much smaller
                    # arithmetic-worker profile.
                    address_space_bytes=16 * 1024 * 1024 * 1024,
                ),
            )
        )
        if completed.termination is ProcessTermination.TIMED_OUT:
            raise TimeoutError("checker execution timed out")
        if completed.termination is ProcessTermination.CANCELLED:
            raise CheckerExecutionCancelledError(_CHECKER_CANCELLED)
        if completed.stdout_exceeded:
            raise CheckerExecutionError(_CHECKER_OUTPUT_TOO_LARGE)
        if completed.stderr_exceeded:
            raise CheckerExecutionError(_CHECKER_DIAGNOSTICS_TOO_LARGE)
        return self._validate_checker_response(
            completed, expected_digest, provider_runtime
        )

    def _validate_checker_response(
        self,
        completed: Any,
        expected_digest: str,
        provider_runtime: CapabilityProviderRuntime | None,
    ) -> CheckerDecision:
        try:
            response = loads_strict_json(completed.stdout)
        except CanonicalizationError as exc:
            raise CheckerExecutionError(_CHECKER_UNREADABLE_RESPONSE) from exc
        if completed.returncode != 0:
            _LOGGER.warning(
                "checker worker stopped: response=%r diagnostics=%r",
                response,
                completed.stderr,
            )
            raise CheckerExecutionError(_checker_failure_detail(response))
        if not isinstance(response, dict):
            raise CheckerExecutionError(_CHECKER_UNREADABLE_RESPONSE)
        if response.get("measured_checker_digest") != expected_digest:
            _LOGGER.warning(
                "checker worker measured an unexpected implementation: %r",
                response,
            )
            raise CheckerExecutionError(_CHECKER_CHANGED)
        expected_runtime_digest = (
            provider_runtime.digest if provider_runtime is not None else None
        )
        if response.get("measured_runtime_digest") != expected_runtime_digest:
            _LOGGER.warning(
                "checker worker measured an unexpected external runtime: %r",
                response,
            )
            raise CheckerExecutionError(_CHECKER_CHANGED)
        try:
            return CheckerDecision.model_validate(response.get("decision"))
        except ValidationError as exc:
            _LOGGER.warning(
                "checker returned an invalid decision: %r",
                response,
                exc_info=exc,
            )
            raise CheckerExecutionError(_CHECKER_INVALID_DECISION) from exc

    def verify_witness(
        self,
        *,
        claim_uri: str,
        candidate_uri: str,
        witness_uri: str,
        checker_id: str,
        timeout_seconds: float | None = None,
        include_artifact_metadata: bool = False,
        include_semantics_artifact: bool = False,
    ) -> ResultEnvelope:
        """Replay a bound witness with the explicitly selected checker."""

        started = time.monotonic()
        try:
            claim = self.store.get(claim_uri)
            candidate = self.store.get(candidate_uri)
            witness_artifact = self.store.get(witness_uri)
            self._validate_artifacts((claim, candidate, witness_artifact))
            witness = WitnessEnvelope.model_validate(witness_artifact.payload)
            scope, expected_bindings, semantics_digest = self._resolve_witness_bindings(
                claim, candidate, witness_artifact, witness
            )
            self._ensure_shared_semantics(
                claim, candidate, witness_artifact, scope, label="witness"
            )
            checker = self.checker_registry.require_compatible(
                checker_id,
                evidence_kind=EvidenceKind.WITNESS,
                format_id=witness.witness_format,
                format_version=witness.format_version,
                claim_schema_uri=claim.manifest.schema_uri,
                semantics_uri=candidate.manifest.semantics_uri,
                candidate_schema_uri=candidate.manifest.schema_uri,
            )
            request = self._build_witness_request(
                claim,
                candidate,
                scope,
                witness_artifact,
                expected_bindings,
                include_artifact_metadata=include_artifact_metadata,
                include_semantics_artifact=include_semantics_artifact,
            )
            request_digest = _digest_bytes(canonicalize_json(request))
            decision = self._run_checker(
                entrypoint=checker.entrypoint,
                expected_digest=checker.executable_digest,
                request=request,
                provider_runtime=checker.provider_runtime,
                timeout_seconds=timeout_seconds,
            )
            runtime_ms = int((time.monotonic() - started) * 1000)
            if not decision.accepted:
                return self._rejected_decision_result(
                    decision,
                    runtime_ms,
                    claim_digest=claim.manifest.object_digest,
                    candidate_digest=candidate.manifest.object_digest,
                    evidence_uri=witness_uri,
                )
            request_artifact_uris = {claim_uri, candidate_uri, witness_uri}
            if scope is not None:
                request_artifact_uris.add(scope.artifact_uri)
            self._ensure_decision_endpoints_in_request(decision, request_artifact_uris)
            record = self._build_verification_record(
                decision,
                checker=checker,
                evidence_kind=EvidenceKind.WITNESS,
                evidence_uri=witness_uri,
                bindings=witness.bindings,
                request_digest=request_digest,
            )
            parent_uris = [claim_uri, candidate_uri, witness_uri]
            if scope is not None:
                parent_uris.append(scope.artifact_uri)
            return self._finalize_verification(
                decision=decision,
                checker=checker,
                runtime_ms=runtime_ms,
                claim_digest=claim.manifest.object_digest,
                semantics_digest=semantics_digest,
                candidate_digest=candidate.manifest.object_digest,
                evidence_uri=witness_uri,
                scope_uri=(scope.artifact_uri if scope is not None else None),
                record=record,
                schema_uri=self.record_schema_uri,
                parents=tuple(parent_uris),
                summary="authorized witness verification",
                execution_detail=decision.detail,
            )
        except (
            TimeoutError,
            CheckerExecutionCancelledError,
            CheckerExecutableChangedError,
            CheckerRegistryError,
            SchemaRegistryError,
            ValueError,
            ValidationError,
            ArtifactNotFoundError,
            StorageError,
            CheckerExecutionError,
        ) as exc:
            return self._verification_failure_result(exc, started)

    def verify_certificate(
        self,
        *,
        certificate_uri: str,
        checker_id: str | None = None,
        timeout_seconds: float | None = None,
        include_artifact_metadata: bool = False,
        supporting_artifact_uris: tuple[str, ...] = (),
    ) -> ResultEnvelope:
        """Run a specified compatible checker or uniquely select one."""

        started = time.monotonic()
        try:
            certificate_artifact = self.store.get(certificate_uri)
            self._validate_artifact(certificate_artifact)
            certificate = CertificateEnvelope.model_validate(
                certificate_artifact.payload
            )
            claim, candidate, scope = self._resolve_certificate_bindings(
                certificate_artifact, certificate
            )
            semantics_digest = self._semantics_digest(candidate)
            if certificate.bindings.semantics_digest != semantics_digest:
                raise ValueError(
                    "The certificate and candidate use different semantics. Recreate "
                    "the certificate from this candidate, then retry."
                )
            self._ensure_shared_semantics(
                claim, candidate, certificate_artifact, scope, label="certificate"
            )
            supporting_artifacts = self._load_supporting_artifacts(
                supporting_artifact_uris
            )
            checker = self._select_certificate_checker(
                checker_id, certificate, claim, candidate
            )
            request = self._build_certificate_request(
                claim,
                candidate,
                scope,
                certificate_artifact,
                certificate,
                supporting_artifacts,
                include_artifact_metadata=include_artifact_metadata,
            )
            request_digest = _digest_bytes(canonicalize_json(request))
            decision = self._run_checker(
                entrypoint=checker.entrypoint,
                expected_digest=checker.executable_digest,
                request=request,
                provider_runtime=checker.provider_runtime,
                timeout_seconds=timeout_seconds,
            )
            runtime_ms = int((time.monotonic() - started) * 1000)
            if not decision.accepted:
                return self._rejected_decision_result(
                    decision,
                    runtime_ms,
                    claim_digest=claim.manifest.object_digest,
                    candidate_digest=candidate.manifest.object_digest,
                    evidence_uri=certificate_uri,
                )
            request_artifact_uris = {
                claim.artifact_uri,
                candidate.artifact_uri,
                certificate_artifact.artifact_uri,
                *(artifact.artifact_uri for artifact in supporting_artifacts),
            }
            if scope is not None:
                request_artifact_uris.add(scope.artifact_uri)
            self._ensure_decision_endpoints_in_request(decision, request_artifact_uris)
            record = self._build_verification_record(
                decision,
                checker=checker,
                evidence_kind=EvidenceKind.CERTIFICATE,
                evidence_uri=certificate_uri,
                bindings=certificate.bindings,
                request_digest=request_digest,
            )
            parent_uris = [claim.artifact_uri, candidate.artifact_uri, certificate_uri]
            if scope is not None:
                parent_uris.append(scope.artifact_uri)
            parent_uris.extend(
                artifact.artifact_uri for artifact in supporting_artifacts
            )
            return self._finalize_verification(
                decision=decision,
                checker=checker,
                runtime_ms=runtime_ms,
                claim_digest=claim.manifest.object_digest,
                semantics_digest=semantics_digest,
                candidate_digest=candidate.manifest.object_digest,
                evidence_uri=certificate_uri,
                scope_uri=(scope.artifact_uri if scope is not None else None),
                record=record,
                schema_uri=self.record_schema_uri,
                parents=tuple(dict.fromkeys(parent_uris)),
                summary="authorized certificate verification",
                execution_detail=decision.detail,
            )
        except (
            TimeoutError,
            CheckerExecutionCancelledError,
            CheckerExecutableChangedError,
            CheckerRegistryError,
            SchemaRegistryError,
            ValueError,
            ValidationError,
            ArtifactNotFoundError,
            StorageError,
            CheckerExecutionError,
        ) as exc:
            return self._verification_failure_result(exc, started)

    def verify_transformation(
        self,
        *,
        transformation_uri: str,
        timeout_seconds: float | None = None,
    ) -> ResultEnvelope:
        """Replay a representation relation with an authorized checker."""

        started = time.monotonic()
        try:
            transformation_artifact = self.store.get(transformation_uri)
            self._validate_artifact(transformation_artifact)
            transformation = TransformationEnvelope.model_validate(
                transformation_artifact.payload
            )
            claim, source, target = self._load_transformation_inputs(transformation)
            self._validate_artifacts((claim, source, target))
            parsed_claim = TransformationClaim.model_validate(claim.payload)
            if (
                parsed_claim.transform_format,
                parsed_claim.format_version,
                parsed_claim.relation,
                parsed_claim.bindings,
            ) != (
                transformation.transform_format,
                transformation.format_version,
                transformation.relation,
                transformation.bindings,
            ):
                raise ValueError(
                    "The transformation record does not match its claim. Recreate the "
                    "transformation from the supplied source and target, then retry."
                )
            expected_bindings = self._transformation_expected_bindings(source, target)
            if transformation.bindings.model_dump(mode="json") != expected_bindings:
                raise ValueError(
                    "The transformation does not match the supplied source and target. "
                    "Recreate it from those exact artifacts, then retry."
                )
            required_parents = {
                claim.artifact_uri,
                source.artifact_uri,
                target.artifact_uri,
            }
            if not required_parents.issubset(transformation_artifact.manifest.parents):
                raise ValueError(
                    "The transformation is missing required source or target lineage. "
                    "Recreate it from the supplied artifacts, then retry."
                )
            checker = self.checker_registry.select_compatible(
                evidence_kind=EvidenceKind.TRANSFORMATION,
                format_id=transformation.transform_format,
                format_version=transformation.format_version,
                claim_schema_uri=claim.manifest.schema_uri,
                semantics_uri=source.manifest.semantics_uri,
                candidate_schema_uri=source.manifest.schema_uri,
                target_schema_uri=target.manifest.schema_uri,
                target_semantics_uri=target.manifest.semantics_uri,
            )
            request = self._build_transformation_request(
                claim,
                source,
                target,
                transformation_artifact,
                transformation,
                expected_bindings,
            )
            request_digest = _digest_bytes(canonicalize_json(request))
            decision = self._run_checker(
                entrypoint=checker.entrypoint,
                expected_digest=checker.executable_digest,
                request=request,
                provider_runtime=checker.provider_runtime,
                timeout_seconds=timeout_seconds,
            )
            runtime_ms = int((time.monotonic() - started) * 1000)
            if not decision.accepted:
                return self._rejected_decision_result(
                    decision,
                    runtime_ms,
                    claim_digest=claim.manifest.object_digest,
                    candidate_digest=source.manifest.object_digest,
                    evidence_uri=transformation_uri,
                )
            record = TransformationVerificationRecord(
                checker_id=checker.checker_id,
                checker_digest=checker.executable_digest,
                transformation_uri=transformation_uri,
                claim_uri=claim.artifact_uri,
                bindings=transformation.bindings,
                relation=transformation.relation,
                conclusion=decision.conclusion,
                arithmetic=decision.arithmetic,
                method=decision.method,
                coverage=decision.coverage,
                request_digest=request_digest,
                environment_digest=_environment_digest(
                    checker.executable_digest,
                    checker.provider_runtime,
                ),
            )
            return self._finalize_verification(
                decision=decision,
                checker=checker,
                runtime_ms=runtime_ms,
                claim_digest=claim.manifest.object_digest,
                semantics_digest=self._semantics_digest(claim),
                candidate_digest=source.manifest.object_digest,
                evidence_uri=transformation_uri,
                scope_uri=transformation_uri,
                record=record,
                schema_uri=self.transformation_record_schema_uri,
                parents=(
                    claim.artifact_uri,
                    source.artifact_uri,
                    target.artifact_uri,
                    transformation_uri,
                ),
                summary="authorized transformation verification",
            )
        except (
            TimeoutError,
            CheckerExecutionCancelledError,
            CheckerExecutableChangedError,
            CheckerRegistryError,
            SchemaRegistryError,
            ValueError,
            ValidationError,
            ArtifactNotFoundError,
            StorageError,
            CheckerExecutionError,
        ) as exc:
            return self._verification_failure_result(exc, started)

    def verify_preservation(
        self,
        *,
        claim_uri: str,
        original_uri: str,
        reduced_uri: str,
        checker_id: str,
        preservation_format: str,
        reducer: str,
    ) -> ResultEnvelope:
        """Verify that a proposed reduction preserves the checked predicate."""

        started = time.monotonic()
        try:
            claim = self.store.get(claim_uri)
            original = self.store.get(original_uri)
            reduced = self.store.get(reduced_uri)
            self._validate_artifacts((claim, original, reduced))
            self._ensure_preservation_semantics(claim, original, reduced)
            if original.manifest.schema_uri != reduced.manifest.schema_uri:
                raise ValueError(
                    "The original and reduced objects use different schemas. "
                    "Run a reducer that preserves the object schema, then retry."
                )
            semantics_digest = self._semantics_digest(reduced)
            bindings = EvidenceBindings(
                claim_digest=claim.manifest.object_digest,
                semantics_digest=semantics_digest,
                candidate_digest=reduced.manifest.object_digest,
                encoding_digest=original.manifest.object_digest,
            )
            evidence = PreservationEnvelope(
                preservation_format=preservation_format,
                format_version="1",
                bindings=bindings,
                reducer=reducer,
            )
            parent_uris = tuple(sorted({claim_uri, original_uri, reduced_uri}))
            evidence_result = self.store.put(
                schema_uri=self.preservation_schema_uri,
                semantics_uri=reduced.manifest.semantics_uri,
                payload=evidence.model_dump(mode="json"),
                parents=parent_uris,
                summary="proposed reduction preservation evidence",
            )
            evidence_artifact = self.store.get(evidence_result.artifact_uri)
            checker = self.checker_registry.require_compatible(
                checker_id,
                evidence_kind=EvidenceKind.PRESERVATION,
                format_id=preservation_format,
                format_version="1",
                claim_schema_uri=claim.manifest.schema_uri,
                semantics_uri=reduced.manifest.semantics_uri,
                candidate_schema_uri=reduced.manifest.schema_uri,
            )
            request = self._build_preservation_request(
                claim, original, reduced, evidence_artifact, evidence, bindings
            )
            request_digest = _digest_bytes(canonicalize_json(request))
            decision = self._run_checker(
                entrypoint=checker.entrypoint,
                expected_digest=checker.executable_digest,
                request=request,
                provider_runtime=checker.provider_runtime,
            )
            runtime_ms = int((time.monotonic() - started) * 1000)
            if not decision.accepted:
                return self._rejected_decision_result(
                    decision,
                    runtime_ms,
                    claim_digest=claim.manifest.object_digest,
                    semantics_digest=semantics_digest,
                    candidate_digest=reduced.manifest.object_digest,
                    evidence_uri=evidence_artifact.artifact_uri,
                )
            record = self._build_verification_record(
                decision,
                checker=checker,
                evidence_kind=EvidenceKind.PRESERVATION,
                evidence_uri=evidence_artifact.artifact_uri,
                bindings=bindings,
                request_digest=request_digest,
            )
            return self._finalize_verification(
                decision=decision,
                checker=checker,
                runtime_ms=runtime_ms,
                claim_digest=claim.manifest.object_digest,
                semantics_digest=semantics_digest,
                candidate_digest=reduced.manifest.object_digest,
                evidence_uri=evidence_artifact.artifact_uri,
                scope_uri=None,
                record=record,
                schema_uri=self.record_schema_uri,
                parents=(*parent_uris, evidence_artifact.artifact_uri),
                summary="authorized reduction preservation verification",
            )
        except (
            TimeoutError,
            CheckerExecutionCancelledError,
            CheckerExecutableChangedError,
            CheckerRegistryError,
            SchemaRegistryError,
            ValueError,
            ValidationError,
            ArtifactNotFoundError,
            StorageError,
            CheckerExecutionError,
        ) as exc:
            return self._verification_failure_result(exc, started)

    def _validate_artifacts(self, artifacts: tuple[StoredArtifact, ...]) -> None:
        for artifact in artifacts:
            self._validate_artifact(artifact)

    def _resolve_witness_bindings(
        self,
        claim: StoredArtifact,
        candidate: StoredArtifact,
        witness_artifact: StoredArtifact,
        witness: WitnessEnvelope,
    ) -> tuple[StoredArtifact | None, dict[str, Any], str]:
        if witness.bindings.encoding_digest is not None:
            raise ValueError(
                "This witness includes an unsupported encoding binding. "
                "Recreate it without encoding_digest, then retry."
            )
        scope = None
        if witness.bindings.scope_digest is not None:
            scope = self._resolve_bound_parent(
                witness_artifact,
                witness.bindings.scope_digest,
                label="scope",
            )
            self._validate_artifact(scope)
        semantics_digest = self._semantics_digest(candidate)
        expected_bindings = {
            "claim_digest": claim.manifest.object_digest,
            "semantics_digest": semantics_digest,
            "candidate_digest": candidate.manifest.object_digest,
            "scope_digest": (
                scope.manifest.object_digest if scope is not None else None
            ),
            "encoding_digest": None,
        }
        if witness.bindings.model_dump(mode="json") != expected_bindings:
            raise ValueError(
                "The witness does not match the supplied claim and candidate. "
                "Recreate the witness from those exact artifacts, then retry."
            )
        required_parents = {claim.artifact_uri, candidate.artifact_uri}
        if scope is not None:
            required_parents.add(scope.artifact_uri)
        if not required_parents.issubset(witness_artifact.manifest.parents):
            raise ValueError(
                "The witness is missing required claim or candidate lineage. "
                "Recreate it from the supplied artifacts, then retry."
            )
        return scope, expected_bindings, semantics_digest

    def _ensure_shared_semantics(
        self,
        claim: StoredArtifact,
        candidate: StoredArtifact,
        evidence_artifact: StoredArtifact,
        scope: StoredArtifact | None,
        *,
        label: str,
    ) -> None:
        if (
            claim.manifest.semantics_uri != candidate.manifest.semantics_uri
            or evidence_artifact.manifest.semantics_uri
            != candidate.manifest.semantics_uri
            or (
                scope is not None
                and scope.manifest.semantics_uri != candidate.manifest.semantics_uri
            )
        ):
            raise ValueError(
                f"The claim, candidate, {label}, and scope use different "
                "semantics. Use artifacts from one reference contract, then retry."
            )

    def _ensure_decision_endpoints_in_request(
        self,
        decision: CheckerDecision,
        request_artifact_uris: set[str],
    ) -> None:
        decision_endpoints = {
            *decision.relationship_source_artifact_uris,
            *decision.relationship_target_artifact_uris,
        }
        if not decision_endpoints.issubset(request_artifact_uris):
            raise ValueError(
                "The checker certified a relationship endpoint outside its "
                "verification request."
            )
        if (
            decision.obligation_uri is not None
            and decision.obligation_uri not in request_artifact_uris
        ):
            raise ValueError(
                "The checker certified an obligation outside its verification request."
            )

    def _build_witness_request(
        self,
        claim: StoredArtifact,
        candidate: StoredArtifact,
        scope: StoredArtifact | None,
        witness_artifact: StoredArtifact,
        expected_bindings: dict[str, Any],
        *,
        include_artifact_metadata: bool,
        include_semantics_artifact: bool,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "request_version": "1",
            "claim": self._checker_artifact(
                claim,
                include_storage_metadata=include_artifact_metadata,
            ),
            "candidate": self._checker_artifact(
                candidate,
                include_storage_metadata=include_artifact_metadata,
            ),
            "scope": (
                self._checker_artifact(
                    scope,
                    include_storage_metadata=include_artifact_metadata,
                )
                if scope is not None
                else None
            ),
            "witness": self._checker_artifact(
                witness_artifact,
                include_storage_metadata=include_artifact_metadata,
            ),
            "expected_bindings": expected_bindings,
        }
        if include_semantics_artifact:
            semantics_artifact = self.store.get(candidate.manifest.semantics_uri)
            request["semantics"] = self._checker_artifact(
                semantics_artifact,
                include_storage_metadata=include_artifact_metadata,
            )
        return request

    def _resolve_certificate_bindings(
        self,
        certificate_artifact: StoredArtifact,
        certificate: CertificateEnvelope,
    ) -> tuple[StoredArtifact, StoredArtifact, StoredArtifact | None]:
        if certificate.bindings.encoding_digest is not None:
            raise ValueError(
                "This certificate includes an unsupported encoding binding. "
                "Recreate it without encoding_digest, then retry."
            )
        claim = self._resolve_bound_parent(
            certificate_artifact,
            certificate.bindings.claim_digest,
            label="claim",
        )
        if certificate.bindings.candidate_digest is None:
            raise ValueError(
                "The certificate does not identify a candidate. Recreate it from "
                "the exact claim and candidate, then retry."
            )
        candidate = self._resolve_bound_parent(
            certificate_artifact,
            certificate.bindings.candidate_digest,
            label="candidate",
        )
        self._validate_artifacts((claim, candidate))
        scope = None
        if certificate.bindings.scope_digest is not None:
            scope = self._resolve_bound_parent(
                certificate_artifact,
                certificate.bindings.scope_digest,
                label="scope",
            )
            self._validate_artifact(scope)
        return claim, candidate, scope

    def _load_supporting_artifacts(
        self, supporting_artifact_uris: tuple[str, ...]
    ) -> tuple[StoredArtifact, ...]:
        artifacts = tuple(
            self.store.get(uri) for uri in dict.fromkeys(supporting_artifact_uris)
        )
        self._validate_artifacts(artifacts)
        return artifacts

    def _select_certificate_checker(
        self,
        checker_id: str | None,
        certificate: CertificateEnvelope,
        claim: StoredArtifact,
        candidate: StoredArtifact,
    ) -> Any:
        compatibility: dict[str, Any] = {
            "evidence_kind": EvidenceKind.CERTIFICATE,
            "format_id": certificate.certificate_type,
            "format_version": certificate.format_version,
            "claim_schema_uri": claim.manifest.schema_uri,
            "semantics_uri": candidate.manifest.semantics_uri,
            "candidate_schema_uri": candidate.manifest.schema_uri,
        }
        if checker_id is None:
            return self.checker_registry.select_compatible(**compatibility)
        return self.checker_registry.require_compatible(checker_id, **compatibility)

    def _build_certificate_request(
        self,
        claim: StoredArtifact,
        candidate: StoredArtifact,
        scope: StoredArtifact | None,
        certificate_artifact: StoredArtifact,
        certificate: CertificateEnvelope,
        supporting_artifacts: tuple[StoredArtifact, ...],
        *,
        include_artifact_metadata: bool,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "request_version": "1",
            "claim": self._checker_artifact(
                claim,
                include_storage_metadata=include_artifact_metadata,
            ),
            "candidate": self._checker_artifact(
                candidate,
                include_storage_metadata=include_artifact_metadata,
            ),
            "scope": (
                self._checker_artifact(
                    scope,
                    include_storage_metadata=include_artifact_metadata,
                )
                if scope is not None
                else None
            ),
            "certificate": {
                **self._checker_artifact(
                    certificate_artifact,
                    include_storage_metadata=include_artifact_metadata,
                ),
                "payload": certificate.model_dump(mode="json"),
            },
            "expected_bindings": certificate.bindings.model_dump(mode="json"),
        }
        if supporting_artifacts:
            request["supporting_artifacts"] = [
                self._checker_artifact(
                    artifact,
                    include_storage_metadata=include_artifact_metadata,
                )
                for artifact in supporting_artifacts
            ]
        return request

    def _load_transformation_inputs(
        self, transformation: TransformationEnvelope
    ) -> tuple[StoredArtifact, StoredArtifact, StoredArtifact]:
        return (
            self.store.get(transformation.claim_uri),
            self.store.get(transformation.source_uri),
            self.store.get(transformation.target_uri),
        )

    def _transformation_expected_bindings(
        self,
        source: StoredArtifact,
        target: StoredArtifact,
    ) -> dict[str, Any]:
        return {
            "source_digest": source.manifest.object_digest,
            "source_schema_uri": source.manifest.schema_uri,
            "source_semantics_digest": self._semantics_digest(source),
            "target_digest": target.manifest.object_digest,
            "target_schema_uri": target.manifest.schema_uri,
            "target_semantics_digest": self._semantics_digest(target),
        }

    def _build_transformation_request(
        self,
        claim: StoredArtifact,
        source: StoredArtifact,
        target: StoredArtifact,
        transformation_artifact: StoredArtifact,
        transformation: TransformationEnvelope,
        expected_bindings: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "request_version": "1",
            "claim": self._checker_artifact(claim),
            "source": self._checker_artifact(source),
            "target": self._checker_artifact(target),
            "transformation": {
                **self._checker_artifact(transformation_artifact),
                "payload": transformation.model_dump(mode="json"),
            },
            "expected_bindings": expected_bindings,
        }

    def _ensure_preservation_semantics(
        self,
        claim: StoredArtifact,
        original: StoredArtifact,
        reduced: StoredArtifact,
    ) -> None:
        if (
            claim.manifest.semantics_uri != original.manifest.semantics_uri
            or original.manifest.semantics_uri != reduced.manifest.semantics_uri
        ):
            raise ValueError(
                "The claim, original object, and reduced object use different "
                "semantics. Use artifacts from one reference contract, then retry."
            )

    def _build_preservation_request(
        self,
        claim: StoredArtifact,
        original: StoredArtifact,
        reduced: StoredArtifact,
        evidence_artifact: StoredArtifact,
        evidence: PreservationEnvelope,
        bindings: EvidenceBindings,
    ) -> dict[str, Any]:
        return {
            "request_version": "1",
            "claim": self._checker_artifact(claim),
            "original": self._checker_artifact(original),
            "reduced": self._checker_artifact(reduced),
            "preservation": {
                **self._checker_artifact(evidence_artifact),
                "payload": evidence.model_dump(mode="json"),
            },
            "expected_bindings": bindings.model_dump(mode="json"),
        }

    def _rejected_decision_result(
        self,
        decision: CheckerDecision,
        runtime_ms: int,
        *,
        claim_digest: str,
        candidate_digest: str,
        evidence_uri: str,
        semantics_digest: str | None = None,
    ) -> ResultEnvelope:
        return ResultEnvelope(
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=runtime_ms,
            ),
            input=InputValidation(
                status=InputStatus.REJECTED,
                errors=(decision.detail,),
            ),
            conclusion=Conclusion.UNKNOWN,
            assurance=Assurance(
                arithmetic=decision.arithmetic,
                method=decision.method,
                coverage=decision.coverage,
                verification=Verification.UNVERIFIED,
            ),
            claim_digest=claim_digest,
            semantics_digest=semantics_digest,
            candidate_digest=candidate_digest,
            evidence_uris=(evidence_uri,),
        )

    def _build_verification_record(
        self,
        decision: CheckerDecision,
        *,
        checker: Any,
        evidence_kind: EvidenceKind,
        evidence_uri: str,
        bindings: EvidenceBindings,
        request_digest: str,
    ) -> VerificationRecord:
        return VerificationRecord(
            checker_id=checker.checker_id,
            checker_digest=checker.executable_digest,
            evidence_kind=evidence_kind,
            evidence_uri=evidence_uri,
            bindings=bindings,
            conclusion=decision.conclusion,
            arithmetic=decision.arithmetic,
            method=decision.method,
            coverage=decision.coverage,
            request_digest=request_digest,
            environment_digest=_environment_digest(
                checker.executable_digest,
                checker.provider_runtime,
            ),
            relation_id=decision.relation_id,
            relationship_source_artifact_uris=(
                decision.relationship_source_artifact_uris
            ),
            relationship_target_artifact_uris=(
                decision.relationship_target_artifact_uris
            ),
            obligation_uri=decision.obligation_uri,
        )

    def _finalize_verification(
        self,
        *,
        decision: CheckerDecision,
        checker: Any,
        runtime_ms: int,
        claim_digest: str,
        candidate_digest: str,
        evidence_uri: str,
        record: VerificationRecord | TransformationVerificationRecord,
        schema_uri: str,
        parents: tuple[str, ...],
        summary: str,
        semantics_digest: str | None = None,
        scope_uri: str | None = None,
        execution_detail: str | None = None,
    ) -> ResultEnvelope:
        assurance = Assurance(
            arithmetic=decision.arithmetic,
            method=decision.method,
            coverage=decision.coverage,
            verification=Verification.VERIFIED,
            checker_id=checker.checker_id,
            checker_digest=checker.executable_digest,
            scope_uri=scope_uri,
        )
        record_artifact = self._commit_verification_record(
            checker_id=checker.checker_id,
            checker_digest=checker.executable_digest,
            schema_uri=schema_uri,
            semantics_uri=self.record_semantics_uri,
            payload=record.model_dump(mode="json"),
            parents=parents,
            summary=summary,
        )
        return ResultEnvelope(
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=runtime_ms,
                detail=execution_detail,
            ),
            input=InputValidation(status=InputStatus.ACCEPTED),
            conclusion=decision.conclusion,
            assurance=assurance,
            claim_digest=claim_digest,
            semantics_digest=semantics_digest,
            candidate_digest=candidate_digest,
            evidence_uris=(evidence_uri,),
            verification_record_uri=record_artifact.artifact_uri,
        )

    def _verification_failure_result(
        self,
        exc: BaseException,
        started: float,
    ) -> ResultEnvelope:
        if isinstance(exc, TimeoutError):
            return self._operational_failure(
                status=ExecutionStatus.TIMEOUT,
                detail=_CHECKER_TIMEOUT,
                started=started,
            )
        if isinstance(exc, CheckerExecutionCancelledError):
            return self._operational_failure(
                status=ExecutionStatus.CANCELLED,
                detail=str(exc),
                started=started,
            )
        if isinstance(exc, CheckerExecutableChangedError):
            return self._operational_failure(
                status=ExecutionStatus.ERROR,
                detail=str(exc),
                started=started,
            )
        if isinstance(exc, ArtifactNotFoundError):
            return self._rejected_input(
                "A required verification artifact is unavailable or invalid. "
                "Check the artifact URIs and retry.",
                started=started,
            )
        if isinstance(
            exc,
            (CheckerRegistryError, SchemaRegistryError, ValueError, ValidationError),
        ):
            return self._rejected_input(
                _verification_input_failure_detail(exc),
                started=started,
            )
        if isinstance(exc, StorageError):
            _LOGGER.warning("verification storage operation failed", exc_info=exc)
            return self._operational_failure(
                status=ExecutionStatus.ERROR,
                detail=_verification_storage_failure_detail(exc),
                started=started,
            )
        return self._operational_failure(
            status=ExecutionStatus.ERROR,
            detail=str(exc),
            started=started,
        )

    def _resolve_bound_parent(
        self,
        evidence_artifact: StoredArtifact,
        object_digest: str,
        *,
        label: str,
    ) -> StoredArtifact:
        parent_set = set(evidence_artifact.manifest.parents)
        matches = [
            uri
            for uri in self.store.find_by_object_digest(object_digest)
            if uri in parent_set
        ]
        if not matches:
            raise ValueError(
                f"The certificate is missing its bound {label} artifact. Recreate "
                "the certificate from the exact verification inputs, then retry."
            )
        return self.store.get(sorted(matches)[0])

    @staticmethod
    def _checker_artifact(
        artifact: StoredArtifact | None,
        *,
        include_storage_metadata: bool = False,
    ) -> dict[str, Any]:
        if artifact is None:
            raise ValueError(
                "Verification evidence is incomplete. Recreate it from the exact "
                "claim and candidate, then retry."
            )
        result = {
            "artifact_uri": artifact.artifact_uri,
            "object_digest": artifact.manifest.object_digest,
            "schema_uri": artifact.manifest.schema_uri,
            "semantics_uri": artifact.manifest.semantics_uri,
            "payload": artifact.payload,
        }
        if include_storage_metadata:
            result["payload_digest"] = artifact.manifest.payload_digest
            result["parents"] = list(artifact.manifest.parents)
        return result

    def _commit_verification_record(
        self,
        *,
        checker_id: str,
        checker_digest: str,
        schema_uri: str,
        semantics_uri: str,
        payload: dict[str, Any],
        parents: tuple[str, ...],
        summary: str,
    ) -> ArtifactPutResult:
        with self.checker_registry.verification_guard(
            checker_id,
            expected_digest=checker_digest,
        ):
            return self.store.put(
                schema_uri=schema_uri,
                semantics_uri=semantics_uri,
                payload=payload,
                parents=parents,
                summary=summary,
            )

    def _rejected_input(self, detail: str, *, started: float) -> ResultEnvelope:
        return ResultEnvelope(
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=int((time.monotonic() - started) * 1000),
            ),
            input=InputValidation(
                status=InputStatus.REJECTED,
                errors=(detail,),
            ),
            conclusion=Conclusion.UNKNOWN,
            assurance=Assurance(
                arithmetic=Arithmetic.SYMBOLIC,
                method=Method.DIRECT_WITNESS,
                coverage=Coverage.NOT_APPLICABLE,
                verification=Verification.UNVERIFIED,
            ),
        )

    def _operational_failure(
        self,
        *,
        status: ExecutionStatus,
        detail: str,
        started: float,
    ) -> ResultEnvelope:
        return ResultEnvelope(
            execution=Execution(
                status=status,
                runtime_ms=int((time.monotonic() - started) * 1000),
                detail=detail,
            ),
            input=InputValidation(status=InputStatus.ACCEPTED),
            conclusion=Conclusion.UNKNOWN,
            assurance=Assurance(
                arithmetic=Arithmetic.SYMBOLIC,
                method=Method.DIRECT_WITNESS,
                coverage=Coverage.NOT_APPLICABLE,
                verification=Verification.UNVERIFIED,
            ),
        )
