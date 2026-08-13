"""Authorized witness and certificate replay services."""

from __future__ import annotations

import logging
import time

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
from jacobian.contracts.checkers import (
    CheckerDecision,
    CheckerRegistration,
    EvidenceKind,
)
from jacobian.contracts.evidence import (
    EvidenceBindings,
)
from jacobian.contracts.exact_domain_verification import (
    InlineExactVerificationRecord,
    inline_exact_value_digest,
)
from jacobian.contracts.results import (
    Conclusion,
    Execution,
    ExecutionStatus,
    InputStatus,
    InputValidation,
    VerificationResult,
)
from jacobian.contracts.verification import VerificationRecord
from jacobian.registry import (
    CheckerExecutableChangedError,
    CheckerRegistry,
    CheckerRegistryError,
)
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError, model_schema
from jacobian.storage.errors import ArtifactNotFoundError, StorageError
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification._helpers import (
    _CHECKER_TIMEOUT,
    _digest_bytes,
    _environment_digest,
    _verification_input_failure_detail,
    _verification_storage_failure_detail,
)
from jacobian.verification.decision_validation import VerificationDecisionValidator
from jacobian.verification.errors import (
    CheckerExecutionCancelledError,
    CheckerExecutionError,
)
from jacobian.verification.executor import BoundedCheckerExecutor
from jacobian.verification.plan_builder import (
    VerificationPlanBuilder,
    _checker_artifact,
    _VerificationPlan,
)
from jacobian.verification.record_committer import VerificationRecordCommitter

_LOGGER = logging.getLogger(__name__)


_RECOVERABLE_VERIFICATION_ERRORS: tuple[type[Exception], ...] = (
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
)


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
        checker_executor: BoundedCheckerExecutor | None = None,
    ) -> None:
        self.store = store
        self.schemas = SchemaRegistry(store)
        self.checker_registry = checker_registry
        self._checker_executor = checker_executor or BoundedCheckerExecutor(
            checker_timeout_seconds=checker_timeout_seconds,
            max_checker_output_bytes=max_checker_output_bytes,
            max_checker_diagnostic_bytes=max_checker_diagnostic_bytes,
        )
        self._plan_builder = VerificationPlanBuilder(
            store,
            self.schemas,
            checker_registry,
        )
        self._decision_validator = VerificationDecisionValidator()
        self._record_committer = VerificationRecordCommitter(store, checker_registry)
        self.record_schema_uri = store.register_descriptor(
            kind="schema",
            name="jacobian.verification-record",
            version="2",
            definition=model_schema(VerificationRecord),
        )
        self.record_semantics_uri = store.register_descriptor(
            kind="semantics",
            name="jacobian.verification-record",
            version="2",
            definition={
                "description": (
                    "authorized checker result bound to exact evidence and its "
                    "measured execution manifest"
                )
            },
        )
        self.inline_exact_record_schema_uri = store.register_descriptor(
            kind="schema",
            name="jacobian.inline-exact-verification-record",
            version="2",
            definition=model_schema(InlineExactVerificationRecord),
        )
        self.inline_exact_record_semantics_uri = store.register_descriptor(
            kind="semantics",
            name="jacobian.inline-exact-verification-record",
            version="2",
            definition={
                "description": (
                    "authorized checker decision bound to canonical inline exact "
                    "input and candidate values plus its measured execution manifest"
                )
            },
        )

    def verify_inline_exact(
        self,
        *,
        operation_id: str,
        claim_schema_uri: str,
        candidate_schema_uri: str,
        semantics_uri: str,
        claim_payload: dict[str, object],
        candidate_payload: dict[str, object],
        checker_id: str,
        witness_format: str,
        timeout_seconds: float | None = None,
    ) -> VerificationResult:
        """Verify exact inline values without materializing them as artifacts.

        Only an accepted immutable record is persisted.  The record binds the
        value digests, schemas, semantics, checker, measured runtime, request,
        and the checker decision; the ordinary input and candidate never enter
        the artifact store.
        """

        started = time.monotonic()
        try:
            self.schemas.validate(claim_schema_uri, claim_payload)
            self.schemas.validate(candidate_schema_uri, candidate_payload)
            semantics = self.store.get(semantics_uri)
            semantics_digest = semantics.manifest.object_digest
            claim_digest = inline_exact_value_digest(
                schema_uri=claim_schema_uri,
                semantics_uri=semantics_uri,
                payload=claim_payload,
            )
            candidate_digest = inline_exact_value_digest(
                schema_uri=candidate_schema_uri,
                semantics_uri=semantics_uri,
                payload=candidate_payload,
            )
            bindings = EvidenceBindings(
                claim_digest=claim_digest,
                semantics_digest=semantics_digest,
                candidate_digest=candidate_digest,
            )
            request = {
                "request_version": "2",
                "operation_id": operation_id,
                "claim": {
                    "schema_uri": claim_schema_uri,
                    "semantics_uri": semantics_uri,
                    "payload": claim_payload,
                },
                "candidate": {
                    "schema_uri": candidate_schema_uri,
                    "semantics_uri": semantics_uri,
                    "payload": candidate_payload,
                },
                "semantics": _checker_artifact(
                    semantics,
                    include_storage_metadata=True,
                ),
                "scope": None,
                "expected_bindings": bindings.model_dump(mode="json"),
            }
            checker = self.checker_registry.require_compatible(
                checker_id,
                evidence_kind=EvidenceKind.WITNESS,
                format_id=witness_format,
                format_version="1",
                claim_schema_uri=claim_schema_uri,
                semantics_uri=semantics_uri,
                candidate_schema_uri=candidate_schema_uri,
            )
            request_digest = _digest_bytes(canonicalize_json(request))
            decision = self._checker_executor.execute(
                manifest=checker.implementation,
                expected_implementation_digest=checker.implementation_digest,
                request=request,
                timeout_seconds=timeout_seconds,
            )
            runtime_ms = int((time.monotonic() - started) * 1000)
            rejection_detail = self._decision_validator.rejection_detail(
                decision,
                permits_bounded_coverage=False,
            )
            if rejection_detail is not None:
                return VerificationResult(
                    execution=Execution(
                        status=ExecutionStatus.COMPLETED, runtime_ms=runtime_ms
                    ),
                    input=InputValidation(
                        status=InputStatus.REJECTED, errors=(rejection_detail,)
                    ),
                    conclusion=Conclusion.UNKNOWN,
                    claim_digest=claim_digest,
                    semantics_digest=semantics_digest,
                    candidate_digest=candidate_digest,
                )
            self._decision_validator.require_request_bound_endpoints(decision, ())
            record = InlineExactVerificationRecord(
                witness_format=witness_format,
                operation_id=operation_id,
                checker_id=checker.checker_id,
                implementation_digest=checker.implementation_digest,
                checker_manifest=checker.implementation,
                runtime_digest=(
                    checker.implementation.provider_runtime.digest
                    if checker.implementation.provider_runtime is not None
                    else None
                ),
                environment_digest=_environment_digest(
                    checker.implementation_digest,
                    checker.implementation.provider_runtime,
                ),
                input_schema_uri=claim_schema_uri,
                candidate_schema_uri=candidate_schema_uri,
                semantics_uri=semantics_uri,
                bindings=bindings,
                decision=decision,
                request_digest=request_digest,
            )
            record_artifact = self._record_committer.commit(
                checker_id=checker.checker_id,
                implementation_digest=checker.implementation_digest,
                schema_uri=self.inline_exact_record_schema_uri,
                semantics_uri=self.inline_exact_record_semantics_uri,
                payload=record.model_dump(mode="json"),
                parents=(semantics_uri,),
                summary="authorized inline exact verification",
            )
            return VerificationResult(
                execution=Execution(
                    status=ExecutionStatus.COMPLETED,
                    runtime_ms=runtime_ms,
                    detail=decision.detail,
                ),
                input=InputValidation(status=InputStatus.ACCEPTED),
                conclusion=decision.conclusion,
                claim_digest=claim_digest,
                semantics_digest=semantics_digest,
                candidate_digest=candidate_digest,
                evidence_uris=(record_artifact.artifact_uri,),
                verification_record_uri=record_artifact.artifact_uri,
            )
        except _RECOVERABLE_VERIFICATION_ERRORS as exc:
            return self._verification_failure_result(exc, started)

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
    ) -> VerificationResult:
        """Replay a bound witness with the explicitly selected checker."""

        started = time.monotonic()
        try:
            plan = self._plan_builder.build_witness(
                claim_uri=claim_uri,
                candidate_uri=candidate_uri,
                witness_uri=witness_uri,
                checker_id=checker_id,
                include_artifact_metadata=include_artifact_metadata,
                include_semantics_artifact=include_semantics_artifact,
            )
            return self._execute_verification_plan(
                plan,
                started=started,
                timeout_seconds=timeout_seconds,
            )
        except _RECOVERABLE_VERIFICATION_ERRORS as exc:
            return self._verification_failure_result(exc, started)

    def verify_certificate(
        self,
        *,
        certificate_uri: str,
        checker_id: str | None = None,
        timeout_seconds: float | None = None,
        include_artifact_metadata: bool = False,
        supporting_artifact_uris: tuple[str, ...] = (),
    ) -> VerificationResult:
        """Run a specified compatible checker or uniquely select one."""

        started = time.monotonic()
        try:
            plan = self._plan_builder.build_certificate(
                certificate_uri=certificate_uri,
                checker_id=checker_id,
                include_artifact_metadata=include_artifact_metadata,
                supporting_artifact_uris=supporting_artifact_uris,
            )
            return self._execute_verification_plan(
                plan,
                started=started,
                timeout_seconds=timeout_seconds,
            )
        except _RECOVERABLE_VERIFICATION_ERRORS as exc:
            return self._verification_failure_result(exc, started)

    def _execute_verification_plan(
        self,
        plan: _VerificationPlan,
        *,
        started: float,
        timeout_seconds: float | None,
    ) -> VerificationResult:
        """Run one resolved plan and commit only a valid accepted decision."""

        request_digest = _digest_bytes(canonicalize_json(plan.request))
        decision = self._checker_executor.execute(
            manifest=plan.checker.implementation,
            expected_implementation_digest=plan.checker.implementation_digest,
            request=plan.request,
            timeout_seconds=timeout_seconds,
        )
        runtime_ms = int((time.monotonic() - started) * 1000)
        rejection_detail = self._decision_validator.rejection_detail(
            decision,
            permits_bounded_coverage=True,
        )
        if rejection_detail is not None:
            return self._rejected_decision_result(
                decision,
                runtime_ms,
                claim_digest=plan.claim_digest,
                semantics_digest=plan.semantics_digest,
                candidate_digest=plan.candidate_digest,
                evidence_uri=plan.evidence_uri,
            )
        self._decision_validator.require_request_bound_endpoints(
            decision,
            plan.request_artifact_uris,
        )
        record = self._build_verification_record(
            decision,
            checker=plan.checker,
            evidence_kind=plan.evidence_kind,
            evidence_uri=plan.evidence_uri,
            bindings=plan.bindings,
            request_digest=request_digest,
        )
        return self._finalize_verification(
            decision=decision,
            checker=plan.checker,
            runtime_ms=runtime_ms,
            claim_digest=plan.claim_digest,
            semantics_digest=plan.semantics_digest,
            candidate_digest=plan.candidate_digest,
            evidence_uri=plan.evidence_uri,
            scope_uri=plan.scope_uri,
            record=record,
            schema_uri=self.record_schema_uri,
            parents=plan.parents,
            summary=plan.summary,
            execution_detail=decision.detail,
        )

    def _rejected_decision_result(
        self,
        decision: CheckerDecision,
        runtime_ms: int,
        *,
        claim_digest: str,
        candidate_digest: str,
        evidence_uri: str,
        semantics_digest: str | None = None,
    ) -> VerificationResult:
        return VerificationResult(
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=runtime_ms,
            ),
            input=InputValidation(
                status=InputStatus.REJECTED,
                errors=(decision.detail,),
            ),
            conclusion=Conclusion.UNKNOWN,
            claim_digest=claim_digest,
            semantics_digest=semantics_digest,
            candidate_digest=candidate_digest,
            evidence_uris=(evidence_uri,),
        )

    def _build_verification_record(
        self,
        decision: CheckerDecision,
        *,
        checker: CheckerRegistration,
        evidence_kind: EvidenceKind,
        evidence_uri: str,
        bindings: EvidenceBindings,
        request_digest: str,
    ) -> VerificationRecord:
        return VerificationRecord(
            checker_id=checker.checker_id,
            implementation_digest=checker.implementation_digest,
            checker_manifest=checker.implementation,
            evidence_kind=evidence_kind,
            evidence_uri=evidence_uri,
            bindings=bindings,
            conclusion=decision.conclusion,
            arithmetic=decision.arithmetic,
            method=decision.method,
            coverage=decision.coverage,
            request_digest=request_digest,
            environment_digest=_environment_digest(
                checker.implementation_digest,
                checker.implementation.provider_runtime,
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
        checker: CheckerRegistration,
        runtime_ms: int,
        claim_digest: str,
        candidate_digest: str,
        evidence_uri: str,
        record: VerificationRecord,
        schema_uri: str,
        parents: tuple[str, ...],
        summary: str,
        semantics_digest: str | None = None,
        scope_uri: str | None = None,
        execution_detail: str | None = None,
    ) -> VerificationResult:
        record_artifact = self._record_committer.commit(
            checker_id=checker.checker_id,
            implementation_digest=checker.implementation_digest,
            schema_uri=schema_uri,
            semantics_uri=self.record_semantics_uri,
            payload=record.model_dump(mode="json"),
            parents=parents,
            summary=summary,
        )
        return VerificationResult(
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=runtime_ms,
                detail=execution_detail,
            ),
            input=InputValidation(status=InputStatus.ACCEPTED),
            conclusion=decision.conclusion,
            scope_uri=scope_uri,
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
    ) -> VerificationResult:
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

    def _rejected_input(self, detail: str, *, started: float) -> VerificationResult:
        return VerificationResult(
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=int((time.monotonic() - started) * 1000),
            ),
            input=InputValidation(
                status=InputStatus.REJECTED,
                errors=(detail,),
            ),
            conclusion=Conclusion.UNKNOWN,
        )

    def _operational_failure(
        self,
        *,
        status: ExecutionStatus,
        detail: str,
        started: float,
    ) -> VerificationResult:
        return VerificationResult(
            execution=Execution(
                status=status,
                runtime_ms=int((time.monotonic() - started) * 1000),
                detail=detail,
            ),
            input=InputValidation(status=InputStatus.ACCEPTED),
            conclusion=Conclusion.UNKNOWN,
        )
