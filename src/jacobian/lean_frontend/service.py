"""Pinned Lean certificate construction and verification."""

from __future__ import annotations

import hashlib
import logging
import threading
import weakref
from collections import OrderedDict

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.lean import (
    LeanCandidate,
    LeanClaim,
    LeanEnvironment,
    LeanVerifyResult,
)
from jacobian.contracts.results import (
    Execution,
    ExecutionStatus,
    ResultEnvelope,
    Verification,
)
from jacobian.contracts.verification import VerificationRecord
from jacobian.references import LeanCheckerInstallation
from jacobian.registry import CheckerRegistryError
from jacobian.store import ArtifactStore, StoreError
from jacobian.verification import VerificationService

_LOGGER = logging.getLogger(__name__)
_RESULT_CACHE_SIZE = 128


class LeanService:
    """Build one fully bound Lean certificate and replay its authorized checker."""

    def __init__(
        self,
        store: ArtifactStore,
        artifacts: ArtifactService,
        verification: VerificationService,
        installations: dict[LeanEnvironment, LeanCheckerInstallation],
    ) -> None:
        self.store = store
        self.artifacts = artifacts
        self.verification = verification
        self.installations = installations
        self._cache: OrderedDict[str, tuple[str, ResultEnvelope]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._certificate_locks: weakref.WeakValueDictionary[str, threading.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._warmup_started = False
        self._warmup_thread: threading.Thread | None = None
        self._closing = False
        self._mathlib_warmup_status = "NOT_STARTED"
        self._mathlib_warmup_detail: str | None = None

    def verify(
        self,
        *,
        statement: str,
        proof: str,
        environment: LeanEnvironment = LeanEnvironment.CORE,
    ) -> LeanVerifyResult:
        installation = self.installations.get(environment)
        if installation is None:
            raise ValueError(
                f"Lean environment {environment.value} is not installed. Call "
                "capability.describe with capability_id='lean.check' to list "
                "installed environments."
            )
        claim_payload = LeanClaim(
            environment=environment,
            statement=statement,
            allowed_axioms=installation.allowed_axioms,
        )
        candidate_payload = LeanCandidate(
            environment=environment,
            statement=statement,
            proof=proof,
        )
        claim = self.artifacts.put(
            schema_uri=installation.claim_schema_uri,
            semantics_uri=installation.semantics_uri,
            payload=claim_payload.model_dump(mode="json"),
            summary="exact Lean proposition",
        )
        candidate = self.artifacts.put(
            schema_uri=installation.candidate_schema_uri,
            semantics_uri=installation.semantics_uri,
            payload=candidate_payload.model_dump(mode="json"),
            parents=(claim.artifact_uri,),
            summary=f"proposed {environment.value} Lean proof",
        )
        claim_artifact = self.store.get(claim.artifact_uri)
        candidate_artifact = self.store.get(candidate.artifact_uri)
        semantics = self.store.get(installation.semantics_uri)
        bindings = EvidenceBindings(
            claim_digest=claim_artifact.manifest.object_digest,
            semantics_digest=semantics.manifest.object_digest,
            candidate_digest=candidate_artifact.manifest.object_digest,
        )
        certificate_payload = {
            "statement": statement,
            "proof": proof,
            "environment": environment.value,
            "declaration_name": "jacobian_theorem",
            "lean_version": installation.lean_version,
            "lean_commit": installation.lean_commit,
            "import_name": installation.import_name,
            "mathlib_commit": installation.mathlib_commit,
            "allowed_axioms": list(installation.allowed_axioms),
        }
        payload_digest = (
            "sha256:"
            + hashlib.sha256(canonicalize_json(certificate_payload)).hexdigest()
        )
        certificate_envelope = CertificateEnvelope(
            certificate_type="lean4.kernel",
            format_version="1",
            bindings=bindings,
            payload_digest=payload_digest,
            payload=certificate_payload,
        )
        certificate = self.artifacts.put(
            schema_uri=installation.certificate_schema_uri,
            semantics_uri=installation.semantics_uri,
            payload=certificate_envelope.model_dump(mode="json"),
            parents=(claim.artifact_uri, candidate.artifact_uri),
            summary=f"{environment.value} Lean proof certificate",
        )
        with self._cache_lock:
            certificate_lock = self._certificate_locks.setdefault(
                certificate.artifact_uri,
                threading.Lock(),
            )
        with certificate_lock:
            result = self._cached_result(
                certificate_uri=certificate.artifact_uri,
                installation=installation,
            )
            cache_hit = result is not None
            if result is None:
                result = self.verification.verify_certificate(
                    certificate_uri=certificate.artifact_uri,
                    checker_id=installation.checker_id,
                    timeout_seconds=installation.checker_timeout_seconds,
                )
                if result.execution.status is ExecutionStatus.COMPLETED:
                    try:
                        registration = (
                            self.verification.checker_registry.require_active(
                                installation.checker_id
                            )
                        )
                    except CheckerRegistryError:
                        pass
                    else:
                        with self._cache_lock:
                            self._cache[certificate.artifact_uri] = (
                                registration.executable_digest,
                                result,
                            )
                            self._cache.move_to_end(certificate.artifact_uri)
                            while len(self._cache) > _RESULT_CACHE_SIZE:
                                self._cache.popitem(last=False)
        return LeanVerifyResult(
            claim_uri=claim.artifact_uri,
            candidate_uri=candidate.artifact_uri,
            certificate_uri=certificate.artifact_uri,
            result=result,
            cache_hit=cache_hit,
        )

    def start_mathlib_warmup(self) -> bool:
        """Warm the pinned Mathlib runtime once without delaying server startup."""

        thread = threading.Thread(
            target=self._warm_mathlib,
            name="jacobian-lean-mathlib-warmup",
            daemon=True,
        )
        with self._cache_lock:
            if self._warmup_started or self._closing:
                return False
            self._warmup_started = True
            self._mathlib_warmup_status = "RUNNING"
            self._warmup_thread = thread
            thread.start()
        return True

    def close(self, *, timeout_seconds: float = 120) -> None:
        """Wait for the optional warm-up before releasing its shared store."""

        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        with self._cache_lock:
            self._closing = True
            thread = self._warmup_thread
        if thread is not None:
            thread.join(timeout=timeout_seconds)
            if thread.is_alive():
                raise RuntimeError("Lean Mathlib warm-up did not quiesce")
        with self._cache_lock:
            self._warmup_thread = None
            self._cache.clear()
            self._certificate_locks.clear()

    def _warm_mathlib(self) -> None:
        try:
            checked = self.verify(
                statement="True",
                proof="by trivial",
                environment=LeanEnvironment.MATHLIB,
            )
            healthy = (
                checked.result.execution.status is ExecutionStatus.COMPLETED
                and checked.result.assurance.verification is Verification.VERIFIED
            )
            with self._cache_lock:
                self._mathlib_warmup_status = "HEALTHY" if healthy else "UNHEALTHY"
                self._mathlib_warmup_detail = (
                    None
                    if healthy
                    else (
                        checked.result.input.errors[0]
                        if checked.result.input.errors
                        else "the MATHLIB smoke proof was not accepted"
                    )
                )
        except Exception as exc:
            with self._cache_lock:
                self._mathlib_warmup_status = "UNHEALTHY"
                self._mathlib_warmup_detail = type(exc).__name__
            _LOGGER.exception("Lean Mathlib warm-up failed")

    def mathlib_warmup_health(self) -> dict[str, str | None]:
        """Return model-facing health without exposing runtime paths."""

        with self._cache_lock:
            return {
                "status": self._mathlib_warmup_status,
                "detail": self._mathlib_warmup_detail,
            }

    def _cached_result(
        self,
        *,
        certificate_uri: str,
        installation: LeanCheckerInstallation,
    ) -> ResultEnvelope | None:
        with self._cache_lock:
            cached = self._cache.get(certificate_uri)
            if cached is not None:
                self._cache.move_to_end(certificate_uri)
        if cached is None:
            return None
        checker_digest, result = cached
        try:
            registration = self.verification.checker_registry.require_active(
                installation.checker_id
            )
            if registration.executable_digest != checker_digest:
                return None
            certificate_artifact = self.store.get(certificate_uri)
            certificate = CertificateEnvelope.model_validate(
                certificate_artifact.payload
            )
            if result.verification_record_uri is not None:
                record = VerificationRecord.model_validate(
                    self.store.get(result.verification_record_uri).payload
                )
                if (
                    record.checker_id != installation.checker_id
                    or record.checker_digest != checker_digest
                    or record.evidence_uri != certificate_uri
                    or record.bindings != certificate.bindings
                ):
                    return None
        except (CheckerRegistryError, StoreError, ValueError):
            return None
        return result.model_copy(
            update={
                "execution": Execution(
                    status=ExecutionStatus.COMPLETED,
                    runtime_ms=0,
                    detail="reused exact certificate result from active-checker cache",
                )
            }
        )
