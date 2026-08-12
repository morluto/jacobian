"""Pinned Lean certificate construction and verification."""

from __future__ import annotations

import hashlib
import logging
import threading
import weakref
from collections import OrderedDict

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.checker_authorization import LeanCheckerInstallation
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
    InputStatus,
    VerificationResult,
)
from jacobian.contracts.verification import VerificationRecord
from jacobian.lean_frontend.diagnostics import checker_diagnostics
from jacobian.registry import CheckerRegistryError
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService

_LOGGER = logging.getLogger(__name__)
_RESULT_CACHE_SIZE = 128


class LeanService:
    """Build one fully bound Lean certificate and replay its authorized checker."""

    def __init__(
        self,
        store: ArtifactRepository,
        artifacts: ArtifactService,
        verification: VerificationService,
        installations: dict[LeanEnvironment, LeanCheckerInstallation],
    ) -> None:
        self.store = store
        self.artifacts = artifacts
        self.verification = verification
        self.installations = installations
        self._cache: OrderedDict[str, tuple[str, VerificationResult]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._certificate_locks: weakref.WeakValueDictionary[str, threading.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._warmup_started = False
        self._warmup_thread: threading.Thread | None = None
        self._closing = False

    def verify(
        self,
        *,
        statement: str,
        proof: str,
        environment: LeanEnvironment = LeanEnvironment.CORE,
    ) -> LeanVerifyResult:
        installation = self.installations.get(environment)
        if installation is None or installation.checker_id is None:
            raise ValueError(
                f"Lean environment {environment.value} is not installed. Call "
                "math.find with capability_id='lean.check' to list "
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
                if (
                    result.execution.status is ExecutionStatus.COMPLETED
                    and result.input.status is InputStatus.ACCEPTED
                ):
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
                                registration.implementation_digest,
                                result,
                            )
                            self._cache.move_to_end(certificate.artifact_uri)
                            while len(self._cache) > _RESULT_CACHE_SIZE:
                                self._cache.popitem(last=False)
        diagnostics = checker_diagnostics(
            result,
            statement=statement,
            proof=proof,
            environment=environment,
        )
        return LeanVerifyResult(
            claim_uri=claim.artifact_uri,
            candidate_uri=candidate.artifact_uri,
            certificate_uri=certificate.artifact_uri,
            result=result,
            diagnostics=diagnostics,
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
        try:
            if thread is not None:
                thread.join(timeout=timeout_seconds)
                if thread.is_alive():
                    raise RuntimeError("Lean Mathlib warm-up did not quiesce")
            with self._cache_lock:
                self._warmup_thread = None
                self._cache.clear()
                self._certificate_locks.clear()
        except BaseException:
            with self._cache_lock:
                self._closing = False
            raise

    def _warm_mathlib(self) -> None:
        try:
            self.verify(
                statement="True",
                proof="by trivial",
                environment=LeanEnvironment.MATHLIB,
            )
        except Exception:
            _LOGGER.exception("Lean Mathlib warm-up failed")

    def _cached_result(
        self,
        *,
        certificate_uri: str,
        installation: LeanCheckerInstallation,
    ) -> VerificationResult | None:
        if installation.checker_id is None:
            return None
        with self._cache_lock:
            cached = self._cache.get(certificate_uri)
            if cached is not None:
                self._cache.move_to_end(certificate_uri)
        if cached is None:
            return None
        implementation_digest, result = cached
        try:
            registration = self.verification.checker_registry.require_active(
                installation.checker_id
            )
            if registration.implementation_digest != implementation_digest:
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
                    or record.implementation_digest != implementation_digest
                    or record.evidence_uri != certificate_uri
                    or record.bindings != certificate.bindings
                ):
                    return None
        except (CheckerRegistryError, StorageError, ValueError):
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
