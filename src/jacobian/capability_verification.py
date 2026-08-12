"""Fail-closed artifact closure and verification-record validation."""

from __future__ import annotations

from typing import Any, Protocol

from jacobian.capability_errors import CapabilityError
from jacobian.contracts.capabilities import CapabilityResult
from jacobian.contracts.exact_domain_verification import InlineExactVerificationRecord
from jacobian.contracts.verification import VerificationRecord
from jacobian.storage.errors import StorageError


class VerificationOwner(Protocol):
    store: Any


class CapabilityVerificationMixin:
    """Own the trust boundary between result projections and local records."""

    def _validate_verified_result(
        self: VerificationOwner,
        result: CapabilityResult,
    ) -> None:
        record_uri = result.verification_record_uri
        if record_uri is None:
            return
        try:
            record_artifact = self.store.get(record_uri)
            record = VerificationRecord.model_validate(record_artifact.payload)
        except (StorageError, ValueError) as exc:
            if isinstance(exc, StorageError):
                raise CapabilityError(
                    "verified capability result has no valid local verification record"
                ) from exc
            try:
                inline_record = InlineExactVerificationRecord.model_validate(
                    record_artifact.payload
                )
            except ValueError as inline_exc:
                raise CapabilityError(
                    "verified capability result has no valid local verification record"
                ) from inline_exc
            _validate_inline_exact_record(
                result, record_uri, record_artifact, inline_record
            )
            return
        if record.evidence_uri not in result.artifact_uris:
            raise CapabilityError(
                "verified capability result does not expose its checked evidence"
            )
        missing_parents = set(record_artifact.manifest.parents) - set(
            result.artifact_uris
        )
        if missing_parents:
            raise CapabilityError(
                "verified capability result omits verification-bound artifacts"
            )
        _validate_projected_output(result, record_uri, record)


def _validate_inline_exact_record(
    result: CapabilityResult,
    record_uri: str,
    record_artifact: Any,
    record: InlineExactVerificationRecord,
) -> None:
    """Validate a digest-bound replay without inventing input/result artifacts."""

    if record_uri not in result.artifact_uris:
        raise CapabilityError(
            "verified capability result does not expose its verification record"
        )
    if tuple(record_artifact.manifest.parents) != (record.semantics_uri,):
        raise CapabilityError(
            "inline exact verification record has unexpected artifact parents"
        )
    if record.semantics_uri not in result.artifact_uris:
        raise CapabilityError(
            "verified capability result does not expose the inline record semantics"
        )
    if result.output.get("operation_id") != record.operation_id:
        raise CapabilityError(
            "verified capability output projects a different operation"
        )
    if result.output.get("checker_id") != record.checker_id:
        raise CapabilityError("verified capability output projects a different checker")
    for field in ("claim_digest", "semantics_digest", "candidate_digest"):
        if result.output.get(field) != getattr(record.bindings, field):
            raise CapabilityError(
                f"verified capability output projects a different {field}"
            )
    projected_record = result.output.get("verification_record_uri")
    if projected_record is not None and projected_record != record_uri:
        raise CapabilityError(
            "verified capability output projects a different verification record"
        )
    projected_conclusion = result.output.get("conclusion")
    if (
        projected_conclusion is not None
        and projected_conclusion != record.decision.conclusion.value
    ):
        raise CapabilityError(
            "verified capability output differs from the checked conclusion"
        )


def _validate_projected_output(
    result: CapabilityResult,
    record_uri: str,
    record: VerificationRecord,
) -> None:
    projected_checker = result.output.get("checker_id")
    if projected_checker is not None and projected_checker != record.checker_id:
        raise CapabilityError("verified capability output projects a different checker")
    for field in (
        "claim_digest",
        "semantics_digest",
        "candidate_digest",
        "scope_digest",
        "encoding_digest",
    ):
        projected_digest = result.output.get(field)
        if projected_digest is not None and projected_digest != getattr(
            record.bindings, field
        ):
            raise CapabilityError(
                f"verified capability output projects a different {field}"
            )
    projected_record = result.output.get("verification_record_uri")
    if projected_record is not None and projected_record != record_uri:
        raise CapabilityError(
            "verified capability output projects a different verification record"
        )
    projected_conclusion = result.output.get("conclusion")
    if (
        projected_conclusion is not None
        and projected_conclusion != record.conclusion.value
    ):
        raise CapabilityError(
            "verified capability output differs from the checked conclusion"
        )


__all__ = ["CapabilityVerificationMixin"]
