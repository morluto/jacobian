"""Fail-closed artifact closure and verification-record validation."""

from __future__ import annotations

from typing import Any, Protocol

from jacobian.capability_errors import CapabilityError
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityObligationStatus,
    CapabilityRelationshipStatus,
    CapabilityResult,
)
from jacobian.contracts.exact_domain_verification import InlineExactVerificationRecord
from jacobian.contracts.results import Coverage
from jacobian.contracts.verification import VerificationRecord
from jacobian.storage.errors import StorageError


class VerificationOwner(Protocol):
    store: Any


class CapabilityVerificationMixin:
    """Own the trust boundary between result projections and local records."""

    @staticmethod
    def _validate_artifact_references(result: CapabilityResult) -> None:
        exposed = set(result.artifact_uris)
        referenced: set[str] = set()
        if result.scope is not None and result.scope.artifact_uri is not None:
            referenced.add(result.scope.artifact_uri)
        if result.completeness.verification_record_uri is not None:
            referenced.add(result.completeness.verification_record_uri)
        for relationship in result.relationships:
            referenced.update(relationship.source_artifact_uris)
            referenced.update(relationship.target_artifact_uris)
            referenced.update(relationship.obligation_uris)
            if relationship.verification_record_uri is not None:
                referenced.add(relationship.verification_record_uri)
        for obligation in result.obligations:
            referenced.add(obligation.obligation_uri)
            if obligation.verification_record_uri is not None:
                referenced.add(obligation.verification_record_uri)
        if referenced - exposed:
            raise CapabilityError(
                "capability result has first-class references missing from artifact_uris"
            )

    def _validate_verified_result(
        self: VerificationOwner,
        result: CapabilityResult,
    ) -> None:
        if result.assurance.level is not CapabilityAssuranceLevel.VERIFIED:
            return
        record_uri = result.assurance.verification_record_uri
        if record_uri is None:
            raise CapabilityError(
                "verified capability result has no verification record URI"
            )
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
        record_parents = set(record_artifact.manifest.parents)
        _validate_verified_relationships(
            result, record_artifact, record, record_parents
        )
        _validate_discharged_obligations(result, record_artifact, record_parents)
        _validate_verified_completeness(result, record, record_parents)


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
    _validate_inline_exact_scope(result, record)
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
    if any(
        relationship.status is CapabilityRelationshipStatus.VERIFIED
        for relationship in result.relationships
    ):
        raise CapabilityError(
            "inline exact verification cannot certify artifact relationships"
        )
    if any(
        obligation.status is CapabilityObligationStatus.DISCHARGED
        for obligation in result.obligations
    ):
        raise CapabilityError(
            "inline exact verification cannot discharge artifact obligations"
        )
    if result.completeness.assurance_level is CapabilityAssuranceLevel.VERIFIED:
        raise CapabilityError(
            "inline exact verification cannot certify artifact completeness"
        )


def _validate_inline_exact_scope(
    result: CapabilityResult,
    record: InlineExactVerificationRecord,
) -> None:
    scope = result.scope
    if scope is None or scope.parameters is None:
        raise CapabilityError("verified inline replay has no bound scope parameters")
    expected = {
        "operation_id": record.operation_id,
        "claim_digest": record.bindings.claim_digest,
        "candidate_digest": record.bindings.candidate_digest,
        "semantics_digest": record.bindings.semantics_digest,
        "checker_id": record.checker_id,
        "witness_format": record.witness_format,
    }
    if scope.parameters != expected:
        raise CapabilityError("verified inline replay scope does not bind its record")


def _validate_projected_output(
    result: CapabilityResult,
    record_uri: str,
    record: VerificationRecord,
) -> None:
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


def _validate_verified_relationships(
    result: CapabilityResult,
    record_artifact: Any,
    record: VerificationRecord,
    record_parents: set[str],
) -> None:
    for relationship in result.relationships:
        if relationship.status is not CapabilityRelationshipStatus.VERIFIED:
            continue
        bound_artifacts = {
            *relationship.source_artifact_uris,
            *relationship.target_artifact_uris,
            *relationship.obligation_uris,
        }
        if not bound_artifacts.issubset(record_parents):
            raise CapabilityError(
                "verified relationship record does not bind its artifacts"
            )
        if record_artifact.payload.get("relation_id") != relationship.relation_id:
            raise CapabilityError(
                "verified relationship differs from the checked relation"
            )
        if (
            record.relationship_source_artifact_uris
            != relationship.source_artifact_uris
            or record.relationship_target_artifact_uris
            != relationship.target_artifact_uris
        ):
            raise CapabilityError(
                "verified relationship endpoints differ from the checked relation"
            )
        checked_obligations = (
            (record.obligation_uri,) if record.obligation_uri is not None else ()
        )
        if relationship.obligation_uris != checked_obligations:
            raise CapabilityError(
                "verified relationship obligations differ from the checked relation"
            )


def _validate_discharged_obligations(
    result: CapabilityResult,
    record_artifact: Any,
    record_parents: set[str],
) -> None:
    for obligation in result.obligations:
        if obligation.status is not CapabilityObligationStatus.DISCHARGED:
            continue
        if (
            obligation.obligation_uri not in record_parents
            or record_artifact.payload.get("obligation_uri")
            != obligation.obligation_uri
        ):
            raise CapabilityError(
                "discharged obligation differs from the checked obligation"
            )


def _validate_verified_completeness(
    result: CapabilityResult,
    record: VerificationRecord,
    record_parents: set[str],
) -> None:
    if (
        result.completeness.status is CapabilityCompletenessStatus.COMPLETE
        and result.completeness.assurance_level is CapabilityAssuranceLevel.VERIFIED
    ):
        if (
            result.scope is None
            or result.scope.artifact_uri is None
            or result.scope.artifact_uri not in record_parents
        ):
            raise CapabilityError(
                "verified completeness requires a checker-bound scope artifact"
            )
        if record.coverage not in {Coverage.EXHAUSTIVE, Coverage.BOUNDED}:
            raise CapabilityError("verified completeness differs from checked coverage")


__all__ = ["CapabilityVerificationMixin"]
