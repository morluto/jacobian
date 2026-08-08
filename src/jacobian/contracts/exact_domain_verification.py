"""Contracts for independent verification of exact domain-operation results.

The exact-domain verifier migration moves from grouped ``result_uri``
verification to per-producer typed verifier capability contracts for inline
results. A verification request carries the producer's exact input and
candidate (result) values directly; the verifier validates both, checks the
authorized checker's scope, and binds canonical inline value digests into an
independent worker request and immutable replay record without materializing
ordinary values as artifacts.

Materialized and bounded-search producers retain their typed ``result_uri``
verifier input. The verifier resolves the producer's exact persisted lineage;
the search's separate optimality-obligation artifact remains outside the exact
replay relation.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from jacobian.canonical import canonicalize_json, sha256_digest
from jacobian.contracts.capabilities import CapabilityId
from jacobian.contracts.checkers import CheckerDecision, EvidenceKind
from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest
from jacobian.contracts.evidence import EvidenceBindings
from jacobian.contracts.results import ContractModel


def inline_exact_value_digest(
    *, schema_uri: str, semantics_uri: str, payload: dict[str, object]
) -> str:
    """Bind one inline value to its contract without assigning it an artifact URI."""

    return sha256_digest(
        canonicalize_json(
            {
                "inline_exact_value_version": "1",
                "schema_uri": schema_uri,
                "semantics_uri": semantics_uri,
                "payload": payload,
            }
        )
    )


class ExactComputedVerificationRequest[
    RequestT: ContractModel,
    ResultT: ContractModel,
](ContractModel):
    """Inline exact input and candidate for one per-producer verifier.

    ``input`` is the producer's validated request payload and ``candidate`` is
    the producer's result payload. Both are required: the verifier validates
    them against the producer's input and result schemas and checks the
    authorized checker's scope before any artifact write.
    """

    input: RequestT
    candidate: ResultT


class ExactComputedVerificationOutput(ContractModel):
    status: Literal[
        "VERIFIED",
        "REJECTED",
        "UNSUPPORTED",
        "TIMEOUT",
        "CANCELLED",
        "ERROR",
    ]
    conclusion: Literal["TRUE", "UNKNOWN"]
    operation_id: CapabilityId
    # Inline replay deliberately leaves these absent: ordinary values remain
    # inline, while a successful record binds their canonical digests.
    input_uri: ArtifactUri | None = None
    result_uri: ArtifactUri | None = None
    witness_uri: ArtifactUri | None = None
    checker_id: CheckerUri
    verification_record_uri: ArtifactUri | None = None
    detail: str


class ExactDomainResultVerificationRequest(ContractModel):
    """Exact replay input for a producer whose result is already durable."""

    result_uri: ArtifactUri


class InlineExactVerificationRecord(ContractModel):
    """Durable accepted decision for a replay over non-durable values.

    The input and candidate are deliberately represented only by canonical
    digests.  They remain ordinary inline mathematical values rather than
    being promoted into artifacts merely to support verification.
    """

    record_schema_version: Literal["1"] = "1"
    evidence_kind: Literal[EvidenceKind.WITNESS] = EvidenceKind.WITNESS
    witness_format: str
    operation_id: CapabilityId
    format_version: Literal["1"] = "1"
    checker_id: CheckerUri
    checker_digest: Sha256Digest
    runtime_digest: Sha256Digest | None = None
    environment_digest: Sha256Digest
    input_schema_uri: ArtifactUri
    candidate_schema_uri: ArtifactUri
    semantics_uri: ArtifactUri
    bindings: EvidenceBindings
    decision: CheckerDecision
    request_digest: Sha256Digest

    @model_validator(mode="after")
    def accepted_inline_replay_is_fully_bound(self) -> Self:
        if (
            not self.decision.accepted
            or self.bindings.candidate_digest is None
            or self.bindings.scope_digest is not None
            or self.bindings.encoding_digest is not None
        ):
            raise ValueError("inline exact records require an accepted unscoped replay")
        return self


__all__ = [
    "ExactComputedVerificationOutput",
    "ExactComputedVerificationRequest",
    "ExactDomainResultVerificationRequest",
    "InlineExactVerificationRecord",
    "inline_exact_value_digest",
]
