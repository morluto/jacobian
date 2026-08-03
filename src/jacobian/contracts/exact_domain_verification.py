"""Contracts for independent verification of exact domain-operation results.

The exact-domain verifier migration moves from grouped ``result_uri``
verification to per-producer typed verifier capability contracts for inline
results. A verification request carries the producer's exact input and
candidate (result) values directly; the verifier validates both, checks the
authorized checker's scope, materializes the exact input and candidate as
artifacts only within verification, and then reuses the existing witness,
checker, ``VerificationService``, and immutable replay-record path.

Materialized and bounded-search producers retain their typed ``result_uri``
verifier input. The verifier resolves the producer's exact persisted lineage;
the search's separate optimality-obligation artifact remains outside the exact
replay relation.
"""

from __future__ import annotations

from typing import Literal

from jacobian.contracts.capabilities import CapabilityId
from jacobian.contracts.common import ArtifactUri, CheckerUri
from jacobian.contracts.results import ContractModel


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
    # ``input_uri`` and ``result_uri`` are absent for ``UNSUPPORTED`` because
    # the verifier checks scope before any artifact write and materializes the
    # exact input and candidate only within verification.
    input_uri: ArtifactUri | None = None
    result_uri: ArtifactUri | None = None
    witness_uri: ArtifactUri | None = None
    checker_id: CheckerUri
    verification_record_uri: ArtifactUri | None = None
    detail: str


class ExactDomainResultVerificationRequest(ContractModel):
    """Exact replay input for a producer whose result is already durable."""

    result_uri: ArtifactUri


__all__ = [
    "ExactComputedVerificationOutput",
    "ExactComputedVerificationRequest",
    "ExactDomainResultVerificationRequest",
]
