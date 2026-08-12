"""Passive codec helpers for checker evidence artifacts.

These helpers encode stored artifacts for operator-authorized checkers. They
assemble witness envelopes bound by digest. They never authorize a checker,
replay mathematics, or interpret verification outcomes; each caller owns its
error type, messages, and result interpretation.

The single helper here is the only demonstrably identical passive codec step
shared by the determinant and SAT assignment verification paths (with the
linear rational and integer row-HNF paths as confirming evidence): building a
SUPPORTS_CLAIM witness envelope bound by claim, semantics, and candidate
object digests and committing it with the claim and candidate as parents.

Authorization, mathematical replay, status interpretation, and verification
wording remain specialized and operator-owned in each caller.
"""

from __future__ import annotations

from typing import Any

from jacobian.artifacts import ArtifactService
from jacobian.contracts.artifacts import ArtifactPutResult
from jacobian.contracts.evidence import (
    EvidenceBindings,
    WitnessEnvelope,
    WitnessRole,
)
from jacobian.storage.models import StoredArtifact


def put_witness_envelope(
    artifacts: ArtifactService,
    *,
    witness_schema_uri: str,
    witness_format: str,
    claim_artifact: StoredArtifact,
    semantics_artifact: StoredArtifact,
    candidate_artifact: StoredArtifact,
    payload: dict[str, Any],
    summary: str,
) -> ArtifactPutResult:
    """Build and store one SUPPORTS_CLAIM witness envelope bound by digest.

    The witness binds the claim, semantics, and candidate by their stored
    object digests and lists the claim and candidate artifacts as parents in
    that order. The caller owns the witness format, payload, and summary so
    mathematical semantics stay specialized. This helper only encodes and
    commits; it performs no replay and interprets no outcome.
    """

    bindings = EvidenceBindings(
        claim_digest=claim_artifact.manifest.object_digest,
        semantics_digest=semantics_artifact.manifest.object_digest,
        candidate_digest=candidate_artifact.manifest.object_digest,
    )
    witness = WitnessEnvelope(
        witness_format=witness_format,
        format_version="1",
        role=WitnessRole.SUPPORTS_CLAIM,
        bindings=bindings,
        payload=payload,
    )
    return artifacts.put(
        schema_uri=witness_schema_uri,
        semantics_uri=semantics_artifact.artifact_uri,
        payload=witness.model_dump(mode="json"),
        parents=(
            claim_artifact.artifact_uri,
            candidate_artifact.artifact_uri,
        ),
        summary=summary,
    )
