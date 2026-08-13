"""Shared fail-closed invariants for accepted verification evidence."""

from __future__ import annotations

from jacobian.contracts.common import ArtifactUri
from jacobian.contracts.operations import OperationId
from jacobian.contracts.results import Arithmetic, Conclusion, Coverage, Method


def validate_decisive_replayable_evidence(
    conclusion: Conclusion,
    arithmetic: Arithmetic,
    coverage: Coverage,
    method: Method,
) -> None:
    if conclusion not in {Conclusion.TRUE, Conclusion.FALSE}:
        raise ValueError("accepted checker evidence requires a decisive conclusion")
    if arithmetic is Arithmetic.FLOATING_HEURISTIC:
        raise ValueError("a checker cannot accept floating heuristic evidence")
    if coverage in {Coverage.RESTRICTED, Coverage.SAMPLED}:
        raise ValueError("a checker cannot accept restricted or sampled evidence")
    if method is Method.DIRECT_WITNESS and coverage is not Coverage.NOT_APPLICABLE:
        raise ValueError("a direct witness checker cannot claim coverage")
    if method is Method.EXHAUSTIVE_FINITE and coverage is not Coverage.EXHAUSTIVE:
        raise ValueError("exhaustive checker acceptance requires exhaustive coverage")


def validate_certified_relationship_endpoints(
    relation_id: OperationId | None,
    source_artifact_uris: tuple[ArtifactUri, ...],
    target_artifact_uris: tuple[ArtifactUri, ...],
) -> None:
    if relation_id is None and (source_artifact_uris or target_artifact_uris):
        raise ValueError("relationship endpoints require a relation ID")
    if relation_id is not None and (
        not source_artifact_uris or not target_artifact_uris
    ):
        raise ValueError("a certified relationship requires exact endpoints")
    if len(set(source_artifact_uris)) != len(source_artifact_uris) or len(
        set(target_artifact_uris)
    ) != len(target_artifact_uris):
        raise ValueError("certified relationship endpoints must be unique")
