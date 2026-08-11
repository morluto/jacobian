"""Deterministic MiniF2F and ProofNet row materialization."""

from __future__ import annotations

import hashlib

from jacobian.canonical import canonicalize_json
from jacobian.contracts.capabilities import (
    CapabilityDiagnostic,
)
from jacobian.contracts.formal_datasets import (
    FormalDatasetArtifact,
    FormalDatasetDiagnosticBaseline,
    FormalDatasetMaterializeRequest,
    formal_dataset_diagnostics,
    formal_dataset_preprocessing,
)
from jacobian.operations import OperationRefusalError
from jacobian_checkers.lean4 import LEAN_VERSION, MATHLIB_COMMIT


def _json_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonicalize_json(value)).hexdigest()


def _text_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    if value == "":
        return ""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return normalized if normalized.endswith("\n") else normalized + "\n"


def _materialize_payload(
    validated: FormalDatasetMaterializeRequest,
) -> FormalDatasetArtifact:
    row_payload = validated.row.model_dump(mode="json")
    row_digest = _json_digest(row_payload)
    header = _normalize_text(validated.row.header) if validated.row.header else ""
    formal_statement = _normalize_text(validated.row.formal_statement)
    normalized_source = f"{header}{formal_statement}"
    environment_payload = validated.environment.model_dump(mode="json")
    return FormalDatasetArtifact(
        dataset_id=validated.row.dataset_id,
        dataset_revision=validated.dataset_revision,
        sample_id=validated.sample_id,
        source_url=validated.source_url,
        split=validated.row.split,
        canonical_row=validated.row,
        row_digest=row_digest,
        normalized_source_digest=_text_digest(normalized_source),
        normalized_source=normalized_source,
        formal_statement=formal_statement,
        informal_statement=(
            _normalize_text(validated.row.informal_statement)
            if validated.row.informal_statement is not None
            else None
        ),
        informal_proof=(
            _normalize_text(validated.row.informal_proof)
            if validated.row.informal_proof is not None
            else None
        ),
        header=header,
        environment=validated.environment,
        environment_digest=_json_digest(environment_payload),
        preprocessing=formal_dataset_preprocessing(validated.row),
        diagnostic_baseline=FormalDatasetDiagnosticBaseline(
            lean_version=LEAN_VERSION,
            mathlib_revision=MATHLIB_COMMIT,
        ),
        diagnostics=formal_dataset_diagnostics(validated.environment),
    )


def _materialize_operation(
    validated: FormalDatasetMaterializeRequest,
) -> FormalDatasetArtifact:
    payload = _materialize_payload(validated)
    if (
        validated.expected_row_digest is not None
        and validated.expected_row_digest != payload.row_digest
    ):
        raise OperationRefusalError(
            CapabilityDiagnostic(
                code="FORMAL_DATASET_ROW_DIGEST_MISMATCH",
                stage="source_binding",
                message="The supplied row does not match expected_row_digest.",
                hint="Re-fetch the pinned row or update the digest explicitly.",
            )
        )
    return payload
