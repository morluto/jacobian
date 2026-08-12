"""Contracts for deterministic formal-dataset row materialization."""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import PureWindowsPath
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    model_validator,
)

from jacobian.canonical import canonicalize_json
from jacobian.contracts.common import Sha256Digest
from jacobian.contracts.results import ContractModel
from jacobian.contracts.urls import normalize_http_url
from jacobian_checkers.lean4 import LEAN_VERSION, MATHLIB_COMMIT


def _require_nfc(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("value must be valid UTF-8 text") from exc
    if value != unicodedata.normalize("NFC", value):
        raise ValueError("value must be NFC-normalized")
    return value


def _require_nonblank_nfc(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return _require_nfc(value)


def _normalize_http_url(value: object) -> str:
    """Normalize a source URL before its bounded contract is enforced."""

    if not isinstance(value, str):
        raise ValueError("source URL must be a valid HTTP(S) URL")
    _require_nonblank_nfc(value)
    return normalize_http_url(value, label="source URL")


BoundedFormalString = Annotated[
    str,
    Field(min_length=1, max_length=2_000),
    AfterValidator(_require_nonblank_nfc),
]

DatasetRevision = Annotated[
    str,
    Field(min_length=7, max_length=128),
    AfterValidator(_require_nonblank_nfc),
]
DatasetSampleId = Annotated[
    str,
    Field(min_length=1, max_length=512),
    AfterValidator(_require_nonblank_nfc),
]
DatasetSourceUrl = Annotated[
    str,
    BeforeValidator(_normalize_http_url),
    Field(min_length=1, max_length=2_000),
    AfterValidator(_require_nonblank_nfc),
]


def _validate_row_text(row: MiniF2FRow | ProofNetRow) -> None:
    required = {
        "name": row.name,
        "split": row.split,
        "formal_statement": row.formal_statement,
    }
    if isinstance(row, MiniF2FRow):
        required["goal"] = row.goal
    else:
        required["informal_statement"] = row.informal_statement
    for field_name, required_value in required.items():
        if not required_value.strip():
            raise ValueError(f"{field_name} must not be blank")
    for value in (
        row.name,
        row.split,
        row.header,
        row.formal_statement,
        row.informal_statement,
        row.informal_proof,
        row.goal if isinstance(row, MiniF2FRow) else None,
    ):
        if value is not None:
            _require_nfc(value)


class MiniF2FRow(ContractModel):
    dataset_id: Literal["MINIF2F"]
    name: str = Field(min_length=1, max_length=256)
    split: Literal["train", "valid", "test"]
    formal_statement: str = Field(min_length=1, max_length=40_000)
    informal_statement: str | None = Field(default=None, max_length=40_000)
    informal_proof: str | None = Field(default=None, max_length=80_000)
    goal: str = Field(min_length=1, max_length=40_000)
    header: str = Field(default="", max_length=20_000)

    @model_validator(mode="after")
    def require_replay_safe_text(self) -> Self:
        _validate_row_text(self)
        return self


class ProofNetRow(ContractModel):
    dataset_id: Literal["PROOFNET"]
    name: str = Field(min_length=1, max_length=256)
    split: str = Field(min_length=1, max_length=64)
    formal_statement: str = Field(min_length=1, max_length=80_000)
    informal_statement: str = Field(min_length=1, max_length=80_000)
    informal_proof: str | None = Field(default=None, max_length=120_000)
    header: str = Field(default="", max_length=20_000)

    @model_validator(mode="after")
    def require_replay_safe_text(self) -> Self:
        _validate_row_text(self)
        return self


FormalDatasetRow = Annotated[
    MiniF2FRow | ProofNetRow,
    Field(discriminator="dataset_id"),
]


class FormalProjectFile(ContractModel):
    path: str = Field(min_length=1, max_length=512)
    digest: Sha256Digest

    @model_validator(mode="after")
    def require_safe_relative_path(self) -> Self:
        parts = self.path.split("/")
        _require_nfc(self.path)
        if (
            self.path.startswith("/")
            or "\\" in self.path
            or "\x00" in self.path
            or PureWindowsPath(self.path).drive
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError("project file path must be canonical NFC and relative")
        return self


class FormalDatasetEnvironment(ContractModel):
    lean_version: str = Field(min_length=1, max_length=64)
    project_source_url: DatasetSourceUrl
    project_revision: str = Field(min_length=1, max_length=128)
    mathlib_revision: str | None = Field(default=None, max_length=128)
    imports: tuple[BoundedFormalString, ...] = Field(default=(), max_length=128)
    namespace: str | None = Field(default=None, max_length=512)
    theorem_context: tuple[BoundedFormalString, ...] = Field(default=(), max_length=128)
    project_files: tuple[FormalProjectFile, ...] = Field(default=(), max_length=512)

    @model_validator(mode="after")
    def require_unique_ordered_bindings(self) -> Self:
        for field_name in ("lean_version", "project_source_url", "project_revision"):
            value = getattr(self, field_name)
            if not value.strip():
                raise ValueError(f"{field_name} must not be blank")
        for field_name in ("mathlib_revision", "namespace"):
            value = getattr(self, field_name)
            if value is not None:
                if not value.strip():
                    raise ValueError(f"{field_name} must not be blank")
                _require_nfc(value)
        _require_nfc(self.lean_version)
        _require_nfc(self.project_source_url)
        _require_nfc(self.project_revision)
        if len(set(self.imports)) != len(self.imports):
            raise ValueError("imports must be unique and ordered")
        if len(set(self.theorem_context)) != len(self.theorem_context):
            raise ValueError("theorem_context entries must be unique and ordered")
        paths = [item.path for item in self.project_files]
        if len(set(paths)) != len(paths):
            raise ValueError("project file paths must be unique")
        return self


class FormalPreprocessingDecision(ContractModel):
    operation: Literal[
        "NORMALIZE_NEWLINES",
        "TRIM_TRAILING_WHITESPACE",
        "ENSURE_FINAL_NEWLINE",
    ]
    applied: bool


class FormalDatasetMaterializeRequest(ContractModel):
    dataset_revision: DatasetRevision
    sample_id: DatasetSampleId
    source_url: DatasetSourceUrl
    row: FormalDatasetRow
    environment: FormalDatasetEnvironment
    expected_row_digest: Sha256Digest | None = None

    @model_validator(mode="after")
    def bind_dataset_identity(self) -> Self:
        if self.sample_id != self.row.name:
            raise ValueError("sample_id must equal the dataset row name")
        return self


class FormalDatasetDiagnostic(ContractModel):
    code: Literal[
        "EXECUTION_NOT_REQUESTED",
        "LEAN_VERSION_NOT_PINNED_RUNTIME",
        "MATHLIB_REVISION_NOT_PINNED_RUNTIME",
        "PROJECT_FILES_UNDECLARED",
    ]
    message: str = Field(min_length=1, max_length=2_000)


class FormalDatasetDiagnosticBaseline(ContractModel):
    """Pinned runtime baseline used to derive replay-stable diagnostics."""

    lean_version: str = Field(min_length=1, max_length=64)
    mathlib_revision: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def require_pinned_nonblank_values(self) -> Self:
        _require_nonblank_nfc(self.lean_version)
        _require_nonblank_nfc(self.mathlib_revision)
        return self


class FormalDatasetArtifact(ContractModel):
    artifact_version: Literal["1"] = "1"
    dataset_id: Literal["MINIF2F", "PROOFNET"]
    dataset_revision: DatasetRevision
    sample_id: DatasetSampleId
    source_url: DatasetSourceUrl
    split: str
    canonical_row: FormalDatasetRow
    row_digest: Sha256Digest
    normalized_source_digest: Sha256Digest
    normalized_source: str
    formal_statement: str
    informal_statement: str | None
    informal_proof: str | None
    header: str
    environment: FormalDatasetEnvironment
    environment_digest: Sha256Digest
    preprocessing: tuple[FormalPreprocessingDecision, ...]
    diagnostic_baseline: FormalDatasetDiagnosticBaseline
    diagnostics: tuple[FormalDatasetDiagnostic, ...]

    @model_validator(mode="after")
    def verify_provenance_bindings(self) -> Self:
        self._verify_identity_and_digests()
        self._verify_normalized_content()
        self._verify_derived_provenance()
        return self

    def _verify_identity_and_digests(self) -> None:
        row_payload = self.canonical_row.model_dump(mode="json")
        if self.dataset_id != self.canonical_row.dataset_id:
            raise ValueError("dataset_id must match canonical_row")
        if self.sample_id != self.canonical_row.name:
            raise ValueError("sample_id must match canonical_row")
        if self.split != self.canonical_row.split:
            raise ValueError("split must match canonical_row")
        if self.row_digest != _json_digest(row_payload):
            raise ValueError("row_digest must bind canonical_row")
        if self.normalized_source != unicodedata.normalize(
            "NFC", self.normalized_source
        ):
            raise ValueError("normalized_source must be NFC-normalized")
        if self.normalized_source_digest != _text_digest(self.normalized_source):
            raise ValueError("normalized_source_digest must bind normalized_source")
        if self.environment_digest != _json_digest(
            self.environment.model_dump(mode="json")
        ):
            raise ValueError("environment_digest must bind environment")

    def _verify_normalized_content(self) -> None:
        expected_header = (
            _normalize_text(self.canonical_row.header)
            if self.canonical_row.header
            else ""
        )
        expected_formal = _normalize_text(self.canonical_row.formal_statement)
        if self.header != expected_header:
            raise ValueError("header must be the normalized canonical-row header")
        if self.formal_statement != expected_formal:
            raise ValueError(
                "formal_statement must be the normalized canonical-row statement"
            )
        if self.normalized_source != f"{expected_header}{expected_formal}":
            raise ValueError("normalized_source must bind header and formal_statement")
        expected_informal = (
            _normalize_text(self.canonical_row.informal_statement)
            if self.canonical_row.informal_statement is not None
            else None
        )
        if self.informal_statement != expected_informal:
            raise ValueError("informal_statement must bind canonical_row")
        expected_proof = (
            _normalize_text(self.canonical_row.informal_proof)
            if self.canonical_row.informal_proof is not None
            else None
        )
        if self.informal_proof != expected_proof:
            raise ValueError("informal_proof must bind canonical_row")

    def _verify_derived_provenance(self) -> None:
        expected_preprocessing = formal_dataset_preprocessing(self.canonical_row)
        if self.preprocessing != expected_preprocessing:
            raise ValueError("preprocessing must describe the canonical pipeline")
        if self.diagnostics != formal_dataset_diagnostics(
            self.environment,
            baseline=self.diagnostic_baseline,
        ):
            raise ValueError("diagnostics must match the declared environment")


def _json_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonicalize_json(value)).hexdigest()


def _text_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    if value == "":
        return ""
    lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized = "\n".join(lines)
    if not normalized.endswith("\n"):
        normalized += "\n"
    return unicodedata.normalize("NFC", normalized)


def formal_dataset_preprocessing(
    row: MiniF2FRow | ProofNetRow,
) -> tuple[FormalPreprocessingDecision, ...]:
    values = (
        row.header,
        row.formal_statement,
        row.informal_statement,
        row.informal_proof,
    )
    present = tuple(value for value in values if value is not None and value != "")
    return (
        FormalPreprocessingDecision(
            operation="NORMALIZE_NEWLINES",
            applied=any("\r" in value for value in present),
        ),
        FormalPreprocessingDecision(
            operation="TRIM_TRAILING_WHITESPACE",
            applied=False,
        ),
        FormalPreprocessingDecision(
            operation="ENSURE_FINAL_NEWLINE",
            applied=any(not value.endswith(("\n", "\r")) for value in present),
        ),
    )


def formal_dataset_diagnostics(
    environment: FormalDatasetEnvironment,
    *,
    baseline: FormalDatasetDiagnosticBaseline | None = None,
) -> tuple[FormalDatasetDiagnostic, ...]:
    effective_baseline = baseline or FormalDatasetDiagnosticBaseline(
        lean_version=LEAN_VERSION,
        mathlib_revision=MATHLIB_COMMIT,
    )
    diagnostics = [
        FormalDatasetDiagnostic(
            code="EXECUTION_NOT_REQUESTED",
            message=(
                "The row was materialized but not executed; submit the normalized "
                "source to a compatible Lean project or verification capability."
            ),
        )
    ]
    if environment.lean_version != effective_baseline.lean_version:
        diagnostics.append(
            FormalDatasetDiagnostic(
                code="LEAN_VERSION_NOT_PINNED_RUNTIME",
                message=(
                    f"The row requires Lean {environment.lean_version}; Jacobian's "
                    f"pinned runtime is Lean {effective_baseline.lean_version}."
                ),
            )
        )
    if (
        environment.mathlib_revision is not None
        and environment.mathlib_revision != effective_baseline.mathlib_revision
    ):
        diagnostics.append(
            FormalDatasetDiagnostic(
                code="MATHLIB_REVISION_NOT_PINNED_RUNTIME",
                message=(
                    "The declared Mathlib revision differs from Jacobian's pinned "
                    "runtime; execution requires the declared project checkout."
                ),
            )
        )
    if not environment.project_files:
        diagnostics.append(
            FormalDatasetDiagnostic(
                code="PROJECT_FILES_UNDECLARED",
                message=(
                    "No project-file digests were supplied; materialization is "
                    "deterministic, but project compatibility is not established."
                ),
            )
        )
    return tuple(diagnostics)
