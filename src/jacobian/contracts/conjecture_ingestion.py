"""Contracts for license-aware external conjecture ingestion."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from jacobian.contracts.common import ArtifactUri, Sha256Digest
from jacobian.contracts.results import ContractModel
from jacobian.contracts.urls import normalize_http_url


class ConjectureLicenseClass(StrEnum):
    CC0_1_0 = "CC0-1.0"
    CC_BY_4_0 = "CC-BY-4.0"
    APACHE_2_0 = "Apache-2.0"
    MIT = "MIT"
    CC_BY_NC_4_0 = "CC-BY-NC-4.0"
    CC_BY_ND_4_0 = "CC-BY-ND-4.0"
    RESTRICTED = "RESTRICTED"
    PROPRIETARY = "PROPRIETARY"
    UNKNOWN = "UNKNOWN"
    MISSING = "MISSING"


class ConjectureLicenseDecision(StrEnum):
    ALLOW_TEXT = "ALLOW_TEXT"
    METADATA_ONLY = "METADATA_ONLY"


class ExternalConjectureMetadata(ContractModel):
    title: str = Field(min_length=1, max_length=2_000)
    domain: str | None = Field(default=None, max_length=256)
    source_name: str | None = Field(default=None, max_length=512)
    source_item_url: str | None = Field(default=None, max_length=2_000)

    @field_validator("source_item_url", mode="before")
    @classmethod
    def normalize_source_item_url(cls, value: str | None) -> str | None:
        return (
            _normalize_http_url(value, "source item URL") if value is not None else None
        )

    @model_validator(mode="after")
    def require_nonblank_title(self) -> Self:
        if not self.title.strip():
            raise ValueError("title must not be blank")
        return self


class ExternalConjectureIngestRequest(ContractModel):
    corpus_id: str = Field(min_length=1, max_length=256)
    corpus_revision: str = Field(min_length=7, max_length=128)
    source_url: str = Field(min_length=1, max_length=2_000)
    item_id: str = Field(min_length=1, max_length=512)
    metadata: ExternalConjectureMetadata
    statement: str | None = Field(default=None, min_length=1, max_length=120_000)
    source_license: ConjectureLicenseClass
    license_evidence_url: str | None = Field(default=None, max_length=2_000)
    license_evidence_text: str | None = Field(default=None, max_length=20_000)
    license_evidence_digest: Sha256Digest | None = None
    policy_id: Literal["jacobian.external-conjecture-publication/v1"] = (
        "jacobian.external-conjecture-publication/v1"
    )
    expected_record_digest: Sha256Digest | None = None
    expected_content_digest: Sha256Digest | None = None

    @field_validator("source_url", mode="before")
    @classmethod
    def normalize_source_url(cls, value: str) -> str:
        return _normalize_http_url(value, "source_url")

    @field_validator("license_evidence_url", mode="before")
    @classmethod
    def normalize_license_evidence_url(cls, value: str | None) -> str | None:
        return (
            _normalize_http_url(value, "license evidence URL")
            if value is not None
            else None
        )

    @model_validator(mode="after")
    def require_complete_nonblank_evidence(self) -> Self:
        for field_name in (
            "corpus_id",
            "corpus_revision",
            "item_id",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be blank")
        evidence = (
            self.license_evidence_url,
            self.license_evidence_text,
            self.license_evidence_digest,
        )
        if any(value is not None for value in evidence) and not all(
            value is not None for value in evidence
        ):
            raise ValueError("license evidence URL, text, and digest must be complete")
        if (
            self.license_evidence_text is not None
            and not self.license_evidence_text.strip()
        ):
            raise ValueError("license evidence text must not be blank")
        if self.statement is not None and not self.statement.strip():
            raise ValueError("statement must not be blank")
        return self


class ExternalConjectureIngestArtifact(ContractModel):
    artifact_version: Literal["1"] = "1"
    corpus_id: str
    corpus_revision: str
    source_url: str
    item_id: str
    metadata: ExternalConjectureMetadata
    record_digest: Sha256Digest
    supplied_content_digest: Sha256Digest | None
    indexed_content_digest: Sha256Digest | None
    source_license: ConjectureLicenseClass
    license_evidence_url: str | None
    license_evidence_digest: Sha256Digest | None
    policy_id: Literal["jacobian.external-conjecture-publication/v1"]
    license_decision: ConjectureLicenseDecision
    license_reason: str
    indexed_statement: str | None
    withheld_fields: tuple[Literal["statement"], ...]
    ingestion_status: Literal[
        "INDEXED",
        "METADATA_INDEXED_TEXT_WITHHELD",
        "METADATA_INDEXED_NO_TEXT",
    ]
    assurance: Literal["HEURISTIC"] = "HEURISTIC"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

    @model_validator(mode="after")
    def enforce_publication_invariants(self) -> Self:
        if self.license_decision is ConjectureLicenseDecision.ALLOW_TEXT:
            _validate_text_publication(self)
            return self

        _validate_metadata_only_publication(self)
        return self


def _validate_text_publication(artifact: ExternalConjectureIngestArtifact) -> None:
    text_allowed = artifact.source_license in {
        ConjectureLicenseClass.CC0_1_0,
        ConjectureLicenseClass.CC_BY_4_0,
        ConjectureLicenseClass.APACHE_2_0,
        ConjectureLicenseClass.MIT,
    }
    if not text_allowed:
        raise ValueError("ALLOW_TEXT requires an allowlisted source license")
    if (
        artifact.license_evidence_url is None
        or artifact.license_evidence_digest is None
    ):
        raise ValueError("ALLOW_TEXT requires bound license evidence")
    if artifact.indexed_statement is None:
        raise ValueError("ALLOW_TEXT requires indexed_statement")
    expected_digest = _text_digest(artifact.indexed_statement)
    if artifact.supplied_content_digest != expected_digest:
        raise ValueError("supplied_content_digest must bind indexed_statement")
    if artifact.indexed_content_digest != expected_digest:
        raise ValueError("indexed_content_digest must bind indexed_statement")
    if artifact.withheld_fields:
        raise ValueError("indexed text cannot also be withheld")
    if artifact.ingestion_status != "INDEXED":
        raise ValueError("ALLOW_TEXT requires INDEXED status")


def _validate_metadata_only_publication(
    artifact: ExternalConjectureIngestArtifact,
) -> None:
    if (
        artifact.indexed_statement is not None
        or artifact.indexed_content_digest is not None
    ):
        raise ValueError("METADATA_ONLY cannot contain indexed text")
    if artifact.supplied_content_digest is None:
        if artifact.withheld_fields:
            raise ValueError("no-text records cannot report withheld fields")
        if artifact.ingestion_status != "METADATA_INDEXED_NO_TEXT":
            raise ValueError("absent text requires METADATA_INDEXED_NO_TEXT")
        return
    if artifact.withheld_fields != ("statement",):
        raise ValueError("supplied metadata-only text must be withheld")
    if artifact.ingestion_status != "METADATA_INDEXED_TEXT_WITHHELD":
        raise ValueError("withheld text requires the withheld status")


class ExternalConjectureIngestOutput(ExternalConjectureIngestArtifact):
    artifact_uri: ArtifactUri


def _text_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_http_url(value: str, label: str) -> str:
    return normalize_http_url(value, label=label)
