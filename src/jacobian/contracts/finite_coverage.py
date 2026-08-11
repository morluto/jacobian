"""Typed contracts for bounded exactly-once finite coverage verification."""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Self

from pydantic import Field, StrictInt, StrictStr, model_validator

from jacobian.canonical import canonicalize_json
from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest
from jacobian.contracts.results import ContractModel

FiniteItem = StrictStr | StrictInt
FiniteCanonicalizerId = Literal[
    "finite.integer.decimal@1",
    "finite.string.nfc@1",
]
FiniteCanonicalKey = Sha256Digest
_MAX_ITEMS = 4096
_MAX_PAGES = 64
_MAX_PAGE_ITEMS = 1024


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonicalize_json(value)).hexdigest()


class FiniteCoveragePageInput(ContractModel):
    items: tuple[FiniteItem, ...] = Field(max_length=_MAX_PAGE_ITEMS)


class FiniteCoverageVerifyRequest(ContractModel):
    request_version: Literal["1"] = "1"
    canonicalizer_id: FiniteCanonicalizerId
    scope_items: tuple[FiniteItem, ...] = Field(
        min_length=1,
        max_length=_MAX_ITEMS,
    )
    pages: tuple[FiniteCoveragePageInput, ...] = Field(
        min_length=1,
        max_length=_MAX_PAGES,
    )

    @model_validator(mode="after")
    def require_typed_bounded_items(self) -> Self:
        expected_type = str if self.canonicalizer_id == "finite.string.nfc@1" else int
        items = (
            *self.scope_items,
            *(item for page in self.pages for item in page.items),
        )
        if any(type(item) is not expected_type for item in items):
            raise ValueError(
                "scope and archive items must match the canonicalizer item type"
            )
        if sum(len(page.items) for page in self.pages) > _MAX_ITEMS:
            raise ValueError("paged archive exceeds 4096 total items")
        return self


class FiniteCanonicalizerRegistration(ContractModel):
    registration_version: Literal["1"] = "1"
    canonicalizer_id: FiniteCanonicalizerId
    item_type: Literal["INTEGER", "STRING"]
    algorithm: Literal["DECIMAL_INTEGER", "NFC_STRING"]
    key_format: Literal["SHA256_RFC8785_TAGGED_VALUE"] = "SHA256_RFC8785_TAGGED_VALUE"
    specification_digest: Sha256Digest

    @model_validator(mode="after")
    def bind_registered_specification(self) -> Self:
        expected = {
            "finite.integer.decimal@1": ("INTEGER", "DECIMAL_INTEGER"),
            "finite.string.nfc@1": ("STRING", "NFC_STRING"),
        }[self.canonicalizer_id]
        if (self.item_type, self.algorithm) != expected:
            raise ValueError("canonicalizer ID, item type, and algorithm disagree")
        specification = {
            "canonicalizer_id": self.canonicalizer_id,
            "item_type": self.item_type,
            "algorithm": self.algorithm,
            "key_format": self.key_format,
        }
        if self.specification_digest != _digest(specification):
            raise ValueError("canonicalizer specification digest is invalid")
        return self


class FiniteCoverageScopeArtifact(ContractModel):
    scope_version: Literal["1"] = "1"
    canonicalizer_uri: ArtifactUri
    canonicalizer_object_digest: Sha256Digest
    canonicalizer_specification_digest: Sha256Digest
    canonicalizer_id: FiniteCanonicalizerId
    items: tuple[FiniteItem, ...] = Field(min_length=1, max_length=_MAX_ITEMS)
    canonical_keys: tuple[FiniteCanonicalKey, ...] = Field(
        min_length=1,
        max_length=_MAX_ITEMS,
    )
    scope_keys_digest: Sha256Digest

    @model_validator(mode="after")
    def require_aligned_unique_scope(self) -> Self:
        if len(self.items) != len(self.canonical_keys):
            raise ValueError("scope items and canonical keys must align")
        if len(set(self.canonical_keys)) != len(self.canonical_keys):
            raise ValueError("finite scope canonical keys must be unique")
        if self.scope_keys_digest != _digest(list(self.canonical_keys)):
            raise ValueError("scope key digest is invalid")
        return self


class FiniteCoveragePageArtifact(ContractModel):
    page_version: Literal["1"] = "1"
    page_index: StrictInt = Field(ge=0, lt=_MAX_PAGES)
    canonicalizer_uri: ArtifactUri
    canonicalizer_object_digest: Sha256Digest
    canonicalizer_id: FiniteCanonicalizerId
    items: tuple[FiniteItem, ...] = Field(max_length=_MAX_PAGE_ITEMS)
    canonical_keys: tuple[FiniteCanonicalKey, ...] = Field(max_length=_MAX_PAGE_ITEMS)
    items_digest: Sha256Digest

    @model_validator(mode="after")
    def require_aligned_page(self) -> Self:
        if len(self.items) != len(self.canonical_keys):
            raise ValueError("page items and canonical keys must align")
        if self.items_digest != _digest(
            {
                "items": list(self.items),
                "canonical_keys": list(self.canonical_keys),
            }
        ):
            raise ValueError("page items digest is invalid")
        return self


class FiniteCoveragePageBinding(ContractModel):
    page_index: StrictInt = Field(ge=0, lt=_MAX_PAGES)
    page_uri: ArtifactUri
    page_object_digest: Sha256Digest
    page_payload_digest: Sha256Digest
    items_digest: Sha256Digest
    item_count: StrictInt = Field(ge=0, le=_MAX_PAGE_ITEMS)


class FiniteCoverageArchiveArtifact(ContractModel):
    archive_version: Literal["1"] = "1"
    scope_uri: ArtifactUri
    scope_object_digest: Sha256Digest
    canonicalizer_uri: ArtifactUri
    canonicalizer_object_digest: Sha256Digest
    canonicalizer_specification_digest: Sha256Digest
    canonicalizer_id: FiniteCanonicalizerId
    page_bindings: tuple[FiniteCoveragePageBinding, ...] = Field(
        min_length=1,
        max_length=_MAX_PAGES,
    )
    total_item_count: StrictInt = Field(ge=0, le=_MAX_ITEMS)
    archive_digest: Sha256Digest

    @model_validator(mode="after")
    def require_contiguous_bound_pages(self) -> Self:
        if tuple(binding.page_index for binding in self.page_bindings) != tuple(
            range(len(self.page_bindings))
        ):
            raise ValueError("archive page indices must be contiguous from zero")
        if self.total_item_count != sum(
            binding.item_count for binding in self.page_bindings
        ):
            raise ValueError("archive total must equal bound page item counts")
        digest_payload = {
            "scope_uri": self.scope_uri,
            "scope_object_digest": self.scope_object_digest,
            "canonicalizer_uri": self.canonicalizer_uri,
            "canonicalizer_object_digest": self.canonicalizer_object_digest,
            "canonicalizer_specification_digest": (
                self.canonicalizer_specification_digest
            ),
            "canonicalizer_id": self.canonicalizer_id,
            "page_bindings": [
                binding.model_dump(mode="json") for binding in self.page_bindings
            ],
            "total_item_count": self.total_item_count,
        }
        if self.archive_digest != _digest(digest_payload):
            raise ValueError("archive digest is invalid")
        return self


class FiniteCoverageClaim(ContractModel):
    claim_version: Literal["1"] = "1"
    predicate: Literal["FINITE_EXACTLY_ONCE_COVERAGE"] = "FINITE_EXACTLY_ONCE_COVERAGE"
    scope_uri: ArtifactUri
    archive_uri: ArtifactUri
    canonicalizer_uri: ArtifactUri
    canonicalizer_id: FiniteCanonicalizerId
    scope_keys_digest: Sha256Digest
    archive_digest: Sha256Digest


class FiniteCoverageOccurrence(ContractModel):
    canonical_key: FiniteCanonicalKey
    page_index: StrictInt = Field(ge=0, lt=_MAX_PAGES)
    item_index: StrictInt = Field(ge=0, lt=_MAX_PAGE_ITEMS)


class FiniteCoverageDiagnostics(ContractModel):
    missing_keys: tuple[FiniteCanonicalKey, ...] = ()
    duplicate_keys: tuple[FiniteCanonicalKey, ...] = ()
    outside_keys: tuple[FiniteCanonicalKey, ...] = ()
    duplicate_occurrences: tuple[FiniteCoverageOccurrence, ...] = ()


class FiniteCoverageVerifyOutput(ContractModel):
    coverage_status: Literal["EXACTLY_ONCE", "INVALID"]
    conclusion: Literal["TRUE", "UNKNOWN"]
    canonicalizer_id: FiniteCanonicalizerId
    canonicalizer_uri: ArtifactUri
    scope_uri: ArtifactUri
    archive_uri: ArtifactUri
    page_uris: tuple[ArtifactUri, ...]
    claim_uri: ArtifactUri
    certificate_uri: ArtifactUri
    verification_record_uri: ArtifactUri | None = None
    diagnostics: FiniteCoverageDiagnostics
    scope_keys_digest: Sha256Digest
    archive_digest: Sha256Digest
    checker_id: CheckerUri
    detail: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def bind_verified_conclusion_shape(self) -> Self:
        if (self.coverage_status == "EXACTLY_ONCE") != (self.conclusion == "TRUE"):
            raise ValueError("only exactly-once coverage may carry a true conclusion")
        if self.conclusion == "TRUE" and self.verification_record_uri is None:
            raise ValueError("true coverage requires a verification record")
        if self.conclusion == "UNKNOWN" and self.verification_record_uri is not None:
            raise ValueError("unknown coverage cannot carry a verification record")
        return self
