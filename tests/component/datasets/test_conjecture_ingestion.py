from __future__ import annotations

from pathlib import Path

import pytest

from jacobian.artifacts import ArtifactService
from jacobian.capability_service import CapabilityInvocationError
from jacobian.conjecture_ingestion import install_conjecture_ingestion_capability
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository


def _adapter(tmp_path: Path):
    store = ArtifactRepository(tmp_path)
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    adapter, _ = install_conjecture_ingestion_capability(store, schemas, artifacts)
    return adapter


def _request(*, license_id: str = "CC-BY-4.0") -> dict[str, object]:
    return {
        "corpus_id": "davisrbr/openconjecture",
        "corpus_revision": "f665b46c93a6a1d505ef9109417902d7b2973ab8",
        "source_url": "https://huggingface.co/datasets/davisrbr/openconjecture",
        "item_id": "fixture-1",
        "metadata": {
            "title": "Fixture conjecture",
            "domain": "number theory",
            "source_name": "Fixture source",
            "source_item_url": "https://example.invalid/conjecture/fixture-1",
        },
        "statement": "Every fixture prime has the fixture property.  \r\n",
        "source_license": license_id,
        "license_evidence_url": "https://example.invalid/license/fixture-1",
        "license_evidence_text": "Creative Commons Attribution 4.0",
        "license_evidence_digest": (
            "sha256:14cf6e4efc51a33be0438483f0bc0d53963cedad7406282e331b3f797779cc11"
        ),
    }


def test_allowed_record_indexes_normalized_text_as_heuristic(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.conjecture.ingest",
            input=_request(),
        )
    )

    assert result.output["license_decision"] == "ALLOW_TEXT"
    assert result.output["ingestion_status"] == "INDEXED"
    assert (
        result.output["indexed_statement"]
        == "Every fixture prime has the fixture property."
    )
    assert (
        result.output["supplied_content_digest"]
        == (result.output["indexed_content_digest"])
    )
    assert result.output["assurance"] == "HEURISTIC"
    assert result.output["verification"] == "UNVERIFIED"
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC


def test_ingestion_persists_normalized_provenance_urls(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    payload = _request()
    payload["source_url"] = " https://example.invalid/source path "
    payload["license_evidence_url"] = " https://example.invalid/license path "
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    metadata["source_item_url"] = " https://example.invalid/item path "

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.conjecture.ingest",
            input=payload,
        )
    )
    stored = adapter.store.get(result.output["artifact_uri"])

    assert stored.payload["source_url"] == "https://example.invalid/source%20path"
    assert (
        stored.payload["license_evidence_url"]
        == "https://example.invalid/license%20path"
    )
    assert (
        stored.payload["metadata"]["source_item_url"]
        == "https://example.invalid/item%20path"
    )


@pytest.mark.parametrize("source_item_url", ["not a URL", "   "])
def test_ingestion_rejects_invalid_source_item_urls(
    tmp_path: Path,
    source_item_url: str,
) -> None:
    adapter = _adapter(tmp_path)
    payload = _request()
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    metadata["source_item_url"] = source_item_url

    with pytest.raises(CapabilityInvocationError):
        adapter.invoke(
            CapabilityRequest(
                capability_id="dataset.conjecture.ingest",
                input=payload,
            )
        )


def test_ingestion_nfc_normalizes_statement_before_digesting(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    payload = _request()
    payload["statement"] = "Cafe\u0301"

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.conjecture.ingest",
            input=payload,
        )
    )

    assert result.output["indexed_statement"] == "Café"
    assert (
        result.output["indexed_content_digest"]
        == result.output["supplied_content_digest"]
    )


@pytest.mark.parametrize(
    "license_id",
    ["CC-BY-NC-4.0", "CC-BY-ND-4.0", "RESTRICTED", "PROPRIETARY"],
)
def test_restricted_record_withholds_text_but_retains_metadata(
    tmp_path: Path,
    license_id: str,
) -> None:
    adapter = _adapter(tmp_path)

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.conjecture.ingest",
            input=_request(license_id=license_id),
        )
    )

    assert result.output["license_decision"] == "METADATA_ONLY"
    assert result.output["indexed_statement"] is None
    assert result.output["indexed_content_digest"] is None
    assert result.output["withheld_fields"] == ["statement"]
    assert result.output["metadata"]["title"] == "Fixture conjecture"
    assert result.output["supplied_content_digest"].startswith("sha256:")


@pytest.mark.parametrize("license_id", ["MISSING", "UNKNOWN"])
def test_missing_or_unclear_license_withholds_text(
    tmp_path: Path,
    license_id: str,
) -> None:
    adapter = _adapter(tmp_path)

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.conjecture.ingest",
            input=_request(license_id=license_id),
        )
    )

    assert result.output["license_decision"] == "METADATA_ONLY"
    assert result.output["indexed_statement"] is None
    assert license_id in result.output["license_reason"]


def test_allowed_license_without_evidence_withholds_text(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    payload = _request()
    payload["license_evidence_url"] = None
    payload["license_evidence_text"] = None
    payload["license_evidence_digest"] = None

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.conjecture.ingest",
            input=payload,
        )
    )

    assert result.output["license_decision"] == "METADATA_ONLY"
    assert "lacks a URL-and-digest" in result.output["license_reason"]


def test_changed_content_fails_expected_digest_binding(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    payload = _request()
    payload["expected_content_digest"] = "sha256:" + "0" * 64

    with pytest.raises(CapabilityInvocationError) as exc_info:
        adapter.invoke(
            CapabilityRequest(
                capability_id="dataset.conjecture.ingest",
                input=payload,
            )
        )

    assert (
        exc_info.value.diagnostic.code == "EXTERNAL_CONJECTURE_CONTENT_DIGEST_MISMATCH"
    )


def test_tampered_record_fails_expected_digest_binding(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    payload = _request()
    payload["expected_record_digest"] = "sha256:" + "f" * 64

    with pytest.raises(CapabilityInvocationError) as exc_info:
        adapter.invoke(
            CapabilityRequest(
                capability_id="dataset.conjecture.ingest",
                input=payload,
            )
        )

    assert (
        exc_info.value.diagnostic.code == "EXTERNAL_CONJECTURE_RECORD_DIGEST_MISMATCH"
    )


def test_tampered_license_evidence_digest_is_rejected(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    payload = _request()
    payload["license_evidence_digest"] = "sha256:" + "0" * 64

    with pytest.raises(CapabilityInvocationError) as exc_info:
        adapter.invoke(
            CapabilityRequest(
                capability_id="dataset.conjecture.ingest",
                input=payload,
            )
        )

    assert (
        exc_info.value.diagnostic.code == "EXTERNAL_CONJECTURE_LICENSE_DIGEST_MISMATCH"
    )


def test_metadata_only_source_without_statement_is_retained(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    payload = _request(license_id="MISSING")
    payload["statement"] = None

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.conjecture.ingest",
            input=payload,
        )
    )

    assert result.output["supplied_content_digest"] is None
    assert result.output["withheld_fields"] == []
    assert result.output["ingestion_status"] == "METADATA_INDEXED_NO_TEXT"
    assert result.output["metadata"]["source_item_url"].endswith("fixture-1")


def test_allowlisted_source_without_statement_uses_no_text_status(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    payload = _request()
    payload["statement"] = None

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.conjecture.ingest",
            input=payload,
        )
    )

    assert result.output["license_decision"] == "METADATA_ONLY"
    assert result.output["ingestion_status"] == "METADATA_INDEXED_NO_TEXT"
    assert result.output["indexed_statement"] is None
    assert result.output["withheld_fields"] == []


def test_generic_artifact_write_rejects_policy_bypass(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.conjecture.ingest",
            input=_request(license_id="PROPRIETARY"),
        )
    )
    stored = adapter.store.get(result.output["artifact_uri"])
    tampered = dict(stored.payload)
    statement = "restricted text"
    digest = "sha256:7c0c7787821e7fbbba7a1800d41c046eb55dc2f6758cec309c2c28fe9c7b24f4"
    tampered.update(
        {
            "license_decision": "ALLOW_TEXT",
            "indexed_statement": statement,
            "supplied_content_digest": digest,
            "indexed_content_digest": digest,
            "withheld_fields": [],
            "ingestion_status": "INDEXED",
        }
    )

    with pytest.raises(ValueError, match="producer-only"):
        adapter.artifacts.put(
            schema_uri=adapter.artifact_schema_uri,
            semantics_uri=adapter.semantics_uri,
            payload=tampered,
        )


def test_maximum_identifiers_keep_summary_within_store_limit(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    payload = _request()
    payload["corpus_id"] = "c" * 256
    payload["item_id"] = "i" * 512

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.conjecture.ingest",
            input=payload,
        )
    )
    stored = adapter.store.get(result.output["artifact_uri"])

    assert len(stored.manifest.summary) <= 512
