"""License-aware ingestion of external conjecture records."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.conjecture_ingestion import (
    ConjectureLicenseClass,
    ConjectureLicenseDecision,
    ExternalConjectureIngestArtifact,
    ExternalConjectureIngestOutput,
    ExternalConjectureIngestRequest,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.domains._examples import example
from jacobian.provider_runtime import jacobian_provider_runtime
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository

_TEXT_ALLOWED = frozenset(
    {
        ConjectureLicenseClass.CC0_1_0,
        ConjectureLicenseClass.CC_BY_4_0,
        ConjectureLicenseClass.APACHE_2_0,
        ConjectureLicenseClass.MIT,
    }
)


def _json_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonicalize_json(value)).hexdigest()


def _text_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_statement(value: str) -> str:
    lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized = "\n".join(line.rstrip() for line in lines).strip()
    return unicodedata.normalize("NFC", normalized)


def _license_decision(
    request: ExternalConjectureIngestRequest,
) -> tuple[ConjectureLicenseDecision, str]:
    license_class = request.source_license
    if request.statement is None:
        return (
            ConjectureLicenseDecision.METADATA_ONLY,
            "No statement text was supplied; metadata was retained without withholding.",
        )
    if license_class in _TEXT_ALLOWED:
        if (
            request.license_evidence_url is None
            or request.license_evidence_text is None
            or request.license_evidence_digest is None
        ):
            return (
                ConjectureLicenseDecision.METADATA_ONLY,
                (
                    f"{license_class.value} permits text publication, but the "
                    "record lacks a URL-and-digest license evidence binding."
                ),
            )
        return (
            ConjectureLicenseDecision.ALLOW_TEXT,
            f"{license_class.value} is in the registered text-publication allowlist.",
        )
    return (
        ConjectureLicenseDecision.METADATA_ONLY,
        (
            f"{license_class.value} is not in the registered text-publication "
            "allowlist; provenance and metadata may be retained but text is withheld."
        ),
    )


@dataclass(frozen=True, slots=True)
class ConjectureIngestionInstallation:
    semantics_uri: str
    artifact_schema_uri: str


class ExternalConjectureIngestAdapter:
    """Apply a registered license policy to one external conjecture record."""

    def __init__(
        self,
        store: ArtifactRepository,
        artifacts: ArtifactService,
        *,
        semantics_uri: str,
        artifact_schema_uri: str,
    ) -> None:
        self.store = store
        self.artifacts = artifacts
        self.semantics_uri = semantics_uri
        self.artifact_schema_uri = artifact_schema_uri
        self._descriptor = CapabilityDescriptor(
            capability_id="dataset.conjecture.ingest",
            version="1",
            title="Ingest one license-reviewed external conjecture",
            description=(
                "Apply a registered publication policy, bind provenance and hashes, "
                "and index permitted conjecture text or metadata only."
            ),
            provider="jacobian.conjecture-ingestion",
            provider_runtime=jacobian_provider_runtime(
                "jacobian.conjecture-ingestion",
                features=("license-policy", "metadata-withholding", "provenance"),
            ),
            input_schema=ExternalConjectureIngestRequest.model_json_schema(),
            output_schema=ExternalConjectureIngestOutput.model_json_schema(),
            tags=("conjecture", "dataset", "ingestion", "license", "provenance"),
            produced_artifact_types=(self.artifact_schema_uri,),
            invocation_examples=(
                example(
                    "openconjecture_cc_by",
                    "Ingest an OpenConjecture fixture whose text is CC-BY-4.0.",
                    {
                        "corpus_id": "davisrbr/openconjecture",
                        "corpus_revision": "fixture-revision-1",
                        "source_url": (
                            "https://huggingface.co/datasets/davisrbr/openconjecture"
                        ),
                        "item_id": "fixture-1",
                        "metadata": {
                            "title": "Fixture conjecture",
                            "domain": "number theory",
                        },
                        "statement": "Every fixture prime has the fixture property.",
                        "source_license": "CC-BY-4.0",
                        "license_evidence_url": (
                            "https://example.invalid/license/fixture-1"
                        ),
                        "license_evidence_text": "Creative Commons Attribution 4.0",
                        "license_evidence_digest": (
                            "sha256:14cf6e4efc51a33be0438483f0bc0d53963cedad"
                            "7406282e331b3f797779cc11"
                        ),
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = ExternalConjectureIngestRequest.model_validate(request.input)
        except ValidationError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_EXTERNAL_CONJECTURE_RECORD",
                    stage="request_validation",
                    message="The external conjecture record is invalid.",
                    hint="Provide pinned corpus, item, license, and provenance fields.",
                )
            ) from exc

        record_payload = validated.model_dump(
            mode="json",
            exclude={"expected_record_digest", "expected_content_digest"},
        )
        record_digest = _json_digest(record_payload)
        if (
            validated.expected_record_digest is not None
            and validated.expected_record_digest != record_digest
        ):
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="EXTERNAL_CONJECTURE_RECORD_DIGEST_MISMATCH",
                    stage="source_binding",
                    message="The record does not match expected_record_digest.",
                    hint="Re-fetch the pinned item before retrying ingestion.",
                )
            )

        normalized_statement = (
            _normalize_statement(validated.statement)
            if validated.statement is not None
            else None
        )
        content_digest = (
            _text_digest(normalized_statement)
            if normalized_statement is not None
            else None
        )
        if (
            validated.expected_content_digest is not None
            and validated.expected_content_digest != content_digest
        ):
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="EXTERNAL_CONJECTURE_CONTENT_DIGEST_MISMATCH",
                    stage="source_binding",
                    message="The statement does not match expected_content_digest.",
                    hint="Withhold changed content until its provenance is reviewed.",
                )
            )

        if (
            validated.license_evidence_text is not None
            and validated.license_evidence_digest is not None
            and _text_digest(validated.license_evidence_text)
            != validated.license_evidence_digest
        ):
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="EXTERNAL_CONJECTURE_LICENSE_DIGEST_MISMATCH",
                    stage="license_validation",
                    message="The license evidence does not match its declared digest.",
                    hint="Re-fetch and review the license evidence before ingestion.",
                )
            )

        decision, reason = _license_decision(validated)
        index_text = (
            decision is ConjectureLicenseDecision.ALLOW_TEXT
            and normalized_statement is not None
        )
        indexed_statement = normalized_statement if index_text else None
        indexed_content_digest = content_digest if index_text else None
        withheld_fields: tuple[Literal["statement"], ...] = (
            ("statement",)
            if normalized_statement is not None and not index_text
            else ()
        )
        status: Literal[
            "INDEXED",
            "METADATA_INDEXED_TEXT_WITHHELD",
            "METADATA_INDEXED_NO_TEXT",
        ]
        if index_text:
            status = "INDEXED"
        elif normalized_statement is not None:
            status = "METADATA_INDEXED_TEXT_WITHHELD"
        else:
            status = "METADATA_INDEXED_NO_TEXT"
        artifact_payload = ExternalConjectureIngestArtifact(
            corpus_id=validated.corpus_id,
            corpus_revision=validated.corpus_revision,
            source_url=validated.source_url,
            item_id=validated.item_id,
            metadata=validated.metadata,
            record_digest=record_digest,
            supplied_content_digest=content_digest,
            indexed_content_digest=indexed_content_digest,
            source_license=validated.source_license,
            license_evidence_url=validated.license_evidence_url,
            license_evidence_digest=validated.license_evidence_digest,
            policy_id=validated.policy_id,
            license_decision=decision,
            license_reason=reason,
            indexed_statement=indexed_statement,
            withheld_fields=withheld_fields,
            ingestion_status=status,
        )
        artifact = self.artifacts.put(
            schema_uri=self.artifact_schema_uri,
            semantics_uri=self.semantics_uri,
            payload=artifact_payload.model_dump(mode="json"),
            summary=_artifact_summary(validated, decision),
            producer_write=True,
        )
        output = ExternalConjectureIngestOutput(
            **artifact_payload.model_dump(mode="python"),
            artifact_uri=artifact.artifact_uri,
        )
        result = CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="one external conjecture record under one policy version",
                parameters={
                    "corpus_id": validated.corpus_id,
                    "corpus_revision": validated.corpus_revision,
                    "item_id": validated.item_id,
                    "record_digest": record_digest,
                    "policy_id": validated.policy_id,
                },
                artifact_uri=artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.COMPLETE,
                basis="the registered license policy was applied to the complete record",
                assurance_level=CapabilityAssuranceLevel.HEURISTIC,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.HEURISTIC,
                basis=(
                    "ingestion preserves a sourced conjecture for research only; "
                    "it does not establish truth, formal correspondence, or proof"
                ),
            ),
            artifact_uris=(artifact.artifact_uri,),
            provider=self.descriptor.provider,
            provider_digest=(
                self.descriptor.provider_runtime.digest
                if self.descriptor.provider_runtime is not None
                else None
            ),
        )
        return result


def install_conjecture_ingestion_capability(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
) -> tuple[ExternalConjectureIngestAdapter, ConjectureIngestionInstallation]:
    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.external-conjecture-ingestion",
        version="1",
        definition={
            "description": (
                "license-aware external conjecture indexing with text withholding"
            ),
            "policy_id": "jacobian.external-conjecture-publication/v1",
            "verification": (
                "none; ingestion is heuristic provenance and never verifies a claim"
            ),
        },
    )
    artifact_schema_uri = schemas.register_model(
        name="jacobian.external-conjecture-record",
        version="1",
        model=ExternalConjectureIngestArtifact,
        producer_only=True,
    )
    return (
        ExternalConjectureIngestAdapter(
            store,
            artifacts,
            semantics_uri=semantics_uri,
            artifact_schema_uri=artifact_schema_uri,
        ),
        ConjectureIngestionInstallation(
            semantics_uri=semantics_uri,
            artifact_schema_uri=artifact_schema_uri,
        ),
    )


def _artifact_summary(
    request: ExternalConjectureIngestRequest,
    decision: ConjectureLicenseDecision,
) -> str:
    prefix = f"{request.corpus_id} item "
    suffix = f": {decision.value}"
    available = max(0, 512 - len(prefix) - len(suffix))
    return f"{prefix}{request.item_id[:available]}{suffix}"
