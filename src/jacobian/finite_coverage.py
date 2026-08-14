"""Bounded exactly-once finite coverage verification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.checker_authorization import authorize_checker_operation
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.finite_coverage import (
    FiniteCanonicalizerId,
    FiniteCanonicalizerRegistration,
    FiniteCoverageArchiveArtifact,
    FiniteCoverageClaim,
    FiniteCoverageDiagnostics,
    FiniteCoverageOccurrence,
    FiniteCoveragePageArtifact,
    FiniteCoveragePageBinding,
    FiniteCoverageScopeArtifact,
    FiniteCoverageVerifyOutput,
    FiniteCoverageVerifyRequest,
)
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiagnostic,
    OperationExample,
    OperationRequest,
)
from jacobian.contracts.results import Conclusion, ExecutionStatus
from jacobian.operation_adapters import OperationAdapter, parse_operation_input
from jacobian.operation_catalog import OperationCatalog, OperationCatalogError
from jacobian.operation_errors import (
    OperationInvocationError,
    enriched_invalid_request,
)
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed, Failed
from jacobian.provider_runtime import known_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService


@dataclass(frozen=True, slots=True)
class FiniteCoverageInstallation:
    semantics_uri: str
    canonicalizer_schema_uri: str
    scope_schema_uri: str
    page_schema_uri: str
    archive_schema_uri: str
    claim_schema_uri: str
    certificate_schema_uri: str
    canonicalizer_uris: dict[str, str]
    checker_id: str | None


_CANONICALIZER_SPECS: dict[str, dict[str, str]] = {
    "finite.integer.decimal@1": {
        "item_type": "INTEGER",
        "algorithm": "DECIMAL_INTEGER",
        "key_format": "SHA256_RFC8785_TAGGED_VALUE",
    },
    "finite.string.nfc@1": {
        "item_type": "STRING",
        "algorithm": "NFC_STRING",
        "key_format": "SHA256_RFC8785_TAGGED_VALUE",
    },
}


def register_finite_coverage_resources(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
) -> FiniteCoverageInstallation:
    """Register passive finite-coverage contracts and canonicalizers."""

    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.finite-exactly-once-coverage",
        version="1",
        definition={
            "domain": "bounded explicit finite scopes and paged archives",
            "coverage": "every canonical scope key appears exactly once",
            "maximum_scope_items": 4096,
            "maximum_pages": 64,
            "maximum_page_items": 1024,
        },
    )
    canonicalizer_schema_uri = schemas.register_model(
        name="jacobian.finite-canonicalizer-registration",
        version="1",
        model=FiniteCanonicalizerRegistration,
    )
    scope_schema_uri = schemas.register_model(
        name="jacobian.finite-coverage-scope",
        version="1",
        model=FiniteCoverageScopeArtifact,
    )
    page_schema_uri = schemas.register_model(
        name="jacobian.finite-coverage-page",
        version="1",
        model=FiniteCoveragePageArtifact,
    )
    archive_schema_uri = schemas.register_model(
        name="jacobian.finite-coverage-archive",
        version="1",
        model=FiniteCoverageArchiveArtifact,
    )
    claim_schema_uri = schemas.register_model(
        name="jacobian.finite-coverage-claim",
        version="1",
        model=FiniteCoverageClaim,
    )
    certificate_schema_uri = schemas.register(
        name="jacobian.certificate-envelope",
        version="1",
        schema=model_schema(CertificateEnvelope),
    )
    canonicalizer_uris: dict[str, str] = {}
    for canonicalizer_id, spec in sorted(_CANONICALIZER_SPECS.items()):
        specification_digest = _digest({"canonicalizer_id": canonicalizer_id, **spec})
        registration = FiniteCanonicalizerRegistration.model_validate(
            {
                "canonicalizer_id": canonicalizer_id,
                "specification_digest": specification_digest,
                **spec,
            }
        )
        stored = artifacts.put(
            schema_uri=canonicalizer_schema_uri,
            semantics_uri=semantics_uri,
            payload=registration.model_dump(mode="json"),
            summary=f"registered finite canonicalizer {canonicalizer_id}",
        )
        canonicalizer_uris[canonicalizer_id] = stored.artifact_uri
    return FiniteCoverageInstallation(
        semantics_uri=semantics_uri,
        canonicalizer_schema_uri=canonicalizer_schema_uri,
        scope_schema_uri=scope_schema_uri,
        page_schema_uri=page_schema_uri,
        archive_schema_uri=archive_schema_uri,
        claim_schema_uri=claim_schema_uri,
        certificate_schema_uri=certificate_schema_uri,
        canonicalizer_uris=canonicalizer_uris,
        checker_id=None,
    )


def bind_selected_finite_coverage(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    catalog: OperationCatalog,
) -> OperationAdapter[Any]:
    """Bind finite coverage from persisted checker authority."""

    operation_id = "finite.coverage.verify"
    binding = catalog.checker_binding(operation_id)
    if binding is None:
        raise OperationCatalogError(
            f"checker binding is missing; run `jacobian update`: {operation_id}"
        )
    checkers.require_catalog_binding(
        binding.checker_id,
        implementation_digest=binding.manifest_digest,
    )
    installation = replace(
        register_finite_coverage_resources(store, schemas, artifacts),
        checker_id=binding.checker_id,
    )
    return FiniteCoverageVerifyAdapter(
        store=store,
        artifacts=artifacts,
        verification=verification,
        installation=installation,
    )


def install_finite_coverage(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[OperationAdapter[Any] | None, FiniteCoverageInstallation]:
    """Register v1 finite-coverage artifacts and optionally authorize replay."""

    resources = register_finite_coverage_resources(store, schemas, artifacts)

    checker_id = authorize_checker_operation(
        checkers,
        CheckerOperation(
            name="finite exactly-once paged coverage checker",
            entrypoint=("jacobian_checkers.finite_coverage:check_finite_coverage"),
            evidence_kind=EvidenceKind.CERTIFICATE,
            format_id="finite.coverage",
            format_version="1",
            claim_schema_uris=(resources.claim_schema_uri,),
            semantics_uris=(resources.semantics_uri,),
            candidate_schema_uris=(resources.archive_schema_uri,),
            reason=(
                "operator requested independent standard-library replay of "
                "every canonical scope and archive item"
            ),
        ),
        authorize=authorize_checker,
    ).checker_id
    installation = replace(resources, checker_id=checker_id)
    adapter = (
        FiniteCoverageVerifyAdapter(
            store=store,
            artifacts=artifacts,
            verification=verification,
            installation=installation,
        )
        if checker_id is not None
        else None
    )
    return adapter, installation


class FiniteCoverageVerifyAdapter:
    """Materialize and independently verify one bounded paged archive."""

    def __init__(
        self,
        *,
        store: ArtifactRepository,
        artifacts: ArtifactService,
        verification: VerificationService,
        installation: FiniteCoverageInstallation,
    ) -> None:
        checker_id = installation.checker_id
        if checker_id is None:
            raise ValueError("finite coverage checker is not authorized")
        self.store = store
        self.artifacts = artifacts
        self.verification = verification
        self.installation = installation
        self._descriptor = OperationDescriptor(
            operation_id="finite.coverage.verify",
            version="1",
            title="Verify exactly-once coverage of a finite paged archive",
            description=(
                "Materialize one typed finite scope and bounded page archive, then "
                "independently replay canonical-key coverage exactly once."
            ),
            provider="jacobian.finite",
            provider_runtime=known_provider_runtime(
                "jacobian.finite",
                features=(
                    "finite-coverage",
                    "paged-archive",
                    "registered-canonicalizers",
                    "exactly-once",
                    "clean-process-checker",
                ),
                checker_ids=(checker_id,),
            ),
            input_schema=model_schema(FiniteCoverageVerifyRequest),
            output_schema=model_schema(FiniteCoverageVerifyOutput),
            tags=("finite", "coverage", "verification", "paged-archive"),
            examples=(
                OperationExample(
                    name="two_page_exact_coverage",
                    description=(
                        "Verify that two pages cover a three-item finite scope "
                        "exactly once."
                    ),
                    input=FiniteCoverageVerifyRequest.model_validate(
                        {
                            "canonicalizer_id": "finite.string.nfc@1",
                            "scope_items": ["alpha", "beta", "gamma"],
                            "pages": [
                                {"items": ["alpha"]},
                                {"items": ["beta", "gamma"]},
                            ],
                        }
                    ).model_dump(mode="json"),
                ),
            ),
        )

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> FiniteCoverageVerifyRequest:
        try:
            return parse_operation_input(FiniteCoverageVerifyRequest, request.input)
        except ValidationError as exc:
            raise OperationInvocationError(
                enriched_invalid_request(
                    OperationDiagnostic(
                        code="INVALID_FINITE_COVERAGE_REQUEST",
                        stage="request_validation",
                        message="The complete finite-coverage request is invalid.",
                    ),
                    exc,
                )
            ) from exc

    def invoke(self, validated: FiniteCoverageVerifyRequest) -> OperationProjection:
        canonicalizer_id = validated.canonicalizer_id
        canonicalizer_uri = self.installation.canonicalizer_uris[canonicalizer_id]
        canonicalizer = self.store.get(canonicalizer_uri)
        registration = FiniteCanonicalizerRegistration.model_validate(
            canonicalizer.payload
        )
        scope_keys = tuple(
            _canonical_key(canonicalizer_id, item) for item in validated.scope_items
        )
        if len(set(scope_keys)) != len(scope_keys):
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="DUPLICATE_FINITE_SCOPE_KEY",
                    stage="scope_canonicalization",
                    message=(
                        "The finite scope contains items that collide under the "
                        "selected registered canonicalizer."
                    ),
                    path="scope_items",
                    hint="Remove duplicate or canonically equivalent scope items.",
                )
            )
        scope_payload = FiniteCoverageScopeArtifact(
            canonicalizer_uri=canonicalizer_uri,
            canonicalizer_object_digest=canonicalizer.manifest.object_digest,
            canonicalizer_specification_digest=registration.specification_digest,
            canonicalizer_id=canonicalizer_id,
            items=validated.scope_items,
            canonical_keys=scope_keys,
            scope_keys_digest=_digest(list(scope_keys)),
        )
        scope = self.artifacts.put(
            schema_uri=self.installation.scope_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=scope_payload.model_dump(mode="json"),
            parents=(canonicalizer_uri,),
            summary="typed finite coverage scope",
        )

        page_uris: list[str] = []
        page_bindings: list[FiniteCoveragePageBinding] = []
        page_payloads: list[FiniteCoveragePageArtifact] = []
        for page_index, page_input in enumerate(validated.pages):
            keys = tuple(
                _canonical_key(canonicalizer_id, item) for item in page_input.items
            )
            items_digest = _digest(
                {"items": list(page_input.items), "canonical_keys": list(keys)}
            )
            page_payload = FiniteCoveragePageArtifact(
                page_index=page_index,
                canonicalizer_uri=canonicalizer_uri,
                canonicalizer_object_digest=canonicalizer.manifest.object_digest,
                canonicalizer_id=canonicalizer_id,
                items=page_input.items,
                canonical_keys=keys,
                items_digest=items_digest,
            )
            page = self.artifacts.put(
                schema_uri=self.installation.page_schema_uri,
                semantics_uri=self.installation.semantics_uri,
                payload=page_payload.model_dump(mode="json"),
                parents=(canonicalizer_uri,),
                summary=f"finite coverage archive page {page_index}",
            )
            stored_page = self.store.get(page.artifact_uri)
            page_uris.append(page.artifact_uri)
            page_payloads.append(page_payload)
            page_bindings.append(
                FiniteCoveragePageBinding(
                    page_index=page_index,
                    page_uri=page.artifact_uri,
                    page_object_digest=stored_page.manifest.object_digest,
                    page_payload_digest=stored_page.manifest.payload_digest,
                    items_digest=items_digest,
                    item_count=len(page_input.items),
                )
            )

        stored_scope = self.store.get(scope.artifact_uri)
        archive_digest_payload = {
            "scope_uri": scope.artifact_uri,
            "scope_object_digest": stored_scope.manifest.object_digest,
            "canonicalizer_uri": canonicalizer_uri,
            "canonicalizer_object_digest": canonicalizer.manifest.object_digest,
            "canonicalizer_specification_digest": registration.specification_digest,
            "canonicalizer_id": canonicalizer_id,
            "page_bindings": [
                binding.model_dump(mode="json") for binding in page_bindings
            ],
            "total_item_count": sum(len(page.items) for page in validated.pages),
        }
        archive_payload = FiniteCoverageArchiveArtifact(
            scope_uri=scope.artifact_uri,
            scope_object_digest=stored_scope.manifest.object_digest,
            canonicalizer_uri=canonicalizer_uri,
            canonicalizer_object_digest=canonicalizer.manifest.object_digest,
            canonicalizer_specification_digest=registration.specification_digest,
            canonicalizer_id=canonicalizer_id,
            page_bindings=tuple(page_bindings),
            total_item_count=sum(len(page.items) for page in validated.pages),
            archive_digest=_digest(archive_digest_payload),
        )
        archive = self.artifacts.put(
            schema_uri=self.installation.archive_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=archive_payload.model_dump(mode="json"),
            parents=(scope.artifact_uri, canonicalizer_uri, *page_uris),
            summary="bounded finite coverage archive manifest",
        )
        claim_payload = FiniteCoverageClaim(
            scope_uri=scope.artifact_uri,
            archive_uri=archive.artifact_uri,
            canonicalizer_uri=canonicalizer_uri,
            canonicalizer_id=canonicalizer_id,
            scope_keys_digest=scope_payload.scope_keys_digest,
            archive_digest=archive_payload.archive_digest,
        )
        claim = self.artifacts.put(
            schema_uri=self.installation.claim_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=claim_payload.model_dump(mode="json"),
            parents=(scope.artifact_uri, archive.artifact_uri, canonicalizer_uri),
            summary="finite exactly-once coverage claim",
        )
        diagnostics = _coverage_diagnostics(scope_keys, page_payloads)
        stored_archive = self.store.get(archive.artifact_uri)
        semantics = self.store.get(self.installation.semantics_uri)
        certificate_payload = {
            "relation_id": "finite.relation.covers-exactly-once",
            "obligation_uri": claim.artifact_uri,
            "canonicalizer_uri": canonicalizer_uri,
            "canonicalizer_object_digest": canonicalizer.manifest.object_digest,
            "canonicalizer_specification_digest": registration.specification_digest,
            "scope_keys_digest": scope_payload.scope_keys_digest,
            "archive_digest": archive_payload.archive_digest,
            "page_binding_digest": _digest(
                [
                    binding.model_dump(mode="json")
                    for binding in archive_payload.page_bindings
                ]
            ),
        }
        envelope = CertificateEnvelope(
            certificate_type="finite.coverage",
            format_version="1",
            bindings=EvidenceBindings(
                claim_digest=self.store.get(claim.artifact_uri).manifest.object_digest,
                semantics_digest=semantics.manifest.object_digest,
                candidate_digest=stored_archive.manifest.object_digest,
                scope_digest=stored_scope.manifest.object_digest,
            ),
            payload_digest=_digest(certificate_payload),
            payload=certificate_payload,
        )
        certificate = self.artifacts.put(
            schema_uri=self.installation.certificate_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=envelope.model_dump(mode="json"),
            parents=(
                claim.artifact_uri,
                archive.artifact_uri,
                scope.artifact_uri,
                canonicalizer_uri,
                *page_uris,
            ),
            summary="finite exactly-once coverage replay certificate",
        )
        checker_id = self.installation.checker_id
        if checker_id is None:
            raise RuntimeError("checker is not installed")
        checked = self.verification.verify_certificate(
            certificate_uri=certificate.artifact_uri,
            checker_id=checker_id,
            include_artifact_metadata=True,
            supporting_artifact_uris=(canonicalizer_uri, *page_uris),
        )
        verified = (
            checked.execution.status is ExecutionStatus.COMPLETED
            and checked.conclusion is Conclusion.TRUE
            and checked.verification_record_uri is not None
        )
        detail = checked.execution.detail
        if detail is None and checked.input.errors:
            detail = checked.input.errors[0]
        if detail is None:
            detail = (
                "the authorized checker accepted exactly-once finite coverage"
                if verified
                else "the archive did not establish exactly-once finite coverage"
            )
        output = FiniteCoverageVerifyOutput(
            coverage_status="EXACTLY_ONCE" if verified else "INVALID",
            conclusion="TRUE" if verified else "UNKNOWN",
            canonicalizer_id=canonicalizer_id,
            canonicalizer_uri=canonicalizer_uri,
            scope_uri=scope.artifact_uri,
            archive_uri=archive.artifact_uri,
            page_uris=tuple(page_uris),
            claim_uri=claim.artifact_uri,
            certificate_uri=certificate.artifact_uri,
            verification_record_uri=(
                checked.verification_record_uri if verified else None
            ),
            diagnostics=diagnostics,
            scope_keys_digest=scope_payload.scope_keys_digest,
            archive_digest=archive_payload.archive_digest,
            checker_id=checker_id,
            detail=detail,
        )
        record_uri = checked.verification_record_uri if verified else None
        artifact_uris = [
            canonicalizer_uri,
            scope.artifact_uri,
            *page_uris,
            archive.artifact_uri,
            claim.artifact_uri,
            certificate.artifact_uri,
        ]
        if record_uri is not None:
            artifact_uris.append(record_uri)
        terminal = (
            Completed(
                value=output,
                runtime_ms=checked.execution.runtime_ms,
                detail=checked.execution.detail,
            )
            if checked.execution.status is ExecutionStatus.COMPLETED
            else Failed(
                status=checked.execution.status,
                runtime_ms=checked.execution.runtime_ms,
                diagnostic=OperationDiagnostic(
                    code="FINITE_COVERAGE_VERIFICATION_FAILED",
                    stage="finite_coverage_verification",
                    message=checked.execution.detail or detail,
                ),
            )
        )
        return OperationProjection(
            operation_id=self.descriptor.operation_id,
            version=self.descriptor.version,
            terminal=terminal,
            publication=PublishedOperation(
                output=output if isinstance(terminal, Completed) else None,
                artifact_uris=tuple(artifact_uris),
            ),
            verification_record_uri=record_uri,
        )


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonicalize_json(value)).hexdigest()


def _canonical_key(canonicalizer_id: FiniteCanonicalizerId, item: Any) -> str:
    return _digest({"canonicalizer_id": canonicalizer_id, "value": item})


def _coverage_diagnostics(
    scope_keys: tuple[str, ...],
    pages: list[FiniteCoveragePageArtifact],
) -> FiniteCoverageDiagnostics:
    scope = set(scope_keys)
    occurrences: dict[str, list[FiniteCoverageOccurrence]] = {}
    for page in pages:
        for item_index, key in enumerate(page.canonical_keys):
            occurrences.setdefault(key, []).append(
                FiniteCoverageOccurrence(
                    canonical_key=key,
                    page_index=page.page_index,
                    item_index=item_index,
                )
            )
    present = set(occurrences)
    duplicate_keys = tuple(
        sorted(key for key, locations in occurrences.items() if len(locations) > 1)
    )
    return FiniteCoverageDiagnostics(
        missing_keys=tuple(sorted(scope - present)),
        duplicate_keys=duplicate_keys,
        outside_keys=tuple(sorted(present - scope)),
        duplicate_occurrences=tuple(
            location for key in duplicate_keys for location in occurrences[key]
        ),
    )
