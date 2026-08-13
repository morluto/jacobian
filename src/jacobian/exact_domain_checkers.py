"""Operator-controlled declarations for independent exact-operation replay."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.capability_adapters import CapabilityAdapter
from jacobian.capability_errors import CapabilityError, CapabilityInvocationError
from jacobian.checker_artifacts import put_witness_envelope
from jacobian.checker_identity import batch_checker_manifest_measurement
from jacobian.checker_installation import CheckerInstaller
from jacobian.checker_operations import CheckerOperation, ExactReplayCheckerDeclaration
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInputKind,
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
    CapabilityRequest,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import EvidenceBindings, WitnessEnvelope
from jacobian.contracts.exact_domain_verification import (
    ExactComputedVerificationOutput,
    ExactComputedVerificationRequest,
    ExactDomainResultVerificationRequest,
    InlineExactVerificationRecord,
    inline_exact_value_digest,
)
from jacobian.contracts.results import (
    Conclusion,
    ContractModel,
    ExecutionStatus,
)
from jacobian.domain_bundles import DomainBundle
from jacobian.operation_bindings import DurablePublication
from jacobian.operation_installation import InstalledDomainBundle
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed, Failed
from jacobian.providers.flint_runtime import (
    exact_domain_checker_source_provider_runtime,
)
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository
from jacobian.validation_diagnostics import bounded_validation_exception_message
from jacobian.verification.service import VerificationService

_LOGGER = logging.getLogger(__name__)
_OPTIONAL_EXACT_REPLAY_PROVIDERS = frozenset({"jacobian.exact-domain-checkers"})


@dataclass(frozen=True, slots=True)
class ExactDomainCheckerInstallation:
    """Exact replay identities and non-conclusive installation diagnostics."""

    checker_ids: dict[str, str | None]
    provider_runtimes: dict[str, CapabilityProviderRuntime]
    declaration_providers: dict[str, str]
    witness_schema_uri: str | None = None
    diagnostics: tuple[CapabilityDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class _InstalledDeclaration:
    declaration: ExactReplayCheckerDeclaration
    result_model: type[ContractModel]
    input_schema_uri: str
    result_schema_uri: str
    semantics_uri: str
    checker_id: str


@dataclass(frozen=True, slots=True)
class _DeclaredRuntimeGroup:
    probe: CapabilityProviderRuntime
    factory: Callable[..., CapabilityProviderRuntime]
    members: tuple[tuple[InstalledDomainBundle, ExactReplayCheckerDeclaration], ...]
    factories: tuple[Callable[..., CapabilityProviderRuntime], ...]


def _declaration_factory(
    declaration: ExactReplayCheckerDeclaration,
) -> Callable[..., CapabilityProviderRuntime]:
    factory = declaration.provider_runtime_factory
    if factory is None:
        raise ValueError(
            "exact replay checker declaration requires a provider runtime factory"
        )
    return factory


def _declared_runtime_groups(
    pairs: tuple[tuple[InstalledDomainBundle, ExactReplayCheckerDeclaration], ...],
) -> tuple[_DeclaredRuntimeGroup, ...]:
    probes: dict[
        Callable[..., CapabilityProviderRuntime], CapabilityProviderRuntime
    ] = {}
    grouped: dict[
        str,
        tuple[
            CapabilityProviderRuntime,
            list[Callable[..., CapabilityProviderRuntime]],
            list[tuple[InstalledDomainBundle, ExactReplayCheckerDeclaration]],
        ],
    ] = {}
    for installed, declaration in pairs:
        factory = _declaration_factory(declaration)
        probe = probes.setdefault(factory, factory())
        current = grouped.get(probe.provider)
        if current is None:
            grouped[probe.provider] = (probe, [factory], [(installed, declaration)])
            continue
        existing_probe, factories, members = current
        if existing_probe != probe:
            raise ValueError(
                "exact replay grouped distinct probes under one provider "
                f"identity: {probe.provider}"
            )
        if factory not in factories:
            factories.append(factory)
        members.append((installed, declaration))
    return tuple(
        _DeclaredRuntimeGroup(
            probe=probe,
            factory=factories[0],
            members=tuple(members),
            factories=tuple(factories),
        )
        for probe, factories, members in grouped.values()
    )


def _authorize_replay_operation(
    installer: CheckerInstaller,
    operation: CheckerOperation,
    *,
    authorize: bool,
    provider_runtime: CapabilityProviderRuntime,
    source_available: bool,
    capability_id: str,
    diagnostics: list[CapabilityDiagnostic],
) -> str | None:
    if provider_runtime.availability is CapabilityProviderAvailability.AVAILABLE:
        return installer.install(operation, authorize=authorize).checker_id
    can_omit = (
        provider_runtime.provider in _OPTIONAL_EXACT_REPLAY_PROVIDERS
        and source_available
    )
    if not can_omit:
        return installer.install(operation, authorize=authorize).checker_id
    diagnostic = CapabilityDiagnostic(
        code="EXACT_REPLAY_PROVIDER_UNAVAILABLE",
        stage="provider_availability",
        message=(
            f"Independent replay for {capability_id!r} is not installed: "
            f"{provider_runtime.diagnostic or 'the provider is unavailable.'}"
        ),
        hint="Install or repair the optional python-flint backend, then retry.",
        details={
            "capability_id": capability_id,
            "provider": provider_runtime.provider,
            "checker_authorization_affected": True,
        },
    )
    diagnostics.append(diagnostic)
    _LOGGER.warning("%s", diagnostic.message)
    return None


def _authorized_provider_runtimes(
    groups: tuple[_DeclaredRuntimeGroup, ...],
    checker_ids: Mapping[str, str | None],
) -> dict[str, CapabilityProviderRuntime]:
    provider_runtimes: dict[str, CapabilityProviderRuntime] = {}
    for group in groups:
        authorized = tuple(
            checker_id
            for _installed, declaration in group.members
            if (checker_id := checker_ids[declaration.capability_id]) is not None
        )
        runtime = group.factory(checker_ids=authorized)
        for factory in group.factories:
            if factory is group.factory:
                continue
            other = factory(checker_ids=authorized)
            if other != runtime:
                raise ValueError(
                    "exact replay grouped distinct runtimes under one provider "
                    f"identity: {group.probe.provider}"
                )
        existing = provider_runtimes.get(runtime.provider)
        if existing is not None and existing != runtime:
            raise ValueError(
                "exact replay grouped distinct runtimes under one provider "
                f"identity: {runtime.provider}"
            )
        if runtime.provider != group.probe.provider:
            raise ValueError(
                "exact replay authorized runtime changed provider identity: "
                f"{group.probe.provider} -> {runtime.provider}"
            )
        provider_runtimes[runtime.provider] = runtime
    return provider_runtimes


def install_exact_domain_checkers(
    checkers: CheckerRegistry,
    *,
    bundles: Mapping[str, tuple[DomainBundle, InstalledDomainBundle]],
    authorize: bool,
) -> ExactDomainCheckerInstallation:
    """Install independent exact replay against dynamically registered schemas."""

    installer = CheckerInstaller(checkers)
    groups = _declared_runtime_groups(_available_declaration_bundles(bundles))
    checker_ids: dict[str, str | None] = {}
    declaration_providers: dict[str, str] = {}
    diagnostics: list[CapabilityDiagnostic] = []
    source_available = (
        exact_domain_checker_source_provider_runtime().availability
        is CapabilityProviderAvailability.AVAILABLE
    )
    with batch_checker_manifest_measurement():
        for group in groups:
            for installed, declaration in group.members:
                declaration_providers[declaration.capability_id] = group.probe.provider
                operation = CheckerOperation(
                    name=(
                        f"{declaration.capability_id} independent "
                        f"{declaration.replay_method}"
                    ),
                    entrypoint=(
                        f"{declaration.entrypoint_module}:{declaration.function}"
                    ),
                    evidence_kind=EvidenceKind.WITNESS,
                    format_id=declaration.format_id,
                    format_version="1",
                    claim_schema_uris=(
                        installed.input_schema_uris[declaration.request_model],
                    ),
                    semantics_uris=(installed.semantics_uri,),
                    candidate_schema_uris=(
                        installed.result_schema_uris[declaration.capability_id],
                    ),
                    reason=declaration.reason,
                    provider_runtime=group.probe,
                )
                checker_ids[declaration.capability_id] = _authorize_replay_operation(
                    installer,
                    operation,
                    authorize=authorize,
                    provider_runtime=group.probe,
                    source_available=source_available,
                    capability_id=declaration.capability_id,
                    diagnostics=diagnostics,
                )
    return ExactDomainCheckerInstallation(
        checker_ids=checker_ids,
        diagnostics=tuple(diagnostics),
        provider_runtimes=_authorized_provider_runtimes(groups, checker_ids),
        declaration_providers=declaration_providers,
    )


def install_exact_domain_verification(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    bundles: Mapping[str, tuple[DomainBundle, InstalledDomainBundle]],
    authorize: bool,
) -> tuple[tuple[CapabilityAdapter, ...], ExactDomainCheckerInstallation]:
    """Authorize exact replay and expose per-producer verification capabilities.

    Each authorized exact replay declaration becomes one
    :class:`ExactComputedVerificationAdapter` exposing a per-producer typed
    verifier contract for inline results. The verifier capability ID, title,
    description, and tags come from the declaration's verification metadata,
    which is always complete after construction (explicit or strictly derived
    by stripping the producer verb and appending ``.verify``).
    """

    installed = install_exact_domain_checkers(
        checkers,
        bundles=bundles,
        authorize=authorize,
    )
    witness_schema_uri = schemas.register_model(
        name="jacobian.witness-envelope",
        version="1",
        model=WitnessEnvelope,
    )
    installation = ExactDomainCheckerInstallation(
        checker_ids=installed.checker_ids,
        witness_schema_uri=witness_schema_uri,
        diagnostics=installed.diagnostics,
        provider_runtimes=installed.provider_runtimes,
        declaration_providers=installed.declaration_providers,
    )
    if not any(
        checker_id is not None for checker_id in installation.checker_ids.values()
    ):
        return (), installation
    adapters: list[CapabilityAdapter] = []
    result_models = {
        operation.spec.operation_id: operation.spec.result_type
        for bundle, _installed_bundle in bundles.values()
        for operation in bundle.capabilities
    }
    stored_producers = {
        operation.spec.operation_id
        for bundle, _installed_bundle in bundles.values()
        for operation in bundle.capabilities
        if isinstance(operation.publication, DurablePublication)
    }
    for installed_bundle, declaration in _available_declaration_bundles(bundles):
        if declaration.capability_id not in installed_bundle.result_schema_uris:
            continue
        if installation.checker_ids.get(declaration.capability_id) is None:
            continue
        installed_declaration = _installed_declaration(
            installed_bundle,
            declaration,
            installation,
            result_models[declaration.capability_id],
        )
        adapters.append(
            ExactComputedVerificationAdapter(
                declaration=installed_declaration,
                store=store,
                schemas=schemas,
                artifacts=artifacts,
                verification=verification,
                witness_schema_uri=witness_schema_uri,
                provider_runtime=installation.provider_runtimes[
                    installation.declaration_providers[declaration.capability_id]
                ],
                stored_result_input=declaration.capability_id in stored_producers,
            )
        )
    return tuple(adapters), installation


def _available_declaration_bundles(
    bundles: Mapping[str, tuple[DomainBundle, InstalledDomainBundle]],
) -> tuple[tuple[InstalledDomainBundle, ExactReplayCheckerDeclaration], ...]:
    """Pair domain-owned declarations with their unique installed producer."""

    available: list[tuple[InstalledDomainBundle, ExactReplayCheckerDeclaration]] = []
    owners: dict[str, str] = {}
    for domain_id, (bundle, installed) in bundles.items():
        producer_capability_ids = {
            operation.spec.operation_id for operation in bundle.capabilities
        }
        installed_producer_ids = {
            adapter.descriptor.capability_id for adapter in installed.adapters
        }
        for declaration in bundle.checker_declarations:
            if declaration.capability_id not in producer_capability_ids:
                raise ValueError(
                    "exact replay declaration is not backed by a domain producer "
                    f"schema: {domain_id}/{declaration.capability_id}"
                )
            if declaration.capability_id not in installed.result_schema_uris:
                continue
            if (
                installed.adapters
                and declaration.capability_id not in installed_producer_ids
            ):
                continue
            previous = owners.setdefault(declaration.capability_id, domain_id)
            if previous != domain_id:
                raise ValueError(
                    "exact replay declaration is owned by multiple bundles: "
                    f"{declaration.capability_id}"
                )
            available.append((installed, declaration))
    capability_ids = [declaration.capability_id for _, declaration in available]
    if len(capability_ids) != len(set(capability_ids)):
        duplicates = sorted(
            capability_id
            for capability_id in set(capability_ids)
            if capability_ids.count(capability_id) > 1
        )
        if duplicates:
            raise ValueError(
                "bundle repeats exact replay declarations: " + ", ".join(duplicates)
            )
    return tuple(available)


def _installed_declaration(
    bundle: InstalledDomainBundle,
    declaration: ExactReplayCheckerDeclaration,
    installation: ExactDomainCheckerInstallation,
    result_model: type[ContractModel],
) -> _InstalledDeclaration:
    checker_id = installation.checker_ids[declaration.capability_id]
    if checker_id is None:
        raise ValueError("exact-domain checker is not authorized")
    return _InstalledDeclaration(
        declaration=declaration,
        result_model=result_model,
        input_schema_uri=bundle.input_schema_uris[declaration.request_model],
        result_schema_uri=bundle.result_schema_uris[declaration.capability_id],
        semantics_uri=bundle.semantics_uri,
        checker_id=checker_id,
    )


class ExactComputedVerificationAdapter:
    """Verify one exact producer result from inline input and candidate.

    The verifier validates the inline input and candidate against the
    producer's input and result schemas and checks the authorized checker's
    bounded input scope before any artifact write. Computed operations use
    the v2 inline replay envelope; only materialized and bounded-search
    operations resolve their existing stored lineage.
    """

    def __init__(
        self,
        *,
        declaration: _InstalledDeclaration,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        artifacts: ArtifactService,
        verification: VerificationService,
        witness_schema_uri: str,
        provider_runtime: CapabilityProviderRuntime,
        stored_result_input: bool,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.artifacts = artifacts
        self.verification = verification
        self.declaration = declaration
        self.witness_schema_uri = witness_schema_uri
        self.stored_result_input = stored_result_input
        generic_request_model: Any = ExactComputedVerificationRequest
        self.input_model = generic_request_model[
            declaration.declaration.request_model,
            declaration.result_model,
        ]
        verification_capability_id = declaration.declaration.verification_capability_id
        verification_title = declaration.declaration.verification_title
        verification_description = declaration.declaration.verification_description
        if (
            verification_capability_id is None
            or verification_title is None
            or verification_description is None
        ):
            raise ValueError(
                "exact replay declaration has incomplete verifier metadata"
            )
        self._descriptor = CapabilityDescriptor(
            capability_id=verification_capability_id,
            version="1",
            title=verification_title,
            description=verification_description,
            provider=provider_runtime.provider,
            provider_runtime=provider_runtime,
            input_schema=model_schema(
                ExactDomainResultVerificationRequest
                if stored_result_input
                else self.input_model
            ),
            output_schema=model_schema(ExactComputedVerificationOutput),
            tags=declaration.declaration.verification_tags,
            accepted_input_kinds=(
                (CapabilityInputKind.TYPED_ARTIFACT,)
                if stored_result_input
                else (CapabilityInputKind.STRUCTURED_REQUEST,)
            ),
            accepted_artifact_types=(
                (declaration.result_schema_uri,) if stored_result_input else ()
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> OperationProjection:
        declaration = self.declaration
        source_artifacts: tuple[StoredArtifact, StoredArtifact, StoredArtifact] | None
        normalized_candidate: dict[str, object] | None
        if self.stored_result_input:
            source_artifacts = self._resolve_stored_result(request)
            normalized_input = source_artifacts[0].payload
            normalized_candidate = None
        else:
            normalized_input, normalized_candidate = self._validated_inline_payloads(
                request
            )
            source_artifacts = None
        # Check the authorized checker's bounded input scope before any artifact write.
        supports_input = declaration.declaration.supports_input
        if supports_input is not None and not supports_input(normalized_input):
            output = ExactComputedVerificationOutput(
                status="UNSUPPORTED",
                conclusion="UNKNOWN",
                operation_id=declaration.declaration.capability_id,
                input_uri=(
                    source_artifacts[0].artifact_uri
                    if source_artifacts is not None
                    else None
                ),
                result_uri=(
                    source_artifacts[1].artifact_uri
                    if source_artifacts is not None
                    else None
                ),
                checker_id=declaration.checker_id,
                detail=(
                    "The authorized checker does not support this input's bounded "
                    "scope; no mathematical conclusion follows."
                ),
            )
            return OperationProjection(
                operation_id=self.descriptor.capability_id,
                version=self.descriptor.version,
                terminal=Completed(value=output),
                publication=PublishedOperation(
                    output=output,
                    artifact_uris=(
                        (
                            source_artifacts[0].artifact_uri,
                            source_artifacts[1].artifact_uri,
                        )
                        if source_artifacts is not None
                        else ()
                    ),
                ),
            )
        if source_artifacts is None:
            if normalized_candidate is None:
                raise AssertionError("inline replay candidate was not validated")
            return self._verify_inline_relation(normalized_input, normalized_candidate)
        input_artifact, result_artifact, semantics_artifact = source_artifacts
        witness = put_witness_envelope(
            self.artifacts,
            witness_schema_uri=self.witness_schema_uri,
            witness_format=declaration.declaration.format_id,
            claim_artifact=input_artifact,
            semantics_artifact=semantics_artifact,
            candidate_artifact=result_artifact,
            payload={
                "operation_id": declaration.declaration.capability_id,
                "input_uri": input_artifact.artifact_uri,
                "result_uri": result_artifact.artifact_uri,
            },
            summary=(
                f"{declaration.declaration.capability_id} independent replay witness"
            ),
        )
        return self._verify_materialized_relation(
            input_artifact,
            result_artifact,
            self.store.get(witness.artifact_uri),
        )

    def _verify_inline_relation(
        self,
        normalized_input: dict[str, object],
        normalized_candidate: dict[str, object],
    ) -> OperationProjection:
        """Replay ordinary values through the checker without storing them."""

        declaration = self.declaration
        semantics = self.store.get(declaration.semantics_uri)
        bindings = EvidenceBindings(
            claim_digest=inline_exact_value_digest(
                schema_uri=declaration.input_schema_uri,
                semantics_uri=declaration.semantics_uri,
                payload=normalized_input,
            ),
            semantics_digest=semantics.manifest.object_digest,
            candidate_digest=inline_exact_value_digest(
                schema_uri=declaration.result_schema_uri,
                semantics_uri=declaration.semantics_uri,
                payload=normalized_candidate,
            ),
        )
        checked = self.verification.verify_inline_exact(
            operation_id=declaration.declaration.capability_id,
            claim_schema_uri=declaration.input_schema_uri,
            candidate_schema_uri=declaration.result_schema_uri,
            semantics_uri=declaration.semantics_uri,
            claim_payload=normalized_input,
            candidate_payload=normalized_candidate,
            checker_id=declaration.checker_id,
            witness_format=declaration.declaration.format_id,
        )
        verified = (
            checked.execution.status is ExecutionStatus.COMPLETED
            and checked.conclusion is Conclusion.TRUE
            and checked.verification_record_uri is not None
        )
        status = self._verification_status(checked.execution.status, verified)
        record_uri = checked.verification_record_uri if verified else None
        if record_uri is not None:
            record = InlineExactVerificationRecord.model_validate(
                self.store.get(record_uri).payload
            )
            if record.bindings != bindings:
                raise CapabilityError(
                    "inline exact record does not bind the verified values"
                )
        detail = checked.execution.detail or (
            checked.input.errors[0]
            if checked.input.errors
            else (
                "the authorized independent checker accepted the exact result"
                if verified
                else "the exact result was not independently accepted"
            )
        )
        output = ExactComputedVerificationOutput(
            status=status,
            conclusion="TRUE" if verified else "UNKNOWN",
            operation_id=declaration.declaration.capability_id,
            checker_id=declaration.checker_id,
            claim_digest=bindings.claim_digest if verified else None,
            semantics_digest=bindings.semantics_digest if verified else None,
            candidate_digest=bindings.candidate_digest if verified else None,
            verification_record_uri=record_uri,
            detail=detail,
        )
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
                diagnostic=CapabilityDiagnostic(
                    code="EXACT_REPLAY_EXECUTION_FAILED",
                    stage="exact_replay",
                    message=checked.execution.detail or detail,
                ),
            )
        )
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=terminal,
            publication=PublishedOperation(
                output=output,
                artifact_uris=(
                    (record_uri, declaration.semantics_uri)
                    if record_uri is not None
                    else ()
                ),
            ),
            verification_record_uri=record_uri,
        )

    def _verify_materialized_relation(
        self,
        input_artifact: StoredArtifact,
        result_artifact: StoredArtifact,
        witness: StoredArtifact,
    ) -> OperationProjection:
        checked = self.verification.verify_witness(
            claim_uri=input_artifact.artifact_uri,
            candidate_uri=result_artifact.artifact_uri,
            witness_uri=witness.artifact_uri,
            checker_id=self.declaration.checker_id,
            include_artifact_metadata=True,
            include_semantics_artifact=True,
        )
        verified = (
            checked.execution.status is ExecutionStatus.COMPLETED
            and checked.conclusion is Conclusion.TRUE
            and checked.verification_record_uri is not None
        )
        status = self._verification_status(checked.execution.status, verified)
        detail = checked.execution.detail or (
            checked.input.errors[0]
            if checked.input.errors
            else (
                "the authorized independent checker accepted the exact result"
                if verified
                else "the exact result was not independently accepted"
            )
        )
        record_uri = checked.verification_record_uri if verified else None
        artifact_uris = (
            input_artifact.artifact_uri,
            result_artifact.artifact_uri,
            witness.artifact_uri,
            *((record_uri,) if record_uri is not None else ()),
        )
        output = ExactComputedVerificationOutput(
            status=status,
            conclusion="TRUE" if verified else "UNKNOWN",
            operation_id=self.declaration.declaration.capability_id,
            input_uri=input_artifact.artifact_uri,
            result_uri=result_artifact.artifact_uri,
            witness_uri=witness.artifact_uri,
            checker_id=self.declaration.checker_id,
            claim_digest=checked.claim_digest if verified else None,
            semantics_digest=checked.semantics_digest if verified else None,
            candidate_digest=checked.candidate_digest if verified else None,
            verification_record_uri=record_uri,
            detail=detail,
        )
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
                diagnostic=CapabilityDiagnostic(
                    code="EXACT_REPLAY_EXECUTION_FAILED",
                    stage="exact_replay",
                    message=checked.execution.detail or detail,
                ),
            )
        )
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=terminal,
            publication=PublishedOperation(
                output=output,
                artifact_uris=artifact_uris,
            ),
            verification_record_uri=record_uri,
        )

    @staticmethod
    def _verification_status(
        execution_status: ExecutionStatus, verified: bool
    ) -> Literal["VERIFIED", "REJECTED", "TIMEOUT", "CANCELLED", "ERROR"]:
        if verified:
            return "VERIFIED"
        statuses: dict[ExecutionStatus, Literal["REJECTED", "TIMEOUT", "CANCELLED"]] = {
            ExecutionStatus.COMPLETED: "REJECTED",
            ExecutionStatus.TIMEOUT: "TIMEOUT",
            ExecutionStatus.CANCELLED: "CANCELLED",
        }
        return statuses.get(execution_status, "ERROR")

    def _validated_inline_payloads(
        self, request: CapabilityRequest
    ) -> tuple[dict[str, object], dict[str, object]]:
        declaration = self.declaration
        try:
            validated = self.input_model.model_validate(request.input)
            normalized_input = self.schemas.validate(
                declaration.input_schema_uri,
                validated.input.model_dump(mode="json"),
            )
            normalized_candidate = self.schemas.validate(
                declaration.result_schema_uri,
                validated.candidate.model_dump(mode="json"),
            )
        except (SchemaRegistryError, ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_EXACT_DOMAIN_INPUT",
                    stage="request_validation",
                    message=bounded_validation_exception_message(exc),
                    hint=(
                        "input must satisfy the producer request contract and "
                        "candidate must satisfy its result contract."
                    ),
                )
            ) from exc
        return normalized_input, normalized_candidate

    def _resolve_stored_result(
        self, request: CapabilityRequest
    ) -> tuple[StoredArtifact, StoredArtifact, StoredArtifact]:
        """Resolve the declared producer's exact materialized lineage."""

        declaration = self.declaration
        try:
            result_uri = ExactDomainResultVerificationRequest.model_validate(
                request.input
            ).result_uri
            result_artifact = self.store.get(result_uri)
            if (
                result_artifact.manifest.schema_uri != declaration.result_schema_uri
                or result_artifact.manifest.semantics_uri != declaration.semantics_uri
                or len(result_artifact.manifest.parents) != 1
            ):
                raise ValueError("result_uri is not this producer's exact result")
            input_artifact = self.store.get(result_artifact.manifest.parents[0])
            if (
                input_artifact.manifest.schema_uri != declaration.input_schema_uri
                or input_artifact.manifest.semantics_uri != declaration.semantics_uri
            ):
                raise ValueError("result lineage does not identify the producer input")
            self.schemas.validate(declaration.input_schema_uri, input_artifact.payload)
            self.schemas.validate(
                declaration.result_schema_uri, result_artifact.payload
            )
            semantics_artifact = self.store.get(declaration.semantics_uri)
        except (SchemaRegistryError, StorageError, ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_EXACT_DOMAIN_RESULT",
                    stage="artifact_resolution",
                    message=bounded_validation_exception_message(exc),
                    path="result_uri",
                    hint="Pass the result_uri returned by this exact producer.",
                )
            ) from exc
        return input_artifact, result_artifact, semantics_artifact


__all__ = [
    "ExactComputedVerificationAdapter",
    "ExactDomainCheckerInstallation",
    "install_exact_domain_checkers",
    "install_exact_domain_verification",
]
