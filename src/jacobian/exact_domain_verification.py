"""Exact verification adapters and selected-operation binding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.checker_artifacts import put_witness_envelope
from jacobian.checker_operations import AuthorizedChecker
from jacobian.contracts.evidence import EvidenceBindings, WitnessEnvelope
from jacobian.contracts.exact_domain_verification import (
    ExactComputedVerificationOutput,
    ExactComputedVerificationRequest,
    ExactDomainResultVerificationRequest,
    InlineExactVerificationRecord,
    inline_exact_value_digest,
)
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiagnostic,
    OperationInputKind,
    OperationRequest,
    OperationValuePort,
    ProviderObservation,
)
from jacobian.contracts.results import Conclusion, ContractModel, ExecutionStatus
from jacobian.exact_domain_checkers import (
    ExactDomainCheckerInstallation,
    ExactOperationGroup,
    _available_declaration_groups,
    install_exact_domain_checkers,
)
from jacobian.operation_adapters import OperationAdapter, parse_operation_input
from jacobian.operation_binding import BoundOperationGroup, OperationBinder
from jacobian.operation_catalog import OperationCatalog, OperationCatalogError
from jacobian.operation_declarations import (
    DurablePublication,
    InlineOperation,
    OperationDeclaration,
    OperationDeclarations,
)
from jacobian.operation_errors import OperationError, OperationInvocationError
from jacobian.operation_ports import InputPort
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed, Failed
from jacobian.providers.flint_runtime import exact_domain_checker_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository
from jacobian.validation_diagnostics import bounded_validation_exception_message
from jacobian.value_references import ValueReferenceError, ValueReferenceStore
from jacobian.verification.service import VerificationService


def _operation_spec(operation: Any) -> Any:
    if isinstance(operation, (InlineOperation, OperationDeclaration)):
        return operation
    return operation.spec


def install_exact_domain_verification(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    values: ValueReferenceStore,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    groups: Mapping[str, ExactOperationGroup],
    authorize: bool,
) -> tuple[tuple[OperationAdapter[Any], ...], ExactDomainCheckerInstallation]:
    """Authorize exact replay and expose per-producer verification operations.

    Each authorized exact replay declaration becomes one
    :class:`ExactComputedVerificationAdapter` exposing a per-producer typed
    verifier contract for inline results. The verifier operation ID, title,
    description, and tags come from the declaration's verification metadata,
    which is always complete after construction (explicit or strictly derived
    by stripping the producer verb and appending ``.verify``).
    """

    installed = install_exact_domain_checkers(
        checkers,
        groups=groups,
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
    adapters: list[OperationAdapter[Any]] = []
    result_models = {
        _operation_spec(operation).operation_id: _operation_spec(operation).result_type
        for operations, _bound_group, _declarations in groups.values()
        for operation in operations
    }
    stored_producers = {
        _operation_spec(operation).operation_id
        for operations, _bound_group, _declarations in groups.values()
        for operation in operations
        if isinstance(operation, OperationDeclaration)
        and isinstance(operation.publication, DurablePublication)
    }
    referenceable_results = {
        _operation_spec(operation).operation_id: _operation_spec(operation).result_type
        for operations, _bound_group, _declarations in groups.values()
        for operation in operations
        if isinstance(operation, OperationDeclaration)
        and operation.output_ports
        and any(
            port.value_type is _operation_spec(operation).result_type
            for port in operation.output_ports
        )
    }
    for bound_group, declaration in _available_declaration_groups(groups):
        if declaration.operation_id not in bound_group.result_schema_uris:
            continue
        if installation.checker_ids.get(declaration.operation_id) is None:
            continue
        installed_declaration = _installed_declaration(
            bound_group,
            declaration,
            installation,
            result_models[declaration.operation_id],
        )
        provider_runtime = installation.provider_runtimes[
            installation.declaration_providers[declaration.operation_id]
        ].model_copy(update={"checker_ids": (installed_declaration.checker_id,)})
        adapters.append(
            ExactComputedVerificationAdapter(
                declaration=installed_declaration,
                store=store,
                schemas=schemas,
                artifacts=artifacts,
                values=values,
                verification=verification,
                witness_schema_uri=witness_schema_uri,
                provider_runtime=provider_runtime,
                stored_result_input=declaration.operation_id in stored_producers,
                candidate_value_type=referenceable_results.get(
                    declaration.operation_id
                ),
            )
        )
    return tuple(adapters), installation


def bind_selected_exact_verification(
    *,
    catalog: OperationCatalog,
    operation_id: str,
    operations: OperationDeclarations,
    declarations: tuple[AuthorizedChecker, ...],
    binder: OperationBinder,
    verification: VerificationService,
    checkers: CheckerRegistry,
) -> OperationAdapter[Any]:
    """Bind one catalog-selected verifier without probing or measuring providers."""

    matches = tuple(
        declaration
        for declaration in declarations
        if declaration.verification_operation_id == operation_id
    )
    if len(matches) != 1:
        raise OperationCatalogError(
            f"exact verifier locator did not resolve exactly once: {operation_id}"
        )
    declaration = matches[0]
    producers = tuple(
        operation
        for operation in operations
        if operation.operation_id == declaration.operation_id
    )
    if len(producers) != 1:
        raise OperationCatalogError(
            f"exact verifier producer did not resolve exactly once: {operation_id}"
        )
    descriptor = catalog.inspect(operation_id)
    binding = catalog.checker_binding(operation_id)
    if descriptor is None or binding is None:
        raise OperationCatalogError(
            f"exact verifier catalog binding is incomplete; run `jacobian update`: {operation_id}"
        )
    checkers.require_catalog_binding(
        binding.checker_id,
        implementation_digest=binding.manifest_digest,
    )
    provider_runtime = exact_domain_checker_provider_runtime(
        checker_ids=(binding.checker_id,)
    )

    producer = producers[0]
    bound = binder.bind(operations)
    installed = _InstalledDeclaration(
        declaration=declaration,
        result_model=producer.result_type,
        input_schema_uri=bound.input_schema_uris[producer.request_type],
        result_schema_uri=bound.result_schema_uris[producer.operation_id],
        semantics_uri=bound.semantics_uri,
        checker_id=binding.checker_id,
    )
    witness_schema_uri = binder.schemas.register_model(
        name="jacobian.witness-envelope",
        version="1",
        model=WitnessEnvelope,
    )
    stored_result_input = isinstance(producer, OperationDeclaration) and isinstance(
        producer.publication, DurablePublication
    )
    candidate_value_type = (
        producer.result_type
        if not stored_result_input
        and isinstance(producer, OperationDeclaration)
        and any(
            port.value_type is producer.result_type for port in producer.output_ports
        )
        else None
    )
    return ExactComputedVerificationAdapter(
        declaration=installed,
        store=binder.store,
        schemas=binder.schemas,
        artifacts=binder.artifacts,
        values=binder.values,
        verification=verification,
        witness_schema_uri=witness_schema_uri,
        provider_runtime=provider_runtime,
        stored_result_input=stored_result_input,
        candidate_value_type=candidate_value_type,
    )


def _installed_declaration(
    group: BoundOperationGroup,
    declaration: AuthorizedChecker,
    installation: ExactDomainCheckerInstallation,
    result_model: type[ContractModel],
) -> _InstalledDeclaration:
    checker_id = installation.checker_ids[declaration.operation_id]
    if checker_id is None:
        raise ValueError("exact-domain checker is not authorized")
    return _InstalledDeclaration(
        declaration=declaration,
        result_model=result_model,
        input_schema_uri=group.input_schema_uris[declaration.request_model],
        result_schema_uri=group.result_schema_uris[declaration.operation_id],
        semantics_uri=group.semantics_uri,
        checker_id=checker_id,
    )


@dataclass(frozen=True, slots=True)
class _InstalledDeclaration:
    declaration: AuthorizedChecker
    result_model: type[ContractModel]
    input_schema_uri: str
    result_schema_uri: str
    semantics_uri: str
    checker_id: str


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
        values: ValueReferenceStore,
        verification: VerificationService,
        witness_schema_uri: str,
        provider_runtime: ProviderObservation,
        stored_result_input: bool,
        candidate_value_type: type[ContractModel] | None,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.artifacts = artifacts
        self.values = values
        self.verification = verification
        self.declaration = declaration
        self.witness_schema_uri = witness_schema_uri
        self.stored_result_input = stored_result_input
        generic_request_model: Any = ExactComputedVerificationRequest
        self.input_model: type[ContractModel] = cast(
            type[ContractModel],
            generic_request_model[
                declaration.declaration.request_model,
                declaration.result_model,
            ],
        )
        self.candidate_port = (
            InputPort(
                name="candidate",
                value_type=candidate_value_type,
                request_field="candidate",
            )
            if candidate_value_type is not None and not stored_result_input
            else None
        )
        verification_operation_id = declaration.declaration.verification_operation_id
        verification_title = declaration.declaration.verification_title
        verification_description = declaration.declaration.verification_description
        if (
            verification_operation_id is None
            or verification_title is None
            or verification_description is None
        ):
            raise ValueError(
                "exact replay declaration has incomplete verifier metadata"
            )
        self._descriptor = OperationDescriptor(
            operation_id=verification_operation_id,
            version="2" if self.candidate_port is not None else "1",
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
                (OperationInputKind.TYPED_ARTIFACT,)
                if stored_result_input
                else (OperationInputKind.STRUCTURED_REQUEST,)
            ),
            accepted_artifact_types=(
                (declaration.result_schema_uri,) if stored_result_input else ()
            ),
            input_ports=(
                (
                    OperationValuePort(
                        name=self.candidate_port.name,
                        value_type=self.candidate_port.value_type.__name__,
                    ),
                )
                if self.candidate_port is not None
                else ()
            ),
        )

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> ContractModel:
        request_model: type[ContractModel] = (
            ExactDomainResultVerificationRequest
            if self.stored_result_input
            else self.input_model
        )
        try:
            payload = request.input
            if request.inputs:
                if self.candidate_port is None:
                    raise ValueError("this exact checker declares no value inputs")
                unknown = sorted(set(request.inputs) - {self.candidate_port.name})
                if unknown:
                    raise ValueError(
                        "unknown exact-checker input port: " + ", ".join(unknown)
                    )
                candidate = self.values.resolve(
                    request.inputs[self.candidate_port.name],
                    self.candidate_port.value_type,
                )
                payload = self.candidate_port.bind_to_request(payload, candidate)
            return parse_operation_input(request_model, payload)
        except (ValidationError, ValueError, ValueReferenceError) as exc:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="INVALID_EXACT_DOMAIN_INPUT",
                    stage="request_validation",
                    message=bounded_validation_exception_message(exc),
                    hint=(
                        "input must satisfy the producer request contract and "
                        "candidate must satisfy its result contract."
                    ),
                )
            ) from exc

    def invoke(self, request: ContractModel) -> OperationProjection:
        declaration = self.declaration
        source_artifacts: tuple[StoredArtifact, StoredArtifact, StoredArtifact] | None
        normalized_candidate: dict[str, object] | None
        if self.stored_result_input:
            source_artifacts = self._resolve_stored_result(
                cast(ExactDomainResultVerificationRequest, request)
            )
            normalized_input = source_artifacts[0].payload
            normalized_candidate = None
        else:
            normalized_input, normalized_candidate = self._validated_inline_payloads(
                cast(
                    ExactComputedVerificationRequest[ContractModel, ContractModel],
                    request,
                )
            )
            source_artifacts = None
        # Check the authorized checker's bounded input scope before any artifact write.
        supports_input = declaration.declaration.supports_input
        if supports_input is not None and not supports_input(normalized_input):
            output = ExactComputedVerificationOutput(
                status="UNSUPPORTED",
                conclusion="UNKNOWN",
                operation_id=declaration.declaration.operation_id,
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
                operation_id=self.descriptor.operation_id,
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
                "operation_id": declaration.declaration.operation_id,
                "input_uri": input_artifact.artifact_uri,
                "result_uri": result_artifact.artifact_uri,
            },
            summary=(
                f"{declaration.declaration.operation_id} independent replay witness"
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
            operation_id=declaration.declaration.operation_id,
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
                raise OperationError(
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
            operation_id=declaration.declaration.operation_id,
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
                diagnostic=OperationDiagnostic(
                    code="EXACT_REPLAY_EXECUTION_FAILED",
                    stage="exact_replay",
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
            operation_id=self.declaration.declaration.operation_id,
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
                diagnostic=OperationDiagnostic(
                    code="EXACT_REPLAY_EXECUTION_FAILED",
                    stage="exact_replay",
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
        self,
        request: ExactComputedVerificationRequest[ContractModel, ContractModel],
    ) -> tuple[dict[str, object], dict[str, object]]:
        declaration = self.declaration
        try:
            normalized_input = self.schemas.validate(
                declaration.input_schema_uri,
                request.input.model_dump(mode="json"),
            )
            normalized_candidate = self.schemas.validate(
                declaration.result_schema_uri,
                request.candidate.model_dump(mode="json"),
            )
        except (
            SchemaRegistryError,
            ValidationError,
            ValueError,
            ValueReferenceError,
        ) as exc:
            raise OperationInvocationError(
                OperationDiagnostic(
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
        self, request: ExactDomainResultVerificationRequest
    ) -> tuple[StoredArtifact, StoredArtifact, StoredArtifact]:
        """Resolve the declared producer's exact materialized lineage."""

        declaration = self.declaration
        try:
            result_uri = request.result_uri
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
            raise OperationInvocationError(
                OperationDiagnostic(
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
    "bind_selected_exact_verification",
    "install_exact_domain_verification",
]
