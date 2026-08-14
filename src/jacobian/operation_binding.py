"""Bind declared mathematical operations to runtime-owned resources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from pydantic import ValidationError

from jacobian import __version__
from jacobian.artifacts import ArtifactService
from jacobian.canonical import CanonicalizationError, encode_strict_json
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiagnostic,
    OperationExample,
    OperationRequest,
    OperationValuePort,
)
from jacobian.contracts.results import ContractModel, ExecutionStatus
from jacobian.operation_adapters import OperationAdapter, parse_operation_input
from jacobian.operation_declarations import (
    Effect as DeclarationEffect,
)
from jacobian.operation_declarations import (
    InlinePublication,
    OperationDeclaration,
    OperationDeclarations,
)
from jacobian.operation_errors import (
    OperationInvocationError,
    enriched_invalid_request,
)
from jacobian.operation_execution import execute_operation
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import (
    PublicationContext,
    PublicationLimitError,
    publish_operation,
)
from jacobian.operation_runtime import (
    DomainOperation,
    OperationResources,
)
from jacobian.operations import Completed, Effect, Failed
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository
from jacobian.value_references import ValueReferenceError, ValueReferenceStore


@dataclass(frozen=True, slots=True)
class BoundOperationGroup:
    """Runtime-owned schemas and adapters for declared operations."""

    adapters: tuple[OperationAdapter[Any], ...]
    semantics_uri: str
    input_schema_uris: dict[type[ContractModel], str]
    result_schema_uris: dict[str, str]
    named_schema_uris: dict[str, str]


def _spec(
    operation: DomainOperation,
) -> OperationDeclaration[Any, Any]:
    return operation


def _operation_id(operation: DomainOperation) -> str:
    return _spec(operation).operation_id


def _operation_version(operation: DomainOperation) -> str:
    return _spec(operation).version


def _request_type(operation: DomainOperation) -> type[ContractModel]:
    return _spec(operation).request_type


def _result_type(operation: DomainOperation) -> type[ContractModel]:
    return _spec(operation).result_type


class OperationBinder:
    """Bind domain declarations to execution and publication resources."""

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        artifacts: ArtifactService,
        values: ValueReferenceStore | None = None,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.artifacts = artifacts
        self.values = values or ValueReferenceStore()

    def bind(self, operations: OperationDeclarations) -> BoundOperationGroup:
        self._validate_operations(operations)
        namespace = self._schema_namespace(operations)
        semantics_uri = self.store.register_descriptor(
            kind="semantics",
            name=f"jacobian.operations.{namespace}",
            version="1",
            definition={
                "operations": [
                    {
                        "operation_id": operation.operation_id,
                        "version": operation.version,
                    }
                    for operation in operations
                ]
            },
        )
        request_models = {_request_type(operation) for operation in operations}
        input_schema_uris = {
            model: self.schemas.register_model(
                name=f"{namespace}-input.{model.__name__}",
                version="1",
                model=model,
            )
            for model in request_models
        }
        result_schema_uris = {
            _operation_id(operation): self.schemas.register_model(
                name=(f"{namespace}-result.{_operation_id(operation)}"),
                version=_operation_version(operation),
                model=_result_type(operation),
            )
            for operation in operations
        }
        resources = OperationResources(
            artifacts=self.artifacts,
            values=self.values,
            semantics_uri=semantics_uri,
            input_schema_uris=input_schema_uris,
            result_schema_uris=result_schema_uris,
        )
        adapters = tuple(
            self._adapter(operation, resources) for operation in operations
        )
        return BoundOperationGroup(
            adapters=adapters,
            semantics_uri=semantics_uri,
            input_schema_uris=input_schema_uris,
            result_schema_uris=result_schema_uris,
            named_schema_uris={},
        )

    @staticmethod
    def _adapter(
        operation: DomainOperation,
        resources: OperationResources,
    ) -> OperationAdapter[Any]:
        return DeclaredOperationAdapter(operation, resources)

    @staticmethod
    def _validate_operations(operations: OperationDeclarations) -> None:
        if not operations:
            raise ValueError("operation declarations must not be empty")
        ids = tuple(_operation_id(operation) for operation in operations)
        if len(ids) != len(set(ids)):
            raise ValueError("operation declarations contain duplicate IDs")

    @staticmethod
    def _schema_namespace(operations: OperationDeclarations) -> str:
        first_id = operations[0].operation_id
        return first_id.split(".", maxsplit=1)[0].replace("_", "-")


class DeclaredOperationAdapter:
    """Execute one semantic operation, then apply its publication policy."""

    def __init__(
        self,
        operation: DomainOperation,
        resources: OperationResources,
    ) -> None:
        self.operation = operation
        self.spec = _spec(operation)
        self.resources = resources
        publication = operation.publication
        if isinstance(publication, InlinePublication):
            from jacobian.contracts.domain_operations import (
                InlineOperationOutput,
                ReferencedInlineOperationOutput,
            )

            inline_output = (
                ReferencedInlineOperationOutput
                if operation.output_ports
                else InlineOperationOutput
            )
            output_model = cast(
                type[ContractModel],
                inline_output[self.spec.result_type],
            )
            produced_artifact_types: tuple[str, ...] = ()
        else:
            from jacobian.contracts.domain_operations import DurableOperationOutput

            preview_type = publication.preview_type or self.spec.result_type
            output_model = cast(
                type[ContractModel],
                DurableOperationOutput[preview_type],  # type: ignore[valid-type]
            )
            produced_artifact_types = (
                resources.result_schema_uris[self.spec.operation_id],
            )
        self._descriptor = OperationDescriptor(
            operation_id=self.spec.operation_id,
            version=self.spec.version,
            title=self.spec.title,
            description=self.spec.description,
            provider="built-in",
            input_schema=model_schema(self.spec.request_type),
            output_schema=model_schema(output_model),
            read_only=(
                self.spec.effect in {Effect.READ_ONLY, DeclarationEffect.READ_ONLY}
            ),
            tags=self.spec.tags,
            produced_artifact_types=produced_artifact_types,
            input_ports=tuple(
                OperationValuePort(
                    name=port.name,
                    value_type=port.value_type.__name__,
                )
                for port in operation.input_ports
            ),
            output_ports=tuple(
                OperationValuePort(
                    name=port.name,
                    value_type=port.value_type.__name__,
                )
                for port in operation.output_ports
            ),
            examples=tuple(
                OperationExample(
                    name=example.name,
                    description=example.description,
                    input=dict(example.input),
                )
                for example in self.spec.examples
            ),
        )

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> ContractModel:
        maximum_bytes = self.resources.artifacts.store.limits.max_artifact_bytes
        try:
            assembled_input = self._bind_inputs(request)
            bounded_input = {
                key: (
                    value.model_dump(mode="json")
                    if isinstance(value, ContractModel)
                    else value
                )
                for key, value in assembled_input.items()
            }
            request_bytes = encode_strict_json(bounded_input)
        except CanonicalizationError as exc:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="REQUEST_RESOURCE_LIMIT_EXCEEDED",
                    stage="operation_request",
                    message=str(exc),
                    hint="Reduce the request or use a typed artifact input operation.",
                )
            ) from exc
        if len(request_bytes) > maximum_bytes:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="REQUEST_RESOURCE_LIMIT_EXCEEDED",
                    stage="operation_request",
                    message=(f"request exceeds {maximum_bytes} canonical bytes"),
                    hint="Reduce the request or use a typed artifact input operation.",
                )
            )
        try:
            parsed_request = cast(
                ContractModel,
                parse_operation_input(self.spec.request_type, assembled_input),
            )
        except ValidationError as exc:
            invalid_request = self.spec.invalid_request
            base = invalid_request or OperationDiagnostic(
                code="INVALID_REQUEST",
                stage="operation_input_validation",
                message="The operation request is invalid.",
                hint="Inspect the operation schema and provide a strictly valid request.",
            )
            raise OperationInvocationError(
                enriched_invalid_request(base, exc)
                if self.spec.enrich_invalid_request or invalid_request is None
                else base
            ) from exc

        return parsed_request

    def invoke(self, prepared: ContractModel) -> OperationProjection:
        parsed_request = prepared
        try:
            terminal = execute_operation(self.spec, parsed_request)
        except ValidationError as exc:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="ADAPTER_EXECUTION_FAILED",
                    stage="operation_result_validation",
                    message="The operation returned an invalid typed result.",
                    hint="Fix the operation implementation to satisfy its result contract.",
                )
            ) from exc
        publication = None
        if isinstance(terminal, Completed):
            try:
                publication = publish_operation(
                    self.operation,
                    parsed_request,
                    terminal.value,
                    PublicationContext(
                        artifacts=self.resources.artifacts,
                        values=self.resources.values,
                        semantics_uri=self.resources.semantics_uri,
                        input_schema_uri=self.resources.input_schema_uris[
                            self.spec.request_type
                        ],
                        result_schema_uri=self.resources.result_schema_uris[
                            self.spec.operation_id
                        ],
                        backend_version=__version__,
                    ),
                )
            except PublicationLimitError as exc:
                terminal = Failed(
                    status=ExecutionStatus.ERROR,
                    diagnostic=OperationDiagnostic(
                        code="PUBLICATION_RESOURCE_LIMIT_EXCEEDED",
                        stage="operation_publication",
                        message=str(exc),
                        hint="Use an operation with durable publication for this result.",
                    ),
                )
        return OperationProjection(
            operation_id=self.spec.operation_id,
            version=self.spec.version,
            terminal=terminal,
            publication=publication,
        )

    def _bind_inputs(self, request: OperationRequest) -> dict[str, Any]:
        if not request.inputs:
            return request.input
        ports = {port.name: port for port in self.operation.input_ports}
        unknown = sorted(set(request.inputs) - set(ports))
        if unknown:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="UNKNOWN_INPUT_PORT",
                    stage="operation_input_binding",
                    message="Unknown operation input port: " + ", ".join(unknown),
                    hint="Inspect the operation and use one of its declared input ports.",
                )
            )
        assembled = request.input
        try:
            for name, value_ref in request.inputs.items():
                port = ports[name]
                value = self.resources.values.resolve(value_ref, port.value_type)
                assembled = port.bind_to_request(assembled, value)
        except (TypeError, ValueError, ValueReferenceError) as exc:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="INCOMPATIBLE_VALUE_REFERENCE",
                    stage="operation_input_binding",
                    message=str(exc),
                    hint="Use a value reference produced for this exact input port.",
                )
            ) from exc
        return assembled
