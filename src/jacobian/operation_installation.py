"""Install typed domain operations into Jacobian's runtime protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import CanonicalizationError, canonicalize_json
from jacobian.capability_service import CapabilityAdapter, CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityProviderAvailability,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.results import ContractModel, ExecutionStatus
from jacobian.operation_bindings import (
    InlinePublication,
    InstalledOperation,
)
from jacobian.operation_execution import execute_operation
from jacobian.operation_projection import project_operation_result
from jacobian.operation_publication import (
    PublicationContext,
    PublicationLimitError,
    publish_operation,
)
from jacobian.operation_runtime import (
    DomainOperation,
    OperationResources,
    enriched_invalid_request,
    operation_runtime,
)
from jacobian.operations import Completed, DomainBundle, Effect, Failed, OperationSpec
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository


@dataclass(frozen=True, slots=True)
class InstalledDomainBundle:
    """Resources and adapters created for one installed domain bundle."""

    adapters: tuple[CapabilityAdapter, ...]
    semantics_uri: str
    input_schema_uris: dict[type[ContractModel], str]
    result_schema_uris: dict[str, str]
    named_schema_uris: dict[str, str]


def _spec(operation: InstalledOperation[Any, Any]) -> OperationSpec[Any, Any]:
    return operation.spec


def _operation_id(operation: DomainOperation) -> str:
    return operation.spec.operation_id


def _operation_version(operation: DomainOperation) -> str:
    return operation.spec.version


def _request_type(operation: DomainOperation) -> type[ContractModel]:
    return operation.spec.request_type


def _result_type(operation: DomainOperation) -> type[ContractModel]:
    return operation.spec.result_type


class OperationInstaller:
    """Bind domain declarations to schemas, providers, and publication."""

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        artifacts: ArtifactService,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.artifacts = artifacts

    def install(self, bundle: DomainBundle) -> InstalledDomainBundle:
        self._validate_bundle(bundle)
        semantics_uri = self.store.register_descriptor(
            kind="semantics",
            name=bundle.semantics.name,
            version=bundle.semantics.version,
            definition=bundle.semantics.definition,
        )
        request_models = {_request_type(operation) for operation in bundle.capabilities}
        input_schema_uris = {
            model: self.schemas.register_model(
                name=f"{bundle.schema_namespace}-input.{model.__name__}",
                version="1",
                model=model,
            )
            for model in request_models
        }
        result_schema_uris = {
            _operation_id(operation): self.schemas.register_model(
                name=(f"{bundle.schema_namespace}-result.{_operation_id(operation)}"),
                version=_operation_version(operation),
                model=_result_type(operation),
            )
            for operation in bundle.capabilities
        }
        resources = OperationResources(
            artifacts=self.artifacts,
            semantics_uri=semantics_uri,
            input_schema_uris=input_schema_uris,
            result_schema_uris=result_schema_uris,
        )
        adapters = tuple(
            self._adapter(operation, bundle, resources)
            for operation in bundle.capabilities
            if self._operation_available(operation, bundle)
        )
        return InstalledDomainBundle(
            adapters=adapters,
            semantics_uri=semantics_uri,
            input_schema_uris=input_schema_uris,
            result_schema_uris=result_schema_uris,
            named_schema_uris={},
        )

    @staticmethod
    def _operation_available(
        operation: DomainOperation,
        bundle: DomainBundle,
    ) -> bool:
        runtime = operation_runtime(operation, bundle)
        return runtime.availability is CapabilityProviderAvailability.AVAILABLE

    @staticmethod
    def _adapter(
        operation: DomainOperation,
        bundle: DomainBundle,
        resources: OperationResources,
    ) -> CapabilityAdapter:
        return InstalledOperationAdapter(operation, bundle, resources)

    @staticmethod
    def _validate_bundle(bundle: DomainBundle) -> None:
        if not bundle.capabilities:
            raise ValueError("capability bundle must not be empty")
        ids = tuple(_operation_id(operation) for operation in bundle.capabilities)
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate capability ID in bundle {bundle.domain_id}")
        if bundle.provider_runtime.provider == "":
            raise ValueError("capability bundle provider must not be empty")


class InstalledOperationAdapter:
    """Execute one semantic operation, then apply its publication policy."""

    typed_input = True

    def __init__(
        self,
        operation: InstalledOperation[Any, Any],
        bundle: DomainBundle,
        resources: OperationResources,
    ) -> None:
        self.operation = operation
        self.spec = _spec(operation)
        self.bundle = bundle
        self.resources = resources
        publication = operation.publication
        if isinstance(publication, InlinePublication):
            from jacobian.contracts.domain_operations import InlineOperationOutput

            output_model = cast(
                type[ContractModel],
                InlineOperationOutput[self.spec.result_type],  # type: ignore[name-defined]
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
        runtime = operation_runtime(operation, bundle)
        self._descriptor = CapabilityDescriptor(
            capability_id=self.spec.operation_id,
            version=self.spec.version,
            title=self.spec.title,
            description=self.spec.description,
            provider=runtime.provider,
            provider_runtime=runtime,
            input_schema=model_schema(self.spec.request_type),
            output_schema=model_schema(output_model),
            read_only=(self.spec.effect is Effect.READ_ONLY),
            tags=self.spec.tags,
            produced_artifact_types=produced_artifact_types,
            invocation_examples=self.spec.invocation_examples,
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        maximum_bytes = self.resources.artifacts.store.limits.max_artifact_bytes
        try:
            request_bytes = canonicalize_json(request.input)
        except CanonicalizationError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="REQUEST_RESOURCE_LIMIT_EXCEEDED",
                    stage="operation_request",
                    message=str(exc),
                    hint="Reduce the request or use a typed artifact input operation.",
                )
            ) from exc
        if len(request_bytes) > maximum_bytes:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="REQUEST_RESOURCE_LIMIT_EXCEEDED",
                    stage="operation_request",
                    message=(f"request exceeds {maximum_bytes} canonical bytes"),
                    hint="Reduce the request or use a typed artifact input operation.",
                )
            )
        try:
            parsed_request = self.spec.request_type.model_validate(request.input)
        except ValidationError as exc:
            base = self.spec.invalid_request or self.bundle.diagnostics.invalid_request
            raise CapabilityInvocationError(
                base
                if self.spec.invalid_request is not None
                else enriched_invalid_request(base, exc)
            ) from exc

        terminal = execute_operation(self.spec, parsed_request)
        publication = None
        if isinstance(terminal, Completed):
            runtime = self.operation.provider_binding.runtime
            backend_version = (
                runtime.version
                if runtime is not None and runtime.version is not None
                else self.bundle.backend_version
            )
            try:
                publication = publish_operation(
                    self.operation,
                    parsed_request,
                    terminal.value,
                    PublicationContext(
                        artifacts=self.resources.artifacts,
                        semantics_uri=self.resources.semantics_uri,
                        input_schema_uri=self.resources.input_schema_uris[
                            self.spec.request_type
                        ],
                        result_schema_uri=self.resources.result_schema_uris[
                            self.spec.operation_id
                        ],
                        backend_version=backend_version,
                    ),
                )
            except PublicationLimitError as exc:
                terminal = Failed(
                    status=ExecutionStatus.ERROR,
                    diagnostic=CapabilityDiagnostic(
                        code="PUBLICATION_RESOURCE_LIMIT_EXCEEDED",
                        stage="operation_publication",
                        message=str(exc),
                        hint="Use an operation with durable publication for this result.",
                    ),
                )
        return project_operation_result(
            operation_id=self.spec.operation_id,
            version=self.spec.version,
            terminal=terminal,
            publication=publication,
        )
