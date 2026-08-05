"""Install domain operations into Jacobian's runtime protocol."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.capability_service import CapabilityAdapter, CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityMode,
    CapabilityObligation,
    CapabilityObligationStatus,
    CapabilityRelationship,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.domain_operations import (
    ComputedOperationOutput,
    MaterializedOperationOutput,
)
from jacobian.contracts.results import ContractModel, Execution, ExecutionStatus
from jacobian.operations import (
    BoundedSearchInterrupted,
    BoundedSearchNotApplicable,
    BoundedSearchOperation,
    BoundedSearchWitness,
    ComputedNotApplicable,
    ComputedOperation,
    DomainBundle,
    DomainOperation,
    MaterializedOperation,
    OperationExecutionFailure,
)
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository


@dataclass(frozen=True, slots=True)
class InstalledDomainBundle:
    """Resources and adapters created for one installed domain bundle."""

    adapters: tuple[CapabilityAdapter, ...]
    semantics_uri: str
    input_schema_uris: dict[type[ContractModel], str]
    result_schema_uris: dict[str, str]
    obligation_schema_uris: dict[str, str]


@dataclass(frozen=True, slots=True)
class _OperationResources:
    artifacts: ArtifactService
    semantics_uri: str
    input_schema_uris: dict[type[ContractModel], str]
    result_schema_uris: dict[str, str]
    obligation_schema_uris: dict[str, str]


def _execution_failure_result(
    *,
    operation: DomainOperation,
    request: CapabilityRequest,
    outcome: OperationExecutionFailure,
    started: float,
) -> CapabilityResult:
    return CapabilityResult(
        capability_id=operation.capability_id,
        capability_version=operation.version,
        mode=request.mode,
        execution=Execution(
            status=outcome.status,
            runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
            detail=outcome.diagnostic.message,
        ),
        output={
            "error": outcome.diagnostic.model_dump(
                mode="json",
                exclude_none=True,
            )
        },
        diagnostics=(outcome.diagnostic,),
        assurance=CapabilityAssurance(
            level=CapabilityAssuranceLevel.HEURISTIC,
            basis="operation did not complete; no mathematical conclusion",
        ),
    )


class OperationInstaller:
    """Own schema, semantics, artifact, and result-envelope mechanics."""

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
        request_models = {operation.request_model for operation in bundle.capabilities}
        input_schema_uris = {
            model: self.schemas.register_model(
                name=(f"{bundle.schema_namespace}-input.{model.__name__}"),
                version="1",
                model=model,
            )
            for model in request_models
        }
        result_schema_uris = {
            operation.capability_id: self.schemas.register_model(
                name=(f"{bundle.schema_namespace}-result.{operation.capability_id}"),
                version=operation.version,
                model=operation.result_model,
            )
            for operation in bundle.capabilities
        }
        obligation_schema_uris = {
            operation.capability_id: self.schemas.register_model(
                name=(
                    f"{bundle.schema_namespace}-obligation.{operation.capability_id}"
                ),
                version=operation.version,
                model=operation.obligation_model,
            )
            for operation in bundle.capabilities
            if isinstance(operation, BoundedSearchOperation)
        }
        resources = _OperationResources(
            artifacts=self.artifacts,
            semantics_uri=semantics_uri,
            input_schema_uris=input_schema_uris,
            result_schema_uris=result_schema_uris,
            obligation_schema_uris=obligation_schema_uris,
        )
        adapters = tuple(
            self._adapter(operation, bundle, resources)
            for operation in bundle.capabilities
        )
        return InstalledDomainBundle(
            adapters=adapters,
            semantics_uri=semantics_uri,
            input_schema_uris=input_schema_uris,
            result_schema_uris=result_schema_uris,
            obligation_schema_uris=obligation_schema_uris,
        )

    @staticmethod
    def _adapter(
        operation: DomainOperation,
        bundle: DomainBundle,
        resources: _OperationResources,
    ) -> CapabilityAdapter:
        if isinstance(operation, ComputedOperation):
            return ComputedOperationAdapter(operation, bundle, resources)
        if isinstance(operation, MaterializedOperation):
            return MaterializedOperationAdapter(operation, bundle, resources)
        if isinstance(operation, BoundedSearchOperation):
            return BoundedSearchOperationAdapter(operation, bundle, resources)
        raise TypeError(f"unsupported domain operation: {type(operation).__name__}")

    @staticmethod
    def _validate_bundle(bundle: DomainBundle) -> None:
        if not bundle.capabilities:
            raise ValueError("capability bundle must not be empty")
        ids = tuple(operation.capability_id for operation in bundle.capabilities)
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate capability ID in bundle {bundle.domain_id}")
        if bundle.provider_runtime.provider == "":
            raise ValueError("capability bundle provider must not be empty")


class ComputedOperationAdapter:
    """Run one typed finite producer without granting verification authority."""

    def __init__(
        self,
        operation: ComputedOperation[Any, Any],
        bundle: DomainBundle,
        resources: _OperationResources,
    ) -> None:
        self.operation = operation
        self.bundle = bundle
        self.resources = resources
        # Pydantic supports runtime specialization of generic response models.
        # Static type checkers cannot resolve a model class held in a value.
        self.output_model = ComputedOperationOutput[operation.result_model]  # type: ignore[name-defined]
        self._descriptor = CapabilityDescriptor(
            capability_id=operation.capability_id,
            version=operation.version,
            title=operation.title,
            description=operation.description,
            provider=bundle.provider_runtime.provider,
            provider_runtime=bundle.provider_runtime,
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(operation.request_model),
            output_schema=model_schema(self.output_model),
            read_only=True,
            tags=operation.tags,
            invocation_examples=operation.invocation_examples,
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated_request = self.operation.request_model.model_validate(
                request.input
            )
        except ValidationError as exc:
            raise CapabilityInvocationError(
                self.operation.invalid_request
                or self.bundle.diagnostics.invalid_request
            ) from exc

        started = time.monotonic()
        outcome = self.operation.implementation(validated_request)
        if isinstance(outcome, ComputedNotApplicable):
            raise CapabilityInvocationError(outcome.diagnostic)
        if isinstance(outcome, OperationExecutionFailure):
            return _execution_failure_result(
                operation=self.operation,
                request=request,
                outcome=outcome,
                started=started,
            )
        validated_result = self.operation.result_model.model_validate(
            outcome.value.model_dump(mode="python")
        )
        output = self.output_model(
            result=validated_result,
            backend_version=self.bundle.backend_version,
        )
        return CapabilityResult(
            capability_id=self.operation.capability_id,
            capability_version=self.operation.version,
            mode=request.mode,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
            ),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description=self.bundle.scope_description,
                parameters=validated_request.model_dump(mode="json"),
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.COMPLETE,
                basis=self.bundle.completeness_basis,
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis=self.bundle.assurance_basis,
            ),
        )


class MaterializedOperationAdapter:
    """Materialize a finite producer result with exact input/result lineage."""

    def __init__(
        self,
        operation: MaterializedOperation[Any, Any, Any],
        bundle: DomainBundle,
        resources: _OperationResources,
    ) -> None:
        self.operation = operation
        self.bundle = bundle
        self.resources = resources
        self.preview_model = operation.preview_model or operation.result_model
        self.output_model = MaterializedOperationOutput[self.preview_model]  # type: ignore[name-defined]
        self._descriptor = CapabilityDescriptor(
            capability_id=operation.capability_id,
            version=operation.version,
            title=operation.title,
            description=operation.description,
            provider=bundle.provider_runtime.provider,
            provider_runtime=bundle.provider_runtime,
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(operation.request_model),
            output_schema=model_schema(self.output_model),
            tags=operation.tags,
            invocation_examples=operation.invocation_examples,
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated_request = self.operation.request_model.model_validate(
                request.input
            )
        except ValidationError as exc:
            raise CapabilityInvocationError(
                self.operation.invalid_request
                or self.bundle.diagnostics.invalid_request
            ) from exc
        started = time.monotonic()
        outcome = self.operation.implementation(validated_request)
        if isinstance(outcome, ComputedNotApplicable):
            raise CapabilityInvocationError(outcome.diagnostic)
        if isinstance(outcome, OperationExecutionFailure):
            return _execution_failure_result(
                operation=self.operation,
                request=request,
                outcome=outcome,
                started=started,
            )
        result = self.operation.result_model.model_validate(
            outcome.value.model_dump(mode="python")
        )
        input_uri = self.resources.artifacts.put(
            schema_uri=self.resources.input_schema_uris[self.operation.request_model],
            semantics_uri=self.resources.semantics_uri,
            payload=validated_request.model_dump(mode="json"),
            summary=f"{self.operation.capability_id} materialized input",
        ).artifact_uri
        result_uri = self.resources.artifacts.put(
            schema_uri=self.resources.result_schema_uris[self.operation.capability_id],
            semantics_uri=self.resources.semantics_uri,
            payload=result.model_dump(mode="json"),
            parents=(input_uri,),
            summary=f"{self.operation.capability_id} materialized result",
        ).artifact_uri
        preview = (
            self.preview_model.model_validate(
                self.operation.preview(result).model_dump(mode="python")
            )
            if self.operation.preview is not None
            else None
        )
        output = self.output_model(
            input_uri=input_uri,
            result_uri=result_uri,
            preview=preview,
            preview_complete=self.operation.preview_complete,
            backend_version=self.bundle.backend_version,
        )
        return CapabilityResult(
            capability_id=self.operation.capability_id,
            capability_version=self.operation.version,
            mode=request.mode,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
            ),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description=self.bundle.scope_description,
                parameters={"input_uri": input_uri},
                artifact_uri=input_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.COMPLETE,
                basis=self.bundle.completeness_basis,
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            relationships=(
                CapabilityRelationship(
                    relation_id=self.operation.relation_id,
                    source_artifact_uris=(input_uri,),
                    target_artifact_uris=(result_uri,),
                ),
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis=self.bundle.assurance_basis,
            ),
            artifact_uris=(input_uri, result_uri),
        )


class BoundedSearchOperationAdapter:
    """Run one budgeted producer without granting verification authority."""

    def __init__(
        self,
        operation: BoundedSearchOperation[Any, Any, Any],
        bundle: DomainBundle,
        resources: _OperationResources,
    ) -> None:
        self.operation = operation
        self.bundle = bundle
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id=operation.capability_id,
            version=operation.version,
            title=operation.title,
            description=operation.description,
            provider=bundle.provider_runtime.provider,
            provider_runtime=bundle.provider_runtime,
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(operation.request_model),
            output_schema=model_schema(operation.result_model),
            tags=operation.tags,
            invocation_examples=operation.invocation_examples,
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated_request = self.operation.request_model.model_validate(
                request.input
            )
        except ValidationError as exc:
            raise CapabilityInvocationError(
                self.operation.invalid_request
                or self.bundle.diagnostics.invalid_request
            ) from exc

        started = time.monotonic()
        outcome = self.operation.implementation(validated_request)
        if isinstance(outcome, BoundedSearchNotApplicable):
            raise CapabilityInvocationError(outcome.diagnostic)
        if isinstance(outcome, OperationExecutionFailure):
            return _execution_failure_result(
                operation=self.operation,
                request=request,
                outcome=outcome,
                started=started,
            )
        validated_result = self.operation.result_model.model_validate(
            outcome.value.model_dump(mode="python")
        )
        validated_obligation = self.operation.obligation_model.model_validate(
            self.operation.obligation(validated_request, validated_result)
        )
        witness_outcome = isinstance(outcome, BoundedSearchWitness)
        interrupted_outcome = (
            outcome if isinstance(outcome, BoundedSearchInterrupted) else None
        )
        complete = self.operation.is_complete(validated_result)
        if complete != witness_outcome:
            raise ValueError(
                "bounded-search outcome contradicts its completion predicate"
            )
        request_payload = validated_request.model_dump(mode="json")
        result_payload = validated_result.model_dump(mode="json")
        scope_parameters = self.operation.scope_parameters(
            validated_request,
            validated_result,
        )
        input_uri = self.resources.artifacts.put(
            schema_uri=self.resources.input_schema_uris[self.operation.request_model],
            semantics_uri=self.resources.semantics_uri,
            payload=request_payload,
            summary=f"{self.operation.capability_id} bounded-search input",
        ).artifact_uri
        result_uri = self.resources.artifacts.put(
            schema_uri=self.resources.result_schema_uris[self.operation.capability_id],
            semantics_uri=self.resources.semantics_uri,
            payload=result_payload,
            parents=(input_uri,),
            summary=f"{self.operation.capability_id} bounded-search result",
        ).artifact_uri
        obligation_uri = self.resources.artifacts.put(
            schema_uri=self.resources.obligation_schema_uris[
                self.operation.capability_id
            ],
            semantics_uri=self.resources.semantics_uri,
            payload=validated_obligation.model_dump(mode="json"),
            parents=(input_uri, result_uri),
            summary=f"{self.operation.capability_id} optimality obligation",
        ).artifact_uri
        return CapabilityResult(
            capability_id=self.operation.capability_id,
            capability_version=self.operation.version,
            mode=request.mode,
            execution=Execution(
                status=(
                    interrupted_outcome.status
                    if interrupted_outcome is not None
                    else ExecutionStatus.COMPLETED
                ),
                runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
                detail=(
                    interrupted_outcome.diagnostic.message
                    if interrupted_outcome is not None
                    else None
                ),
            ),
            output=result_payload,
            scope=CapabilityScope(
                description=self.bundle.scope_description,
                parameters=scope_parameters,
                artifact_uri=input_uri,
            ),
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.COMPLETE
                    if complete
                    else CapabilityCompletenessStatus.UNKNOWN
                ),
                basis=(
                    self.bundle.completeness_basis
                    if complete
                    else self.operation.incomplete_basis
                ),
                assurance_level=(
                    CapabilityAssuranceLevel.HEURISTIC
                    if interrupted_outcome is not None
                    else CapabilityAssuranceLevel.COMPUTED
                ),
            ),
            relationships=(
                CapabilityRelationship(
                    relation_id=self.operation.relation_id,
                    source_artifact_uris=(input_uri,),
                    target_artifact_uris=(result_uri,),
                    obligation_uris=(obligation_uri,),
                ),
            ),
            obligations=(
                CapabilityObligation(
                    obligation_uri=obligation_uri,
                    status=CapabilityObligationStatus.OPEN,
                ),
            ),
            diagnostics=(
                (interrupted_outcome.diagnostic,)
                if interrupted_outcome is not None
                else ()
            ),
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.HEURISTIC
                    if interrupted_outcome is not None
                    else CapabilityAssuranceLevel.COMPUTED
                ),
                basis=(
                    "bounded execution was interrupted; partial artifacts carry "
                    "no mathematical conclusion"
                    if interrupted_outcome is not None
                    else self.bundle.assurance_basis
                ),
            ),
            artifact_uris=(input_uri, result_uri, obligation_uri),
        )
