from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Self, cast

import pytest
from pydantic import ConfigDict, Field, model_validator
from tests.support.services import DomainTestServices, open_domain_services

from jacobian import __version__
from jacobian.contracts.operations import (
    OperationDiagnostic,
    OperationRequest,
)
from jacobian.contracts.results import ContractModel, ExecutionStatus
from jacobian.operation_binding import OperationBinder
from jacobian.operation_declarations import (
    Effect,
    OperationAbortError,
    OperationDeclaration,
    OperationDeclarations,
    OperationRefusalError,
    PreflightResult,
    PreflightStatus,
    durable_operation,
    inline_operation,
)
from jacobian.operation_ports import InputPort, OutputPort
from jacobian.operations import Failed


@pytest.fixture
def operation_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Open only the production services exercised by synthetic operations."""

    with open_domain_services(tmp_path / "state") as services:
        yield services


class _SyntheticRequest(ContractModel):
    value: int = Field(ge=0, le=100)


class _CrossFieldRequest(ContractModel):
    """A request whose cross-field validator rejects otherwise valid fields."""

    value: int = Field(ge=0, le=100)
    limit: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def require_value_below_limit(self) -> Self:
        if self.value >= self.limit:
            raise ValueError("value must be strictly less than limit")
        return self


class _SyntheticResult(ContractModel):
    doubled: int


class _SyntheticPreview(ContractModel):
    summary: str


class _ComposedRequest(ContractModel):
    source: _SyntheticResult


class _SchemaDiscoveryOnlyRequest(ContractModel):
    """Advertise an impossible schema while retaining a valid typed parser."""

    model_config = ConfigDict(json_schema_extra={"not": {}})

    value: int


def _synthetic_bundle() -> OperationDeclarations:
    not_applicable = OperationDiagnostic(
        code="SYNTHETIC_NOT_APPLICABLE",
        stage="synthetic_computation",
        message="Thirteen is excluded from this synthetic operation.",
    )

    def compute(
        request: _SyntheticRequest,
    ) -> _SyntheticResult:
        if request.value == 13:
            raise OperationRefusalError(not_applicable)
        return _SyntheticResult(doubled=request.value * 2)

    return (
        inline_operation(
            OperationDeclaration(
                operation_id="synthetic.compute.double",
                version="2",
                title="Double a bounded integer",
                description="Double one bounded nonnegative integer.",
                request_type=_SyntheticRequest,
                result_type=_SyntheticResult,
                execute=compute,
                tags=("synthetic",),
            )
        ),
    )


def _install(runtime: DomainTestServices, bundle: OperationDeclarations) -> None:
    installation = OperationBinder(
        runtime.core.store,
        runtime.core.schemas,
        runtime.core.artifacts,
        runtime.core.values,
    ).bind(bundle)
    for adapter in installation.adapters:
        runtime.core.operations.register(adapter)


def test_synthetic_bundle_returns_an_inline_typed_result(
    operation_services,
) -> None:
    _install(operation_services, _synthetic_bundle())

    descriptor = next(
        descriptor
        for descriptor in operation_services.core.operations.snapshot().operations
        if descriptor.operation_id == "synthetic.compute.double"
    )
    assert descriptor.provider == "built-in"
    assert descriptor.input_schema["additionalProperties"] is False
    result_schema = descriptor.output_schema["properties"]["result"]
    assert result_schema == {"$ref": "#/$defs/_SyntheticResult"}
    assert "value_refs" not in descriptor.output_schema["properties"]

    result = operation_services.core.operations.invoke(
        OperationRequest(
            operation_id="synthetic.compute.double",
            input={"value": 6},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"doubled": 12}
    assert result.artifact_uris == ()


def test_declared_ports_publish_and_resolve_one_opaque_typed_value(
    operation_services,
) -> None:
    bundle = _synthetic_bundle()
    producer = inline_operation(
        replace(
            bundle[0],
            operation_id="synthetic.compute.produce_double",
        ),
        output_ports=(OutputPort(name="value", value_type=_SyntheticResult),),
    )
    consumer = inline_operation(
        OperationDeclaration(
            operation_id="synthetic.compute.increment",
            version="2",
            title="Increment a typed synthetic value",
            description="Consume one exact value reference without JSON rewriting.",
            request_type=_ComposedRequest,
            result_type=_SyntheticResult,
            execute=lambda request: _SyntheticResult(
                doubled=request.source.doubled + 1
            ),
        ),
        input_ports=(
            InputPort(
                name="source",
                value_type=_SyntheticResult,
                request_field="source",
            ),
        ),
    )
    _install(
        operation_services,
        (producer, consumer),
    )

    descriptors = {
        descriptor.operation_id: descriptor
        for descriptor in operation_services.core.operations.snapshot().operations
    }
    assert descriptors[producer.operation_id].output_ports[0].model_dump() == {
        "name": "value",
        "value_type": "_SyntheticResult",
    }
    assert descriptors[consumer.operation_id].input_ports[0].model_dump() == {
        "name": "source",
        "value_type": "_SyntheticResult",
    }

    produced = operation_services.core.operations.invoke(
        OperationRequest(
            operation_id=producer.operation_id,
            input={"value": 6},
        )
    )
    value_ref = produced.output["value_refs"]["value"]
    stored = operation_services.core.values.inspect(value_ref)
    assert stored.source_operation_id == producer.operation_id
    assert stored.source_operation_version == producer.version
    assert stored.source_port == "value"
    assert stored.digest.startswith("sha256:")

    consumed = operation_services.core.operations.invoke(
        OperationRequest(
            operation_id=consumer.operation_id,
            input={},
            inputs={"source": value_ref},
        )
    )

    assert consumed.execution.status is ExecutionStatus.COMPLETED
    assert consumed.output["result"] == {"doubled": 13}


def test_operation_without_input_ports_rejects_value_references(
    operation_services,
) -> None:
    bundle = _synthetic_bundle()
    _install(operation_services, bundle)

    result = operation_services.core.operations.invoke(
        OperationRequest(
            operation_id=bundle[0].operation_id,
            input={"value": 2},
            inputs={"undeclared": "value://AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_REQUEST"
    assert result.diagnostics[0].path == "inputs"
    assert result.diagnostics[0].details["unknown_input_ports"] == ["undeclared"]


def test_value_reference_binding_rejects_unknown_ports_and_payload_conflicts(
    operation_services,
) -> None:
    bundle = _synthetic_bundle()
    producer = inline_operation(
        replace(
            bundle[0],
            operation_id="synthetic.compute.produce_for_rejection",
        ),
        output_ports=(OutputPort(name="value", value_type=_SyntheticResult),),
    )
    consumer = inline_operation(
        OperationDeclaration(
            operation_id="synthetic.compute.consume_for_rejection",
            version="2",
            title="Consume a synthetic value",
            description="Exercise fail-closed typed value binding.",
            request_type=_ComposedRequest,
            result_type=_SyntheticResult,
            execute=lambda request: request.source,
        ),
        input_ports=(
            InputPort(
                name="source",
                value_type=_SyntheticResult,
                request_field="source",
            ),
        ),
    )
    _install(
        operation_services,
        (producer, consumer),
    )
    produced = operation_services.core.operations.invoke(
        OperationRequest(
            operation_id=producer.operation_id,
            input={"value": 2},
        )
    )
    value_ref = produced.output["value_refs"]["value"]

    unknown_port = operation_services.core.operations.invoke(
        OperationRequest(
            operation_id=consumer.operation_id,
            input={},
            inputs={"other": value_ref},
        )
    )
    assert unknown_port.diagnostics[0].code == "UNKNOWN_INPUT_PORT"

    conflicting_payload = operation_services.core.operations.invoke(
        OperationRequest(
            operation_id=consumer.operation_id,
            input={"source": {"doubled": 99}},
            inputs={"source": value_ref},
        )
    )
    assert conflicting_payload.diagnostics[0].code == "INCOMPATIBLE_VALUE_REFERENCE"


def test_operation_uses_one_typed_parse_not_its_discovery_schema(
    operation_services,
) -> None:
    operation = inline_operation(
        OperationDeclaration(
            operation_id="synthetic.compute.schema_discovery_only",
            version="2",
            request_type=_SchemaDiscoveryOnlyRequest,
            result_type=_SyntheticResult,
            execute=lambda request: _SyntheticResult(doubled=request.value * 2),
            title="Typed parse boundary",
            description="Prove JSON Schema is discovery metadata, not execution.",
        )
    )
    _install(operation_services, (operation,))

    result = operation_services.core.operations.invoke(
        OperationRequest(
            operation_id=operation.operation_id,
            input={"value": 7},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"doubled": 14}


def test_preflight_refusal_runs_before_execution_or_publication(
    operation_services,
) -> None:
    def must_not_execute(_request: _SyntheticRequest) -> _SyntheticResult:
        raise AssertionError("execution ran after preflight refusal")

    operation = durable_operation(
        OperationDeclaration(
            operation_id="synthetic.compute.preflight_refused",
            version="2",
            request_type=_SyntheticRequest,
            result_type=_SyntheticResult,
            execute=must_not_execute,
            preflight=lambda _request: PreflightResult(
                PreflightStatus.RESOURCE_LIMIT_EXCEEDED,
                "synthetic allocation budget exceeded",
            ),
            title="Preflight refusal",
            description="Refuse before execution and publication.",
        ),
        resource_reason="the test exercises preflight before durable publication",
    )
    _install(operation_services, (operation,))

    result = operation_services.core.operations.invoke(
        OperationRequest(
            operation_id=operation.operation_id,
            input={"value": 4},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "RESOURCE_LIMIT_EXCEEDED"
    assert result.artifact_uris == ()


def test_postcondition_failure_publishes_no_artifacts(operation_services) -> None:
    def reject_result(
        _request: _SyntheticRequest,
        _result: _SyntheticResult,
    ) -> None:
        raise ValueError("synthetic postcondition failed")

    operation = durable_operation(
        OperationDeclaration(
            operation_id="synthetic.compute.postcondition_failed",
            version="2",
            request_type=_SyntheticRequest,
            result_type=_SyntheticResult,
            execute=lambda request: _SyntheticResult(doubled=request.value * 2),
            postcondition=reject_result,
            title="Postcondition failure",
            description="Reject before durable publication.",
        ),
        resource_reason="the test exercises postcondition before publication",
    )
    _install(operation_services, (operation,))

    result = operation_services.core.operations.invoke(
        OperationRequest(
            operation_id=operation.operation_id,
            input={"value": 4},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "ADAPTER_EXECUTION_FAILED"
    assert result.artifact_uris == ()


def test_materialized_operation_retains_artifacts_lineage_and_typed_preview(
    operation_services,
) -> None:
    materialized = durable_operation(
        OperationDeclaration(
            operation_id="synthetic.materialize.double",
            version="2",
            title="Materialize a doubled integer",
            description="Materialize one doubled integer with lineage and a typed preview.",
            request_type=_SyntheticRequest,
            result_type=_SyntheticResult,
            execute=lambda request: _SyntheticResult(doubled=request.value * 2),
            tags=("synthetic",),
        ),
        resource_reason="the test exercises durable lineage and preview behavior",
        preview_type=_SyntheticPreview,
        preview=lambda result: _SyntheticPreview(summary=f"doubled={result.doubled}"),
        preview_complete=True,
    )
    _install(operation_services, (materialized,))

    descriptor = next(
        descriptor
        for descriptor in operation_services.core.operations.snapshot().operations
        if descriptor.operation_id == "synthetic.materialize.double"
    )
    assert descriptor.read_only is True
    assert set(descriptor.output_schema["properties"]) == {
        "input_uri",
        "result_uri",
        "preview",
        "preview_complete",
        "backend_version",
    }

    result = operation_services.core.operations.invoke(
        OperationRequest(
            operation_id="synthetic.materialize.double",
            input={"value": 6},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert len(result.artifact_uris) == 2
    input_uri, result_uri = result.artifact_uris
    assert result.output["input_uri"] == input_uri
    assert result.output["result_uri"] == result_uri
    assert result.output["backend_version"] == __version__
    assert result.output["preview"] == {"summary": "doubled=12"}
    assert result.output["preview_complete"] is True
    input_artifact = operation_services.core.store.get(input_uri)
    output_artifact = operation_services.core.store.get(result_uri)
    assert input_artifact.payload == {"value": 6}
    assert output_artifact.payload == {"doubled": 12}
    assert output_artifact.manifest.parents == (input_uri,)


def test_catalog_effect_comes_from_operation_spec(
    operation_services,
) -> None:
    stateful = inline_operation(
        OperationDeclaration(
            operation_id="synthetic.stateful.double",
            version="2",
            title="Stateful double",
            description="A synthetic stateful operation published inline.",
            request_type=_SyntheticRequest,
            result_type=_SyntheticResult,
            execute=lambda request: _SyntheticResult(doubled=request.value * 2),
            effect=Effect.STATEFUL,
        )
    )
    _install(operation_services, (stateful,))

    descriptor = next(
        descriptor
        for descriptor in operation_services.core.operations.snapshot().operations
        if descriptor.operation_id == stateful.operation_id
    )

    assert descriptor.read_only is False


def test_materialized_operation_omits_preview_without_projection(
    operation_services,
) -> None:
    materialized = durable_operation(
        OperationDeclaration(
            operation_id="synthetic.materialize.no_preview",
            version="2",
            title="Materialize a doubled integer without preview",
            description="Materialize one doubled integer with lineage but no inline preview.",
            request_type=_SyntheticRequest,
            result_type=_SyntheticResult,
            execute=lambda request: _SyntheticResult(doubled=request.value * 2),
            tags=("synthetic",),
        ),
        resource_reason="the test exercises durable lineage without a preview",
    )
    _install(operation_services, (materialized,))

    result = operation_services.core.operations.invoke(
        OperationRequest(
            operation_id="synthetic.materialize.no_preview",
            input={"value": 4},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert len(result.artifact_uris) == 2
    assert result.output["preview"] is None
    assert result.output["preview_complete"] is False
    assert result.output["backend_version"] == __version__
    assert operation_services.core.store.get(result.artifact_uris[1]).payload == {
        "doubled": 8
    }


def test_materialized_operation_fails_closed_before_artifact_writes(
    operation_services,
) -> None:
    not_applicable = OperationDiagnostic(
        code="SYNTHETIC_NOT_APPLICABLE",
        stage="synthetic_computation",
        message="Thirteen is excluded from this synthetic operation.",
    )

    def refuse(_request: _SyntheticRequest) -> _SyntheticResult:
        raise OperationRefusalError(not_applicable)

    materialized = durable_operation(
        OperationDeclaration(
            operation_id="synthetic.materialize.excluded",
            version="2",
            title="Materialize a doubled integer that excludes thirteen",
            description="Exercise fail-closed materialization.",
            request_type=_SyntheticRequest,
            result_type=_SyntheticResult,
            execute=refuse,
            tags=("synthetic",),
        ),
        resource_reason="the test exercises fail-closed durable computation",
        preview_type=_SyntheticPreview,
        preview=lambda result: _SyntheticPreview(summary=f"doubled={result.doubled}"),
        preview_complete=True,
    )
    _install(operation_services, (materialized,))

    result = operation_services.core.operations.invoke(
        OperationRequest(
            operation_id="synthetic.materialize.excluded",
            input={"value": 13},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "SYNTHETIC_NOT_APPLICABLE"
    assert result.artifact_uris == ()


def test_synthetic_bundle_fails_closed_before_artifact_writes(
    operation_services,
) -> None:
    _install(operation_services, _synthetic_bundle())

    result = operation_services.core.operations.invoke(
        OperationRequest(
            operation_id="synthetic.compute.double",
            input={"value": 13},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "SYNTHETIC_NOT_APPLICABLE"
    assert result.artifact_uris == ()


def test_computed_adapter_preserves_operational_failure_status(
    operation_services,
) -> None:
    bundle = _synthetic_bundle()
    diagnostic = OperationDiagnostic(
        code="SYNTHETIC_OPERATION_FAILED",
        stage="synthetic_computation",
        message="The synthetic operation did not complete.",
    )
    statuses = (
        ExecutionStatus.ERROR,
        ExecutionStatus.TIMEOUT,
        ExecutionStatus.CANCELLED,
    )

    def abort(
        status: ExecutionStatus,
    ) -> Any:
        def execute(_request: _SyntheticRequest) -> _SyntheticResult:
            raise OperationAbortError(status, diagnostic)

        return execute

    failed_operations = tuple(
        replace(
            bundle[0],
            operation_id=f"synthetic.compute.failure.{status.value.lower()}",
            run=abort(status),
        )
        for status in statuses
    )
    _install(operation_services, failed_operations)

    for status, operation in zip(statuses, failed_operations, strict=True):
        result = operation_services.core.operations.invoke(
            OperationRequest(
                operation_id=operation.operation_id,
                input={"value": 2},
            )
        )

        assert result.execution.status is status
        assert result.diagnostics == (diagnostic,), status
        assert result.artifact_uris == (), status


def test_computed_failure_rejects_conclusive_status() -> None:
    diagnostic = OperationDiagnostic(
        code="SYNTHETIC_OPERATION_FAILED",
        stage="synthetic_computation",
        message="The synthetic operation did not complete.",
    )

    with pytest.raises(ValueError, match="operational failure status"):
        Failed(ExecutionStatus.COMPLETED, diagnostic)


def test_computed_adapter_rejects_invalid_implementation_result(
    operation_services,
) -> None:
    bundle = _synthetic_bundle()
    original = bundle[0]
    invalid = inline_operation(
        OperationDeclaration(
            operation_id="synthetic.compute.invalid",
            version="2",
            title=original.title,
            description=original.description,
            request_type=_SyntheticRequest,
            result_type=_SyntheticResult,
            execute=lambda _request: cast(Any, {"doubled": "not-an-integer"}),
        )
    )
    _install(
        operation_services,
        (invalid,),
    )
    result = operation_services.core.operations.invoke(
        OperationRequest(
            operation_id="synthetic.compute.invalid",
            input={"value": 2},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "ADAPTER_EXECUTION_FAILED"
    assert result.artifact_uris == ()


def test_operation_specific_invalid_request_is_not_enriched(
    operation_services,
) -> None:
    """An operation that sets its own ``invalid_request`` diagnostic must
    receive it verbatim on ``ValidationError`` — the enrichment helper must
    not overwrite its ``path``, ``hint``, or any other field.

    ``_CrossFieldRequest`` fails its Pydantic ``model_validator`` at the one
    typed parse owned by the installed adapter.
    """

    operation_diagnostic = OperationDiagnostic(
        code="SYNTHETIC_OPERATION_INVALID_REQUEST",
        stage="synthetic_operation_validation",
        message="This operation-specific diagnostic must be preserved verbatim.",
        hint="Use a value between 0 and 100.",
        path="value",
    )
    operation = inline_operation(
        OperationDeclaration(
            operation_id="synthetic.compute.operation_diagnostic",
            version="2",
            title="Cross-field validated synthetic operation",
            description="Exercise operation-specific invalid_request preservation.",
            request_type=_CrossFieldRequest,
            result_type=_SyntheticResult,
            execute=lambda request: _SyntheticResult(doubled=request.value * 2),
            tags=("synthetic",),
            invalid_request=operation_diagnostic,
        )
    )
    _install(operation_services, (operation,))

    result = operation_services.core.operations.invoke(
        OperationRequest(
            operation_id="synthetic.compute.operation_diagnostic",
            input={"value": 50, "limit": 10},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0] == operation_diagnostic
    assert result.artifact_uris == ()


def test_binder_rejects_empty_and_duplicate_operation_groups(
    operation_services,
) -> None:
    installer = OperationBinder(
        operation_services.core.store,
        operation_services.core.schemas,
        operation_services.core.artifacts,
    )
    bundle = _synthetic_bundle()

    with pytest.raises(ValueError, match="must not be empty"):
        installer.bind(())
    with pytest.raises(ValueError, match="duplicate IDs"):
        installer.bind((bundle[0],) * 2)
