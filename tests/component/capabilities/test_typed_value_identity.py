from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import CapabilityDiagnostic, CapabilityRequest
from jacobian.contracts.results import ContractModel, ExecutionStatus
from jacobian.domain_bundles import DomainBundle
from jacobian.operation_bindings import inline_operation
from jacobian.operation_ports import InputPort, OutputPort
from jacobian.operations import DomainDiagnostics, DomainSemantics, OperationSpec
from jacobian.provider_runtime import known_provider_runtime


class _SourceRequest(ContractModel):
    value: int


class _ExactValue(ContractModel):
    value: int


class _ConsumerRequest(ContractModel):
    source: _ExactValue
    increment: int


@pytest.fixture
def typed_value_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(tmp_path / "state") as services:
        yield services


def test_resolved_typed_value_retains_identity_while_payload_stays_strict(
    typed_value_services: DomainTestServices,
) -> None:
    produced_values: list[_ExactValue] = []
    consumed_values: list[_ExactValue] = []

    def produce(request: _SourceRequest) -> _ExactValue:
        value = _ExactValue(value=request.value)
        produced_values.append(value)
        return value

    def consume(request: _ConsumerRequest) -> _ExactValue:
        consumed_values.append(request.source)
        return _ExactValue(value=request.source.value + request.increment)

    producer = inline_operation(
        OperationSpec(
            operation_id="typed_identity.compute.produce",
            version="1",
            title="Produce an exact typed value",
            description="Produce one exact in-process value for composition.",
            request_type=_SourceRequest,
            result_type=_ExactValue,
            execute=produce,
        ),
        output_ports=(OutputPort(name="value", value_type=_ExactValue),),
    )
    consumer = inline_operation(
        OperationSpec(
            operation_id="typed_identity.compute.consume",
            version="1",
            title="Consume an exact typed value",
            description="Bind one resolved value without a JSON round trip.",
            request_type=_ConsumerRequest,
            result_type=_ExactValue,
            execute=consume,
        ),
        input_ports=(
            InputPort(
                name="source",
                value_type=_ExactValue,
                request_field="source",
            ),
        ),
    )
    bundle = DomainBundle(
        domain_id="typed-identity",
        schema_namespace="jacobian.typed-identity",
        semantics=DomainSemantics(
            name="jacobian.typed-identity",
            version="1",
            definition={"description": "typed identity regression semantics"},
        ),
        provider_runtime=known_provider_runtime(
            "jacobian.typed-identity",
            features=("deterministic",),
        ),
        backend_version="typed-identity-1",
        capabilities=(producer, consumer),
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_TYPED_IDENTITY_REQUEST",
                stage="typed_identity_input_validation",
                message="Input does not satisfy the typed identity contract.",
            )
        ),
    )
    installation = typed_value_services.core.operations.install(bundle)
    for adapter in installation.adapters:
        typed_value_services.core.capabilities.register(adapter)

    produced = typed_value_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=producer.spec.operation_id,
            input={"value": 12},
        )
    )
    assert produced.execution.status is ExecutionStatus.COMPLETED
    value_ref = produced.output["value_refs"]["value"]

    stringly = typed_value_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=consumer.spec.operation_id,
            input={"increment": "1"},
            inputs={"source": value_ref},
        )
    )
    assert stringly.execution.status is ExecutionStatus.ERROR
    assert consumed_values == []

    consumed = typed_value_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=consumer.spec.operation_id,
            input={"increment": 1},
            inputs={"source": value_ref},
        )
    )

    assert consumed.execution.status is ExecutionStatus.COMPLETED
    assert consumed.output["result"] == {"value": 13}
    assert consumed_values == [produced_values[0]]
    assert consumed_values[0] is produced_values[0]
