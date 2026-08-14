from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import ContractModel, ExecutionStatus
from jacobian.operation_binding import OperationBinder
from jacobian.operation_bindings import inline_operation
from jacobian.operation_declarations import OperationDeclaration
from jacobian.operation_ports import InputPort, OutputPort


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
        OperationDeclaration(
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
        OperationDeclaration(
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
    operations = (producer, consumer)
    installation = OperationBinder(
        typed_value_services.core.store,
        typed_value_services.core.schemas,
        typed_value_services.core.artifacts,
        typed_value_services.core.values,
    ).bind(operations)
    for adapter in installation.adapters:
        typed_value_services.core.operations.register(adapter)

    produced = typed_value_services.core.operations.invoke(
        OperationRequest(
            operation_id=producer.operation_id,
            input={"value": 12},
        )
    )
    assert produced.execution.status is ExecutionStatus.COMPLETED
    value_ref = produced.output["value_refs"]["value"]

    stringly = typed_value_services.core.operations.invoke(
        OperationRequest(
            operation_id=consumer.operation_id,
            input={"increment": "1"},
            inputs={"source": value_ref},
        )
    )
    assert stringly.execution.status is ExecutionStatus.ERROR
    assert consumed_values == []

    consumed = typed_value_services.core.operations.invoke(
        OperationRequest(
            operation_id=consumer.operation_id,
            input={"increment": 1},
            inputs={"source": value_ref},
        )
    )

    assert consumed.execution.status is ExecutionStatus.COMPLETED
    assert consumed.output["result"] == {"value": 13}
    assert consumed_values == [produced_values[0]]
    assert consumed_values[0] is produced_values[0]
