from __future__ import annotations

import pytest

from jacobian.contracts.base import ContractModel
from jacobian.operation_declarations import (
    DurablePublication,
    InlinePublication,
    OperationDeclaration,
    OperationExample,
)


class Request(ContractModel):
    value: int


class Result(ContractModel):
    value: int


def declaration(**changes: object) -> OperationDeclaration[Request, Result]:
    arguments = {
        "operation_id": "arithmetic.increment",
        "version": "1",
        "title": "Increment an integer",
        "description": "Compute an exact integer increment.",
        "tags": ("arithmetic", "integer"),
        "request_type": Request,
        "result_type": Result,
        "execute": lambda request: Result(value=request.value + 1),
        "publication": InlinePublication(),
    }
    arguments.update(changes)
    return OperationDeclaration(**arguments)  # type: ignore[arg-type]


def test_declaration_is_pure_typed_data_plus_kernel_binding() -> None:
    operation = declaration()

    assert operation.execute(Request(value=4)) == Result(value=5)


def test_declaration_rejects_duplicate_examples() -> None:
    example = OperationExample(
        name="small",
        description="A small input.",
        input={"value": 1},
    )

    with pytest.raises(ValueError, match="example names must be unique"):
        declaration(examples=(example, example))


def test_declaration_rejects_durable_output_reference() -> None:
    from jacobian.operation_ports import OutputPort

    with pytest.raises(ValueError, match="durable operations cannot publish"):
        declaration(
            publication=DurablePublication[Result, Result](
                resource_reason="durable identity",
            ),
            output_ports=(OutputPort(name="result", value_type=Result),),
        )
