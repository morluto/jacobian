from __future__ import annotations

import pytest

from jacobian.contracts.base import ContractModel
from jacobian.operation_bindings import (
    DurablePublication,
    InlinePublication,
    InstalledOperation,
)
from jacobian.operation_ports import InputPort, OutputPort
from jacobian.operations import OperationSpec


class MatrixValue(ContractModel):
    entries: tuple[tuple[int, ...], ...]


class OtherValue(ContractModel):
    value: int


class RankRequest(ContractModel):
    matrix: MatrixValue


class RankResult(ContractModel):
    rank: int
    matrix: MatrixValue


def _spec() -> OperationSpec[RankRequest, RankResult]:
    return OperationSpec(
        operation_id="test.matrix.rank",
        version="1",
        request_type=RankRequest,
        result_type=RankResult,
        execute=lambda request: RankResult(rank=1, matrix=request.matrix),
        title="Rank",
        description="Compute a test matrix rank.",
    )


def test_ports_bind_and_extract_exact_typed_fields() -> None:
    input_port = InputPort[MatrixValue](
        name="matrix",
        value_type=MatrixValue,
        request_field="matrix",
    )
    output_port = OutputPort[RankResult](
        name="rank",
        value_type=RankResult,
    )
    InstalledOperation(
        spec=_spec(),
        publication=InlinePublication(),
        input_ports=(input_port,),
        output_ports=(output_port,),
    )
    matrix = MatrixValue(entries=((1, 0), (0, 1)))

    assert input_port.bind_to_request({}, matrix) == {"matrix": matrix}
    result = RankResult(rank=2, matrix=matrix)
    assert output_port.extract_from_result(result) is result


def test_ports_reject_conflicts_and_same_shape_wrong_types() -> None:
    input_port = InputPort[MatrixValue](
        name="matrix",
        value_type=MatrixValue,
        request_field="matrix",
    )
    with pytest.raises(ValueError, match="conflicts"):
        input_port.bind_to_request(
            {"matrix": {"entries": []}},
            MatrixValue(entries=()),
        )
    with pytest.raises(TypeError, match="requires MatrixValue"):
        input_port.bind_to_request({}, OtherValue(value=1))  # type: ignore[arg-type]


def test_installation_rejects_ports_that_disagree_with_model_fields() -> None:
    with pytest.raises(ValueError, match="does not match"):
        InstalledOperation(
            spec=_spec(),
            publication=InlinePublication(),
            input_ports=(
                InputPort[OtherValue](
                    name="matrix",
                    value_type=OtherValue,
                    request_field="matrix",
                ),
            ),
        )
    with pytest.raises(ValueError, match="does not match"):
        InstalledOperation(
            spec=_spec(),
            publication=InlinePublication(),
            output_ports=(
                OutputPort[MatrixValue](
                    name="matrix",
                    value_type=MatrixValue,
                ),
            ),
        )


def test_installation_rejects_multiple_whole_result_output_ports() -> None:
    with pytest.raises(ValueError, match="at most one output port"):
        InstalledOperation(
            spec=_spec(),
            publication=InlinePublication(),
            output_ports=(
                OutputPort[RankResult](name="first", value_type=RankResult),
                OutputPort[RankResult](name="second", value_type=RankResult),
            ),
        )


def test_durable_publication_rejects_request_local_output_references() -> None:
    with pytest.raises(ValueError, match="durable operations cannot publish"):
        InstalledOperation(
            spec=_spec(),
            publication=DurablePublication(resource_reason="large result"),
            output_ports=(OutputPort[RankResult](name="rank", value_type=RankResult),),
        )
