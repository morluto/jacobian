"""Tests for symmetric function operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.symmetric_functions._models import (
    IntegerPartition,
    PartitionRequest,
    SchurExpansionRequest,
)
from jacobian.math.symmetric_functions._operations import (
    compute_partition_conjugate,
    compute_schur_evaluation,
)
from jacobian.math.symmetric_functions._tools import TOOLS


def test_operations_in_catalog() -> None:
    tools = {tool.operation_id: tool for tool in TOOLS}
    assert "symmetric_function.schur.evaluate.compute" in tools
    # The narrowed request envelope (50 parts) is a versioned contract change.
    # conjugate is NATIVE_ONLY via algebraic_combinatorics; not a distinct public operation
    assert "symmetric_function.partition.conjugate.compute" not in tools


def test_conjugate_self_conjugate_partition() -> None:
    result = compute_partition_conjugate(
        PartitionRequest(partition=IntegerPartition(parts=(3, 2, 1)))
    )
    assert result.conjugate.parts == (3, 2, 1)


def test_conjugate_non_self_conjugate() -> None:
    result = compute_partition_conjugate(
        PartitionRequest(partition=IntegerPartition(parts=(4, 2)))
    )
    assert result.conjugate.parts == (2, 2, 1, 1)


def test_conjugate_single_row() -> None:
    result = compute_partition_conjugate(
        PartitionRequest(partition=IntegerPartition(parts=(5,)))
    )
    assert result.conjugate.parts == (1, 1, 1, 1, 1)


def test_partition_domain_is_closed_under_boundary_conjugation() -> None:
    source = IntegerPartition(parts=(500,))
    result = compute_partition_conjugate(PartitionRequest(partition=source))
    assert result.conjugate.parts == (1,) * 500
    assert (
        compute_partition_conjugate(
            PartitionRequest(partition=result.conjugate)
        ).conjugate
        == source
    )


def test_conjugate_empty() -> None:
    result = compute_partition_conjugate(
        PartitionRequest(partition=IntegerPartition(parts=()))
    )
    assert result.conjugate.parts == ()


def test_schur_single_variable_one() -> None:
    result = compute_schur_evaluation(
        SchurExpansionRequest(
            partition=IntegerPartition(parts=(1,)),
            variables=("x",),
            point=(1,),
        )
    )
    assert result.value == "1"


def test_schur_complete_homogeneous_at_ones() -> None:
    # s_(2)(1,1) = h_2(1,1) = 3
    result = compute_schur_evaluation(
        SchurExpansionRequest(
            partition=IntegerPartition(parts=(2,)),
            variables=("x1", "x2"),
            point=(1, 1),
        )
    )
    assert result.value == "3"


def test_schur_elementary_at_ones() -> None:
    # s_(1,1)(1,1) = e_2(1,1) = x1*x2 = 1
    result = compute_schur_evaluation(
        SchurExpansionRequest(
            partition=IntegerPartition(parts=(1, 1)),
            variables=("x1", "x2"),
            point=(1, 1),
        )
    )
    assert result.value == "1"


def test_schur_at_origin() -> None:
    result = compute_schur_evaluation(
        SchurExpansionRequest(
            partition=IntegerPartition(parts=(3, 2, 1)),
            variables=("x1", "x2", "x3"),
            point=(0, 0, 0),
        )
    )
    assert result.value == "0"


def test_partition_rejects_non_decreasing() -> None:
    with pytest.raises(ValidationError) as error:
        IntegerPartition(parts=(1, 2, 3))
    assert (
        error.value.errors()[0]["type"]
        == "symmetric_function.partition_not_weakly_decreasing"
    )


def test_partition_rejects_non_positive() -> None:
    with pytest.raises(ValidationError) as error:
        IntegerPartition(parts=(3, 0, 1))
    assert (
        error.value.errors()[0]["type"]
        == "symmetric_function.partition_parts_not_positive"
    )


def test_schur_rejects_mismatched_dimensions() -> None:
    with pytest.raises(ValidationError) as error:
        SchurExpansionRequest(
            partition=IntegerPartition(parts=(1,)),
            variables=("x1", "x2"),
            point=(1,),
        )
    assert (
        error.value.errors()[0]["type"]
        == "symmetric_function.schur_dimensions_mismatch"
    )


def test_request_schema_publishes_schur_invariants() -> None:
    """math.find must expose the Schur preconditions without a failed call."""
    schema = SchurExpansionRequest.model_json_schema()
    variables = schema["properties"]["variables"]
    point = schema["properties"]["point"]
    assert "istinct" in variables["description"]
    assert "equal" in variables["description"]
    assert variables.get("uniqueItems") is True
    assert variables["minItems"] == 1 and variables["maxItems"] == 20
    assert point["items"]["minimum"] == -999_999
    assert point["items"]["maximum"] == 999_999
    assert "decimal digits" in point["items"]["description"]
    partition = schema["properties"]["partition"]
    assert partition["title"] == "IntegerPartition"
    assert "500" in partition["description"]
    assert "at most 50" in partition["description"]
    parts = partition["properties"]["parts"]
    assert parts["maxItems"] == 50
    assert "at most 50" in parts["description"]
    assert "500" in parts["description"]


def test_schur_rejects_coordinate_exceeding_digit_bound() -> None:
    with pytest.raises(ValueError):
        SchurExpansionRequest(
            partition=IntegerPartition(parts=(1,)),
            variables=("x1",),
            point=(1_000_000,),
        )


def test_schur_accepts_boundary_coordinate() -> None:
    request = SchurExpansionRequest(
        partition=IntegerPartition(parts=(1,)),
        variables=("x1",),
        point=(-999_999,),
    )
    assert request.point == (-999_999,)


def test_partition_schema_rejects_size_above_cap() -> None:
    with pytest.raises(ValidationError) as error:
        IntegerPartition(parts=(251, 250))
    assert (
        error.value.errors()[0]["type"] == "symmetric_function.partition_size_exceeded"
    )


def test_schur_request_retains_its_operation_specific_length_bound() -> None:
    with pytest.raises(ValidationError) as error:
        SchurExpansionRequest(
            partition=IntegerPartition(parts=(1,) * 51),
            variables=("x",),
            point=(1,),
        )
    assert (
        error.value.errors()[0]["type"]
        == "symmetric_function.schur_partition_length_exceeded"
    )


def test_schur_accepts_boundary_partition_length() -> None:
    request = SchurExpansionRequest(
        partition=IntegerPartition(parts=(1,) * 50),
        variables=("x",),
        point=(0,),
    )
    assert isinstance(request.partition, IntegerPartition)
    assert compute_schur_evaluation(request).value == "0"
