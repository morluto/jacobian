from __future__ import annotations

from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics._partition_models import (
    MAX_ENUMERATED_PARTITIONS,
    MAX_PARTITION_N,
    IntegerPartitionEnumerationRequest,
    IntegerPartitionEnumerationResult,
)


@contextmanager
def raises_code(code: str):
    with pytest.raises(ValidationError) as exc_info:
        yield
    assert exc_info.value.errors()[0]["type"] == code


def test_partition_enumeration_schema_retains_its_admission_bounds() -> None:
    request_schema = IntegerPartitionEnumerationRequest.model_json_schema()
    result_schema = IntegerPartitionEnumerationResult.model_json_schema()

    assert request_schema["properties"]["n"]["maximum"] == MAX_PARTITION_N
    assert request_schema["properties"]["max_parts"]["maximum"] == MAX_PARTITION_N
    assert result_schema["properties"]["partitions"]["maxItems"] == (
        MAX_ENUMERATED_PARTITIONS
    )


def test_partition_enumeration_result_retains_canonical_order() -> None:
    result = IntegerPartitionEnumerationResult(
        n=5,
        max_parts=2,
        partitions=((5,), (4, 1), (3, 2)),
    )

    assert result.partitions == ((5,), (4, 1), (3, 2))

    with raises_code("combinatorics.partition_invariant"):
        IntegerPartitionEnumerationResult(
            n=5,
            max_parts=2,
            partitions=((3, 2), (4, 1), (5,)),
        )


def test_zero_partition_enumeration_requires_the_empty_partition() -> None:
    with raises_code("combinatorics.partition_invariant"):
        IntegerPartitionEnumerationResult(n=0, max_parts=1, partitions=())
