from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics._partition_models import (
    IncreasingPartsObstruction,
    NonpositivePartObstruction,
    PartitionCheckRequest,
    PartitionFound,
    PartitionRejected,
)
from jacobian.math.combinatorics._partition_tools import check_partition
from jacobian.math.combinatorics.symmetric_functions.values import IntegerPartition


def test_partition_check_returns_canonical_value_and_ferrers_data() -> None:
    result = check_partition(PartitionCheckRequest(parts=[4, 2, 1]))

    assert isinstance(result.outcome, PartitionFound)
    assert result.outcome.partition == IntegerPartition(parts=(4, 2, 1))
    assert result.outcome.size == 7
    assert result.outcome.length == 3
    assert result.outcome.conjugate == IntegerPartition(parts=(3, 2, 1, 1))
    assert result.outcome.cells == (
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 1),
        (2, 2),
        (3, 1),
    )


def test_empty_sequence_is_the_partition_of_zero() -> None:
    result = check_partition(PartitionCheckRequest(parts=[]))

    assert isinstance(result.outcome, PartitionFound)
    assert result.outcome.partition.parts == ()
    assert result.outcome.conjugate.parts == ()
    assert result.outcome.cells == ()
    assert result.outcome.size == result.outcome.length == 0


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        ((0, 4), NonpositivePartObstruction(index=0, value=0)),
        ((3, -1, 5), NonpositivePartObstruction(index=1, value=-1)),
        ((4, 2, 3, 0), IncreasingPartsObstruction(index=2, previous_value=2, value=3)),
        ((4, 5, 0), IncreasingPartsObstruction(index=1, previous_value=4, value=5)),
    ],
)
def test_nonpartition_result_identifies_first_left_to_right_obstruction(
    parts: tuple[int, ...], expected: object
) -> None:
    result = check_partition(PartitionCheckRequest(parts=parts))

    assert isinstance(result.outcome, PartitionRejected)
    assert result.outcome.obstruction == expected


def _first_obstruction(parts: tuple[int, ...]) -> tuple[str, int, int | None]:
    previous: int | None = None
    for index, part in enumerate(parts):
        if part <= 0:
            return "NONPOSITIVE_PART", index, part
        if previous is not None and previous < part:
            return "INCREASING_ADJACENT_PARTS", index, previous
        previous = part
    return "PARTITION", len(parts), None


def test_small_candidate_sequences_match_independent_classifier() -> None:
    for length in range(5):
        for parts in product(range(-1, 4), repeat=length):
            expected, index, prior = _first_obstruction(parts)
            result = check_partition(PartitionCheckRequest(parts=parts))
            if expected == "PARTITION":
                assert isinstance(result.outcome, PartitionFound)
                assert result.outcome.partition.parts == parts
            else:
                assert isinstance(result.outcome, PartitionRejected)
                assert result.outcome.obstruction.kind == expected
                assert result.outcome.obstruction.index == index
                if expected == "INCREASING_ADJACENT_PARTS":
                    assert isinstance(
                        result.outcome.obstruction, IncreasingPartsObstruction
                    )
                    assert result.outcome.obstruction.previous_value == prior


def test_raw_candidate_is_bounded_before_request_model_construction() -> None:
    with pytest.raises(ValidationError) as exc_info:
        PartitionCheckRequest.model_validate({"parts": [1] * 501})
    assert exc_info.value.errors()[0]["type"] == (
        "combinatorics.partition_candidate_length"
    )

    with pytest.raises(ValidationError):
        PartitionCheckRequest(parts=("3",))

    with pytest.raises(ValidationError) as exc_info:
        PartitionCheckRequest(parts=(501,))
    assert exc_info.value.errors()[0]["type"] == (
        "combinatorics.partition_candidate_size"
    )


def test_cell_output_at_the_admitted_bound_is_complete() -> None:
    result = check_partition(PartitionCheckRequest(parts=(1,) * 500))

    assert isinstance(result.outcome, PartitionFound)
    assert result.outcome.size == 500
    assert result.outcome.length == 500
    assert len(result.outcome.cells) == 500
    assert result.outcome.cells[0] == (1, 1)
    assert result.outcome.cells[-1] == (500, 1)
