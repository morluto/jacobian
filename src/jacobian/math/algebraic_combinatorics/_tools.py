"""Algebraic combinatorics operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.algebraic_combinatorics._models import (
    ConjugatePartitionRequest,
    ConjugatePartitionResult,
    HookLengthRequest,
    HookLengthResult,
    StandardYoungTableauCountRequest,
    StandardYoungTableauCountResult,
)
from jacobian.math.algebraic_combinatorics._operations import (
    compute_conjugate_partition,
    compute_hook_lengths,
    compute_syt_count,
)


def ac_operation[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


_PARTITION_321 = {"partition": {"parts": [3, 2, 1]}}

ALGEBRAIC_COMBINATORICS_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    ac_operation(
        "combinatorics.hook_length.compute",
        "Compute hook lengths of a Young diagram",
        "Compute the hook length H(i,j) = lambda_i - j + lambda'_j - i + 1 "
        "for each cell (i,j) of the Young diagram of a partition.",
        HookLengthRequest,
        HookLengthResult,
        compute_hook_lengths,
        "combinatorics",
        "hook-length",
        "exact",
        examples=(
            example(
                "partition_321",
                "Hook lengths of partition (3, 2, 1).",
                _PARTITION_321,
            ),
        ),
    ),
    ac_operation(
        "combinatorics.standard_young_tableaux.count",
        "Count standard Young tableaux via the hook length formula",
        "Count the number of standard Young tableaux of a given shape using "
        "the hook length formula: f^lambda = n! / product of hook lengths.",
        StandardYoungTableauCountRequest,
        StandardYoungTableauCountResult,
        compute_syt_count,
        "combinatorics",
        "young-tableaux",
        "exact",
        examples=(
            example(
                "partition_321",
                "Number of SYT for shape (3, 2, 1) is 16.",
                _PARTITION_321,
            ),
        ),
    ),
    ac_operation(
        "combinatorics.conjugate_partition.compute",
        "Compute the conjugate (transpose) partition",
        "Compute the conjugate partition lambda' by transposing the Ferrers "
        "diagram of a partition lambda.",
        ConjugatePartitionRequest,
        ConjugatePartitionResult,
        compute_conjugate_partition,
        "combinatorics",
        "partition",
        "exact",
        examples=(
            example(
                "partition_321",
                "Conjugate of partition (3, 2, 1) is (3, 2, 1).",
                _PARTITION_321,
            ),
        ),
    ),
)

TOOLS = ALGEBRAIC_COMBINATORICS_OPERATIONS

__all__ = ["TOOLS"]
