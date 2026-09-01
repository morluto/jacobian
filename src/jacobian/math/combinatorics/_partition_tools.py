"""Immutable declarations for integer-partition operations."""

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics import operations as native
from jacobian.math.combinatorics._models import (
    IntegerResult,
    NonnegativeIntegerRequest,
    NonnegativePairRequest,
)
from jacobian.math.combinatorics._partition_models import (
    IntegerPartitionEnumerationRequest,
    IntegerPartitionEnumerationResult,
)


def _integer_result(value: int) -> IntegerResult:
    return IntegerResult(value=format_canonical_integer(value))


def stirling_first(request: NonnegativePairRequest) -> IntegerResult:
    return _integer_result(native.stirling_first(request.n, request.k))


def stirling_second(request: NonnegativePairRequest) -> IntegerResult:
    return _integer_result(native.stirling_second(request.n, request.k))


def bell(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.bell_number(request.n))


def partition_number(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.partition_number(request.n))


def enumerate_integer_partitions(
    request: IntegerPartitionEnumerationRequest,
) -> IntegerPartitionEnumerationResult:
    """Enumerate all bounded partitions using the native exact kernel."""
    return IntegerPartitionEnumerationResult(
        n=request.n,
        max_parts=request.max_parts,
        partitions=native.integer_partitions(request.n, max_parts=request.max_parts),
    )


PARTITION_OPERATIONS = (
    MathTool(
        operation_id="combinatorics.compute.stirling_first",
        title="Compute Stirling number of first kind",
        description="Count permutations of n elements with k cycles, unsigned.",
        request_type=NonnegativePairRequest,
        result_type=IntegerResult,
        run=stirling_first,
        tags=("combinatorics", "partition"),
        examples=(
            OperationExample(
                name="stirling_first_5_2",
                description="Compute the unsigned Stirling number for n=5, k=2.",
                input={"n": 5, "k": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.compute.stirling_second",
        title="Compute Stirling number of second kind",
        description="Count partitions of n labeled elements into k nonempty blocks.",
        request_type=NonnegativePairRequest,
        result_type=IntegerResult,
        run=stirling_second,
        tags=("combinatorics", "partition"),
        examples=(
            OperationExample(
                name="stirling_second_5_2",
                description="Compute the Stirling number for n=5, k=2.",
                input={"n": 5, "k": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.compute.bell",
        title="Compute Bell number",
        description="Count set partitions of n labeled elements.",
        request_type=NonnegativeIntegerRequest,
        result_type=IntegerResult,
        run=bell,
        tags=("combinatorics", "partition"),
        examples=(
            OperationExample(
                name="bell_5",
                description="Compute the fifth Bell number.",
                input={"n": 5},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.compute.partition_number",
        title="Compute partition number",
        description="Count unordered additive partitions of n.",
        request_type=NonnegativeIntegerRequest,
        result_type=IntegerResult,
        run=partition_number,
        tags=("combinatorics", "partition"),
        examples=(
            OperationExample(
                name="partition_number_6",
                description="Count the additive partitions of 6.",
                input={"n": 6},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.enumerate.integer_partitions",
        title="Enumerate integer partitions",
        description=(
            "Enumerate every partition of bounded n containing at most "
            "max_parts summands, in canonical descending order."
        ),
        request_type=IntegerPartitionEnumerationRequest,
        result_type=IntegerPartitionEnumerationResult,
        run=enumerate_integer_partitions,
        tags=("combinatorics", "partition", "enumeration"),
        examples=(
            OperationExample(
                name="partitions_of_5_with_two_parts",
                description="Enumerate partitions of 5 using at most two parts.",
                input={"n": 5, "max_parts": 2},
            ),
        ),
    ),
)
