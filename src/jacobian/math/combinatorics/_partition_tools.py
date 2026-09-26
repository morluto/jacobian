"""Immutable declarations for integer-partition operations."""

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics import operations as native
from jacobian.math.combinatorics._models import (
    IntegerResult,
    NonnegativeIntegerRequest,
    NonnegativePairRequest,
)
from jacobian.math.combinatorics._partition_models import (
    MAX_PARTITION_SIZE,
    MAX_PARTITION_ITEM,
    IncreasingPartsObstruction,
    IntegerPartitionEnumerationRequest,
    IntegerPartitionEnumerationResult,
    NonpositivePartObstruction,
    PartitionCheckRequest,
    PartitionCheckResult,
    PartitionFound,
    PartitionRejected,
)
from jacobian.math.combinatorics.symmetric_functions.values import IntegerPartition


def _integer_result(value: int) -> IntegerResult:
    return IntegerResult(value=value)


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


def check_partition(request: PartitionCheckRequest) -> PartitionCheckResult:
    """Return the canonical partition or the first defining obstruction."""
    if not isinstance(request, PartitionCheckRequest):
        raise OperationDomainValidationError(
            location=(),
            code="combinatorics.partition_request_type",
            message="request must be a partition-check request",
        )
    parts = getattr(request, "parts", None)
    if type(parts) is not tuple or len(parts) > MAX_PARTITION_SIZE:
        raise OperationDomainValidationError(
            location=("parts",),
            code="combinatorics.partition_candidate_shape",
            message="candidate must be a bounded tuple of exact integers",
        )
    if any(
        type(part) is not int or abs(part) > MAX_PARTITION_ITEM
        for part in parts
    ):
        raise OperationDomainValidationError(
            location=("parts",),
            code="combinatorics.partition_candidate_integer",
            message="candidate parts must be exact JSON-safe integers",
        )
    try:
        parts = PartitionCheckRequest.model_validate({"parts": parts}).parts
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("parts",),
            code="combinatorics.partition_request_invalid",
            message="request must satisfy the partition-check input contract",
        ) from exc
    previous: int | None = None
    for index, part in enumerate(parts):
        if part <= 0:
            return PartitionCheckResult(
                outcome=PartitionRejected(
                    parts=parts,
                    obstruction=NonpositivePartObstruction(index=index, value=part),
                )
            )
        if previous is not None and previous < part:
            return PartitionCheckResult(
                outcome=PartitionRejected(
                    parts=parts,
                    obstruction=IncreasingPartsObstruction(
                        index=index,
                        previous_value=previous,
                        value=part,
                    ),
                )
            )
        previous = part

    if sum(parts) > MAX_PARTITION_SIZE:
        raise OperationResourceAdmissionError(
            location=("parts",),
            code="combinatorics.partition_candidate_size",
            message="valid partition candidate exceeds the supported size",
        )

    partition = IntegerPartition(parts=parts)
    conjugate = tuple(
        sum(part >= column for part in parts)
        for column in range(1, (parts[0] if parts else 0) + 1)
    )
    cells = tuple(
        (row, column)
        for row, part in enumerate(parts, start=1)
        for column in range(1, part + 1)
    )
    return PartitionCheckResult(
        outcome=PartitionFound._from_checked(
            partition=partition,
            conjugate=IntegerPartition(parts=conjugate),
            cells=cells,
        )
    )


PARTITION_OPERATIONS = (
    MathTool(
        operation_id="combinatorics.partition.check",
        title="Check an integer partition candidate",
        description=(
            "Classify a bounded sequence of exact integers as an integer "
            "partition, returning its canonical value and Ferrers data, or "
            "the first nonpositive part or adjacent increase."
        ),
        request_type=PartitionCheckRequest,
        result_type=PartitionCheckResult,
        run=check_partition,
        tags=("combinatorics", "partition", "exact"),
        examples=(
            OperationExample(
                name="partition_candidate",
                description="Check the partition (4, 2, 1).",
                input={"parts": [4, 2, 1]},
            ),
            OperationExample(
                name="partition_obstruction",
                description="Locate the first increasing adjacent pair.",
                input={"parts": [3, 4, 1]},
            ),
        ),
    ),
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
