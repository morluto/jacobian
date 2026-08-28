"""Immutable declarations for integer-partition operations."""

from jacobian.catalog._examples import example
from jacobian.math.combinatorics._models import (
    IntegerResult,
    NonnegativeIntegerRequest,
    NonnegativePairRequest,
)
from jacobian.math.combinatorics._operations import (
    bell,
    enumerate_integer_partitions,
    partition_number,
    stirling_first,
    stirling_second,
)
from jacobian.math.combinatorics._partition_models import (
    IntegerPartitionEnumerationRequest,
    IntegerPartitionEnumerationResult,
)
from jacobian.math.combinatorics._support import (
    combinatorics_operation,
)

PARTITION_OPERATIONS = (
    combinatorics_operation(
        "combinatorics.compute.stirling_first",
        "Compute Stirling number of first kind",
        "Count permutations of n elements with k cycles, unsigned.",
        NonnegativePairRequest,
        IntegerResult,
        stirling_first,
        "combinatorics",
        "partition",
        examples=(
            example(
                "stirling_first_5_2",
                "Compute the unsigned Stirling number for n=5, k=2.",
                {"n": 5, "k": 2},
            ),
        ),
    ),
    combinatorics_operation(
        "combinatorics.compute.stirling_second",
        "Compute Stirling number of second kind",
        "Count partitions of n labeled elements into k nonempty blocks.",
        NonnegativePairRequest,
        IntegerResult,
        stirling_second,
        "combinatorics",
        "partition",
        examples=(
            example(
                "stirling_second_5_2",
                "Compute the Stirling number for n=5, k=2.",
                {"n": 5, "k": 2},
            ),
        ),
    ),
    combinatorics_operation(
        "combinatorics.compute.bell",
        "Compute Bell number",
        "Count set partitions of n labeled elements.",
        NonnegativeIntegerRequest,
        IntegerResult,
        bell,
        "combinatorics",
        "partition",
        examples=(example("bell_5", "Compute the fifth Bell number.", {"n": 5}),),
    ),
    combinatorics_operation(
        "combinatorics.compute.partition_number",
        "Compute partition number",
        "Count unordered additive partitions of n.",
        NonnegativeIntegerRequest,
        IntegerResult,
        partition_number,
        "combinatorics",
        "partition",
        examples=(
            example(
                "partition_number_6", "Count the additive partitions of 6.", {"n": 6}
            ),
        ),
    ),
    combinatorics_operation(
        "combinatorics.enumerate.integer_partitions",
        "Enumerate integer partitions",
        (
            "Enumerate every partition of bounded n containing at most "
            "max_parts summands, in canonical descending order."
        ),
        IntegerPartitionEnumerationRequest,
        IntegerPartitionEnumerationResult,
        enumerate_integer_partitions,
        "combinatorics",
        "partition",
        "enumeration",
        examples=(
            example(
                "partitions_of_5_with_two_parts",
                "Enumerate partitions of 5 using at most two parts.",
                {"n": 5, "max_parts": 2},
            ),
        ),
    ),
)
