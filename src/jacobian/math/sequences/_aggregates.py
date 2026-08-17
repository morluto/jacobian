"""Sequence aggregate operations over finite integer sequences."""

from jacobian.catalog._examples import example
from jacobian.math.sequences._models import (
    IntegerSequenceRequest,
    IntegerSequenceValueResult,
)
from jacobian.math.sequences._operations import (
    sequence_distinct_count,
    sequence_gcd,
    sequence_lcm,
    sequence_maximum,
    sequence_minimum,
    sequence_product,
    sequence_range,
    sequence_sum,
)
from jacobian.math.sequences._support import sequence_operation

SEQUENCE_AGGREGATE_OPERATIONS = (
    sequence_operation(
        "sequence.compute.sum",
        "Sum integer sequence",
        "Compute the exact sum of a finite integer sequence.",
        IntegerSequenceRequest,
        IntegerSequenceValueResult,
        sequence_sum,
        "sequence",
        "exact",
        examples=(
            example(
                "sum_1_2_3",
                "Sum the sequence 1, 2, and 3.",
                {"values": ["1", "2", "3"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.compute.product",
        "Multiply integer sequence",
        "Compute the exact product of a finite integer sequence.",
        IntegerSequenceRequest,
        IntegerSequenceValueResult,
        sequence_product,
        "sequence",
        "exact",
        examples=(
            example(
                "product_2_3_4",
                "Multiply the sequence 2, 3, and 4.",
                {"values": ["2", "3", "4"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.compute.gcd",
        "Compute sequence gcd",
        "Compute the gcd of every value in a finite integer sequence.",
        IntegerSequenceRequest,
        IntegerSequenceValueResult,
        sequence_gcd,
        "sequence",
        "divisibility",
        examples=(
            example(
                "gcd_12_18", "Compute the gcd of 12 and 18.", {"values": ["12", "18"]}
            ),
        ),
    ),
    sequence_operation(
        "sequence.compute.lcm",
        "Compute sequence lcm",
        "Compute the lcm of every value in a finite integer sequence.",
        IntegerSequenceRequest,
        IntegerSequenceValueResult,
        sequence_lcm,
        "sequence",
        "divisibility",
        examples=(
            example("lcm_4_6", "Compute the lcm of 4 and 6.", {"values": ["4", "6"]}),
        ),
    ),
    sequence_operation(
        "sequence.compute.minimum",
        "Compute sequence minimum",
        "Compute the least value in a finite integer sequence.",
        IntegerSequenceRequest,
        IntegerSequenceValueResult,
        sequence_minimum,
        "sequence",
        "order",
        examples=(
            example(
                "minimum_3_1_2",
                "Find the minimum of 3, 1, and 2.",
                {"values": ["3", "1", "2"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.compute.maximum",
        "Compute sequence maximum",
        "Compute the greatest value in a finite integer sequence.",
        IntegerSequenceRequest,
        IntegerSequenceValueResult,
        sequence_maximum,
        "sequence",
        "order",
        examples=(
            example(
                "maximum_1_3_2",
                "Find the maximum of 1, 3, and 2.",
                {"values": ["1", "3", "2"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.compute.range",
        "Compute sequence range",
        "Compute maximum minus minimum for a finite integer sequence.",
        IntegerSequenceRequest,
        IntegerSequenceValueResult,
        sequence_range,
        "sequence",
        "statistic",
        examples=(
            example(
                "range_1_4_2",
                "Compute the range of 1, 4, and 2.",
                {"values": ["1", "4", "2"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.compute.distinct_count",
        "Count distinct sequence values",
        "Count distinct values in a finite integer sequence.",
        IntegerSequenceRequest,
        IntegerSequenceValueResult,
        sequence_distinct_count,
        "sequence",
        "counting",
        examples=(
            example(
                "distinct_count_1_2_1",
                "Count distinct values in 1, 2, and 1.",
                {"values": ["1", "2", "1"]},
            ),
        ),
    ),
)
