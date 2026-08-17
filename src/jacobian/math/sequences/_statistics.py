"""Sequence statistic operations producing rational results."""

from jacobian.catalog._examples import example
from jacobian.math.sequences._models import (
    IntegerSequenceRationalResult,
    IntegerSequenceRequest,
)
from jacobian.math.sequences._operations import (
    sequence_mean,
    sequence_median,
)
from jacobian.math.sequences._support import sequence_operation

SEQUENCE_STATISTIC_OPERATIONS = (
    sequence_operation(
        "sequence.compute.mean",
        "Compute sequence mean",
        "Compute the reduced arithmetic mean of a finite integer sequence.",
        IntegerSequenceRequest,
        IntegerSequenceRationalResult,
        sequence_mean,
        "sequence",
        "statistic",
        examples=(
            example(
                "mean_1_2_3",
                "Compute the mean of 1, 2, and 3.",
                {"values": ["1", "2", "3"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.compute.median",
        "Compute sequence median",
        "Compute the reduced median of a finite integer sequence.",
        IntegerSequenceRequest,
        IntegerSequenceRationalResult,
        sequence_median,
        "sequence",
        "statistic",
        examples=(
            example(
                "median_1_3_2",
                "Compute the median of 1, 3, and 2.",
                {"values": ["1", "3", "2"]},
            ),
        ),
    ),
)
