"""Sequence search and frequency operations over finite integer sequences."""

from jacobian.catalog._examples import example
from jacobian.math.sequences._models import (
    IntegerSequenceFrequenciesResult,
    IntegerSequenceIndexListResult,
    IntegerSequenceRequest,
)
from jacobian.math.sequences._operations import (
    frequencies,
    zero_indices,
)
from jacobian.math.sequences._support import sequence_operation

SEQUENCE_SEARCH_OPERATIONS = (
    sequence_operation(
        "sequence.compute.frequencies",
        "Compute value frequencies",
        "Count each distinct value in a finite integer sequence.",
        IntegerSequenceRequest,
        IntegerSequenceFrequenciesResult,
        frequencies,
        "sequence",
        "counting",
        examples=(
            example(
                "frequencies_1_2_1",
                "Count frequencies in 1, 2, and 1.",
                {"values": ["1", "2", "1"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.compute.zero_indices",
        "Locate zero terms",
        "Return zero-based indices whose sequence value is zero.",
        IntegerSequenceRequest,
        IntegerSequenceIndexListResult,
        zero_indices,
        "sequence",
        "search",
        examples=(
            example(
                "zero_indices_2_0_3",
                "Locate zero terms in 2, 0, and 3.",
                {"values": ["2", "0", "3"]},
            ),
        ),
    ),
)
