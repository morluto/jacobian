"""Sequence predicate operations over finite integer sequences."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory.sequences.core._models import (
    IntegerSequenceBooleanResult,
    IntegerSequenceRequest,
)
from jacobian.math.number_theory.sequences.core._operations import (
    decide_arithmetic,
    decide_geometric,
    decide_nondecreasing,
    decide_strictly_increasing,
)
from jacobian.math.number_theory.sequences.core._support import sequence_operation

SEQUENCE_PREDICATE_OPERATIONS = (
    sequence_operation(
        "sequence.decide.arithmetic",
        "Decide arithmetic progression",
        "Decide whether consecutive terms have one common difference.",
        IntegerSequenceRequest,
        IntegerSequenceBooleanResult,
        decide_arithmetic,
        "sequence",
        "predicate",
        examples=(
            example(
                "arithmetic_sequence",
                "Recognize an arithmetic sequence.",
                {"values": ["3", "6", "9", "12"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.decide.geometric",
        "Decide geometric progression",
        "Decide whether a finite integer sequence has a consistent rational ratio.",
        IntegerSequenceRequest,
        IntegerSequenceBooleanResult,
        decide_geometric,
        "sequence",
        "predicate",
        examples=(
            example(
                "powers_of_two",
                "Recognize a geometric sequence of powers of two.",
                {"values": ["2", "4", "8", "16"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.decide.nondecreasing",
        "Decide nondecreasing order",
        "Decide whether every term is at least its predecessor.",
        IntegerSequenceRequest,
        IntegerSequenceBooleanResult,
        decide_nondecreasing,
        "sequence",
        "predicate",
        examples=(
            example(
                "nondecreasing_sequence",
                "Check nondecreasing order.",
                {"values": ["1", "1", "3", "5"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.decide.strictly_increasing",
        "Decide strict increase",
        "Decide whether every term is greater than its predecessor.",
        IntegerSequenceRequest,
        IntegerSequenceBooleanResult,
        decide_strictly_increasing,
        "sequence",
        "predicate",
        examples=(
            example(
                "strictly_increasing_sequence",
                "Check strict increase.",
                {"values": ["1", "2", "4", "7"]},
            ),
        ),
    ),
)
