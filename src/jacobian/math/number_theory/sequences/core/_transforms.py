"""Sequence transform operations over finite integer sequences."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory.sequences.core._models import (
    IntegerSequenceListResult,
    IntegerSequenceRequest,
)
from jacobian.math.number_theory.sequences.core._operations import (
    first_differences,
    parities,
    prefix_gcds,
    prefix_lcms,
    prefix_maxima,
    prefix_minima,
    prefix_products,
    prefix_sums,
    reverse_sequence,
    second_differences,
    signs,
    sort_sequence,
    sorted_unique,
)
from jacobian.math.number_theory.sequences.core._support import sequence_operation

SEQUENCE_TRANSFORM_OPERATIONS = (
    sequence_operation(
        "sequence.compute.prefix_sums",
        "Compute prefix sums",
        "Compute every nonempty prefix sum of a finite integer sequence.",
        IntegerSequenceRequest,
        IntegerSequenceListResult,
        prefix_sums,
        "sequence",
        "transform",
        examples=(
            example(
                "prefix_sums_1_2_3",
                "Compute prefix sums of 1, 2, and 3.",
                {"values": ["1", "2", "3"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.compute.first_differences",
        "Compute first differences",
        "Compute adjacent first differences of a finite integer sequence.",
        IntegerSequenceRequest,
        IntegerSequenceListResult,
        first_differences,
        "sequence",
        "transform",
        examples=(
            example(
                "square_first_differences",
                "Compute differences of consecutive squares.",
                {"values": ["1", "4", "9", "16"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.compute.prefix_products",
        "Compute prefix products",
        "Compute every nonempty prefix product of a finite integer sequence.",
        IntegerSequenceRequest,
        IntegerSequenceListResult,
        prefix_products,
        "sequence",
        "transform",
        examples=(
            example(
                "prefix_products_2_3_4",
                "Compute prefix products of 2, 3, and 4.",
                {"values": ["2", "3", "4"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.compute.prefix_minima",
        "Compute prefix minima",
        "Compute the minimum of every nonempty prefix.",
        IntegerSequenceRequest,
        IntegerSequenceListResult,
        prefix_minima,
        "sequence",
        "transform",
        examples=(
            example(
                "prefix_minima_3_1_2",
                "Compute prefix minima of 3, 1, and 2.",
                {"values": ["3", "1", "2"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.compute.prefix_maxima",
        "Compute prefix maxima",
        "Compute the maximum of every nonempty prefix.",
        IntegerSequenceRequest,
        IntegerSequenceListResult,
        prefix_maxima,
        "sequence",
        "transform",
        examples=(
            example(
                "prefix_maxima_1_3_2",
                "Compute prefix maxima of 1, 3, and 2.",
                {"values": ["1", "3", "2"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.compute.prefix_gcds",
        "Compute prefix gcds",
        "Compute the gcd of every nonempty prefix.",
        IntegerSequenceRequest,
        IntegerSequenceListResult,
        prefix_gcds,
        "sequence",
        "divisibility",
        examples=(
            example(
                "prefix_gcds_18_24_15",
                "Compute prefix gcds of 18, 24, and 15.",
                {"values": ["18", "24", "15"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.compute.prefix_lcms",
        "Compute prefix lcms",
        "Compute the lcm of every nonempty prefix.",
        IntegerSequenceRequest,
        IntegerSequenceListResult,
        prefix_lcms,
        "sequence",
        "divisibility",
        examples=(
            example(
                "prefix_lcms_2_3_4",
                "Compute prefix lcms of 2, 3, and 4.",
                {"values": ["2", "3", "4"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.compute.second_differences",
        "Compute second differences",
        "Compute adjacent differences of the first-difference sequence.",
        IntegerSequenceRequest,
        IntegerSequenceListResult,
        second_differences,
        "sequence",
        "transform",
        examples=(
            example(
                "second_differences_squares",
                "Compute second differences of consecutive squares.",
                {"values": ["1", "4", "9", "16"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.transform.sorted_unique",
        "Sort and deduplicate sequence",
        "Return the strictly increasing values occurring in a finite integer sequence.",
        IntegerSequenceRequest,
        IntegerSequenceListResult,
        sorted_unique,
        "sequence",
        "transform",
        examples=(
            example(
                "sorted_unique_3_1_3_2",
                "Sort and deduplicate a sequence.",
                {"values": ["3", "1", "3", "2"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.transform.sort",
        "Sort integer sequence",
        "Return a nondecreasing ordering retaining multiplicities.",
        IntegerSequenceRequest,
        IntegerSequenceListResult,
        sort_sequence,
        "sequence",
        "transform",
        examples=(
            example(
                "sort_3_1_2", "Sort an integer sequence.", {"values": ["3", "1", "2"]}
            ),
        ),
    ),
    sequence_operation(
        "sequence.transform.reverse",
        "Reverse integer sequence",
        "Return the finite integer sequence in reverse order.",
        IntegerSequenceRequest,
        IntegerSequenceListResult,
        reverse_sequence,
        "sequence",
        "transform",
        examples=(
            example(
                "reverse_1_2_3",
                "Reverse an integer sequence.",
                {"values": ["1", "2", "3"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.transform.parities",
        "Compute parity sequence",
        "Return 0 for even and 1 for odd at each position.",
        IntegerSequenceRequest,
        IntegerSequenceListResult,
        parities,
        "sequence",
        "transform",
        examples=(
            example(
                "parities_1_2_3",
                "Compute parities of 1, 2, and 3.",
                {"values": ["1", "2", "3"]},
            ),
        ),
    ),
    sequence_operation(
        "sequence.transform.signs",
        "Compute sign sequence",
        "Return -1, 0, or 1 for each sequence value.",
        IntegerSequenceRequest,
        IntegerSequenceListResult,
        signs,
        "sequence",
        "transform",
        examples=(
            example(
                "signs_negative_zero_positive",
                "Compute signs of negative, zero, and positive values.",
                {"values": ["-2", "0", "5"]},
            ),
        ),
    ),
)
