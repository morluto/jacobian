"""Immutable catalog declarations for finite integer-sequence operations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.sequences.core._models import (
    IntegerSequenceBooleanResult,
    IntegerSequenceFrequenciesResult,
    IntegerSequenceIndexListResult,
    IntegerSequenceListResult,
    IntegerSequenceRationalResult,
    IntegerSequenceValueResult,
)
from jacobian.math.number_theory.sequences.core.operations import (
    decide_arithmetic,
    decide_geometric,
    decide_nondecreasing,
    decide_strictly_increasing,
    first_differences,
    frequencies,
    parities,
    prefix_gcds,
    prefix_lcms,
    prefix_maxima,
    prefix_minima,
    prefix_products,
    prefix_sums,
    reverse_sequence,
    second_differences,
    sequence_distinct_count,
    sequence_gcd,
    sequence_lcm,
    sequence_maximum,
    sequence_mean,
    sequence_median,
    sequence_minimum,
    sequence_product,
    sequence_range,
    sequence_sum,
    signs,
    sort_sequence,
    sorted_unique,
    zero_indices,
)
from jacobian.math.number_theory.sequences.core.values import IntegerSequence

_SEQ = {"values": ["1", "2", "3"]}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="sequence.compute.sum",
        title="Sum integer sequence",
        description="Compute the exact sum of a finite integer sequence.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceValueResult,
        run=sequence_sum,
        tags=("sequence", "exact"),
        examples=(
            OperationExample(name="sum", description="Sum 1, 2, and 3.", input=_SEQ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.product",
        title="Multiply integer sequence",
        description="Compute the exact product of a finite integer sequence.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceValueResult,
        run=sequence_product,
        tags=("sequence", "exact"),
        examples=(
            OperationExample(
                name="product", description="Multiply 1, 2, and 3.", input=_SEQ
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.gcd",
        title="Compute sequence gcd",
        description="Compute the gcd of every value in a finite integer sequence.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceValueResult,
        run=sequence_gcd,
        tags=("sequence", "divisibility"),
        examples=(
            OperationExample(
                name="gcd",
                description="Compute the gcd of 12 and 18.",
                input={"values": ["12", "18"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.lcm",
        title="Compute sequence lcm",
        description="Compute the lcm of every value in a finite integer sequence.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceValueResult,
        run=sequence_lcm,
        tags=("sequence", "divisibility"),
        examples=(
            OperationExample(
                name="lcm",
                description="Compute the lcm of 4 and 6.",
                input={"values": ["4", "6"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.minimum",
        title="Compute sequence minimum",
        description="Compute the least value in a finite integer sequence.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceValueResult,
        run=sequence_minimum,
        tags=("sequence", "order"),
        examples=(
            OperationExample(
                name="minimum",
                description="Find the minimum of 3, 1, and 2.",
                input={"values": ["3", "1", "2"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.maximum",
        title="Compute sequence maximum",
        description="Compute the greatest value in a finite integer sequence.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceValueResult,
        run=sequence_maximum,
        tags=("sequence", "order"),
        examples=(
            OperationExample(
                name="maximum",
                description="Find the maximum of 1, 3, and 2.",
                input={"values": ["1", "3", "2"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.range",
        title="Compute sequence range",
        description="Compute maximum minus minimum for a finite integer sequence.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceValueResult,
        run=sequence_range,
        tags=("sequence", "statistic"),
        examples=(
            OperationExample(
                name="range",
                description="Compute the range of 1, 4, and 2.",
                input={"values": ["1", "4", "2"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.distinct_count",
        title="Count distinct sequence values",
        description="Count distinct values in a finite integer sequence.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceValueResult,
        run=sequence_distinct_count,
        tags=("sequence", "counting"),
        examples=(
            OperationExample(
                name="distinct",
                description="Count distinct values in 1, 2, and 1.",
                input={"values": ["1", "2", "1"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.mean",
        title="Compute sequence mean",
        description="Compute the reduced arithmetic mean of a finite integer sequence.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceRationalResult,
        run=sequence_mean,
        tags=("sequence", "statistic"),
        examples=(
            OperationExample(
                name="mean", description="Compute the mean of 1, 2, and 3.", input=_SEQ
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.median",
        title="Compute sequence median",
        description="Compute the reduced median of a finite integer sequence.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceRationalResult,
        run=sequence_median,
        tags=("sequence", "statistic"),
        examples=(
            OperationExample(
                name="median",
                description="Compute the median of 1, 3, and 2.",
                input={"values": ["1", "3", "2"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.prefix_sums",
        title="Compute prefix sums",
        description="Compute every nonempty prefix sum of a finite integer sequence.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceListResult,
        run=prefix_sums,
        tags=("sequence", "transform"),
        examples=(
            OperationExample(
                name="prefix_sums",
                description="Compute prefix sums of 1, 2, and 3.",
                input=_SEQ,
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.first_differences",
        title="Compute first differences",
        description="Compute adjacent first differences of a finite integer sequence.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceListResult,
        run=first_differences,
        tags=("sequence", "transform"),
        examples=(
            OperationExample(
                name="first_differences",
                description="Compute differences of consecutive squares.",
                input={"values": ["1", "4", "9", "16"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.prefix_products",
        title="Compute prefix products",
        description="Compute every nonempty prefix product of a finite integer sequence.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceListResult,
        run=prefix_products,
        tags=("sequence", "transform"),
        examples=(
            OperationExample(
                name="prefix_products",
                description="Compute prefix products of 2, 3, and 4.",
                input={"values": ["2", "3", "4"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.prefix_minima",
        title="Compute prefix minima",
        description="Compute the minimum of every nonempty prefix.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceListResult,
        run=prefix_minima,
        tags=("sequence", "transform"),
        examples=(
            OperationExample(
                name="prefix_minima",
                description="Compute prefix minima of 3, 1, and 2.",
                input={"values": ["3", "1", "2"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.prefix_maxima",
        title="Compute prefix maxima",
        description="Compute the maximum of every nonempty prefix.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceListResult,
        run=prefix_maxima,
        tags=("sequence", "transform"),
        examples=(
            OperationExample(
                name="prefix_maxima",
                description="Compute prefix maxima of 1, 3, and 2.",
                input={"values": ["1", "3", "2"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.prefix_gcds",
        title="Compute prefix gcds",
        description="Compute the gcd of every nonempty prefix.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceListResult,
        run=prefix_gcds,
        tags=("sequence", "divisibility"),
        examples=(
            OperationExample(
                name="prefix_gcds",
                description="Compute prefix gcds of 18, 24, and 15.",
                input={"values": ["18", "24", "15"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.prefix_lcms",
        title="Compute prefix lcms",
        description="Compute the lcm of every nonempty prefix.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceListResult,
        run=prefix_lcms,
        tags=("sequence", "divisibility"),
        examples=(
            OperationExample(
                name="prefix_lcms",
                description="Compute prefix lcms of 2, 3, and 4.",
                input={"values": ["2", "3", "4"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.second_differences",
        title="Compute second differences",
        description="Compute adjacent differences of the first-difference sequence.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceListResult,
        run=second_differences,
        tags=("sequence", "transform"),
        examples=(
            OperationExample(
                name="second_differences",
                description="Compute second differences of consecutive squares.",
                input={"values": ["1", "4", "9", "16"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.transform.sorted_unique",
        title="Sort and deduplicate sequence",
        description="Return the strictly increasing values occurring in a finite integer sequence.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceListResult,
        run=sorted_unique,
        tags=("sequence", "transform"),
        examples=(
            OperationExample(
                name="sorted_unique",
                description="Sort and deduplicate a sequence.",
                input={"values": ["3", "1", "3", "2"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.transform.sort",
        title="Sort integer sequence",
        description="Return a nondecreasing ordering retaining multiplicities.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceListResult,
        run=sort_sequence,
        tags=("sequence", "transform"),
        examples=(
            OperationExample(
                name="sort",
                description="Sort an integer sequence.",
                input={"values": ["3", "1", "2"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.transform.reverse",
        title="Reverse integer sequence",
        description="Return the finite integer sequence in reverse order.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceListResult,
        run=reverse_sequence,
        tags=("sequence", "transform"),
        examples=(
            OperationExample(
                name="reverse", description="Reverse an integer sequence.", input=_SEQ
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.transform.parities",
        title="Compute parity sequence",
        description="Return 0 for even and 1 for odd at each position.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceListResult,
        run=parities,
        tags=("sequence", "transform"),
        examples=(
            OperationExample(
                name="parities",
                description="Compute parities of 1, 2, and 3.",
                input=_SEQ,
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.transform.signs",
        title="Compute sign sequence",
        description="Return -1, 0, or 1 for each sequence value.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceListResult,
        run=signs,
        tags=("sequence", "transform"),
        examples=(
            OperationExample(
                name="signs",
                description="Compute signs of negative, zero, and positive values.",
                input={"values": ["-2", "0", "5"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.decide.arithmetic",
        title="Decide arithmetic progression",
        description="Decide whether consecutive terms have one common difference.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceBooleanResult,
        run=decide_arithmetic,
        tags=("sequence", "predicate"),
        examples=(
            OperationExample(
                name="arithmetic",
                description="Recognize an arithmetic sequence.",
                input={"values": ["3", "6", "9", "12"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.decide.geometric",
        title="Decide geometric progression",
        description="Decide whether a finite integer sequence has a consistent rational ratio.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceBooleanResult,
        run=decide_geometric,
        tags=("sequence", "predicate"),
        examples=(
            OperationExample(
                name="geometric",
                description="Recognize powers of two.",
                input={"values": ["2", "4", "8", "16"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.decide.nondecreasing",
        title="Decide nondecreasing order",
        description="Decide whether every term is at least its predecessor.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceBooleanResult,
        run=decide_nondecreasing,
        tags=("sequence", "predicate"),
        examples=(
            OperationExample(
                name="nondecreasing",
                description="Check nondecreasing order.",
                input={"values": ["1", "1", "3", "5"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.decide.strictly_increasing",
        title="Decide strict increase",
        description="Decide whether every term is greater than its predecessor.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceBooleanResult,
        run=decide_strictly_increasing,
        tags=("sequence", "predicate"),
        examples=(
            OperationExample(
                name="strictly_increasing",
                description="Check strict increase.",
                input={"values": ["1", "2", "4", "7"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.frequencies",
        title="Compute value frequencies",
        description="Count each distinct value in a finite integer sequence.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceFrequenciesResult,
        run=frequencies,
        tags=("sequence", "counting"),
        examples=(
            OperationExample(
                name="frequencies",
                description="Count frequencies in 1, 2, and 1.",
                input={"values": ["1", "2", "1"]},
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.compute.zero_indices",
        title="Locate zero terms",
        description="Return zero-based indices whose sequence value is zero.",
        request_type=IntegerSequence,
        result_type=IntegerSequenceIndexListResult,
        run=zero_indices,
        tags=("sequence", "search"),
        examples=(
            OperationExample(
                name="zero_indices",
                description="Locate zero terms in 2, 0, and 3.",
                input={"values": ["2", "0", "3"]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
