"""Rational-owned exact arithmetic operations.

The arithmetic domain owns rational arithmetic (reciprocal, negation,
absolute value, sum, difference, product, quotient), rational order
(minimum, maximum), rational rounding (floor, ceiling), rational
representation (continued fraction), and rational predicates (equality,
strict less-than).
"""

from jacobian.catalog._examples import example
from jacobian.math.arithmetic._operations import (
    ceiling,
    continued_fraction,
    difference,
    equal,
    floor,
    less_than,
    maximum,
    minimum,
    negation,
    product,
    quotient,
    rational_absolute_value,
    reciprocal,
    sum_rationals,
)
from jacobian.math.arithmetic._rational_models import (
    NonzeroRationalValueRequest,
    RationalComparisonResult,
    RationalContinuedFractionResult,
    RationalDivisionRequest,
    RationalIntegerResult,
    RationalPairRequest,
    RationalValueRequest,
    RationalValueResult,
)
from jacobian.math.arithmetic._support import arithmetic_operation

RATIONAL_OPERATIONS = (
    arithmetic_operation(
        "rational.compute.reciprocal",
        "Compute rational reciprocal",
        "Compute the reduced reciprocal of one nonzero rational.",
        NonzeroRationalValueRequest,
        RationalValueResult,
        reciprocal,
        "rational",
        "exact",
        examples=(
            example(
                "reciprocal_two_thirds",
                "Compute the reciprocal of two thirds.",
                {"value": {"num": "2", "den": "3"}},
            ),
        ),
    ),
    arithmetic_operation(
        "rational.compute.negation",
        "Negate rational",
        "Compute the exact additive inverse of one rational.",
        RationalValueRequest,
        RationalValueResult,
        negation,
        "rational",
        "exact",
        examples=(
            example(
                "negation_two_thirds",
                "Negate two thirds.",
                {"value": {"num": "2", "den": "3"}},
            ),
        ),
    ),
    arithmetic_operation(
        "rational.compute.absolute_value",
        "Compute rational absolute value",
        "Compute the exact absolute value of one rational.",
        RationalValueRequest,
        RationalValueResult,
        rational_absolute_value,
        "rational",
        "exact",
        examples=(
            example(
                "absolute_value_negative_three_halves",
                "Compute the absolute value of negative three halves.",
                {"value": {"num": "-3", "den": "2"}},
            ),
        ),
    ),
    arithmetic_operation(
        "rational.compute.sum",
        "Add rationals",
        "Compute the reduced sum of two rationals.",
        RationalPairRequest,
        RationalValueResult,
        sum_rationals,
        "rational",
        "exact",
        examples=(
            example(
                "one_half_plus_one_third",
                "Add one half and one third.",
                {"left": {"num": "1", "den": "2"}, "right": {"num": "1", "den": "3"}},
            ),
        ),
    ),
    arithmetic_operation(
        "rational.compute.difference",
        "Subtract rationals",
        "Compute the reduced difference of two rationals.",
        RationalPairRequest,
        RationalValueResult,
        difference,
        "rational",
        "exact",
        examples=(
            example(
                "three_fourths_minus_one_sixth",
                "Subtract one sixth from three fourths.",
                {"left": {"num": "3", "den": "4"}, "right": {"num": "1", "den": "6"}},
            ),
        ),
    ),
    arithmetic_operation(
        "rational.compute.product",
        "Multiply rationals",
        "Compute the reduced product of two rationals.",
        RationalPairRequest,
        RationalValueResult,
        product,
        "rational",
        "exact",
        examples=(
            example(
                "two_thirds_times_three_fifths",
                "Multiply two thirds by three fifths.",
                {"left": {"num": "2", "den": "3"}, "right": {"num": "3", "den": "5"}},
            ),
        ),
    ),
    arithmetic_operation(
        "rational.compute.quotient",
        "Divide rationals",
        "Compute the reduced quotient of two rationals.",
        RationalDivisionRequest,
        RationalValueResult,
        quotient,
        "rational",
        "exact",
        examples=(
            example(
                "three_fourths_divided_by_two_thirds",
                "Divide three fourths by two thirds.",
                {"left": {"num": "3", "den": "4"}, "right": {"num": "2", "den": "3"}},
            ),
        ),
    ),
    arithmetic_operation(
        "rational.compute.minimum",
        "Compute rational minimum",
        "Return the lesser of two exact rationals.",
        RationalPairRequest,
        RationalValueResult,
        minimum,
        "rational",
        "order",
        examples=(
            example(
                "minimum_one_half_two_thirds",
                "Find the lesser of one half and two thirds.",
                {"left": {"num": "1", "den": "2"}, "right": {"num": "2", "den": "3"}},
            ),
        ),
    ),
    arithmetic_operation(
        "rational.compute.maximum",
        "Compute rational maximum",
        "Return the greater of two exact rationals.",
        RationalPairRequest,
        RationalValueResult,
        maximum,
        "rational",
        "order",
        examples=(
            example(
                "maximum_one_half_two_thirds",
                "Find the greater of one half and two thirds.",
                {"left": {"num": "1", "den": "2"}, "right": {"num": "2", "den": "3"}},
            ),
        ),
    ),
    arithmetic_operation(
        "rational.compute.floor",
        "Floor rational",
        "Compute the greatest integer not exceeding one rational.",
        RationalValueRequest,
        RationalIntegerResult,
        floor,
        "rational",
        "rounding",
        examples=(
            example(
                "floor_seven_thirds",
                "Compute the floor of seven thirds.",
                {"value": {"num": "7", "den": "3"}},
            ),
        ),
    ),
    arithmetic_operation(
        "rational.compute.ceiling",
        "Ceil rational",
        "Compute the least integer not below one rational.",
        RationalValueRequest,
        RationalIntegerResult,
        ceiling,
        "rational",
        "rounding",
        examples=(
            example(
                "ceiling_seven_thirds",
                "Compute the ceiling of seven thirds.",
                {"value": {"num": "7", "den": "3"}},
            ),
        ),
    ),
    arithmetic_operation(
        "rational.compute.continued_fraction",
        "Expand rational continued fraction",
        "Compute the finite simple continued fraction of one rational.",
        RationalValueRequest,
        RationalContinuedFractionResult,
        continued_fraction,
        "rational",
        "representation",
        examples=(
            example(
                "negative_seven_fifths",
                "Expand negative seven fifths as a continued fraction.",
                {"value": {"num": "-7", "den": "5"}},
            ),
        ),
    ),
    arithmetic_operation(
        "rational.decide.equal",
        "Decide rational equality",
        "Decide exact equality of two reduced rationals.",
        RationalPairRequest,
        RationalComparisonResult,
        equal,
        "rational",
        "predicate",
        examples=(
            example(
                "equal_negative_one_half_negative_one_half",
                "Check equality of two reduced equivalent rationals.",
                {
                    "left": {"num": "-1", "den": "2"},
                    "right": {"num": "-1", "den": "2"},
                },
            ),
        ),
    ),
    arithmetic_operation(
        "rational.decide.less_than",
        "Compare rationals",
        "Decide whether the first rational is strictly less than the second.",
        RationalPairRequest,
        RationalComparisonResult,
        less_than,
        "rational",
        "predicate",
        examples=(
            example(
                "less_than_one_half_two_thirds",
                "Check whether one half is less than two thirds.",
                {"left": {"num": "1", "den": "2"}, "right": {"num": "2", "den": "3"}},
            ),
        ),
    ),
)
