"""Rational-owned exact arithmetic operation declarations."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory.arithmetic._operations import (
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
from jacobian.math.number_theory.arithmetic._rational_models import (
    NonzeroRationalValueRequest,
    RationalComparisonResult,
    RationalContinuedFractionResult,
    RationalDivisionRequest,
    RationalIntegerResult,
    RationalPairRequest,
    RationalValueRequest,
    RationalValueResult,
)
from jacobian.math.number_theory.arithmetic._support import arithmetic_operation

_ONE_HALF = {"num": "1", "den": "2"}
_TWO_THIRDS = {"num": "2", "den": "3"}

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
        examples=(example("two_thirds", "Invert two thirds.", {"value": _TWO_THIRDS}),),
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
        examples=(example("two_thirds", "Negate two thirds.", {"value": _TWO_THIRDS}),),
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
                "negative_three_halves",
                "Take the absolute value of negative three halves.",
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
                "half_plus_two_thirds",
                "Add one half and two thirds.",
                {"left": _ONE_HALF, "right": _TWO_THIRDS},
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
                "two_thirds_minus_half",
                "Subtract one half from two thirds.",
                {"left": _TWO_THIRDS, "right": _ONE_HALF},
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
                "half_times_two_thirds",
                "Multiply one half by two thirds.",
                {"left": _ONE_HALF, "right": _TWO_THIRDS},
            ),
        ),
    ),
    arithmetic_operation(
        "rational.compute.quotient",
        "Divide rationals",
        "Compute the reduced quotient of two rationals with nonzero divisor.",
        RationalDivisionRequest,
        RationalValueResult,
        quotient,
        "rational",
        "exact",
        examples=(
            example(
                "two_thirds_divided_by_half",
                "Divide two thirds by one half.",
                {"left": _TWO_THIRDS, "right": _ONE_HALF},
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
                "half_and_two_thirds",
                "Find the lesser rational.",
                {"left": _ONE_HALF, "right": _TWO_THIRDS},
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
                "half_and_two_thirds",
                "Find the greater rational.",
                {"left": _ONE_HALF, "right": _TWO_THIRDS},
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
                "seven_thirds",
                "Floor seven thirds.",
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
                "seven_thirds",
                "Ceil seven thirds.",
                {"value": {"num": "7", "den": "3"}},
            ),
        ),
    ),
    arithmetic_operation(
        "rational.compute.continued_fraction",
        "Expand rational continued fraction",
        "Compute the canonical finite simple continued fraction of one rational.",
        RationalValueRequest,
        RationalContinuedFractionResult,
        continued_fraction,
        "rational",
        "representation",
        examples=(
            example(
                "negative_seven_fifths",
                "Expand negative seven fifths.",
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
                "equal_halves",
                "Compare one half with itself.",
                {"left": _ONE_HALF, "right": _ONE_HALF},
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
                "half_less_than_two_thirds",
                "Compare one half and two thirds.",
                {"left": _ONE_HALF, "right": _TWO_THIRDS},
            ),
        ),
    ),
)

__all__ = ["RATIONAL_OPERATIONS"]
