"""Rational-owned exact arithmetic operation declarations."""

from fractions import Fraction

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.arithmetic import operations as native
from jacobian.math.number_theory.arithmetic._rational_models import (
    MAX_RATIONAL_CONTINUED_FRACTION_TERMS,
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


def _fraction(value: CanonicalRational) -> Fraction:
    return value.as_fraction()


def _wire(
    value: Fraction, *, location: tuple[str | int, ...] = ("value",)
) -> CanonicalRational:
    numerator = format_canonical_integer(value.numerator)
    denominator = format_canonical_integer(value.denominator)
    if (
        len(numerator.lstrip("-")) > MAX_CANONICAL_RATIONAL_DIGITS
        or len(denominator) > MAX_CANONICAL_RATIONAL_DIGITS
    ):
        raise OperationDomainValidationError(
            location=location,
            code="arithmetic.rational_result_exceeds_component_bound",
            message="exact rational result exceeds the canonical component bound",
        )
    return CanonicalRational(num=numerator, den=denominator)


def reciprocal(request: NonzeroRationalValueRequest) -> RationalValueResult:
    return RationalValueResult(value=_wire(native.reciprocal(_fraction(request.value))))


def negation(request: RationalValueRequest) -> RationalValueResult:
    return RationalValueResult(
        value=_wire(native.negate_rational(_fraction(request.value)))
    )


def rational_absolute_value(request: RationalValueRequest) -> RationalValueResult:
    return RationalValueResult(
        value=_wire(native.rational_absolute_value(_fraction(request.value)))
    )


def sum_rationals(request: RationalPairRequest) -> RationalValueResult:
    return RationalValueResult(
        value=_wire(
            native.sum_rationals(_fraction(request.left), _fraction(request.right)),
            location=("left", "right"),
        )
    )


def difference(request: RationalPairRequest) -> RationalValueResult:
    return RationalValueResult(
        value=_wire(
            native.difference_rationals(
                _fraction(request.left), _fraction(request.right)
            ),
            location=("left", "right"),
        )
    )


def product(request: RationalPairRequest) -> RationalValueResult:
    return RationalValueResult(
        value=_wire(
            native.product_rationals(_fraction(request.left), _fraction(request.right)),
            location=("left", "right"),
        )
    )


def quotient(request: RationalDivisionRequest) -> RationalValueResult:
    return RationalValueResult(
        value=_wire(
            native.quotient(_fraction(request.left), _fraction(request.right)),
            location=("left", "right"),
        )
    )


def minimum(request: RationalPairRequest) -> RationalValueResult:
    return RationalValueResult(
        value=_wire(
            native.minimum_rational(_fraction(request.left), _fraction(request.right))
        )
    )


def maximum(request: RationalPairRequest) -> RationalValueResult:
    return RationalValueResult(
        value=_wire(
            native.maximum_rational(_fraction(request.left), _fraction(request.right))
        )
    )


def floor(request: RationalValueRequest) -> RationalIntegerResult:
    return RationalIntegerResult(
        value=format_canonical_integer(native.floor_rational(_fraction(request.value)))
    )


def ceiling(request: RationalValueRequest) -> RationalIntegerResult:
    return RationalIntegerResult(
        value=format_canonical_integer(
            native.ceiling_rational(_fraction(request.value))
        )
    )


def continued_fraction(
    request: RationalValueRequest,
) -> RationalContinuedFractionResult:
    terms = native.continued_fraction(
        _fraction(request.value), max_terms=MAX_RATIONAL_CONTINUED_FRACTION_TERMS
    )
    return RationalContinuedFractionResult._from_kernel(
        value=request.value,
        terms=tuple(format_canonical_integer(term) for term in terms),
    )


def equal(request: RationalPairRequest) -> RationalComparisonResult:
    return RationalComparisonResult(
        holds=native.equal_rationals(_fraction(request.left), _fraction(request.right))
    )


def less_than(request: RationalPairRequest) -> RationalComparisonResult:
    return RationalComparisonResult(
        holds=native.less_than_rationals(
            _fraction(request.left), _fraction(request.right)
        )
    )


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
