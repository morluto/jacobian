"""Rational-owned exact arithmetic operation declarations."""

from fractions import Fraction

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
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


def _fraction(value: CanonicalRational) -> Fraction:
    return value.as_fraction()


def _wire(
    value: Fraction, *, location: tuple[str | int, ...] = ("value",)
) -> CanonicalRational:
    numerator = value.numerator
    denominator = value.denominator
    if (
        abs(numerator) >= 10**MAX_CANONICAL_RATIONAL_DIGITS
        or abs(denominator) >= 10**MAX_CANONICAL_RATIONAL_DIGITS
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
    return RationalIntegerResult(value=native.floor_rational(_fraction(request.value)))


def ceiling(request: RationalValueRequest) -> RationalIntegerResult:
    return RationalIntegerResult(
        value=native.ceiling_rational(_fraction(request.value))
    )


def continued_fraction(
    request: RationalValueRequest,
) -> RationalContinuedFractionResult:
    terms = native.continued_fraction(
        _fraction(request.value), max_terms=MAX_RATIONAL_CONTINUED_FRACTION_TERMS
    )
    return RationalContinuedFractionResult._from_kernel(
        value=request.value,
        terms=tuple(terms),
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
    MathTool(
        operation_id="rational.compute.reciprocal",
        title="Compute rational reciprocal",
        description="Compute the reduced reciprocal of one nonzero rational.",
        request_type=NonzeroRationalValueRequest,
        result_type=RationalValueResult,
        run=reciprocal,
        tags=("rational", "exact"),
        examples=(
            OperationExample(
                name="two_thirds",
                description="Invert two thirds.",
                input={"value": _TWO_THIRDS},
            ),
        ),
    ),
    MathTool(
        operation_id="rational.compute.negation",
        title="Negate rational",
        description="Compute the exact additive inverse of one rational.",
        request_type=RationalValueRequest,
        result_type=RationalValueResult,
        run=negation,
        tags=("rational", "exact"),
        examples=(
            OperationExample(
                name="two_thirds",
                description="Negate two thirds.",
                input={"value": _TWO_THIRDS},
            ),
        ),
    ),
    MathTool(
        operation_id="rational.compute.absolute_value",
        title="Compute rational absolute value",
        description="Compute the exact absolute value of one rational.",
        request_type=RationalValueRequest,
        result_type=RationalValueResult,
        run=rational_absolute_value,
        tags=("rational", "exact"),
        examples=(
            OperationExample(
                name="negative_three_halves",
                description="Take the absolute value of negative three halves.",
                input={"value": {"num": "-3", "den": "2"}},
            ),
        ),
    ),
    MathTool(
        operation_id="rational.compute.sum",
        title="Add rationals",
        description="Compute the reduced sum of two rationals.",
        request_type=RationalPairRequest,
        result_type=RationalValueResult,
        run=sum_rationals,
        tags=("rational", "exact"),
        examples=(
            OperationExample(
                name="half_plus_two_thirds",
                description="Add one half and two thirds.",
                input={"left": _ONE_HALF, "right": _TWO_THIRDS},
            ),
        ),
    ),
    MathTool(
        operation_id="rational.compute.difference",
        title="Subtract rationals",
        description="Compute the reduced difference of two rationals.",
        request_type=RationalPairRequest,
        result_type=RationalValueResult,
        run=difference,
        tags=("rational", "exact"),
        examples=(
            OperationExample(
                name="two_thirds_minus_half",
                description="Subtract one half from two thirds.",
                input={"left": _TWO_THIRDS, "right": _ONE_HALF},
            ),
        ),
    ),
    MathTool(
        operation_id="rational.compute.product",
        title="Multiply rationals",
        description="Compute the reduced product of two rationals.",
        request_type=RationalPairRequest,
        result_type=RationalValueResult,
        run=product,
        tags=("rational", "exact"),
        examples=(
            OperationExample(
                name="half_times_two_thirds",
                description="Multiply one half by two thirds.",
                input={"left": _ONE_HALF, "right": _TWO_THIRDS},
            ),
        ),
    ),
    MathTool(
        operation_id="rational.compute.quotient",
        title="Divide rationals",
        description="Compute the reduced quotient of two rationals with nonzero divisor.",
        request_type=RationalDivisionRequest,
        result_type=RationalValueResult,
        run=quotient,
        tags=("rational", "exact"),
        examples=(
            OperationExample(
                name="two_thirds_divided_by_half",
                description="Divide two thirds by one half.",
                input={"left": _TWO_THIRDS, "right": _ONE_HALF},
            ),
        ),
    ),
    MathTool(
        operation_id="rational.compute.minimum",
        title="Compute rational minimum",
        description="Return the lesser of two exact rationals.",
        request_type=RationalPairRequest,
        result_type=RationalValueResult,
        run=minimum,
        tags=("rational", "order"),
        examples=(
            OperationExample(
                name="half_and_two_thirds",
                description="Find the lesser rational.",
                input={"left": _ONE_HALF, "right": _TWO_THIRDS},
            ),
        ),
    ),
    MathTool(
        operation_id="rational.compute.maximum",
        title="Compute rational maximum",
        description="Return the greater of two exact rationals.",
        request_type=RationalPairRequest,
        result_type=RationalValueResult,
        run=maximum,
        tags=("rational", "order"),
        examples=(
            OperationExample(
                name="half_and_two_thirds",
                description="Find the greater rational.",
                input={"left": _ONE_HALF, "right": _TWO_THIRDS},
            ),
        ),
    ),
    MathTool(
        operation_id="rational.compute.floor",
        title="Floor rational",
        description="Compute the greatest integer not exceeding one rational.",
        request_type=RationalValueRequest,
        result_type=RationalIntegerResult,
        run=floor,
        tags=("rational", "rounding"),
        examples=(
            OperationExample(
                name="seven_thirds",
                description="Floor seven thirds.",
                input={"value": {"num": "7", "den": "3"}},
            ),
        ),
    ),
    MathTool(
        operation_id="rational.compute.ceiling",
        title="Ceil rational",
        description="Compute the least integer not below one rational.",
        request_type=RationalValueRequest,
        result_type=RationalIntegerResult,
        run=ceiling,
        tags=("rational", "rounding"),
        examples=(
            OperationExample(
                name="seven_thirds",
                description="Ceil seven thirds.",
                input={"value": {"num": "7", "den": "3"}},
            ),
        ),
    ),
    MathTool(
        operation_id="rational.compute.continued_fraction",
        title="Expand rational continued fraction",
        description="Compute the canonical finite simple continued fraction of one rational.",
        request_type=RationalValueRequest,
        result_type=RationalContinuedFractionResult,
        run=continued_fraction,
        tags=("rational", "representation"),
        examples=(
            OperationExample(
                name="negative_seven_fifths",
                description="Expand negative seven fifths.",
                input={"value": {"num": "-7", "den": "5"}},
            ),
        ),
    ),
    MathTool(
        operation_id="rational.decide.equal",
        title="Decide rational equality",
        description="Decide exact equality of two reduced rationals.",
        request_type=RationalPairRequest,
        result_type=RationalComparisonResult,
        run=equal,
        tags=("rational", "predicate"),
        examples=(
            OperationExample(
                name="equal_halves",
                description="Compare one half with itself.",
                input={"left": _ONE_HALF, "right": _ONE_HALF},
            ),
        ),
    ),
    MathTool(
        operation_id="rational.decide.less_than",
        title="Compare rationals",
        description="Decide whether the first rational is strictly less than the second.",
        request_type=RationalPairRequest,
        result_type=RationalComparisonResult,
        run=less_than,
        tags=("rational", "predicate"),
        examples=(
            OperationExample(
                name="half_less_than_two_thirds",
                description="Compare one half and two thirds.",
                input={"left": _ONE_HALF, "right": _TWO_THIRDS},
            ),
        ),
    ),
)

__all__ = ["RATIONAL_OPERATIONS"]
