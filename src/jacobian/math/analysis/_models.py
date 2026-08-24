"""Typed contracts for rigorous real-function point enclosures."""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from typing import Literal, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

MAX_RATIONAL_DIGITS = 128
MAX_EXPRESSION_DEPTH = 16
MAX_EXPRESSION_NODES = 64
MAX_INTEGER_EXPONENT = 64
MAX_DYADIC_EXPONENT = 2**53 - 1
MAX_DYADIC_MANTISSA_DIGITS = 1_235
MAX_POINT_CHECK_DYADIC_EXPONENT = 8_192
MAX_POINT_CHECK_LOG_TERMS = 128
MAX_POINT_CHECK_FRACTION_BITS = 131_072
MAX_POINT_CHECK_FRACTION_UPDATES = 4 * MAX_POINT_CHECK_LOG_TERMS
MAX_POINT_CHECK_OUTPUT_BYTES = 4_096

type IntervalExpressionOp = Literal[
    "const",
    "var",
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "neg",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
]


class IntervalExpressionNode(StrictModel):
    """One node in a bounded univariate expression tree."""

    op: IntervalExpressionOp
    value: CanonicalRational | None = None
    exponent: StrictInt | None = Field(
        default=None, ge=-MAX_INTEGER_EXPONENT, le=MAX_INTEGER_EXPONENT
    )
    children: tuple[IntervalExpressionNode, ...] = Field(default=(), max_length=2)

    @model_validator(mode="after")
    def require_operation_shape(self) -> Self:
        arity = {
            "const": 0,
            "var": 0,
            "neg": 1,
            "pow": 1,
            "exp": 1,
            "log": 1,
            "sqrt": 1,
            "sin": 1,
            "cos": 1,
            "add": 2,
            "sub": 2,
            "mul": 2,
            "div": 2,
        }[self.op]
        if len(self.children) != arity:
            raise ValueError(f"{self.op} node requires exactly {arity} children")
        if self.op == "const":
            if self.value is None:
                raise ValueError("const node requires a value")
            require_bounded_rational(
                self.value,
                max_digits=MAX_RATIONAL_DIGITS,
                label="interval-expression rational",
            )
        elif self.value is not None:
            raise ValueError("only a const node may carry a value")
        if self.op == "pow":
            if self.exponent is None or self.exponent == 0:
                raise ValueError("pow node requires a nonzero bounded integer exponent")
        elif self.exponent is not None:
            raise ValueError("only a pow node may carry an exponent")
        return self


class IntervalExpressionEnclosureRequest(StrictModel):
    """Evaluate a bounded expression at one exact rational argument using Arb."""

    expression: IntervalExpressionNode
    argument: CanonicalRational
    precision_bits: StrictInt = Field(default=128, ge=32, le=4096)

    @model_validator(mode="after")
    def require_bounded_tree(self) -> Self:
        require_bounded_rational(
            self.argument,
            max_digits=MAX_RATIONAL_DIGITS,
            label="interval-enclosure argument",
        )
        stack = [(self.expression, 1)]
        count = 0
        while stack:
            node, depth = stack.pop()
            count += 1
            if depth > MAX_EXPRESSION_DEPTH:
                raise ValueError(f"expression depth exceeds {MAX_EXPRESSION_DEPTH}")
            if count > MAX_EXPRESSION_NODES:
                raise ValueError(
                    f"expression node count exceeds {MAX_EXPRESSION_NODES}"
                )
            stack.extend((child, depth + 1) for child in node.children)
        return self


class IntervalExpressionEnclosureResult(StrictModel):
    status: Literal[
        "ENCLOSED",
        "DOMAIN_ERROR",
        "PRECISION_INSUFFICIENT",
        "NONFINITE",
        "OUTPUT_MAGNITUDE_EXCEEDED",
    ]
    precision_bits: StrictInt = Field(ge=32, le=4096)
    lower: ExactDyadic | None = None
    upper: ExactDyadic | None = None
    relative_accuracy_bits: StrictInt | None = None
    exact: bool = False
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_enclosure_to_status(self) -> Self:
        enclosed = self.status == "ENCLOSED"
        if enclosed != (self.lower is not None and self.upper is not None):
            raise ValueError("only an enclosed result may carry dyadic endpoints")
        if not enclosed and (self.relative_accuracy_bits is not None or self.exact):
            raise ValueError("a non-enclosure cannot claim accuracy or exactness")
        if enclosed:
            assert self.lower is not None and self.upper is not None
            if self.lower.compare(self.upper) > 0:
                raise ValueError("enclosure lower endpoint exceeds upper endpoint")
            if self.exact != (self.relative_accuracy_bits is None):
                raise ValueError(
                    "exact enclosures omit relative accuracy; inexact ones report it"
                )
        return self


class RealUnaryFunction(StrEnum):
    EXP = "EXP"
    LOG = "LOG"
    SQRT = "SQRT"
    SIN = "SIN"
    COS = "COS"


class PointEnclosureCheckFunction(StrEnum):
    """Real functions supported by the independent exact checker."""

    LOG = "LOG"
    SQRT = "SQRT"


class ArbPointEnclosureRequest(StrictModel):
    function: RealUnaryFunction
    argument: CanonicalRational
    precision_bits: StrictInt = Field(default=128, ge=32, le=4096)

    @model_validator(mode="after")
    def bound_argument_size(self) -> Self:
        require_bounded_rational(
            self.argument,
            max_digits=MAX_RATIONAL_DIGITS,
            label="validated-analysis rational",
        )
        return self


class ExactDyadic(StrictModel):
    """The exact value ``mantissa * 2**exponent``."""

    mantissa: str = Field(
        pattern=r"^-?(?:0|[1-9][0-9]*)$", max_length=MAX_DYADIC_MANTISSA_DIGITS
    )
    exponent: StrictInt = Field(ge=-MAX_DYADIC_EXPONENT, le=MAX_DYADIC_EXPONENT)

    @model_validator(mode="after")
    def require_canonical_binary_form(self) -> Self:
        mantissa = int(self.mantissa)
        if mantissa == 0 and self.exponent != 0:
            raise ValueError("canonical dyadic zero must have exponent 0")
        if mantissa != 0 and mantissa % 2 == 0:
            raise ValueError("canonical nonzero dyadic mantissa must be odd")
        return self

    def as_fraction(self) -> Fraction:
        mantissa = Fraction(int(self.mantissa))
        if self.exponent >= 0:
            return mantissa * Fraction(2**self.exponent, 1)
        return mantissa / Fraction(2 ** (-self.exponent), 1)

    def compare(self, other: ExactDyadic) -> int:
        """Compare two dyadics without materializing either power of two."""

        left = int(self.mantissa)
        right = int(other.mantissa)
        if left == 0 or right == 0 or (left < 0) != (right < 0):
            return (left > right) - (left < right)

        left_magnitude = abs(left)
        right_magnitude = abs(right)
        left_top_bit = left_magnitude.bit_length() + self.exponent
        right_top_bit = right_magnitude.bit_length() + other.exponent
        if left_top_bit != right_top_bit:
            magnitude_order = (left_top_bit > right_top_bit) - (
                left_top_bit < right_top_bit
            )
        elif self.exponent >= other.exponent:
            magnitude_order = (
                (left_magnitude << (self.exponent - other.exponent)) > right_magnitude
            ) - ((left_magnitude << (self.exponent - other.exponent)) < right_magnitude)
        else:
            magnitude_order = (
                left_magnitude > (right_magnitude << (other.exponent - self.exponent))
            ) - (left_magnitude < (right_magnitude << (other.exponent - self.exponent)))
        return magnitude_order if left > 0 else -magnitude_order


type PointEnclosureCheckOutcome = Literal["ACCEPTED", "REJECTED", "NON_RESULT"]


def _preflight_point_check_source(data: object) -> object:
    """Reject oversized raw source scalars before canonical integer parsing."""

    if not isinstance(data, dict):
        return data
    argument = data.get("argument")
    if isinstance(argument, CanonicalRational):
        raw_components: tuple[object, ...] = (argument.num, argument.den)
    elif isinstance(argument, dict):
        raw_components = (argument.get("num"), argument.get("den"))
    else:
        raw_components = ()
    if any(
        isinstance(component, str) and len(component.lstrip("-")) > MAX_RATIONAL_DIGITS
        for component in raw_components
    ):
        raise ValueError(
            "point-enclosure checker raw rational component exceeds the "
            f"{MAX_RATIONAL_DIGITS}-digit bound"
        )
    return data


def _point_check_fraction_bound_bits(argument: CanonicalRational) -> int:
    """Bound LOG/SQRT intermediates, including claim comparisons and replay."""

    numerator, denominator = argument.as_integer_ratio()
    source_bits = max(abs(numerator).bit_length(), denominator.bit_length())
    transformed_bits = source_bits + 2
    odd_denominator_bits = (2 * MAX_POINT_CHECK_LOG_TERMS + 1).bit_length()

    # A common denominator for one upper atanh bound divides
    # b**(2*n-1) * lcm(1, 3, ..., 2*n-1) * (2*n+1) * (b*b-a*a).
    # Combining log(y) with k*log(2) adds the independent powers of 3 but
    # shares the odd-denominator factors.  The coefficient k has at most
    # source_bits.bit_length() bits after exact power-of-two range reduction.
    combined_log_bits = (
        (2 * MAX_POINT_CHECK_LOG_TERMS - 1) * (transformed_bits + 2)
        + 2 * transformed_bits
        + (MAX_POINT_CHECK_LOG_TERMS + 1) * odd_denominator_bits
        + source_bits.bit_length()
        + 8
    )
    endpoint_bits = 4 * MAX_DYADIC_MANTISSA_DIGITS + MAX_POINT_CHECK_DYADIC_EXPONENT
    log_comparison_bits = combined_log_bits + endpoint_bits + 32
    sqrt_comparison_bits = 2 * endpoint_bits + source_bits + 8
    return max(log_comparison_bits, sqrt_comparison_bits)


class PointEnclosureCheckRequest(StrictModel):
    """Check one claimed LOG or SQRT enclosure by exact independent replay.

    Rational components have at most 128 decimal digits. Claimed dyadic
    exponents must lie in -8192..8192; reversed or mathematically invalid
    intervals remain valid requests and produce typed checker outcomes. LOG
    replay uses at most 128 terms per series, about 400 worst-case bits after
    range reduction, so tighter claims can produce NON_RESULT even when their
    retained precision metadata is larger.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "point_check_log_term_bound": MAX_POINT_CHECK_LOG_TERMS,
            "point_check_fraction_intermediate_bit_bound": (
                MAX_POINT_CHECK_FRACTION_BITS
            ),
            "point_check_producer_replay_term_update_bound": (
                MAX_POINT_CHECK_FRACTION_UPDATES
            ),
            "point_check_output_byte_bound": MAX_POINT_CHECK_OUTPUT_BYTES,
        }
    )

    function: PointEnclosureCheckFunction = Field(
        description="Function checked independently; exactly LOG or SQRT."
    )
    argument: CanonicalRational = Field(
        description="Exact reduced rational argument with at most 128 digits per component."
    )
    precision_bits: StrictInt = Field(
        default=128,
        ge=32,
        le=4096,
        description=(
            "Precision metadata retained from the source claim; it does not "
            "promise that the independent 128-term LOG replay resolves a "
            "claim at that precision."
        ),
    )
    lower: ExactDyadic = Field(description="Claimed inclusive lower endpoint.")
    upper: ExactDyadic = Field(description="Claimed inclusive upper endpoint.")

    @model_validator(mode="before")
    @classmethod
    def preflight_raw_source(cls, data: object) -> object:
        return _preflight_point_check_source(data)

    @model_validator(mode="after")
    def preflight_exact_checker(self) -> Self:
        require_bounded_rational(
            self.argument,
            max_digits=MAX_RATIONAL_DIGITS,
            label="point-enclosure checker rational",
        )
        if any(
            abs(endpoint.exponent) > MAX_POINT_CHECK_DYADIC_EXPONENT
            for endpoint in (self.lower, self.upper)
        ):
            raise ValueError(
                "point-enclosure checker dyadic exponent exceeds the "
                f"+/-{MAX_POINT_CHECK_DYADIC_EXPONENT} bound"
            )
        if (
            _point_check_fraction_bound_bits(self.argument)
            > MAX_POINT_CHECK_FRACTION_BITS
        ):
            raise ValueError(
                "point-enclosure checker exact rational work exceeds the "
                f"{MAX_POINT_CHECK_FRACTION_BITS}-bit intermediate bound"
            )
        return self


class PointEnclosureCheckResult(StrictModel):
    """A source-bound checker outcome replayed during result validation.

    ACCEPTED means the independently proved interval is contained in the
    claim. REJECTED covers an invalid real-domain or reversed claim, or a claim
    proved disjoint from the true value. NON_RESULT means the independent LOG
    enclosure still partially overlaps the claim after 128 series terms.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "point_check_output_byte_bound": MAX_POINT_CHECK_OUTPUT_BYTES,
        }
    )

    function: PointEnclosureCheckFunction
    argument: CanonicalRational
    precision_bits: StrictInt = Field(
        ge=32,
        le=4096,
        description="Precision metadata retained exactly from the source claim.",
    )
    lower: ExactDyadic
    upper: ExactDyadic
    outcome: PointEnclosureCheckOutcome = Field(
        description=(
            "ACCEPTED when the independent enclosure is contained in the "
            "claim; REJECTED for invalid or provably excluding claims; "
            "NON_RESULT for unresolved partial overlap at the LOG term cap."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def preflight_raw_source(cls, data: object) -> object:
        return _preflight_point_check_source(data)

    @model_validator(mode="after")
    def replay_outcome(self) -> Self:
        from jacobian.math.analysis._point_enclosure_check import (
            point_enclosure_check_outcome,
        )

        request = PointEnclosureCheckRequest(
            function=self.function,
            argument=self.argument,
            precision_bits=self.precision_bits,
            lower=self.lower,
            upper=self.upper,
        )
        if self.outcome != point_enclosure_check_outcome(request):
            raise ValueError(
                "outcome must equal the deterministic enclosure check for the retained source"
            )
        return self


class ArbPointEnclosureResult(StrictModel):
    status: Literal[
        "ENCLOSED", "NONFINITE", "TIMEOUT", "BACKEND_ERROR", "OUTPUT_MAGNITUDE_EXCEEDED"
    ]
    function: RealUnaryFunction
    argument: CanonicalRational
    precision_bits: StrictInt = Field(ge=32, le=4096)
    lower: ExactDyadic | None = None
    upper: ExactDyadic | None = None
    relative_accuracy_bits: StrictInt | None = None
    exact: bool = False
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_enclosure_to_status(self) -> Self:
        enclosed = self.status == "ENCLOSED"
        if enclosed != (self.lower is not None and self.upper is not None):
            raise ValueError("only an enclosed result may carry dyadic endpoints")
        if not enclosed and (self.relative_accuracy_bits is not None or self.exact):
            raise ValueError("a non-enclosure cannot claim accuracy or exactness")
        if enclosed:
            lower = self.lower
            upper = self.upper
            if lower is None or upper is None:
                raise ValueError("only an enclosed result may carry dyadic endpoints")
            if lower.compare(upper) > 0:
                raise ValueError("enclosure lower endpoint exceeds upper endpoint")
            if self.exact != (self.relative_accuracy_bits is None):
                raise ValueError(
                    "exact enclosures omit relative accuracy; inexact ones report it"
                )
        return self
