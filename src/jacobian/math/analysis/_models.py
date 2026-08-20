"""Typed contracts for rigorous real-function point enclosures."""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

MAX_RATIONAL_DIGITS = 128
MAX_EXPRESSION_DEPTH = 16
MAX_EXPRESSION_NODES = 64
MAX_INTEGER_EXPONENT = 64
MAX_DYADIC_EXPONENT = 2**53 - 1
MAX_DYADIC_MANTISSA_DIGITS = 1_235

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
