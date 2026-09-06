"""Canonical exact scalar values and their JSON encodings."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Any, Self

from pydantic import (
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    model_validator,
)
from pydantic_core import PydanticCustomError, core_schema

from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer


@dataclass(frozen=True)
class DecimalIntegerEncoding:
    """JSON codec metadata for native integers, not a second value class.

    Python validation accepts integers only. JSON validation checks canonical
    decimal spelling and the digit envelope before decoding. Python dumps stay
    numeric; JSON dumps use strings at every magnitude.
    """

    max_digits: int

    def __get_pydantic_core_schema__(
        self, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        if source is not int or self.max_digits < 1:
            raise TypeError(
                "decimal integer encoding requires int and a positive digit bound"
            )
        limit = 10**self.max_digits

        def require_bound(value: int) -> int:
            if abs(value) >= limit:
                raise PydanticCustomError(
                    "exact_integer.digit_bound",
                    "integer exceeds the decimal digit bound",
                )
            return value

        wire = core_schema.str_schema(
            strict=True,
            # A negative lookahead expresses absolute end-of-input in both
            # Python and JSON Schema regex syntax (unlike `$` before a newline).
            pattern=rf"^(?:0|-?[1-9][0-9]{{0,{self.max_digits - 1}}})(?![\s\S])",
            regex_engine="python-re",
            max_length=self.max_digits + 1,
        )
        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(
                parse_canonical_integer, wire
            ),
            python_schema=core_schema.no_info_after_validator_function(
                require_bound, core_schema.int_schema(strict=True)
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                format_canonical_integer, when_used="json", return_schema=wire
            ),
        )


MAX_CANONICAL_INTEGER_DIGITS = 32_768
ExactInteger = Annotated[
    int, DecimalIntegerEncoding(max_digits=MAX_CANONICAL_INTEGER_DIGITS)
]

MAX_CANONICAL_RATIONAL_DIGITS = 32_768


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"canonical_rational.{reason}", message)


def format_canonical_rational(value: Fraction) -> str:
    """Return an exact fraction in canonical integer-or-``num/den`` form."""

    if value.denominator == 1:
        return format_canonical_integer(value.numerator)
    return (
        f"{format_canonical_integer(value.numerator)}/"
        f"{format_canonical_integer(value.denominator)}"
    )


def canonical_rational_component_digits(value: CanonicalRational) -> int:
    """Return the greatest decimal width of a canonical rational component."""

    return max(
        len(format_canonical_integer(abs(value.num))),
        len(format_canonical_integer(value.den)),
    )


def require_bounded_rational(
    value: CanonicalRational,
    *,
    max_digits: int,
    label: str,
) -> None:
    """Reject a canonical rational whose components exceed a domain bound."""

    if abs(value.num) >= 10**max_digits or value.den >= 10**max_digits:
        raise ValueError(f"{label} exceeds the {max_digits}-digit bound")


class CanonicalRational(StrictModel):
    """A reduced rational whose denominator is positive and whose zero is 0/1."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"num": "1", "den": "2"}]}
    )

    num: ExactInteger = Field(
        description="Canonical decimal numerator of the reduced rational.",
        examples=["1"],
    )
    den: ExactInteger = Field(
        description=(
            "Positive canonical decimal denominator; together with num it must be "
            "reduced, and integers use den='1'."
        ),
        examples=["2"],
        json_schema_extra={
            "pattern": rf"^[1-9][0-9]{{0,{MAX_CANONICAL_RATIONAL_DIGITS - 1}}}(?![\s\S])",
            "maxLength": MAX_CANONICAL_RATIONAL_DIGITS,
        },
    )

    @model_validator(mode="after")
    def require_reduced_positive_denominator(self) -> Self:
        if self.den == 0:
            raise _validation_error(
                "zero_denominator", "rational denominator cannot be zero"
            )
        value = Fraction(self.num, self.den)
        if (self.num, self.den) != (value.numerator, value.denominator):
            raise _validation_error(
                "noncanonical_representation",
                "rational must be reduced with a positive denominator and canonical zero",
            )
        return self

    def as_fraction(self) -> Fraction:
        return Fraction(*self.as_integer_ratio())

    def as_integer_ratio(self) -> tuple[int, int]:
        return self.num, self.den

    @classmethod
    def from_integer_ratio(cls, numerator: int, denominator: int) -> CanonicalRational:
        try:
            fraction = Fraction(numerator, denominator)
        except ZeroDivisionError:
            raise ValueError("rational denominator cannot be zero") from None
        return cls.from_fraction(fraction)

    @classmethod
    def from_fraction(cls, value: Fraction) -> CanonicalRational:
        return cls(
            num=value.numerator,
            den=value.denominator,
        )
