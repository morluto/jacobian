"""Opt-in request ingress for exact number-theory integer scalars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from jacobian._exact import DecimalIntegerEncoding
from jacobian.math.number_theory._models import MAX_INTEGER_DIGITS

MAX_SAFE_INTEGER = (1 << 53) - 1


def _nonnegative_decimal_pattern(maximum: int) -> str:
    """Match canonical magnitudes up to a fixed bound in O(digits) branches."""

    digits = str(maximum)
    alternatives = ["0"]
    if len(digits) > 1:
        alternatives.append(rf"[1-9][0-9]{{0,{len(digits) - 2}}}")
    for index, digit in enumerate(digits):
        first = 1 if index == 0 else 0
        last = int(digit) - 1
        if last < first:
            continue
        smaller = str(first) if first == last else f"[{first}-{last}]"
        remaining = len(digits) - index - 1
        suffix = rf"[0-9]{{{remaining}}}" if remaining else ""
        alternatives.append(digits[:index] + smaller + suffix)
    alternatives.append(digits)
    return "(?:" + "|".join(alternatives) + ")"


def _integer_at_most_pattern(maximum: int) -> str:
    end = r"(?![\s\S])"
    if maximum >= 0:
        return rf"^(?:-[1-9][0-9]*|{_nonnegative_decimal_pattern(maximum)}){end}"
    excluded = _nonnegative_decimal_pattern(-maximum - 1)
    return rf"^-(?!{excluded}{end})[1-9][0-9]*{end}"


@dataclass(frozen=True)
class IntegerInputEncoding(DecimalIntegerEncoding):
    """Decode canonical strings or safe JSON integers into bounded native ints.

    This metadata is only for operation arguments. Canonical value/result
    codecs remain strict. String spelling and size are checked before decoding;
    the operation's numerical bounds then apply to either JSON representation.
    Python validation remains integer-only, and JSON serialization is canonical.
    """

    minimum: int | None = None
    maximum: int | None = None

    def __get_pydantic_core_schema__(
        self, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        canonical = super().__get_pydantic_core_schema__(source, handler)
        bounded = core_schema.int_schema(strict=True, ge=self.minimum, le=self.maximum)
        exact = core_schema.chain_schema([canonical, bounded])
        safe_limit = min(MAX_SAFE_INTEGER, 10**self.max_digits - 1)
        numeric = core_schema.int_schema(
            strict=True,
            ge=max(
                -safe_limit, self.minimum if self.minimum is not None else -safe_limit
            ),
            le=min(
                safe_limit, self.maximum if self.maximum is not None else safe_limit
            ),
        )
        return core_schema.json_or_python_schema(
            json_schema=core_schema.union_schema(
                [exact, numeric],
                custom_error_type="number_theory.integer_input",
                custom_error_message=(
                    "integer must be a canonical decimal string with at most "
                    f"{self.max_digits} digits, or a JSON integer with absolute value "
                    f"at most {safe_limit}; decoded value must be in "
                    f"[{self.minimum if self.minimum is not None else '-infinity'}, "
                    f"{self.maximum if self.maximum is not None else 'infinity'}]"
                ),
            ),
            python_schema=exact,
            serialization=canonical["serialization"],
        )

    def __get_pydantic_json_schema__(
        self, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        result = handler(schema)
        string = result["anyOf"][0] if handler.mode == "validation" else result
        bounds: list[JsonSchemaValue] = []
        if self.minimum is not None:
            bounds.append(
                {"not": {"pattern": _integer_at_most_pattern(self.minimum - 1)}}
            )
        if self.maximum is not None:
            bounds.append({"pattern": _integer_at_most_pattern(self.maximum)})
        if bounds:
            string["allOf"] = bounds
        return result


IntegerInput = Annotated[int, IntegerInputEncoding(max_digits=MAX_INTEGER_DIGITS)]
