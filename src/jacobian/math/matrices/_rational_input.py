"""Explicit JSON request presentations of bounded exact rational values."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Any

from pydantic import ConfigDict, Field, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError, core_schema

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.matrices.values import (
    MAX_RATIONAL_MATRIX_AXIS,
    MAX_SPARSE_RATIONAL_MATRIX_NONZEROS,
    RationalMatrix,
    SparseRationalMatrix,
    SparseRationalMatrixEntry,
)

MAX_REQUEST_RATIONAL_DIGITS = 256


def _canonical_serialization(value: StrictModel) -> StrictModel:
    return value


@dataclass(frozen=True)
class RationalInputEncoding:
    """Normalize a raw bounded JSON ratio, retaining strict native validation.

    Both components keep the exact integer string grammar. The caller supplies
    the raw digit budget: reduction must never make an oversized input admissible.
    This annotation is only applied to declared rational operation arguments.
    """

    max_digits: int

    def __get_pydantic_core_schema__(
        self, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        if source is not CanonicalRational or self.max_digits < 1:
            raise TypeError(
                "rational input requires CanonicalRational and a digit bound"
            )
        canonical = handler(source)
        magnitude = rf"[1-9][0-9]{{0,{self.max_digits - 1}}}"
        end = r"(?![\s\S])"
        wire = core_schema.typed_dict_schema(
            {
                "num": core_schema.typed_dict_field(
                    core_schema.str_schema(
                        strict=True,
                        pattern=rf"^(?:0|-?{magnitude}){end}",
                        regex_engine="python-re",
                        max_length=self.max_digits + 1,
                    )
                ),
                "den": core_schema.typed_dict_field(
                    core_schema.str_schema(
                        strict=True,
                        pattern=rf"^-?{magnitude}{end}",
                        regex_engine="python-re",
                        max_length=self.max_digits + 1,
                    )
                ),
            },
            extra_behavior="forbid",
        )
        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(
                self._normalize_ratio, wire
            ),
            python_schema=canonical,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _canonical_serialization, return_schema=canonical
            ),
        )

    @staticmethod
    def _normalize_ratio(value: dict[str, str]) -> CanonicalRational:
        numerator = parse_canonical_integer(value["num"])
        denominator = parse_canonical_integer(value["den"])
        if denominator == 0:
            raise PydanticCustomError(
                "canonical_rational.zero_denominator",
                "rational denominator cannot be zero",
            )
        return CanonicalRational.from_fraction(Fraction(numerator, denominator))


RationalInput = Annotated[
    CanonicalRational, RationalInputEncoding(max_digits=MAX_REQUEST_RATIONAL_DIGITS)
]


@dataclass(frozen=True)
class RationalValueInputEncoding:
    """Use an explicit request decoder while returning the canonical native type.

    Inherited value validators retain dimensions, domains, coordinate ordering,
    uniqueness and nonzero rules. Only explicitly overridden rational fields
    normalize. Source/result annotations never use this adapter.
    """

    canonical_type: type[StrictModel]
    input_type: type[StrictModel]

    def _canonical_value(self, value: StrictModel) -> StrictModel:
        return self.canonical_type.model_construct(
            _fields_set=value.model_fields_set,
            **{name: getattr(value, name) for name in self.canonical_type.model_fields},
        )

    def __get_pydantic_core_schema__(
        self, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        if source is not self.canonical_type or not issubclass(
            self.input_type, self.canonical_type
        ):
            raise TypeError(
                "rational value input requires its carrier and decoder subclass"
            )
        canonical = handler(source)
        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(
                self._canonical_value, handler.generate_schema(self.input_type)
            ),
            python_schema=canonical,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _canonical_serialization, return_schema=canonical
            ),
        )


class _RationalMatrixInput(RationalMatrix):
    model_config = ConfigDict(title="RationalMatrixInput")

    entries: tuple[tuple[RationalInput, ...], ...] = Field(
        default=(),
        max_length=MAX_RATIONAL_MATRIX_AXIS,
        description=(
            "Exact rational entries. JSON requests accept reduced or unreduced ratios "
            "with either denominator sign; canonical decimal string components have "
            "at most 256 digits and denominators are nonzero. Ratios normalize before "
            "computation. Declared dimensions and entry order are preserved."
        ),
    )


RationalMatrixInput = Annotated[
    RationalMatrix, RationalValueInputEncoding(RationalMatrix, _RationalMatrixInput)
]


class _SparseRationalMatrixEntryInput(SparseRationalMatrixEntry):
    value: RationalInput


_SparseRationalMatrixEntryInputValue = Annotated[
    SparseRationalMatrixEntry,
    RationalValueInputEncoding(
        SparseRationalMatrixEntry, _SparseRationalMatrixEntryInput
    ),
]


class _SparseRationalMatrixInput(SparseRationalMatrix):
    model_config = ConfigDict(title="SparseRationalMatrixInput")

    entries: tuple[_SparseRationalMatrixEntryInputValue, ...] = Field(
        default=(),
        max_length=MAX_SPARSE_RATIONAL_MATRIX_NONZEROS,
        description=(
            "Unique row-major nonzero entries. Only rational values normalize from "
            "bounded JSON ratios; coordinates are never reordered or merged, and "
            "zero ratios (including 0/5) remain forbidden stored entries."
        ),
    )


SparseRationalMatrixInput = Annotated[
    SparseRationalMatrix,
    RationalValueInputEncoding(SparseRationalMatrix, _SparseRationalMatrixInput),
]


@dataclass(frozen=True)
class RationalMatrixInputEnvelope:
    """Expose an owner's existing raw dense-axis preflight in request schemas.

    This metadata does not admit kernel work or alter the shared matrix carrier.
    Copy the resolved request schema so another owner's limit cannot leak into it.
    """

    maximum_axis: int

    def __get_pydantic_json_schema__(
        self, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        result = handler(schema)
        if handler.mode == "serialization":
            return result
        result = deepcopy(handler.resolve_ref_schema(result))
        properties = result["properties"]
        for axis in ("row_count", "column_count"):
            properties[axis]["maximum"] = self.maximum_axis
        entries = properties["entries"]
        entries["maxItems"] = self.maximum_axis
        entries["items"]["maxItems"] = self.maximum_axis
        return result
