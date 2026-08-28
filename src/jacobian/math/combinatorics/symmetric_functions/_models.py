"""Typed wire contracts for symmetric function operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, WithJsonSchema, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.combinatorics.symmetric_functions.values import (
    MAX_PARTITION_SIZE,
    IntegerPartition,
)

_MAX_POINT_COORDINATE_DIGITS = 6
_MAX_POINT_COORDINATE_ABS = 10**_MAX_POINT_COORDINATE_DIGITS - 1
_MAX_SCHUR_RESULT_DIGITS = 4000
_MAX_SCHUR_PARTITION_LENGTH = 50


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by symmetric-function contracts."""

    return PydanticCustomError(f"symmetric_function.{reason}", message)


PointCoordinate = Annotated[
    int,
    Field(
        ge=-_MAX_POINT_COORDINATE_ABS,
        le=_MAX_POINT_COORDINATE_ABS,
        description=(
            "Canonical integer with at most "
            f"{_MAX_POINT_COORDINATE_DIGITS} decimal digits."
        ),
    ),
]
"""One bounded evaluation coordinate: ``abs(value) <= 10**6 - 1``."""


def _schur_partition_schema() -> JsonSchemaValue:
    """Project the Jacobi-Trudi part bound onto the shared partition schema."""

    schema = IntegerPartition.model_json_schema()
    schema["properties"]["parts"].update(
        maxItems=_MAX_SCHUR_PARTITION_LENGTH,
        description=(
            "Positive weakly-decreasing parts with a total size (sum) of at "
            f"most {MAX_PARTITION_SIZE}; at most {_MAX_SCHUR_PARTITION_LENGTH} "
            "parts for this operation."
        ),
    )
    return schema


class PartitionRequest(StrictModel):
    partition: IntegerPartition


class PartitionConjugateResult(StrictModel):
    conjugate: IntegerPartition


class SchurExpansionRequest(StrictModel):
    """Evaluate one Schur function at a bounded integer point.

    Preconditions published through this schema: ``variables`` and ``point``
    must have equal lengths in ``[1, 20]``, variable names must be distinct,
    each coordinate satisfies ``abs(coordinate) <= 999999``, and the partition
    total size is capped at 500.
    """

    partition: Annotated[
        IntegerPartition,
        WithJsonSchema(_schur_partition_schema()),
    ] = Field(
        description=(
            "A canonical partition of total size at most "
            f"{MAX_PARTITION_SIZE} with at most {_MAX_SCHUR_PARTITION_LENGTH} "
            "parts for the admitted Jacobi-Trudi determinant."
        )
    )
    variables: tuple[str, ...] = Field(
        min_length=1,
        max_length=20,
        description=(
            "Distinct variable names; the length must equal the length of "
            "point (between 1 and 20)."
        ),
        json_schema_extra={"uniqueItems": True},
    )
    point: tuple[PointCoordinate, ...] = Field(
        min_length=1,
        max_length=20,
        description=(
            "Integer evaluation coordinates, one per variable, each with at "
            f"most {_MAX_POINT_COORDINATE_DIGITS} decimal digits; the length "
            "must equal the length of variables."
        ),
    )

    @model_validator(mode="after")
    def require_matching_dimensions(self) -> Self:
        if len(self.partition.parts) > _MAX_SCHUR_PARTITION_LENGTH:
            raise _validation_error(
                "schur_partition_length_exceeded",
                "Schur evaluation partition length must not exceed "
                f"{_MAX_SCHUR_PARTITION_LENGTH}",
            )
        if len(self.variables) != len(self.point):
            raise _validation_error(
                "schur_dimensions_mismatch",
                "variables and point must have the same length",
            )
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "schur_variables_not_distinct",
                "variables must be distinct (duplicate axis)",
            )
        return self


class SchurExpansionResult(StrictModel):
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_bounded_value(self) -> Self:
        if len(self.value.lstrip("-")) > _MAX_SCHUR_RESULT_DIGITS:
            raise _validation_error(
                "schur_value_digits_exceeded",
                "Schur value exceeds the output digit bound",
            )
        return self


__all__ = [
    "IntegerPartition",
    "PartitionConjugateResult",
    "PartitionRequest",
    "SchurExpansionRequest",
    "SchurExpansionResult",
]
