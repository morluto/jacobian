"""Typed result for exact Graver bases of bounded one-row configurations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, WithJsonSchema, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.affine_semigroups.semigroup import AffineConfiguration
from jacobian.math.matrices.values import IntegerMatrix


def _toric_configuration_json_schema() -> JsonSchemaValue:
    schema = AffineConfiguration.model_json_schema()
    properties = schema["properties"]
    properties["row_labels"].update(minItems=1, maxItems=1)
    properties["generator_labels"].update(minItems=1, maxItems=5)
    properties["generator_labels"]["items"].update(
        minLength=1,
        maxLength=32,
        pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$",
    )
    properties["entries"].update(minItems=1, maxItems=1)
    properties["entries"]["items"].update(minItems=1, maxItems=5)
    properties["entries"]["items"]["items"].update(
        maxLength=8,
        pattern=r"^(?:0|[1-9][0-9]{0,7})$",
    )
    schema["description"] = (
        "One row of 1..5 nonnegative integer weights. Generator labels are "
        "ordered polynomial variables; each weight is an exact integer with "
        "at most 8 decimal digits."
    )
    return schema


class IntegerConfigurationGraverBasis(StrictModel):
    """Complete sign-normalized Graver basis on the retained generator axis."""

    configuration: IntegerMatrix
    vectors: tuple[tuple[ExactInteger, ...], ...] = Field(max_length=5_000)
    convention: Literal["PRIMITIVE_CONFORMAL_MINIMA_FIRST_NONZERO_POSITIVE"] = (
        "PRIMITIVE_CONFORMAL_MINIMA_FIRST_NONZERO_POSITIVE"
    )

    @model_validator(mode="after")
    def require_canonical_vectors(self) -> Self:
        n = self.configuration.column_count
        if self.configuration.row_count != 1:
            raise PydanticCustomError(
                "affine_semigroup.graver_rows",
                "Graver results require one-row configurations",
            )
        if any(
            len(vector) != n
            or not any(vector)
            or next(value for value in vector if value) < 0
            for vector in self.vectors
        ):
            raise PydanticCustomError(
                "affine_semigroup.graver_axis",
                "Graver vectors must be nonzero and use the canonical configuration axis",
            )
        if self.vectors != tuple(sorted(set(self.vectors))):
            raise PydanticCustomError(
                "affine_semigroup.graver_order",
                "Graver vectors must be sorted and unique",
            )
        return self


class IntegerConfigurationGraverRequest(StrictModel):
    """Request the complete Graver basis of a bounded one-row matrix."""

    configuration: IntegerMatrix

    @model_validator(mode="after")
    def require_one_row(self) -> Self:
        if self.configuration.row_count != 1:
            raise PydanticCustomError(
                "affine_semigroup.graver_rows",
                "Graver enumeration requires exactly one row",
            )
        if not 1 <= self.configuration.column_count <= 5:
            raise PydanticCustomError(
                "affine_semigroup.graver_columns",
                "Graver enumeration supports 1..5 columns",
            )
        return self


class IntegerConfigurationMarkovBasis(StrictModel):
    """A globally fiber-connecting generating set for a one-row configuration."""

    configuration: IntegerMatrix
    moves: tuple[tuple[ExactInteger, ...], ...] = Field(max_length=5_000)
    property: Literal["GENERATES_TORIC_IDEAL_AND_CONNECTS_EVERY_NONNEGATIVE_FIBER"] = (
        "GENERATES_TORIC_IDEAL_AND_CONNECTS_EVERY_NONNEGATIVE_FIBER"
    )
    construction: Literal["COMPLETE_GRAVER_BASIS"] = "COMPLETE_GRAVER_BASIS"

    @model_validator(mode="after")
    def require_canonical_moves(self) -> Self:
        n = self.configuration.column_count
        if self.configuration.row_count != 1 or not 1 <= n <= 5:
            raise PydanticCustomError(
                "affine_semigroup.markov_shape",
                "Markov results require a one-row configuration with 1..5 columns",
            )
        if any(
            len(move) != n
            or not any(move)
            or next(value for value in move if value) < 0
            for move in self.moves
        ):
            raise PydanticCustomError(
                "affine_semigroup.markov_axis",
                "Markov moves must be nonzero, sign-normalized vectors on the configuration axis",
            )
        if self.moves != tuple(sorted(set(self.moves))):
            raise PydanticCustomError(
                "affine_semigroup.markov_order",
                "Markov moves must be sorted and unique",
            )
        return self


class IntegerConfigurationMarkovBasisRequest(StrictModel):
    """Request a globally connecting move family for a bounded one-row matrix."""

    configuration: IntegerMatrix

    @model_validator(mode="after")
    def require_one_row(self) -> Self:
        if self.configuration.row_count != 1:
            raise PydanticCustomError(
                "affine_semigroup.markov_rows",
                "Markov basis enumeration requires exactly one row",
            )
        if not 1 <= self.configuration.column_count <= 5:
            raise PydanticCustomError(
                "affine_semigroup.markov_columns",
                "Markov basis enumeration supports 1..5 columns",
            )
        return self


class IntegerConfigurationToricIdealRequest(StrictModel):
    """Return the QQ toric ideal for an admitted one-row configuration."""

    configuration: Annotated[
        AffineConfiguration,
        WithJsonSchema(_toric_configuration_json_schema()),
    ] = Field(
        description=(
            "A one-row nonnegative integer configuration with 1..5 labelled "
            "generators. Generator labels must be polynomial variable names. "
            "Admission bounds the Graver candidate work, the number of possible "
            "ideal generators, polynomial exponents, and serialized output."
        )
    )


__all__ = [
    "IntegerConfigurationGraverBasis",
    "IntegerConfigurationGraverRequest",
    "IntegerConfigurationMarkovBasis",
    "IntegerConfigurationMarkovBasisRequest",
    "IntegerConfigurationToricIdealRequest",
]
