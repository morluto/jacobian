"""Typed wire contracts for Boolean truth-table operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel

MAX_WALSH_VARIABLES = 12
MAX_TRUTH_TABLE_LENGTH = 1 << MAX_WALSH_VARIABLES


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"boolean.{reason}", message)


class BooleanTruthTable(StrictModel):
    """Complete Boolean function values in natural little-endian order."""

    values: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_TRUTH_TABLE_LENGTH
    )

    @property
    def variable_count(self) -> int:
        return len(self.values).bit_length() - 1

    @model_validator(mode="after")
    def require_boolean_cube_shape(self) -> Self:
        if len(self.values) & (len(self.values) - 1):
            raise _validation_error(
                "truth_table_power", "truth table length must be a power of two"
            )
        if any(value.as_fraction() not in (0, 1) for value in self.values):
            raise _validation_error(
                "truth_table_boolean", "truth table entries must be 0 or 1"
            )
        return self


class BooleanTruthTableRequest(StrictModel):
    """A finite Boolean truth table indexed in natural (little-endian) order.

    The truth table is a list of ``0``/``1`` values whose length must be a
    positive power of two.  Entry ``i`` is the value of the Boolean function
    at the row whose integer index is ``i``.
    """

    truth_table: tuple[Literal[0, 1], ...] = Field(
        min_length=1,
        max_length=MAX_TRUTH_TABLE_LENGTH,
    )


class BooleanWalshTransformResult(StrictModel):
    """The exact Boolean Walsh spectrum of a Boolean truth table.

    The spectrum is computed from the sign vector ``(-1)^f = 1 - 2f``,
    using the fast Walsh-Hadamard transform in Hadamard (natural) order.
    """

    spectrum: tuple[ExactInteger, ...] = Field(
        min_length=1,
        max_length=MAX_TRUTH_TABLE_LENGTH,
    )
    variable_count: int = Field(ge=0, le=MAX_WALSH_VARIABLES)
    ordering: Literal["HADAMARD"] = "HADAMARD"
    convention: Literal["BOOLEAN_SIGN"] = "BOOLEAN_SIGN"

    @model_validator(mode="after")
    def require_spectrum_shape(self) -> Self:
        if len(self.spectrum) != 1 << self.variable_count:
            raise _validation_error(
                "spectrum_length_mismatch",
                "spectrum length must equal 2 ** variable_count",
            )
        return self


class BooleanRationalVector(StrictModel):
    """Exact coefficient vector indexed by Boolean subset masks."""

    values: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_TRUTH_TABLE_LENGTH
    )

    @property
    def variable_count(self) -> int:
        return len(self.values).bit_length() - 1

    @model_validator(mode="after")
    def require_cube_shape(self) -> Self:
        if len(self.values) & (len(self.values) - 1):
            raise _validation_error(
                "coefficient_power", "coefficient vector length must be a power of two"
            )
        return self
