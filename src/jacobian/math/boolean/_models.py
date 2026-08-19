"""Typed wire contracts for Boolean truth-table operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel

MAX_TRUTH_TABLE_LENGTH = 4096


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

    @model_validator(mode="after")
    def require_power_of_two_length(self) -> Self:
        n = len(self.truth_table)
        if n & (n - 1) != 0:
            raise ValueError("truth table length must be a power of two")
        return self


class BooleanWalshTransformResult(StrictModel):
    """The exact Boolean Walsh spectrum of a Boolean truth table.

    The spectrum is computed from the sign vector ``(-1)^f = 1 - 2f``,
    using the fast Walsh-Hadamard transform in Hadamard (natural) order.
    """

    spectrum: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_TRUTH_TABLE_LENGTH,
    )
    variable_count: int = Field(ge=0, le=12)
    ordering: Literal["HADAMARD"] = "HADAMARD"
    convention: Literal["BOOLEAN_SIGN"] = "BOOLEAN_SIGN"

    @model_validator(mode="after")
    def require_spectrum_shape(self) -> Self:
        if len(self.spectrum) != 1 << self.variable_count:
            raise ValueError("spectrum length must equal 2 ** variable_count")
        return self
