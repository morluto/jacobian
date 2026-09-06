"""Typed wire contracts for Boolean function analysis operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.analysis.boolean._models import (
    MAX_TRUTH_TABLE_LENGTH,
    BooleanRationalVector,
    BooleanTruthTable,
)

MAX_VARIABLES = 10
MIN_VARIABLES = 1


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"boolean_analysis.{reason}", message)


class TruthTableRequest(StrictModel):
    """Evaluate a Boolean function over all 2^n inputs.

    The truth table is a tuple of ``0``/``1`` values given as canonical
    rationals whose length must be exactly ``2 ** n`` for some ``n`` in
    ``[1, 10]``.  Entry ``i`` is the value of the Boolean function at the row
    whose integer index is ``i`` (little-endian / natural ordering).
    """

    truth_table: tuple[CanonicalRational, ...] = Field(
        min_length=2,
        max_length=1 << MAX_VARIABLES,
    )

    def as_int_list(self) -> list[int]:
        """Return the truth table as a list of plain 0/1 integers."""
        return [int(entry.as_fraction()) for entry in self.truth_table]


class TruthTableResult(StrictModel):
    """A complete source-bound Boolean truth table."""

    truth_table: BooleanTruthTable

    @property
    def variable_count(self) -> int:
        return self.truth_table.variable_count

    convention: Literal["NATURAL_ORDER"] = "NATURAL_ORDER"


class FourierSpectrumRequest(TruthTableRequest):
    """Transform the 0/1 values, including the zero-variable Boolean cube."""

    truth_table: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_TRUTH_TABLE_LENGTH
    )


class FourierSpectrumResult(StrictModel):
    """The exact rational Fourier spectrum of a source truth table."""

    source: BooleanTruthTable
    spectrum: BooleanRationalVector

    @property
    def variable_count(self) -> int:
        return self.source.variable_count

    convention: Literal["BOOLEAN_VALUES"] = "BOOLEAN_VALUES"

    @model_validator(mode="after")
    def require_spectrum_shape(self) -> Self:
        if len(self.spectrum.values) != len(self.source.values):
            raise _validation_error(
                "spectrum_length", "spectrum length must equal 2 ** variable_count"
            )
        return self


class MultilinearExtensionRequest(TruthTableRequest):
    """Compute the multilinear extension of a Boolean function."""


class MultilinearExtensionResult(StrictModel):
    """Exact coefficients of prod(x_i for i in S), indexed by subset mask.

    Bit i names coordinate x_i. Coefficients include zero entries and retain
    every ambient coordinate, even for the zero or constant polynomial.
    """

    source: BooleanTruthTable
    coefficients: BooleanRationalVector
    convention: Literal["SUBSET_MONOMIALS"] = "SUBSET_MONOMIALS"

    @property
    def variable_count(self) -> int:
        return self.source.variable_count

    @model_validator(mode="after")
    def require_coefficient_shape(self) -> Self:
        if len(self.coefficients.values) != len(self.source.values):
            raise _validation_error(
                "coefficient_shape", "coefficient count must equal 2 ** variable_count"
            )
        return self


class ErasureNoiseRequest(StrictModel):
    """Compute the expected value of a Boolean function under erasure noise."""

    truth_table: tuple[CanonicalRational, ...] = Field(
        min_length=2,
        max_length=1 << MAX_VARIABLES,
    )
    probability: CanonicalRational = Field(
        description=(
            "Probability p (0 <= p <= 1) that each coordinate is kept; "
            "the remaining coordinates are replaced with independent "
            "uniform random bits."
        ),
    )
    base_input: tuple[int, ...] = Field(
        min_length=MIN_VARIABLES,
        max_length=MAX_VARIABLES,
        description="Original Boolean assignment at which the noise operator is evaluated.",
    )

    def as_int_list(self) -> list[int]:
        return [int(entry.as_fraction()) for entry in self.truth_table]


class ErasureNoiseResult(StrictModel):
    """A source-bound expected value under erasure noise."""

    source: BooleanTruthTable
    expected_value: CanonicalRational
    probability: CanonicalRational
    base_input: tuple[int, ...]

    @property
    def variable_count(self) -> int:
        return self.source.variable_count

    convention: Literal["FOURIER_WEIGHTED"] = "FOURIER_WEIGHTED"
