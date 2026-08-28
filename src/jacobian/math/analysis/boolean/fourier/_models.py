"""Typed wire contracts for Boolean function analysis operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

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

    @model_validator(mode="after")
    def require_valid_truth_table(self) -> Self:
        n = len(self.truth_table)
        if n & (n - 1) != 0:
            raise _validation_error(
                "truth_table_power", "truth table length must be a power of two"
            )
        variable_count = n.bit_length() - 1
        if not (MIN_VARIABLES <= variable_count <= MAX_VARIABLES):
            raise _validation_error(
                "variable_count", "variable count must be between 1 and 10"
            )
        for entry in self.truth_table:
            value = entry.as_fraction()
            if value not in (0, 1):
                raise _validation_error(
                    "truth_table_boolean", "truth table entry must be 0 or 1"
                )
        return self

    def as_int_list(self) -> list[int]:
        """Return the truth table as a list of plain 0/1 integers."""
        return [int(entry.as_fraction()) for entry in self.truth_table]


class TruthTableResult(StrictModel):
    """Result of evaluating a Boolean function over all 2^n inputs."""

    truth_table: tuple[CanonicalRational, ...] = Field(
        min_length=2,
        max_length=1 << MAX_VARIABLES,
    )
    variable_count: int = Field(ge=1, le=MAX_VARIABLES)
    convention: Literal["NATURAL_ORDER"] = "NATURAL_ORDER"


class FourierSpectrumRequest(TruthTableRequest):
    """Compute the Fourier/Walsh-Hadamard spectrum of a Boolean function."""


class FourierSpectrumResult(StrictModel):
    """The exact integer Fourier/Walsh-Hadamard spectrum of a truth table."""

    spectrum: tuple[CanonicalRational, ...] = Field(
        min_length=2,
        max_length=1 << MAX_VARIABLES,
    )
    variable_count: int = Field(ge=1, le=MAX_VARIABLES)
    convention: Literal["WALSH_HADAMARD"] = "WALSH_HADAMARD"

    @model_validator(mode="after")
    def require_spectrum_shape(self) -> Self:
        if len(self.spectrum) != 1 << self.variable_count:
            raise _validation_error(
                "spectrum_length", "spectrum length must equal 2 ** variable_count"
            )
        return self


class MultilinearExtensionRequest(TruthTableRequest):
    """Compute the multilinear extension of a Boolean function."""


class MultilinearExtensionResult(StrictModel):
    """The multilinear extension polynomial as a SymPy string."""

    polynomial: str = Field(min_length=1)
    variable_count: int = Field(ge=1, le=MAX_VARIABLES)


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

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        n = len(self.truth_table)
        if n & (n - 1) != 0:
            raise _validation_error(
                "truth_table_power", "truth table length must be a power of two"
            )
        variable_count = n.bit_length() - 1
        if not (MIN_VARIABLES <= variable_count <= MAX_VARIABLES):
            raise _validation_error(
                "variable_count", "variable count must be between 1 and 10"
            )
        for entry in self.truth_table:
            value = entry.as_fraction()
            if value not in (0, 1):
                raise _validation_error(
                    "truth_table_boolean", "truth table entry must be 0 or 1"
                )
        p = self.probability.as_fraction()
        if not (0 <= p <= 1):
            raise _validation_error(
                "probability_range", "probability must be in [0, 1]"
            )
        if len(self.base_input) != variable_count:
            raise _validation_error(
                "base_input_length", "base_input must have one bit per variable"
            )
        if any(bit not in (0, 1) for bit in self.base_input):
            raise _validation_error(
                "base_input_boolean", "base_input bits must be 0 or 1"
            )
        return self

    def as_int_list(self) -> list[int]:
        return [int(entry.as_fraction()) for entry in self.truth_table]


class ErasureNoiseResult(StrictModel):
    """The expected value of a Boolean function under erasure noise."""

    expected_value: CanonicalRational
    variable_count: int = Field(ge=1, le=MAX_VARIABLES)
    probability: CanonicalRational
    convention: Literal["FOURIER_WEIGHTED"] = "FOURIER_WEIGHTED"
