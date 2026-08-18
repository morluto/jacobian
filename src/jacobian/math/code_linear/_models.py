"""Typed wire contracts for linear code structural operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_CODEWORDS = 4096
MAX_LENGTH = 32


def _validate_prime_matrix(
    field_order: int,
    generator_matrix: tuple[tuple[int, ...], ...],
) -> int:
    from sympy import isprime

    if not isprime(field_order):
        raise ValueError("field_order must be prime")
    width = len(generator_matrix[0])
    if width == 0 or width > MAX_LENGTH:
        raise ValueError("generator rows must have between 1 and 32 entries")
    if any(len(row) != width for row in generator_matrix):
        raise ValueError("generator matrix rows must have equal length")
    if any(not 0 <= entry < field_order for row in generator_matrix for entry in row):
        raise ValueError("generator entries must be canonical field residues")
    return width


class GeneratorMatrixRequest(StrictModel):
    """A linear code given by a generator matrix over a bounded prime field."""

    field_order: int = Field(ge=2, le=251)
    generator_matrix: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def require_bounded_prime_matrix(self) -> Self:
        width = _validate_prime_matrix(self.field_order, self.generator_matrix)
        if self.field_order ** len(self.generator_matrix) > MAX_CODEWORDS:
            raise ValueError("generator matrix exceeds exact enumeration bound")
        if width > MAX_LENGTH:
            raise ValueError("code length exceeds bound")
        return self


class MacWilliamsRequest(StrictModel):
    """Primal weight distribution for MacWilliams transform."""

    field_order: int = Field(ge=2, le=251)
    code_cardinality: int = Field(ge=1)
    length: int = Field(ge=1, le=MAX_LENGTH)
    weights: tuple[int, ...] = Field(min_length=1, max_length=MAX_LENGTH + 1)

    @model_validator(mode="after")
    def require_valid_distribution(self) -> Self:
        if any(w < 0 or w > self.length for w in self.weights):
            raise ValueError("weights must be between 0 and length")
        if self.weights[0] != 1:
            raise ValueError("first weight count must be 1 (zero codeword)")
        if sum(self.weights) != self.code_cardinality:
            raise ValueError("weight counts must sum to code cardinality")
        return self


class PunctureRequest(StrictModel):
    """Puncture a linear code by deleting one coordinate."""

    field_order: int = Field(ge=2, le=251)
    generator_matrix: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)
    coordinate: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        _validate_prime_matrix(self.field_order, self.generator_matrix)
        width = len(self.generator_matrix[0])
        if self.coordinate >= width:
            raise ValueError("coordinate index out of range")
        return self


class DualCodeResult(StrictModel):
    dual_generator: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=0)
    dual_dimension: int = Field(ge=0)
    length: int = Field(ge=0)
    method: str = "NULLSPACE"


class MacWilliamsResult(StrictModel):
    dual_weights: tuple[int, ...]
    method: str = "MACWILLIAMS_IDENTITY"


class PunctureResult(StrictModel):
    generator: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=0)
    length: int = Field(ge=0)
    method: str = "PUNCTURE"
