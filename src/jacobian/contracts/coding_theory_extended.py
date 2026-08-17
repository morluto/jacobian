"""Typed wire contracts for extended coding theory operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel

MAX_CODE_LENGTH = 256
MAX_CODE_DIMENSION = 64
MAX_FIELD_ORDER = 251


class GeneratorMatrix(ContractModel):
    """A generator matrix over a prime field."""

    field_order: int = Field(ge=2, le=MAX_FIELD_ORDER)
    generator_matrix: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_CODE_DIMENSION
    )

    @model_validator(mode="after")
    def require_valid_matrix(self) -> Self:
        from sympy import isprime

        if not isprime(self.field_order):
            raise ValueError("field_order must be prime")
        width = len(self.generator_matrix[0])
        if width == 0 or width > MAX_CODE_LENGTH:
            raise ValueError("code length must be between 1 and 256")
        for row in self.generator_matrix:
            if len(row) != width:
                raise ValueError("generator matrix rows must have equal length")
            for entry in row:
                if not (0 <= entry < self.field_order):
                    raise ValueError(
                        "generator entries must be canonical field residues"
                    )
        return self


class DualCodeRequest(ContractModel):
    """Compute the dual code's parity-check matrix from a generator matrix."""

    code: GeneratorMatrix


class ParityCheckResult(ContractModel):
    """A parity-check matrix H such that GH^T = 0."""

    parity_check_matrix: tuple[tuple[int, ...], ...] = Field(default=())
    field_order: int = Field(ge=2, le=MAX_FIELD_ORDER)
    code_length: int = Field(ge=1, le=MAX_CODE_LENGTH)
    code_dimension: int = Field(ge=1, le=MAX_CODE_DIMENSION)
    method: Literal["RREF_NULLSPACE"] = "RREF_NULLSPACE"


class PunctureRequest(ContractModel):
    """Puncture a code by deleting one coordinate position."""

    code: GeneratorMatrix
    position: int = Field(ge=0, le=MAX_CODE_LENGTH - 1)

    @model_validator(mode="after")
    def require_valid_position(self) -> Self:
        width = len(self.code.generator_matrix[0])
        if self.position >= width:
            raise ValueError("position must be within the code length")
        return self


class PunctureResult(ContractModel):
    """A punctured code's generator matrix."""

    generator_matrix: tuple[tuple[int, ...], ...] = Field(min_length=1)
    field_order: int = Field(ge=2, le=MAX_FIELD_ORDER)
    code_length: int = Field(ge=0, le=MAX_CODE_LENGTH)
    method: Literal["COLUMN_DELETION"] = "COLUMN_DELETION"


class ShortenRequest(ContractModel):
    """Shorten a code by fixing one coordinate to zero and deleting it.

    Only the zero value is supported: a nonzero value would select an
    affine coset rather than a linear subcode, which cannot be represented
    by a generator matrix.
    """

    code: GeneratorMatrix
    position: int = Field(ge=0, le=MAX_CODE_LENGTH - 1)

    @model_validator(mode="after")
    def require_valid_position(self) -> Self:
        width = len(self.code.generator_matrix[0])
        if self.position >= width:
            raise ValueError("position must be within the code length")
        return self


class ShortenResult(ContractModel):
    """A shortened code's generator matrix."""

    generator_matrix: tuple[tuple[int, ...], ...] = Field(min_length=0)
    field_order: int = Field(ge=2, le=MAX_FIELD_ORDER)
    code_length: int = Field(ge=0, le=MAX_CODE_LENGTH)
    method: Literal["COORDINATE_FIX_AND_DELETE"] = "COORDINATE_FIX_AND_DELETE"


__all__ = [
    "DualCodeRequest",
    "GeneratorMatrix",
    "ParityCheckResult",
    "PunctureRequest",
    "PunctureResult",
    "ShortenRequest",
    "ShortenResult",
]
