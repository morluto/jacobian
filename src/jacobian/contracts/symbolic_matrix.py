"""Typed wire contracts for symbolic matrix operations over QQ(t_1, ..., t_n)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel

MAX_SYMBOLIC_MATRIX_DIMENSION = 32
MAX_SYMBOLIC_VARIABLES = 16
MAX_SYMBOLIC_ENTRY_LENGTH = 4096


class SymbolicMatrix(ContractModel):
    """One nonempty rectangular matrix over a multivariate rational-function field.

    Each entry is a string SymPy expression in the declared variables, for
    example ``"a*c"`` or ``"f/e"``.  The expression is parsed with SymPy and
    reduced over ``QQ(t_1, ..., t_n)``.
    """

    matrix_schema_version: Literal["1"] = "1"
    variables: tuple[str, ...] = Field(
        min_length=0,
        max_length=MAX_SYMBOLIC_VARIABLES,
    )
    entries: tuple[tuple[str, ...], ...] = Field(
        min_length=1,
        max_length=MAX_SYMBOLIC_MATRIX_DIMENSION,
    )

    @model_validator(mode="after")
    def require_rectangular_nonempty_rows(self) -> Self:
        column_count = len(self.entries[0])
        if column_count == 0 or column_count > MAX_SYMBOLIC_MATRIX_DIMENSION:
            raise ValueError(
                "matrix rows must contain between 1 and "
                f"{MAX_SYMBOLIC_MATRIX_DIMENSION} entries"
            )
        if any(len(row) != column_count for row in self.entries):
            raise ValueError("matrix rows must all have the same length")
        for row in self.entries:
            for value in row:
                if len(value) > MAX_SYMBOLIC_ENTRY_LENGTH:
                    raise ValueError("symbolic matrix entry exceeds the length bound")
        return self


class SymbolicMatrixRequest(ContractModel):
    """A symbolic matrix over a declared variable list."""

    matrix: SymbolicMatrix

    @model_validator(mode="after")
    def require_request_consistency(self) -> Self:
        return self


class SymbolicDeterminantResult(ContractModel):
    """The exact symbolic determinant as a canonical SymPy expression string."""

    determinant: str
    method: Literal["SYMPY_BAREISS"] = "SYMPY_BAREISS"


class SymbolicRankResult(ContractModel):
    """The exact symbolic rank and the canonical pivot columns."""

    rank: int = Field(ge=0, le=MAX_SYMBOLIC_MATRIX_DIMENSION)
    pivot_columns: tuple[int, ...] = Field(max_length=MAX_SYMBOLIC_MATRIX_DIMENSION)
    method: Literal["EXACT_SYMBOLIC_ROW_REDUCTION"] = "EXACT_SYMBOLIC_ROW_REDUCTION"


class SymbolicCharacteristicPolynomialResult(ContractModel):
    """The dense monic characteristic polynomial coefficients (descending)."""

    variable: Literal["lambda"] = "lambda"
    degree: int = Field(ge=1, le=MAX_SYMBOLIC_MATRIX_DIMENSION)
    coefficients_descending: tuple[str, ...] = Field(
        min_length=2,
        max_length=MAX_SYMBOLIC_MATRIX_DIMENSION + 1,
    )
    convention: Literal["DET_LAMBDA_I_MINUS_A"] = "DET_LAMBDA_I_MINUS_A"


class SymbolicEigenvaluesResult(ContractModel):
    """The exact eigenvalues with algebraic multiplicities."""

    eigenvalues: tuple[str, ...] = Field(
        min_length=1,
        max_length=MAX_SYMBOLIC_MATRIX_DIMENSION,
    )
    multiplicities: tuple[int, ...] = Field(
        min_length=1,
        max_length=MAX_SYMBOLIC_MATRIX_DIMENSION,
    )
    convention: Literal["SYMPY_EIGENVALS"] = "SYMPY_EIGENVALS"

    @model_validator(mode="after")
    def require_matching_lengths(self) -> Self:
        if len(self.eigenvalues) != len(self.multiplicities):
            raise ValueError("eigenvalues and multiplicities must have the same length")
        return self
