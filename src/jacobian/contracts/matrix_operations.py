"""Bounded exact matrix-operation contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from jacobian.contracts.exact import CanonicalInteger, CanonicalRational
from jacobian.contracts.matrices import (
    MAX_MATRIX_DIMENSION,
    MAX_MATRIX_SCALAR_DIGITS,
    IntegerMatrix,
    RationalMatrix,
    require_matrix_scalar_digits,
)
from jacobian.contracts.results import ContractModel

MAX_INPUT_SCALAR_DIGITS = 256
MAX_DETERMINANT_MATRIX_DIMENSION = 64

DeterminantRow = Annotated[
    tuple[CanonicalRational, ...],
    Field(min_length=1, max_length=MAX_DETERMINANT_MATRIX_DIMENSION),
]


def _check_integer_digits(
    value: str, *, maximum: int = MAX_INPUT_SCALAR_DIGITS
) -> None:
    if len(value.lstrip("-")) > maximum:
        raise ValueError(f"matrix scalars are limited to {maximum} decimal digits")


class RationalMatrixRequest(ContractModel):
    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_rref_input_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
        )
        return self


class RationalMatrixProductRequest(ContractModel):
    """Two compatible bounded matrices over the exact rational domain."""

    left: RationalMatrix
    right: RationalMatrix

    @model_validator(mode="after")
    def require_compatible_shapes(self) -> Self:
        if len(self.left.entries[0]) != len(self.right.entries):
            raise ValueError(
                "matrix multiplication requires the left column count to equal "
                "the right row count"
            )
        require_matrix_scalar_digits(
            self.left.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
        )
        require_matrix_scalar_digits(
            self.right.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
        )
        return self


class SquareRationalMatrixRequest(ContractModel):
    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_square(self) -> Self:
        if len(self.matrix.entries) != len(self.matrix.entries[0]):
            raise ValueError("characteristic polynomial requires a square matrix")
        require_matrix_scalar_digits(
            self.matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
        )
        return self


class DeterminantRationalMatrix(ContractModel):
    """One determinant-owned rational matrix bounded independently to order 64."""

    matrix_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    entries: tuple[DeterminantRow, ...] = Field(
        min_length=1, max_length=MAX_DETERMINANT_MATRIX_DIMENSION
    )

    @model_validator(mode="after")
    def require_rectangular_nonempty_rows(self) -> Self:
        column_count = len(self.entries[0])
        if not 1 <= column_count <= MAX_DETERMINANT_MATRIX_DIMENSION:
            raise ValueError(
                "determinant matrix rows must contain between 1 and 64 entries"
            )
        if any(len(row) != column_count for row in self.entries):
            raise ValueError("determinant matrix rows must all have the same length")
        require_matrix_scalar_digits(
            self.entries,
            maximum=MAX_INPUT_SCALAR_DIGITS,
            label="determinant input",
        )
        return self


class MatrixDeterminantRequest(ContractModel):
    """One square rational matrix of order at most 64."""

    matrix: DeterminantRationalMatrix

    @model_validator(mode="after")
    def require_square(self) -> Self:
        if len(self.matrix.entries) != len(self.matrix.entries[0]):
            raise ValueError("determinant computation requires a square matrix")
        return self


class MatrixRankRequest(ContractModel):
    """One bounded rectangular matrix whose exact rank is requested."""

    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_input_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.matrix.entries,
            maximum=MAX_INPUT_SCALAR_DIGITS,
            label="rank input",
        )
        return self


class IntegerMatrixRequest(ContractModel):
    matrix: IntegerMatrix

    @model_validator(mode="after")
    def require_integer_input_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
        )
        return self


class SquareIntegerMatrixRequest(ContractModel):
    matrix: IntegerMatrix

    @model_validator(mode="after")
    def require_square(self) -> Self:
        if len(self.matrix.entries) != len(self.matrix.entries[0]):
            raise ValueError("operation requires a square integer matrix")
        require_matrix_scalar_digits(
            self.matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
        )
        return self


class RationalLinearSolveRequest(ContractModel):
    matrix: RationalMatrix
    rhs: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_MATRIX_DIMENSION,
    )

    @model_validator(mode="after")
    def require_square_system(self) -> Self:
        rows = len(self.matrix.entries)
        if len(self.matrix.entries[0]) != rows or len(self.rhs) != rows:
            raise ValueError("linear solve requires a square matrix and matching rhs")
        require_matrix_scalar_digits(
            self.matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
        )
        for value in self.rhs:
            _check_integer_digits(value.num)
            _check_integer_digits(value.den)
        return self


class LatticeReductionRequest(ContractModel):
    basis: IntegerMatrix

    @model_validator(mode="after")
    def require_lattice_input_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.basis.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="basis input"
        )
        return self


class RrefResult(ContractModel):
    reduced_matrix: RationalMatrix
    rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    pivot_columns: tuple[int, ...] = Field(max_length=MAX_MATRIX_DIMENSION)
    free_columns: tuple[int, ...] = Field(max_length=MAX_MATRIX_DIMENSION)
    convention: Literal["UNIQUE_RREF_OVER_QQ"] = "UNIQUE_RREF_OVER_QQ"


class MatrixDeterminantResult(ContractModel):
    """One exact determinant, returned inline for ordinary composition."""

    determinant: CanonicalRational
    method: Literal["FRACTION_FREE_BAREISS"] = "FRACTION_FREE_BAREISS"


class MatrixRankResult(ContractModel):
    """One exact rank with the canonical RREF pivot columns."""

    rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    pivot_columns: tuple[int, ...] = Field(max_length=MAX_MATRIX_DIMENSION)
    method: Literal["EXACT_RATIONAL_ROW_REDUCTION"] = "EXACT_RATIONAL_ROW_REDUCTION"


class NullspaceResult(ContractModel):
    ambient_dimension: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    nullity: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    basis_vectors: tuple[tuple[CanonicalRational, ...], ...] = Field(
        max_length=MAX_MATRIX_DIMENSION
    )
    free_columns: tuple[int, ...] = Field(max_length=MAX_MATRIX_DIMENSION)
    convention: Literal["RREF_FUNDAMENTAL_BASIS"] = "RREF_FUNDAMENTAL_BASIS"

    @model_validator(mode="after")
    def require_basis_shape(self) -> Self:
        if self.rank + self.nullity != self.ambient_dimension:
            raise ValueError("rank plus nullity must equal the ambient dimension")
        if len(self.basis_vectors) != self.nullity:
            raise ValueError("basis vector count must equal nullity")
        if any(len(vector) != self.ambient_dimension for vector in self.basis_vectors):
            raise ValueError("each basis vector must have the ambient dimension")
        if len(self.free_columns) != self.nullity:
            raise ValueError("free column count must equal nullity")
        return self


class CharacteristicPolynomialResult(ContractModel):
    variable: Literal["lambda"] = "lambda"
    degree: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    coefficients_descending: tuple[CanonicalRational, ...] = Field(
        min_length=2,
        max_length=MAX_MATRIX_DIMENSION + 1,
    )
    monic: Literal[True] = True
    convention: Literal["DET_LAMBDA_I_MINUS_A"] = "DET_LAMBDA_I_MINUS_A"

    @model_validator(mode="after")
    def require_dense_monic_coefficients(self) -> Self:
        if len(self.coefficients_descending) != self.degree + 1:
            raise ValueError("dense coefficient count must be degree plus one")
        if self.coefficients_descending[0] != CanonicalRational(num="1", den="1"):
            raise ValueError("characteristic polynomial must be monic")
        return self


class MatrixInverseResult(ContractModel):
    inverse: RationalMatrix
    convention: Literal["TWO_SIDED_INVERSE_OVER_QQ"] = "TWO_SIDED_INVERSE_OVER_QQ"


class MatrixTraceResult(ContractModel):
    trace: CanonicalInteger
    convention: Literal["SUM_OF_DIAGONAL_ENTRIES"] = "SUM_OF_DIAGONAL_ENTRIES"

    @field_validator("trace")
    @classmethod
    def require_bounded_trace(cls, value: CanonicalInteger) -> CanonicalInteger:
        _check_integer_digits(value, maximum=MAX_MATRIX_SCALAR_DIGITS)
        return value


class MatrixProductResult(ContractModel):
    product: RationalMatrix
    left_rows: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    inner_dimension: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    right_columns: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    convention: Literal["STANDARD_ROW_BY_COLUMN_PRODUCT_OVER_QQ"] = (
        "STANDARD_ROW_BY_COLUMN_PRODUCT_OVER_QQ"
    )

    @model_validator(mode="after")
    def require_product_shape(self) -> Self:
        if len(self.product.entries) != self.left_rows:
            raise ValueError("product row count must equal left_rows")
        if len(self.product.entries[0]) != self.right_columns:
            raise ValueError("product column count must equal right_columns")
        return self


class RationalLinearSolveResult(ContractModel):
    solution: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_MATRIX_DIMENSION,
    )
    convention: Literal["UNIQUE_SOLUTION_OVER_QQ"] = "UNIQUE_SOLUTION_OVER_QQ"


class MatrixAdjugateResult(ContractModel):
    adjugate: IntegerMatrix
    convention: Literal["CLASSICAL_ADJUGATE"] = "CLASSICAL_ADJUGATE"


class LatticeReductionResult(ContractModel):
    reduced_basis: IntegerMatrix
    transformation: IntegerMatrix
    rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    relation: Literal["REDUCED_BASIS_EQUALS_TRANSFORMATION_TIMES_BASIS"] = (
        "REDUCED_BASIS_EQUALS_TRANSFORMATION_TIMES_BASIS"
    )
    representation: Literal["INTEGER_ROW_BASIS"] = "INTEGER_ROW_BASIS"
    gram_mode: Literal["EXACT"] = "EXACT"
    delta: Literal["0.99"] = "0.99"
    eta: Literal["0.51"] = "0.51"

    @model_validator(mode="after")
    def require_transformation_shape(self) -> Self:
        rows = len(self.reduced_basis.entries)
        if len(self.transformation.entries) != rows:
            raise ValueError("LLL transformation must have one row per basis row")
        if len(self.transformation.entries[0]) != rows:
            raise ValueError("LLL transformation must be square by basis row count")
        return self
