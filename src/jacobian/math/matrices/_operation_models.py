"""Bounded exact matrix-operation contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.matrices.values import (
    MAX_MATRIX_DIMENSION,
    MAX_MATRIX_SCALAR_DIGITS,
    IntegerMatrix,
    RationalMatrix,
    require_matrix_scalar_digits,
)

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


class RationalMatrixRequest(StrictModel):
    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_rref_input_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
        )
        return self


class RationalMatrixProductRequest(StrictModel):
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


class SquareRationalMatrixRequest(StrictModel):
    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_square(self) -> Self:
        if len(self.matrix.entries) != len(self.matrix.entries[0]):
            raise ValueError("characteristic polynomial requires a square matrix")
        require_matrix_scalar_digits(
            self.matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
        )
        return self


class DeterminantRationalMatrix(StrictModel):
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


class MatrixDeterminantRequest(StrictModel):
    """One square rational matrix of order at most 64."""

    matrix: DeterminantRationalMatrix

    @model_validator(mode="after")
    def require_square(self) -> Self:
        if len(self.matrix.entries) != len(self.matrix.entries[0]):
            raise ValueError("determinant computation requires a square matrix")
        return self


class MatrixRankRequest(StrictModel):
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


class IntegerMatrixRequest(StrictModel):
    matrix: IntegerMatrix

    @model_validator(mode="after")
    def require_integer_input_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
        )
        return self


class SquareIntegerMatrixRequest(StrictModel):
    matrix: IntegerMatrix

    @model_validator(mode="after")
    def require_square(self) -> Self:
        if len(self.matrix.entries) != len(self.matrix.entries[0]):
            raise ValueError("operation requires a square integer matrix")
        require_matrix_scalar_digits(
            self.matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
        )
        return self


class RationalLinearSolveRequest(StrictModel):
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


class RrefResult(StrictModel):
    reduced_matrix: RationalMatrix
    rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    pivot_columns: tuple[int, ...] = Field(max_length=MAX_MATRIX_DIMENSION)
    free_columns: tuple[int, ...] = Field(max_length=MAX_MATRIX_DIMENSION)
    convention: Literal["UNIQUE_RREF_OVER_QQ"] = "UNIQUE_RREF_OVER_QQ"


class MatrixDeterminantResult(StrictModel):
    """One exact determinant, returned inline for ordinary composition."""

    determinant: CanonicalRational
    method: Literal["FRACTION_FREE_BAREISS"] = "FRACTION_FREE_BAREISS"


class MatrixRankResult(StrictModel):
    """One exact rank with the canonical RREF pivot columns."""

    rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    pivot_columns: tuple[int, ...] = Field(max_length=MAX_MATRIX_DIMENSION)
    method: Literal["EXACT_RATIONAL_ROW_REDUCTION"] = "EXACT_RATIONAL_ROW_REDUCTION"


class NullspaceResult(StrictModel):
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


class CharacteristicPolynomialResult(StrictModel):
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


class MatrixInverseResult(StrictModel):
    inverse: RationalMatrix
    convention: Literal["TWO_SIDED_INVERSE_OVER_QQ"] = "TWO_SIDED_INVERSE_OVER_QQ"


class MatrixTraceResult(StrictModel):
    trace: CanonicalInteger
    convention: Literal["SUM_OF_DIAGONAL_ENTRIES"] = "SUM_OF_DIAGONAL_ENTRIES"

    @field_validator("trace")
    @classmethod
    def require_bounded_trace(cls, value: CanonicalInteger) -> CanonicalInteger:
        _check_integer_digits(value, maximum=MAX_MATRIX_SCALAR_DIGITS)
        return value


class MatrixProductResult(StrictModel):
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


class RationalLinearSolveResult(StrictModel):
    solution: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_MATRIX_DIMENSION,
    )
    convention: Literal["UNIQUE_SOLUTION_OVER_QQ"] = "UNIQUE_SOLUTION_OVER_QQ"


class MatrixAdjugateResult(StrictModel):
    adjugate: IntegerMatrix
    convention: Literal["CLASSICAL_ADJUGATE"] = "CLASSICAL_ADJUGATE"


class MatrixPermanentResult(StrictModel):
    """One exact matrix permanent."""

    permanent: CanonicalRational
    method: Literal["SYMPY_PERMANENT"] = "SYMPY_PERMANENT"


class MatrixKroneckerProductRequest(StrictModel):
    """Two bounded matrices for an exact Kronecker product over QQ."""

    left: RationalMatrix
    right: RationalMatrix

    @model_validator(mode="after")
    def require_input_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.left.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
        )
        require_matrix_scalar_digits(
            self.right.entries,
            maximum=MAX_INPUT_SCALAR_DIGITS,
            label="matrix input",
        )
        return self


class MatrixKroneckerProductResult(StrictModel):
    """The Kronecker product of two bounded matrices over QQ."""

    product: RationalMatrix
    left_rows: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    left_columns: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    right_rows: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    right_columns: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    convention: Literal["SYMPY_KRONECKER_PRODUCT_OVER_QQ"] = (
        "SYMPY_KRONECKER_PRODUCT_OVER_QQ"
    )

    @model_validator(mode="after")
    def require_product_shape(self) -> Self:
        if len(self.product.entries) != self.left_rows * self.right_rows:
            raise ValueError(
                "Kronecker product row count must equal left_rows * right_rows"
            )
        if len(self.product.entries[0]) != self.left_columns * self.right_columns:
            raise ValueError(
                "Kronecker product column count must equal left_columns * right_columns"
            )
        return self


class MatrixPartialTraceRequest(StrictModel):
    """A composite matrix (Kronecker product A (x) B) and the subsystem dimensions."""

    matrix: RationalMatrix
    traced_dimension: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    kept_dimension: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)

    @model_validator(mode="after")
    def require_composite_shape(self) -> Self:
        total = self.traced_dimension * self.kept_dimension
        if len(self.matrix.entries) != total:
            raise ValueError(
                "composite matrix row count must equal traced_dimension * kept_dimension"
            )
        if len(self.matrix.entries[0]) != total:
            raise ValueError(
                "composite matrix must be square: traced_dimension * kept_dimension"
            )
        require_matrix_scalar_digits(
            self.matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
        )
        return self


class MatrixPartialTraceResult(StrictModel):
    """The partial trace over the traced subsystem of a composite matrix."""

    reduced_matrix: RationalMatrix
    traced_dimension: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    kept_dimension: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    convention: Literal["BLOCK_TRACE_OVER_QQ"] = "BLOCK_TRACE_OVER_QQ"

    @model_validator(mode="after")
    def require_reduced_shape(self) -> Self:
        if len(self.reduced_matrix.entries) != self.kept_dimension:
            raise ValueError("reduced matrix row count must equal kept_dimension")
        if len(self.reduced_matrix.entries[0]) != self.kept_dimension:
            raise ValueError("reduced matrix must be square of order kept_dimension")
        return self
