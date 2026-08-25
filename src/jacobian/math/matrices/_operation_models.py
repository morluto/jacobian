"""Bounded exact matrix-operation contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.matrices.values import (
    MAX_MATRIX_DIMENSION,
    MAX_MATRIX_SCALAR_DIGITS,
    MAX_RATIONAL_MATRIX_ORDER,
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


def _require_computation_dimensions(
    entries: tuple[tuple[CanonicalRational, ...], ...],
) -> None:
    if len(entries) > MAX_MATRIX_DIMENSION or len(entries[0]) > MAX_MATRIX_DIMENSION:
        raise _validation_error(
            "budget_exceeded",
            "matrix computation dimensions are limited to "
            f"{MAX_MATRIX_DIMENSION} rows and columns",
        )


def _check_integer_digits(
    value: str, *, maximum: int = MAX_INPUT_SCALAR_DIGITS
) -> None:
    if len(value.lstrip("-")) > maximum:
        raise _validation_error(
            "budget_exceeded", f"matrix scalars are limited to {maximum} decimal digits"
        )


def _require_square_system_admission(
    matrix: RationalMatrix, rhs: tuple[CanonicalRational, ...]
) -> None:
    """Apply the linear-solve shape and scalar envelope to one system.

    Shared by the wire request and by result validation, so a retained
    source can never reach replay arithmetic outside this operation's
    admitted work envelope.
    """

    rows = len(matrix.entries)
    if len(matrix.entries[0]) != rows or len(rhs) != rows:
        raise _validation_error(
            "budget_exceeded", "linear solve requires a square matrix and matching rhs"
        )
    require_matrix_scalar_digits(
        matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
    )
    for value in rhs:
        _check_integer_digits(value.num)
        _check_integer_digits(value.den)


class RationalMatrixRequest(StrictModel):
    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_rref_input_budget(self) -> Self:
        _require_computation_dimensions(self.matrix.entries)
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
            raise _validation_error(
                "budget_exceeded",
                "matrix multiplication requires the left column count to equal "
                "the right row count",
            )
        _require_computation_dimensions(self.left.entries)
        _require_computation_dimensions(self.right.entries)
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
            raise _validation_error(
                "budget_exceeded", "characteristic polynomial requires a square matrix"
            )
        _require_computation_dimensions(self.matrix.entries)
        require_matrix_scalar_digits(
            self.matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
        )
        return self


class DeterminantRationalMatrix(StrictModel):
    """One determinant-owned rational matrix bounded independently to order 64."""

    domain: Literal["QQ"] = "QQ"
    entries: tuple[DeterminantRow, ...] = Field(
        min_length=1, max_length=MAX_DETERMINANT_MATRIX_DIMENSION
    )

    @model_validator(mode="after")
    def require_rectangular_nonempty_rows(self) -> Self:
        column_count = len(self.entries[0])
        if not 1 <= column_count <= MAX_DETERMINANT_MATRIX_DIMENSION:
            raise _validation_error(
                "budget_exceeded",
                "determinant matrix rows must contain between 1 and 64 entries",
            )
        if any(len(row) != column_count for row in self.entries):
            raise _validation_error(
                "budget_exceeded",
                "determinant matrix rows must all have the same length",
            )
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
            raise _validation_error(
                "budget_exceeded", "determinant computation requires a square matrix"
            )
        return self


class MatrixRankRequest(StrictModel):
    """One bounded rectangular matrix whose exact rank is requested."""

    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_input_budget(self) -> Self:
        _require_computation_dimensions(self.matrix.entries)
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


class NonsingularIntegerMatrixRequest(StrictModel):
    """A square integer matrix that must be nonsingular (invertible)."""

    matrix: IntegerMatrix

    @model_validator(mode="after")
    def require_square_and_nonsingular(self) -> Self:
        rows = len(self.matrix.entries)
        if rows == 0 or rows != len(self.matrix.entries[0]):
            raise _validation_error(
                "budget_exceeded", "operation requires a square integer matrix"
            )
        require_matrix_scalar_digits(
            self.matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
        )
        from sympy import Matrix

        raw = Matrix([[int(str(v)) for v in row] for row in self.matrix.entries])
        if raw.det() == 0:
            raise _validation_error(
                "budget_exceeded", "matrix is singular; inverse does not exist"
            )
        return self


class SquareIntegerMatrixRequest(StrictModel):
    matrix: IntegerMatrix

    @model_validator(mode="after")
    def require_square(self) -> Self:
        if len(self.matrix.entries) != len(self.matrix.entries[0]):
            raise _validation_error(
                "budget_exceeded", "operation requires a square integer matrix"
            )
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
        _require_square_system_admission(self.matrix, self.rhs)
        return self


class RrefResult(StrictModel):
    """The unique reduced row echelon form bound to its source matrix.

    Retains the source matrix so validation replays the defining relation:
    the claimed form is the unique RREF over QQ of the retained source, the
    rank equals the pivot count, and the pivot and free columns partition the
    source column axis.  The rational matrix domain admits at least one row
    and column, so zero-row shapes are rejected by request admission rather
    than silently dropped.
    """

    matrix: RationalMatrix
    reduced_matrix: RationalMatrix
    rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    pivot_columns: tuple[int, ...] = Field(max_length=MAX_MATRIX_DIMENSION)
    free_columns: tuple[int, ...] = Field(max_length=MAX_MATRIX_DIMENSION)
    convention: Literal["UNIQUE_RREF_OVER_QQ"] = "UNIQUE_RREF_OVER_QQ"

    @model_validator(mode="after")
    def require_source_bound(self) -> Self:
        column_count = len(self.matrix.entries[0])
        if (
            len(self.reduced_matrix.entries) != len(self.matrix.entries)
            or len(self.reduced_matrix.entries[0]) != column_count
        ):
            raise _validation_error(
                "shape_mismatch", "reduced matrix must have the source shape"
            )
        if self.rank != len(self.pivot_columns):
            raise _validation_error(
                "shape_mismatch", "rank must equal the pivot column count"
            )
        if sorted((*self.pivot_columns, *self.free_columns)) != list(
            range(column_count)
        ):
            raise _validation_error(
                "shape_mismatch",
                "pivot and free columns must partition the source columns",
            )
        from jacobian.math.matrices import _conversions as conversions
        from jacobian.math.matrices._operations import _rref_replay

        expected_reduced, pivots = _rref_replay(self.matrix)
        if tuple(int(pivot) for pivot in pivots) != self.pivot_columns or (
            conversions.rational_matrix_to_sympy(self.reduced_matrix)
            != expected_reduced
        ):
            raise _validation_error(
                "budget_exceeded",
                "reduced matrix must be the unique RREF of the source",
            )
        return self


class MatrixDeterminantResult(StrictModel):
    """One exact determinant, returned inline for ordinary composition."""

    determinant: CanonicalRational
    method: Literal["FRACTION_FREE_BAREISS"] = "FRACTION_FREE_BAREISS"


class MatrixRankResult(StrictModel):
    """One exact rank with the canonical RREF pivot columns, bound to its source."""

    matrix: RationalMatrix
    rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    pivot_columns: tuple[int, ...] = Field(max_length=MAX_MATRIX_DIMENSION)
    method: Literal["EXACT_RATIONAL_ROW_REDUCTION"] = "EXACT_RATIONAL_ROW_REDUCTION"

    @model_validator(mode="after")
    def require_source_bound(self) -> Self:
        if self.rank != len(self.pivot_columns):
            raise _validation_error(
                "budget_exceeded", "rank must equal the pivot column count"
            )
        from jacobian.math.matrices._operations import _rank_replay

        expected_rank, pivots = _rank_replay(self.matrix)
        if expected_rank != self.rank or tuple(int(p) for p in pivots) != (
            self.pivot_columns
        ):
            raise _validation_error(
                "budget_exceeded",
                "rank and pivot columns must replay against the source",
            )
        return self


class NullspaceResult(StrictModel):
    """The RREF fundamental nullspace basis bound to its source matrix.

    Retains the source matrix so validation replays the defining relations:
    every basis vector satisfies ``A v = 0`` exactly, each vector carries a
    one in its own free column and zeros in the other free columns (which
    also establishes independence), and the claimed rank equals the exact
    rank of the retained source.
    """

    matrix: RationalMatrix
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
        if self.ambient_dimension != len(self.matrix.entries[0]):
            raise _validation_error(
                "shape_mismatch", "ambient dimension must equal the source column count"
            )
        if self.rank + self.nullity != self.ambient_dimension:
            raise _validation_error(
                "shape_mismatch", "rank plus nullity must equal the ambient dimension"
            )
        if len(self.basis_vectors) != self.nullity:
            raise _validation_error(
                "shape_mismatch", "basis vector count must equal nullity"
            )
        if any(len(vector) != self.ambient_dimension for vector in self.basis_vectors):
            raise _validation_error(
                "shape_mismatch", "each basis vector must have the ambient dimension"
            )
        if len(self.free_columns) != self.nullity or self.free_columns != tuple(
            sorted(self.free_columns)
        ):
            raise _validation_error(
                "shape_mismatch",
                "free column count must equal nullity in ascending order",
            )

        entries = [
            [value.as_fraction() for value in row] for row in self.matrix.entries
        ]
        for index, vector in enumerate(self.basis_vectors):
            components = [value.as_fraction() for value in vector]
            for row in entries:
                if sum(a * b for a, b in zip(row, components, strict=True)) != 0:
                    raise _validation_error(
                        "shape_mismatch",
                        f"basis vector {index} does not satisfy A v = 0 exactly",
                    )
            own = self.free_columns[index]
            if components[own] != 1 or any(
                components[other] != 0 for other in self.free_columns if other != own
            ):
                raise _validation_error(
                    "shape_mismatch",
                    f"basis vector {index} is not the fundamental basis vector "
                    "of its free column",
                )

        from jacobian.math.matrices._operations import _rank_replay

        expected_rank, _pivots = _rank_replay(self.matrix)
        if expected_rank != self.rank:
            raise _validation_error(
                "shape_mismatch", "claimed rank must equal the exact rank of the source"
            )
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
            raise _validation_error(
                "shape_mismatch", "dense coefficient count must be degree plus one"
            )
        if self.coefficients_descending[0] != CanonicalRational(num="1", den="1"):
            raise _validation_error(
                "budget_exceeded", "characteristic polynomial must be monic"
            )
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
            raise _validation_error(
                "budget_exceeded", "product row count must equal left_rows"
            )
        if len(self.product.entries[0]) != self.right_columns:
            raise _validation_error(
                "budget_exceeded", "product column count must equal right_columns"
            )
        return self


class RationalLinearSolveResult(StrictModel):
    """One square-system classification over QQ, bound to its source system.

    Retains the coefficient matrix and right-hand side so validation replays
    the classification with the same exact kernel: a unique solution carries
    one coordinate per column, satisfies ``A x = b`` exactly, and requires
    the retained coefficient matrix to be nonsingular; an inconsistent
    outcome requires ``rank(A) < rank([A | b])`` on the retained system; a
    non-unique outcome requires a consistent, rank-deficient retained system.
    Validation first reapplies the request's squareness and scalar envelope
    to the retained source, so deserializing a relayed payload can never push
    replay arithmetic outside this operation's admitted work envelope.
    The rational matrix domain admits at least one row and column, so
    zero-row shapes are rejected by request admission rather than silently
    dropped.
    """

    matrix: RationalMatrix
    rhs: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_MATRIX_DIMENSION,
    )
    outcome: Literal["UNIQUE", "INCONSISTENT", "NON_UNIQUE"]
    solution: tuple[CanonicalRational, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_MATRIX_DIMENSION,
    )
    convention: Literal["LINEAR_SYSTEM_CLASSIFICATION_OVER_QQ"] = (
        "LINEAR_SYSTEM_CLASSIFICATION_OVER_QQ"
    )

    @model_validator(mode="after")
    def require_source_bound_classification(self) -> Self:
        solution = self.solution
        if self.outcome == "UNIQUE":
            if solution is None:
                raise _validation_error(
                    "shape_mismatch",
                    "a unique solution must populate the solution field",
                )
        elif solution is not None:
            raise _validation_error(
                "shape_mismatch",
                "a non-unique or inconsistent result must not populate the solution field",
            )
        # Deserialized source components pass the canonical rational domain
        # but not this operation's work envelope, so reapply request
        # admission before any exact replay arithmetic runs.
        _require_square_system_admission(self.matrix, self.rhs)
        if len(self.rhs) != len(self.matrix.entries):
            raise _validation_error(
                "shape_mismatch",
                "right-hand side length must equal the source row count",
            )

        from jacobian.math.matrices._operations import _system_rank_replay

        coefficient_rank, augmented_rank = _system_rank_replay(self.matrix, self.rhs)
        columns = len(self.matrix.entries[0])
        if solution is not None:
            components = [value.as_fraction() for value in solution]
            if len(components) != columns:
                raise _validation_error(
                    "budget_exceeded",
                    "solution length must equal the source column count",
                )
            for row, bound in zip(self.matrix.entries, self.rhs, strict=True):
                residual = sum(
                    coefficient.as_fraction() * component
                    for coefficient, component in zip(row, components, strict=True)
                )
                if residual != bound.as_fraction():
                    raise _validation_error(
                        "shape_mismatch", "solution does not satisfy A x = b exactly"
                    )
            if coefficient_rank != columns:
                raise _validation_error(
                    "shape_mismatch",
                    "a unique outcome requires a nonsingular source coefficient matrix",
                )
        elif self.outcome == "INCONSISTENT":
            if coefficient_rank >= augmented_rank:
                raise _validation_error(
                    "shape_mismatch",
                    "an inconsistent outcome requires rank(A) < rank([A | b]) "
                    "on the source system",
                )
        else:
            if coefficient_rank == columns or coefficient_rank != augmented_rank:
                raise _validation_error(
                    "invariant_mismatch",
                    "a non-unique outcome requires a consistent, rank-deficient "
                    "source system",
                )
        return self


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
        _require_computation_dimensions(self.left.entries)
        _require_computation_dimensions(self.right.entries)
        if len(self.left.entries) * len(self.right.entries) > (
            MAX_RATIONAL_MATRIX_ORDER
        ) or len(self.left.entries[0]) * len(self.right.entries[0]) > (
            MAX_RATIONAL_MATRIX_ORDER
        ):
            raise _validation_error(
                "budget_exceeded",
                "kronecker products must fit within "
                f"{MAX_RATIONAL_MATRIX_ORDER} rows and columns",
            )
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
            raise _validation_error(
                "shape_mismatch",
                "Kronecker product row count must equal left_rows * right_rows",
            )
        if len(self.product.entries[0]) != self.left_columns * self.right_columns:
            raise _validation_error(
                "shape_mismatch",
                "Kronecker product column count must equal left_columns * right_columns",
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
            raise _validation_error(
                "budget_exceeded",
                "composite matrix row count must equal traced_dimension * kept_dimension",
            )
        if len(self.matrix.entries[0]) != total:
            raise _validation_error(
                "budget_exceeded",
                "composite matrix must be square: traced_dimension * kept_dimension",
            )
        _require_computation_dimensions(self.matrix.entries)
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
            raise _validation_error(
                "shape_mismatch", "reduced matrix row count must equal kept_dimension"
            )
        if len(self.reduced_matrix.entries[0]) != self.kept_dimension:
            raise _validation_error(
                "shape_mismatch",
                "reduced matrix must be square of order kept_dimension",
            )
        return self


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"matrix.{reason}", message)
