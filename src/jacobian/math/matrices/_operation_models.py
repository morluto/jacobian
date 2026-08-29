"""Bounded exact matrix-operation contracts."""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.matrices.values import (
    MAX_EXACT_LINEAR_MATRIX_AXIS,
    MAX_MATRIX_DIMENSION,
    MAX_MATRIX_SCALAR_DIGITS,
    IntegerMatrix,
    RationalMatrix,
    require_matrix_scalar_digits,
)

MAX_INPUT_SCALAR_DIGITS = 256
MAX_DETERMINANT_MATRIX_DIMENSION = 128
MAX_DETERMINANT_SCALAR_WORK = 500_000_000
MAX_CHARACTERISTIC_POLYNOMIAL_ORDER = 128
MAX_MATRIX_PRODUCT_AXIS = 128
MAX_MATRIX_PRODUCT_MULTIPLY_ADDS = 2_500_000
MAX_MATRIX_PRODUCT_OUTPUT_DIGIT_WORK = 3_000_000
MAX_PERMANENT_RYSER_SUBSETS = 4_096
MAX_PERMANENT_MATRIX_ORDER = MAX_PERMANENT_RYSER_SUBSETS.bit_length() - 1
# The canonical dense rational matrix carries determinant inputs through
# order 128, but Kronecker admission was established only for product axes
# through order 50. Pin each admitted output axis to that envelope.
MAX_KRONECKER_PRODUCT_AXIS = 50
MAX_EXACT_LINEAR_MATRIX_WORK = 100_000_000


def _require_raw_scalar_digits(value: object, *, label: str) -> None:
    """Reject an over-budget scalar before nested canonical parsing."""

    components: tuple[object, ...]
    if isinstance(value, dict):
        components = (value.get("num"), value.get("den"))
    elif isinstance(value, CanonicalRational):
        components = (value.num, value.den)
    else:
        components = (value,)
    for component in components:
        if isinstance(component, (str, int)) and len(str(component).lstrip("-")) > (
            MAX_INPUT_SCALAR_DIGITS
        ):
            raise _validation_error(
                "budget_exceeded",
                f"{label} scalars are limited to {MAX_INPUT_SCALAR_DIGITS} decimal digits",
            )


def _require_raw_matrix(value: object, *, label: str, maximum_axis: int) -> None:
    """Bound raw matrix containers before converting JSON arrays to tuples."""

    if isinstance(value, dict):
        unexpected = set(value).difference({"domain", "entries"})
        if unexpected:
            raise _validation_error(
                "shape_mismatch", f"{label} contains unknown fields"
            )
        entries = value.get("entries")
    else:
        entries = getattr(value, "entries", None)
    if not isinstance(entries, (list, tuple)):
        return
    if len(entries) > maximum_axis:
        raise _validation_error(
            "budget_exceeded",
            f"{label} dimensions are limited to {maximum_axis} rows and columns",
        )
    for row in entries:
        if not isinstance(row, (list, tuple)):
            continue
        if len(row) > maximum_axis:
            raise _validation_error(
                "budget_exceeded",
                f"{label} dimensions are limited to {maximum_axis} rows and columns",
            )
        for scalar in row:
            if isinstance(scalar, dict) and set(scalar).difference({"num", "den"}):
                raise _validation_error(
                    "shape_mismatch", f"{label} rational scalar contains unknown fields"
                )
            _require_raw_scalar_digits(scalar, label=label)


class _MatrixRequest(StrictModel):
    """Raw transport preflight shared by bounded base-matrix requests."""

    _raw_matrix_axis_limit: ClassVar[int] = MAX_MATRIX_DIMENSION

    @model_validator(mode="before")
    @classmethod
    def require_raw_input_envelope(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if set(data).difference(cls.model_fields):
            raise _validation_error(
                "shape_mismatch", "matrix request contains unknown fields"
            )
        for name in ("matrix", "left", "right"):
            if name in data:
                _require_raw_matrix(
                    data[name],
                    label="matrix input",
                    maximum_axis=cls._raw_matrix_axis_limit,
                )
        rhs = data.get("rhs")
        if isinstance(rhs, (list, tuple)):
            if len(rhs) > MAX_MATRIX_DIMENSION:
                raise _validation_error(
                    "budget_exceeded",
                    f"right-hand side has at most {MAX_MATRIX_DIMENSION} entries",
                )
            for scalar in rhs:
                _require_raw_scalar_digits(scalar, label="right-hand side")
        return canonicalize_json_containers(data)


def _require_computation_dimensions(
    entries: tuple[tuple[CanonicalRational, ...], ...],
) -> None:
    if (
        len(entries) > MAX_EXACT_LINEAR_MATRIX_AXIS
        or len(entries[0]) > MAX_EXACT_LINEAR_MATRIX_AXIS
    ):
        raise _validation_error(
            "budget_exceeded",
            "matrix computation dimensions are limited to "
            f"{MAX_EXACT_LINEAR_MATRIX_AXIS} rows and columns",
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


class RationalMatrixRequest(_MatrixRequest):
    matrix: RationalMatrix
    _raw_matrix_axis_limit: ClassVar[int] = MAX_EXACT_LINEAR_MATRIX_AXIS

    @model_validator(mode="after")
    def require_computation_dimensions(self) -> Self:
        _require_computation_dimensions(self.matrix.entries)
        return self


class RationalMatrixProductRequest(_MatrixRequest):
    """Two compatible bounded matrices over the exact rational domain."""

    left: RationalMatrix
    right: RationalMatrix
    _raw_matrix_axis_limit: ClassVar[int] = MAX_MATRIX_PRODUCT_AXIS


class SquareRationalMatrixRequest(_MatrixRequest):
    matrix: RationalMatrix


class CharacteristicPolynomialRequest(_MatrixRequest):
    """One rational matrix for a complete exact characteristic polynomial."""

    matrix: RationalMatrix
    _raw_matrix_axis_limit: ClassVar[int] = MAX_CHARACTERISTIC_POLYNOMIAL_ORDER


class MatrixPermanentRequest(_MatrixRequest):
    """One square matrix charged by the exact Ryser subset enumeration."""

    matrix: RationalMatrix
    _raw_matrix_axis_limit: ClassVar[int] = MAX_PERMANENT_MATRIX_ORDER


class MatrixDeterminantRequest(_MatrixRequest):
    """One square rational matrix of order at most 128."""

    matrix: RationalMatrix
    _raw_matrix_axis_limit: ClassVar[int] = MAX_DETERMINANT_MATRIX_DIMENSION


class MatrixRankRequest(_MatrixRequest):
    """One bounded rectangular matrix whose exact rank is requested."""

    matrix: RationalMatrix
    _raw_matrix_axis_limit: ClassVar[int] = MAX_EXACT_LINEAR_MATRIX_AXIS

    @model_validator(mode="after")
    def require_computation_dimensions(self) -> Self:
        _require_computation_dimensions(self.matrix.entries)
        return self


class IntegerMatrixRequest(_MatrixRequest):
    matrix: IntegerMatrix
    _raw_matrix_axis_limit: ClassVar[int] = MAX_EXACT_LINEAR_MATRIX_AXIS


class NonsingularIntegerMatrixRequest(_MatrixRequest):
    """One bounded square integer matrix for the exact inverse kernel."""

    matrix: IntegerMatrix


class SquareIntegerMatrixRequest(_MatrixRequest):
    matrix: IntegerMatrix


class RationalLinearSolveRequest(_MatrixRequest):
    matrix: RationalMatrix
    rhs: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_MATRIX_DIMENSION,
    )


class RrefResult(StrictModel):
    """A structurally bounded RREF outcome bound to its source matrix.

    Kernel-produced values use :meth:`_from_kernel`; parsing this public wire
    shape performs structural validation and never executes a backend.
    """

    matrix: RationalMatrix
    reduced_matrix: RationalMatrix
    rank: int = Field(ge=0, le=MAX_EXACT_LINEAR_MATRIX_AXIS)
    pivot_columns: tuple[int, ...] = Field(max_length=MAX_EXACT_LINEAR_MATRIX_AXIS)
    free_columns: tuple[int, ...] = Field(max_length=MAX_EXACT_LINEAR_MATRIX_AXIS)
    convention: Literal["UNIQUE_RREF_OVER_QQ"] = "UNIQUE_RREF_OVER_QQ"

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)

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
        if self.pivot_columns != tuple(sorted(set(self.pivot_columns))) or any(
            not 0 <= column < column_count for column in self.pivot_columns
        ):
            raise _validation_error(
                "shape_mismatch",
                "pivot columns must be distinct source columns in order",
            )
        return self


class MatrixDeterminantResult(StrictModel):
    """One exact determinant, returned inline for ordinary composition."""

    determinant: CanonicalRational


class MatrixRankResult(StrictModel):
    """One structurally bounded rank outcome bound to its source matrix."""

    matrix: RationalMatrix
    rank: int = Field(ge=0, le=MAX_EXACT_LINEAR_MATRIX_AXIS)
    pivot_columns: tuple[int, ...] = Field(max_length=MAX_EXACT_LINEAR_MATRIX_AXIS)

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)

    @model_validator(mode="after")
    def require_source_bound(self) -> Self:
        if self.rank != len(self.pivot_columns):
            raise _validation_error(
                "budget_exceeded", "rank must equal the pivot column count"
            )
        column_count = len(self.matrix.entries[0])
        if self.pivot_columns != tuple(sorted(set(self.pivot_columns))) or any(
            not 0 <= column < column_count for column in self.pivot_columns
        ):
            raise _validation_error(
                "shape_mismatch",
                "pivot columns must be distinct source columns in order",
            )
        return self


class NullspaceResult(StrictModel):
    """A structurally bounded fundamental-nullspace outcome.

    Exact kernel output uses :meth:`_from_kernel`; ordinary parsing remains
    structural and does not establish an independently supplied claim.
    """

    matrix: RationalMatrix
    ambient_dimension: int = Field(ge=1, le=MAX_EXACT_LINEAR_MATRIX_AXIS)
    rank: int = Field(ge=0, le=MAX_EXACT_LINEAR_MATRIX_AXIS)
    nullity: int = Field(ge=0, le=MAX_EXACT_LINEAR_MATRIX_AXIS)
    basis_vectors: tuple[tuple[CanonicalRational, ...], ...] = Field(
        max_length=MAX_EXACT_LINEAR_MATRIX_AXIS
    )
    free_columns: tuple[int, ...] = Field(max_length=MAX_EXACT_LINEAR_MATRIX_AXIS)
    convention: Literal["RREF_FUNDAMENTAL_BASIS"] = "RREF_FUNDAMENTAL_BASIS"

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)

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

        if self.free_columns != tuple(sorted(set(self.free_columns))) or any(
            not 0 <= column < self.ambient_dimension for column in self.free_columns
        ):
            raise _validation_error(
                "shape_mismatch",
                "free columns must be distinct ambient columns in order",
            )
        return self


class CharacteristicPolynomialResult(StrictModel):
    variable: Literal["lambda"] = "lambda"
    degree: int = Field(ge=1, le=MAX_CHARACTERISTIC_POLYNOMIAL_ORDER)
    coefficients_descending: tuple[CanonicalRational, ...] = Field(
        min_length=2,
        max_length=MAX_CHARACTERISTIC_POLYNOMIAL_ORDER + 1,
    )
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
    left_rows: int = Field(ge=1, le=MAX_MATRIX_PRODUCT_AXIS)
    inner_dimension: int = Field(ge=1, le=MAX_MATRIX_PRODUCT_AXIS)
    right_columns: int = Field(ge=1, le=MAX_MATRIX_PRODUCT_AXIS)
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
    """A structurally bounded square-system outcome over QQ.

    Kernel output uses :meth:`_from_kernel`; ordinary parsing remains
    structural and does not establish an independently supplied claim.
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

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)

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
        if len(self.rhs) != len(self.matrix.entries):
            raise _validation_error(
                "shape_mismatch",
                "right-hand side length must equal the source row count",
            )

        columns = len(self.matrix.entries[0])
        if solution is not None and len(solution) != columns:
            raise _validation_error(
                "budget_exceeded",
                "solution length must equal the source column count",
            )
        return self


class MatrixAdjugateResult(StrictModel):
    adjugate: IntegerMatrix
    convention: Literal["CLASSICAL_ADJUGATE"] = "CLASSICAL_ADJUGATE"


class MatrixPermanentResult(StrictModel):
    """One exact matrix permanent."""

    permanent: CanonicalRational


class MatrixKroneckerProductRequest(_MatrixRequest):
    """Two bounded matrices for an exact Kronecker product over QQ."""

    left: RationalMatrix
    right: RationalMatrix


class MatrixKroneckerProductResult(StrictModel):
    """The Kronecker product of two bounded matrices over QQ."""

    product: RationalMatrix
    left_rows: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    left_columns: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    right_rows: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    right_columns: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)

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


class MatrixPartialTraceRequest(_MatrixRequest):
    """A composite matrix (Kronecker product A (x) B) and the subsystem dimensions."""

    matrix: RationalMatrix
    traced_dimension: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    kept_dimension: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)


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
