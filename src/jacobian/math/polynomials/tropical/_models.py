"""Typed wire contracts for tropical operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.polynomials.tropical.values import (
    TropicalMatrix,
    TropicalPolynomial,
    TropicalScalar,
    TropicalSemiring,
    TropicalVector,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"tropical.{reason}", message)


class ScalarAddRequest(StrictModel):
    semiring: TropicalSemiring
    left: TropicalScalar
    right: TropicalScalar

    @model_validator(mode="after")
    def require_shared_semiring(self) -> Self:
        if self.left.semiring != self.semiring or self.right.semiring != self.semiring:
            raise _validation_error(
                "semiring_mismatch", "operands must carry the request semiring"
            )
        return self


AddBranch = Literal["LEFT", "RIGHT", "TIE"]
InfinityCase = Literal["NONE", "LEFT_INFINITE", "RIGHT_INFINITE", "BOTH_INFINITE"]


class ScalarAddResult(StrictModel):
    semiring: TropicalSemiring
    left: TropicalScalar
    right: TropicalScalar
    result: TropicalScalar
    branch: AddBranch
    infinity_case: InfinityCase

    @model_validator(mode="after")
    def require_branch_shape(self) -> Self:
        if self.result.semiring != self.semiring:
            raise _validation_error(
                "result_semiring_mismatch", "result must carry the request semiring"
            )
        expected = (
            "BOTH_INFINITE"
            if self.left.kind != "FINITE" and self.right.kind != "FINITE"
            else "LEFT_INFINITE"
            if self.left.kind != "FINITE"
            else "RIGHT_INFINITE"
            if self.right.kind != "FINITE"
            else "NONE"
        )
        if self.infinity_case != expected:
            raise _validation_error(
                "infinity_case_mismatch", "infinity_case must match operand variants"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: ScalarAddRequest,
        *,
        result: TropicalScalar,
        branch: AddBranch,
        infinity_case: InfinityCase,
    ) -> Self:
        return cls.model_construct(
            semiring=request.semiring,
            left=request.left,
            right=request.right,
            result=result,
            branch=branch,
            infinity_case=infinity_case,
        )


class ScalarBinaryRequest(StrictModel):
    semiring: TropicalSemiring
    left: TropicalScalar
    right: TropicalScalar

    @model_validator(mode="after")
    def require_shared(self) -> Self:
        if self.left.semiring != self.semiring or self.right.semiring != self.semiring:
            raise _validation_error(
                "semiring_mismatch", "operands must carry the request semiring"
            )
        return self


class ScalarPowerRequest(StrictModel):
    scalar: TropicalScalar
    exponent: int = Field(ge=0, le=256)


class ScalarResult(StrictModel):
    result: TropicalScalar

    @classmethod
    def _from_kernel(cls, result: TropicalScalar) -> Self:
        return cls.model_construct(result=result)


class VectorBinaryRequest(StrictModel):
    left: TropicalVector
    right: TropicalVector

    @model_validator(mode="after")
    def require_axis(self) -> Self:
        if (
            self.left.semiring != self.right.semiring
            or self.left.axis != self.right.axis
        ):
            raise _validation_error(
                "vector_mismatch", "vectors must have identical semiring and axis"
            )
        return self


class VectorScaleRequest(StrictModel):
    scalar: TropicalScalar
    vector: TropicalVector

    @model_validator(mode="after")
    def require_semiring(self) -> Self:
        if self.scalar.semiring != self.vector.semiring:
            raise _validation_error(
                "semiring_mismatch", "scalar and vector must share semiring"
            )
        return self


class VectorResult(StrictModel):
    result: TropicalVector

    @classmethod
    def _from_kernel(cls, result: TropicalVector) -> Self:
        return cls.model_construct(result=result)


class PolynomialBinaryRequest(StrictModel):
    left: TropicalPolynomial
    right: TropicalPolynomial

    @model_validator(mode="after")
    def require_parent(self) -> Self:
        if (
            self.left.semiring != self.right.semiring
            or self.left.variables != self.right.variables
        ):
            raise _validation_error(
                "polynomial_mismatch",
                "polynomials must share semiring and variable axis",
            )
        return self


class PolynomialEvaluateRequest(StrictModel):
    polynomial: TropicalPolynomial
    point: TropicalVector

    @model_validator(mode="after")
    def require_axis(self) -> Self:
        if (
            self.polynomial.semiring != self.point.semiring
            or self.polynomial.variables != self.point.axis
        ):
            raise _validation_error(
                "polynomial_point_mismatch",
                "point must share polynomial semiring and variable axis",
            )
        return self


class PolynomialResult(StrictModel):
    result: TropicalPolynomial

    @classmethod
    def _from_kernel(cls, result: TropicalPolynomial) -> Self:
        return cls.model_construct(result=result)


class PolynomialEvaluateResult(StrictModel):
    polynomial: TropicalPolynomial
    point: TropicalVector
    value: TropicalScalar
    active_exponents: tuple[tuple[int, ...], ...]

    @classmethod
    def _from_kernel(
        cls,
        *,
        polynomial: TropicalPolynomial,
        point: TropicalVector,
        value: TropicalScalar,
        active_exponents: tuple[tuple[int, ...], ...],
    ) -> Self:
        return cls.model_construct(
            polynomial=polynomial,
            point=point,
            value=value,
            active_exponents=active_exponents,
        )


class MatrixMultiplyRequest(StrictModel):
    left: TropicalMatrix
    right: TropicalMatrix

    @model_validator(mode="after")
    def require_axes(self) -> Self:
        if (
            self.left.semiring != self.right.semiring
            or self.left.column_axis != self.right.row_axis
        ):
            raise _validation_error(
                "matrix_mismatch", "matrix inner axes and semiring must match"
            )
        return self


class MatrixPowerRequest(StrictModel):
    matrix: TropicalMatrix
    exponent: int = Field(ge=0, le=64)

    @model_validator(mode="after")
    def require_square(self) -> Self:
        if self.matrix.row_axis != self.matrix.column_axis:
            raise _validation_error(
                "matrix_square", "matrix power requires equal row and column axes"
            )
        return self


class MatrixFinitePowerSumRequest(StrictModel):
    matrix: TropicalMatrix
    max_power: int = Field(ge=0, le=32)

    @model_validator(mode="after")
    def require_square(self) -> Self:
        if self.matrix.row_axis != self.matrix.column_axis:
            raise _validation_error(
                "matrix_square", "finite power sums require a square matrix"
            )
        return self


class MatrixAssignmentRequest(StrictModel):
    matrix: TropicalMatrix

    @model_validator(mode="after")
    def require_square(self) -> Self:
        if self.matrix.row_axis != self.matrix.column_axis:
            raise _validation_error(
                "matrix_square", "assignment requires a square matrix"
            )
        return self


class MatrixResult(StrictModel):
    result: TropicalMatrix

    @classmethod
    def _from_kernel(cls, result: TropicalMatrix) -> Self:
        return cls.model_construct(result=result)


class FinitePowerSumResult(StrictModel):
    """A finite power sum bound to the matrix and cutoff that define it."""

    source_matrix: TropicalMatrix
    max_power: int = Field(ge=0, le=32)
    matrix: TropicalMatrix
    winning_lengths: tuple[tuple[tuple[int, ...], ...], ...]

    @model_validator(mode="after")
    def require_source_bound_shape(self) -> Self:
        if (
            self.source_matrix.row_axis != self.source_matrix.column_axis
            or self.matrix.row_axis != self.source_matrix.row_axis
            or self.matrix.column_axis != self.source_matrix.column_axis
            or len(self.winning_lengths) != len(self.source_matrix.row_axis)
            or any(
                len(row) != len(self.source_matrix.column_axis)
                for row in self.winning_lengths
            )
        ):
            raise _validation_error(
                "finite_power_sum_shape",
                "power-sum result must retain the source square axes",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_matrix: TropicalMatrix,
        max_power: int,
        matrix: TropicalMatrix,
        winning_lengths: tuple[tuple[tuple[int, ...], ...], ...],
    ) -> Self:
        return cls.model_construct(
            source_matrix=source_matrix,
            max_power=max_power,
            matrix=matrix,
            winning_lengths=winning_lengths,
        )


class AssignmentResult(StrictModel):
    matrix: TropicalMatrix
    value: TropicalScalar
    permutations: tuple[tuple[int, ...], ...]

    @classmethod
    def _from_kernel(
        cls,
        *,
        matrix: TropicalMatrix,
        value: TropicalScalar,
        permutations: tuple[tuple[int, ...], ...],
    ) -> Self:
        return cls.model_construct(
            matrix=matrix, value=value, permutations=permutations
        )


__all__ = [
    "AddBranch",
    "AssignmentResult",
    "FinitePowerSumResult",
    "InfinityCase",
    "MatrixAssignmentRequest",
    "MatrixFinitePowerSumRequest",
    "MatrixMultiplyRequest",
    "MatrixPowerRequest",
    "MatrixResult",
    "PolynomialBinaryRequest",
    "PolynomialEvaluateRequest",
    "PolynomialEvaluateResult",
    "PolynomialResult",
    "ScalarAddRequest",
    "ScalarAddResult",
    "ScalarBinaryRequest",
    "ScalarPowerRequest",
    "ScalarResult",
    "VectorBinaryRequest",
    "VectorResult",
    "VectorScaleRequest",
]
