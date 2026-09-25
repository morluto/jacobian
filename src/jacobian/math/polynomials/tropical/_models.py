"""Typed wire contracts for tropical operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.polynomials.tropical.values import (
    TropicalMatrix,
    TropicalNewtonPolygonProfile,
    TropicalPolynomial,
    TropicalPolynomialTerm,
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


class VectorProjectivizeRequest(StrictModel):
    """A tropical vector modulo common finite tropical translation."""

    vector: TropicalVector


class VectorProjectivizeResult(StrictModel):
    source: TropicalVector
    kind: Literal["PROJECTIVIZED", "NO_PROJECTIVE_CLASS"]
    representative: TropicalVector | None = None
    translation: TropicalScalar | None = None

    @model_validator(mode="after")
    def require_branch(self) -> Self:
        if self.kind == "PROJECTIVIZED":
            if self.representative is None or self.translation is None:
                raise _validation_error(
                    "projective_shape", "normalized result needs its translation"
                )
            if not any(entry.kind == "FINITE" for entry in self.source.entries):
                raise _validation_error(
                    "projective_source",
                    "a projectivized vector needs a finite coordinate",
                )
            if (
                self.representative.semiring != self.source.semiring
                or self.representative.axis != self.source.axis
                or self.translation.semiring != self.source.semiring
                or self.translation.kind != "FINITE"
            ):
                raise _validation_error(
                    "projective_shape",
                    "normalized result must retain source domain and axes",
                )
            original = self.source.entries
            normalized = self.representative.entries
            if len(original) != len(normalized) or any(
                (left.kind == "FINITE") != (right.kind == "FINITE")
                for left, right in zip(original, normalized, strict=True)
            ):
                raise _validation_error(
                    "projective_support", "normalization must preserve infinity support"
                )
        elif (
            self.representative is not None
            or self.translation is not None
            or any(entry.kind == "FINITE" for entry in self.source.entries)
        ):
            raise _validation_error(
                "projective_shape",
                "all-infinity vector has no projective representative",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: TropicalVector,
        kind: Literal["PROJECTIVIZED", "NO_PROJECTIVE_CLASS"],
        representative: TropicalVector | None = None,
        translation: TropicalScalar | None = None,
    ) -> Self:
        return cls.model_construct(
            source=source,
            kind=kind,
            representative=representative,
            translation=translation,
        )


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


class PolynomialPowerRequest(StrictModel):
    """Raise one sparse formal tropical polynomial to a bounded power."""

    polynomial: TropicalPolynomial
    exponent: int = Field(ge=0, le=16)


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


class PolynomialActiveTermsRequest(StrictModel):
    """Request all source monomials attaining the semiring extremum."""

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


class TropicalActiveTerm(StrictModel):
    """One indexed source term and its exact value at the evaluation point."""

    index: StrictInt = Field(ge=0, le=511)
    term: TropicalPolynomialTerm
    value: TropicalScalar


class PolynomialActiveTermsResult(StrictModel):
    """Source-bound witness of every active monomial at one exact point."""

    polynomial: TropicalPolynomial
    point: TropicalVector
    value: TropicalScalar
    active_terms: tuple[TropicalActiveTerm, ...] = Field(max_length=512)

    @model_validator(mode="after")
    def require_source_indices(self) -> Self:
        if (
            self.polynomial.semiring != self.point.semiring
            or self.polynomial.variables != self.point.axis
            or self.value.semiring != self.polynomial.semiring
        ):
            raise _validation_error(
                "active_terms_parent",
                "source, point, and value must share their exact tropical parent",
            )
        indices = tuple(item.index for item in self.active_terms)
        if indices != tuple(sorted(set(indices))) or any(
            index >= len(self.polynomial.terms)
            or self.polynomial.terms[index] != item.term
            or item.value.semiring != self.polynomial.semiring
            or item.value != self.value
            for index, item in zip(indices, self.active_terms, strict=True)
        ):
            raise _validation_error(
                "active_terms_source_indices",
                "active witnesses must preserve canonical source-term indices",
            )
        if self.active_terms and self.value.kind != "FINITE":
            raise _validation_error(
                "active_terms_value", "a nonempty active set has a finite value"
            )
        if not self.active_terms and self.value.kind != (
            "POSITIVE_INFINITY"
            if self.polynomial.semiring.convention == "MIN_PLUS"
            else "NEGATIVE_INFINITY"
        ):
            raise _validation_error(
                "active_terms_value",
                "an empty active set must carry the semiring additive identity",
            )
        return self


class UnivariateRootsRequest(StrictModel):
    polynomial: TropicalPolynomial


class UnivariateSplitFormRequest(StrictModel):
    """Return a univariate polynomial with the same function in split form."""

    polynomial: TropicalPolynomial

    @model_validator(mode="after")
    def require_univariate(self) -> Self:
        if len(self.polynomial.variables) != 1:
            raise _validation_error(
                "split_form_univariate",
                "split form requires exactly one polynomial variable",
            )
        return self


class UnivariateNewtonPolygonRequest(StrictModel):
    """Request an exact coefficient-lifted hull for a univariate polynomial."""

    polynomial: TropicalPolynomial

    @model_validator(mode="after")
    def require_univariate(self) -> Self:
        if len(self.polynomial.variables) != 1:
            raise _validation_error(
                "newton_polygon_univariate",
                "Newton polygon profile requires one variable",
            )
        return self


class UnivariateNewtonPolygonResult(StrictModel):
    profile: TropicalNewtonPolygonProfile

    @classmethod
    def _from_kernel(cls, profile: TropicalNewtonPolygonProfile) -> Self:
        return cls.model_construct(profile=profile)


class BivariateRegularSubdivisionRequest(StrictModel):
    """Compute the bounded exact subdivision of one bivariate polynomial."""

    polynomial: TropicalPolynomial

    @model_validator(mode="after")
    def require_bivariate(self) -> Self:
        if len(self.polynomial.variables) != 2:
            raise _validation_error(
                "regular_subdivision_bivariate",
                "regular subdivision requires exactly two variables",
            )
        return self


class BivariateHypersurfaceRequest(StrictModel):
    """Compute the complete bounded exact corner complex of one polynomial."""

    polynomial: TropicalPolynomial

    @model_validator(mode="after")
    def require_bivariate(self) -> Self:
        if len(self.polynomial.variables) != 2:
            raise _validation_error(
                "hypersurface_bivariate",
                "the exact tropical hypersurface currently requires two variables",
            )
        return self


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
        # Assignment matches rows to columns. Their label sets are distinct
        # axes, so only cardinality (not label identity) is required.
        if len(self.matrix.row_axis) != len(self.matrix.column_axis):
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
    "BivariateHypersurfaceRequest",
    "BivariateRegularSubdivisionRequest",
    "FinitePowerSumResult",
    "InfinityCase",
    "MatrixAssignmentRequest",
    "MatrixFinitePowerSumRequest",
    "MatrixMultiplyRequest",
    "MatrixPowerRequest",
    "MatrixResult",
    "PolynomialActiveTermsRequest",
    "PolynomialActiveTermsResult",
    "PolynomialBinaryRequest",
    "PolynomialEvaluateRequest",
    "PolynomialEvaluateResult",
    "PolynomialPowerRequest",
    "PolynomialResult",
    "ScalarAddRequest",
    "ScalarAddResult",
    "ScalarBinaryRequest",
    "ScalarPowerRequest",
    "ScalarResult",
    "TropicalActiveTerm",
    "UnivariateNewtonPolygonRequest",
    "UnivariateNewtonPolygonResult",
    "UnivariateRootsRequest",
    "UnivariateSplitFormRequest",
    "VectorBinaryRequest",
    "VectorProjectivizeRequest",
    "VectorProjectivizeResult",
    "VectorResult",
    "VectorScaleRequest",
]
