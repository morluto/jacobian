"""Typed exact contracts for finite vector families and frames."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.number_theory.number_fields import GaussianRational
from jacobian.math.topology.frames.values import (
    MAX_COMPLEX_BASIS_COUNT,
    MAX_DIM,
    MAX_VECTOR_CELLS,
    ComplexFrame,
    VectorFamily,
)


class GramResult(VectorFamily):
    gram: IntegerMatrix

    @model_validator(mode="after")
    def require_source_shape(self) -> Self:
        if self.gram.row_count != len(self.vectors) or self.gram.column_count != len(
            self.vectors
        ):
            raise PydanticCustomError(
                "frames.gram_shape",
                "Gram matrix must align with the retained vector-family axes",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        vectors: tuple[tuple[int, ...], ...],
        dimension: int,
        gram: IntegerMatrix,
    ) -> Self:
        return cls.model_construct(
            vectors=vectors,
            gram=gram,
            dimension=dimension,
        )


class CoherenceResult(VectorFamily):
    coherence_squared: CanonicalRational
    maximizing_pair: tuple[int, int] | None

    @classmethod
    def _from_kernel(
        cls,
        *,
        vectors: tuple[tuple[int, ...], ...],
        dimension: int,
        coherence_squared: CanonicalRational,
        maximizing_pair: tuple[int, int] | None,
    ) -> Self:
        return cls.model_construct(
            vectors=vectors,
            dimension=dimension,
            coherence_squared=coherence_squared,
            maximizing_pair=maximizing_pair,
        )


class FramePotentialResult(VectorFamily):
    potential: ExactInteger

    @classmethod
    def _from_kernel(
        cls,
        *,
        vectors: tuple[tuple[int, ...], ...],
        dimension: int,
        potential: ExactInteger,
    ) -> Self:
        return cls.model_construct(
            vectors=vectors,
            dimension=dimension,
            potential=potential,
        )


class TightEquiangularProfileResult(VectorFamily):
    """Exact tightness and equiangularity predicates for one frame."""

    tight: bool
    tight_constant: ExactInteger | None
    equiangular: bool
    common_squared_inner_product: CanonicalRational | None

    @model_validator(mode="after")
    def require_profile_shape(self) -> Self:
        if self.tight != (self.tight_constant is not None):
            raise PydanticCustomError(
                "frames.profile_shape", "tightness must agree with its scalar constant"
            )
        if not self.equiangular and self.common_squared_inner_product is not None:
            raise PydanticCustomError(
                "frames.profile_shape",
                "non-equiangular frames cannot carry a common value",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        vectors: tuple[tuple[int, ...], ...],
        dimension: int,
        tight: bool,
        tight_constant: ExactInteger | None,
        equiangular: bool,
        common_squared_inner_product: CanonicalRational | None,
    ) -> Self:
        return cls.model_construct(
            vectors=vectors,
            dimension=dimension,
            tight=tight,
            tight_constant=tight_constant,
            equiangular=equiangular,
            common_squared_inner_product=common_squared_inner_product,
        )


class MutuallyUnbiasedBasesRequest(StrictModel):
    """Nonzero orthogonal representatives of bases in one complex space.

    MUB status uses normalized squared cross-overlaps, so representatives need
    not have unit coordinate norm.
    """

    dimension: int = Field(ge=1, le=MAX_DIM)
    bases: tuple[ComplexFrame, ...] = Field(
        min_length=1, max_length=MAX_COMPLEX_BASIS_COUNT
    )

    @model_validator(mode="after")
    def require_basis_axes(self) -> Self:
        if any(
            basis.dimension != self.dimension or len(basis.vectors) != self.dimension
            for basis in self.bases
        ):
            raise PydanticCustomError(
                "frames.mub_basis_shape",
                "every basis must have exactly dimension vectors of that dimension",
            )
        return self


class MutuallyUnbiasedBasesResult(MutuallyUnbiasedBasesRequest):
    basis_grams: tuple[tuple[tuple[GaussianRational, ...], ...], ...]
    cross_gram_squared: tuple[tuple[tuple[CanonicalRational, ...], ...], ...]
    is_mutually_unbiased: bool
    basis_pair_count: int

    @model_validator(mode="after")
    def require_profile_shape(self) -> Self:
        if self.basis_pair_count != len(self.bases) * (len(self.bases) - 1) // 2:
            raise PydanticCustomError(
                "frames.mub_profile_shape", "basis pair count is not canonical"
            )
        dimension = self.dimension
        if (
            len(self.basis_grams) != len(self.bases)
            or any(
                len(gram) != dimension or any(len(row) != dimension for row in gram)
                for gram in self.basis_grams
            )
            or len(self.cross_gram_squared) != self.basis_pair_count
            or any(
                len(gram) != dimension or any(len(row) != dimension for row in gram)
                for gram in self.cross_gram_squared
            )
        ):
            raise PydanticCustomError(
                "frames.mub_profile_axes", "MUB ledgers must retain basis axes"
            )
        expected_overlap = Fraction(1, dimension)
        observed = True
        for gram in self.basis_grams:
            for row_index, row in enumerate(gram):
                for column_index, entry in enumerate(row):
                    real, imaginary = entry.as_fractions()
                    if row_index == column_index:
                        if imaginary != 0 or real <= 0:
                            observed = False
                    elif real != 0 or imaginary != 0:
                        observed = False
        for gram in self.cross_gram_squared:
            for row in gram:
                for entry in row:
                    if entry.as_fraction() != expected_overlap:
                        observed = False
        if self.is_mutually_unbiased != observed:
            raise PydanticCustomError(
                "frames.mub_status",
                "MUB status must agree with the retained Gram and overlap ledgers",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        dimension: int,
        bases: tuple[ComplexFrame, ...],
        basis_grams: tuple[tuple[tuple[GaussianRational, ...], ...], ...],
        cross_gram_squared: tuple[tuple[tuple[CanonicalRational, ...], ...], ...],
        is_mutually_unbiased: bool,
    ) -> Self:
        return cls.model_construct(
            dimension=dimension,
            bases=bases,
            basis_grams=basis_grams,
            cross_gram_squared=cross_gram_squared,
            is_mutually_unbiased=is_mutually_unbiased,
            basis_pair_count=len(bases) * (len(bases) - 1) // 2,
        )


class ComplexFrameProfileRequest(StrictModel):
    frame: ComplexFrame


class ComplexFrameProfileResult(ComplexFrameProfileRequest):
    tight: bool
    equiangular: bool
    common_squared_overlap: CanonicalRational | None
    frame_operator: tuple[tuple[GaussianRational, ...], ...]
    tight_residual: tuple[tuple[GaussianRational, ...], ...]

    @model_validator(mode="after")
    def require_profile_shape(self) -> Self:
        if not self.equiangular and self.common_squared_overlap is not None:
            raise PydanticCustomError(
                "frames.complex_profile_shape",
                "non-equiangular profile cannot carry a common overlap",
            )
        d = self.frame.dimension
        if any(
            len(matrix) != d or any(len(row) != d for row in matrix)
            for matrix in (self.frame_operator, self.tight_residual)
        ):
            raise PydanticCustomError(
                "frames.complex_profile_axes",
                "frame operator ledgers must retain ambient axes",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        frame: ComplexFrame,
        *,
        tight: bool,
        equiangular: bool,
        common_squared_overlap: CanonicalRational | None,
        frame_operator: tuple[tuple[GaussianRational, ...], ...],
        tight_residual: tuple[tuple[GaussianRational, ...], ...],
    ) -> Self:
        return cls.model_construct(
            frame=frame,
            tight=tight,
            equiangular=equiangular,
            common_squared_overlap=common_squared_overlap,
            frame_operator=frame_operator,
            tight_residual=tight_residual,
        )


class SicProfileRequest(StrictModel):
    """Complex projective representatives checked through normalized projectors."""

    frame: ComplexFrame


class SicProfileResult(SicProfileRequest):
    """A scale-invariant SIC ledger and its normalized frame operator."""

    is_sic: bool
    cardinality_residual: ExactInteger = Field(
        description="The supplied line count minus dimension squared."
    )
    equiangular: bool = Field(
        description="Whether all observed off-diagonal normalized overlaps agree."
    )
    common_squared_overlap: CanonicalRational | None = Field(
        description="The common off-diagonal normalized overlap when equiangular."
    )
    common_squared_overlap_residual: CanonicalRational | None = Field(
        description="The common overlap minus the SIC target 1/(dimension+1)."
    )
    squared_overlaps: tuple[tuple[CanonicalRational, ...], ...]
    frame_operator: tuple[tuple[GaussianRational, ...], ...]
    tight_residual: tuple[tuple[GaussianRational, ...], ...]

    @model_validator(mode="after")
    def require_profile_shape(self) -> Self:
        expected_cardinality_residual = (
            len(self.frame.vectors) - self.frame.dimension * self.frame.dimension
        )
        if self.cardinality_residual != expected_cardinality_residual:
            raise PydanticCustomError(
                "frames.sic_profile_cardinality",
                "SIC cardinality residual must match the source family",
            )
        if (self.common_squared_overlap is None) != (
            self.common_squared_overlap_residual is None
        ):
            raise PydanticCustomError(
                "frames.sic_profile_shape",
                "common overlap and its residual must be present together",
            )
        if self.equiangular != (self.common_squared_overlap is not None):
            raise PydanticCustomError(
                "frames.sic_profile_shape",
                "equiangular status must agree with the common overlap",
            )
        d = self.frame.dimension
        n = len(self.frame.vectors)
        if (
            len(self.squared_overlaps) != n
            or any(len(row) != n for row in self.squared_overlaps)
            or any(
                len(matrix) != d or any(len(row) != d for row in matrix)
                for matrix in (self.frame_operator, self.tight_residual)
            )
        ):
            raise PydanticCustomError(
                "frames.sic_profile_axes", "SIC ledgers must retain source axes"
            )
        if any(
            overlap.as_fraction() != 1
            for index, row in enumerate(self.squared_overlaps)
            for column, overlap in enumerate(row)
            if index == column
        ):
            raise PydanticCustomError(
                "frames.sic_profile_diagonal",
                "SIC overlap diagonals must equal one",
            )
        if any(
            self.squared_overlaps[left][right].as_fraction()
            != self.squared_overlaps[right][left].as_fraction()
            for left in range(n)
            for right in range(left + 1, n)
        ):
            raise PydanticCustomError(
                "frames.sic_profile_overlap_symmetry",
                "squared overlaps must be symmetric",
            )
        off_diagonal = tuple(
            self.squared_overlaps[left][right].as_fraction()
            for left in range(n)
            for right in range(left + 1, n)
        )
        observed_equiangular = (d == 1 and n == 1) or (
            bool(off_diagonal) and len(set(off_diagonal)) == 1
        )
        if self.equiangular != observed_equiangular:
            raise PydanticCustomError(
                "frames.sic_profile_equiangular",
                "equiangular status must agree with off-diagonal overlaps",
            )
        if self.equiangular:
            assert self.common_squared_overlap is not None
            common = self.common_squared_overlap.as_fraction()
            if off_diagonal and any(overlap != common for overlap in off_diagonal):
                raise PydanticCustomError(
                    "frames.sic_profile_common_overlap",
                    "common overlap must equal every off-diagonal overlap",
                )
            if not off_diagonal and common != Fraction(1, d + 1):
                raise PydanticCustomError(
                    "frames.sic_profile_common_overlap",
                    "common overlap must equal the SIC target when no pair is observed",
                )
            expected_residual = common - Fraction(1, d + 1)
            assert self.common_squared_overlap_residual is not None
            if self.common_squared_overlap_residual.as_fraction() != expected_residual:
                raise PydanticCustomError(
                    "frames.sic_profile_common_residual",
                    "common overlap residual must equal the SIC target difference",
                )
        tight = all(
            entry.as_fractions() == (Fraction(0), Fraction(0))
            for row in self.tight_residual
            for entry in row
        )
        expected_is_sic = (
            self.cardinality_residual == 0
            and self.equiangular
            and self.common_squared_overlap_residual is not None
            and self.common_squared_overlap_residual.as_fraction() == 0
            and tight
        )
        if self.is_sic != expected_is_sic:
            raise PydanticCustomError(
                "frames.sic_profile_status",
                "SIC status must agree with cardinality, overlap, and tight residuals",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        frame: ComplexFrame,
        *,
        is_sic: bool,
        cardinality_residual: ExactInteger,
        equiangular: bool,
        common_squared_overlap: CanonicalRational | None,
        common_squared_overlap_residual: CanonicalRational | None,
        squared_overlaps: tuple[tuple[CanonicalRational, ...], ...],
        frame_operator: tuple[tuple[GaussianRational, ...], ...],
        tight_residual: tuple[tuple[GaussianRational, ...], ...],
    ) -> Self:
        return cls.model_construct(
            frame=frame,
            is_sic=is_sic,
            cardinality_residual=cardinality_residual,
            equiangular=equiangular,
            common_squared_overlap=common_squared_overlap,
            common_squared_overlap_residual=common_squared_overlap_residual,
            squared_overlaps=squared_overlaps,
            frame_operator=frame_operator,
            tight_residual=tight_residual,
        )


__all__ = [
    "MAX_DIM",
    "MAX_VECTOR_CELLS",
    "CoherenceResult",
    "ComplexFrameProfileRequest",
    "ComplexFrameProfileResult",
    "FramePotentialResult",
    "GramResult",
    "MutuallyUnbiasedBasesRequest",
    "MutuallyUnbiasedBasesResult",
    "SicProfileRequest",
    "SicProfileResult",
    "TightEquiangularProfileResult",
]
