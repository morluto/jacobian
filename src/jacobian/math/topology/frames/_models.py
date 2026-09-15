"""Typed exact contracts for finite vector families and frames."""

from __future__ import annotations

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
    CyclotomicFrame,
    CyclotomicScalar,
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
        # An equiangular frame with an observed pair always has a common
        # squared inner product.  Only a singleton (or empty) family has no
        # off-diagonal pair from which to observe one, so a nontrivial
        # equiangular profile without the value is structurally impossible.
        if (
            self.equiangular
            and self.common_squared_inner_product is None
            and len(self.vectors) > 1
        ):
            raise PydanticCustomError(
                "frames.profile_shape",
                "equiangular frames with at least one vector pair require the "
                "common squared inner product",
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
        min_length=1,
        max_length=MAX_COMPLEX_BASIS_COUNT,
        description=(
            "One to MAX_COMPLEX_BASIS_COUNT bases; every base must have "
            "exactly `dimension` vectors, each of ambient dimension `dimension`."
        ),
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
    cross_gram_squared: tuple[tuple[tuple[CanonicalRational, ...], ...], ...] = Field(
        description=(
            "Squared normalized cross-Gram matrices in canonical pair order: "
            "every pair (first, second) with first < second, ordered by first "
            "then second."
        )
    )
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


class SphericalDesignRequest(StrictModel):
    """A weighted point family claiming cubature exactness of a strength."""

    family: VectorFamily
    weights: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        description="One nonnegative cubature weight per point summing to one.",
    )
    strength: int = Field(
        ge=1,
        le=5,
        description="Monomial total degree through which exactness is checked.",
    )

    @model_validator(mode="after")
    def require_weight_axis(self) -> Self:
        if len(self.weights) != len(self.family.vectors):
            raise PydanticCustomError(
                "frames.design_weight_count",
                "weights must have one entry per design point",
            )
        return self


class SphericalDesignResult(SphericalDesignRequest):
    """The cubature verdict with its complete moment ledger size."""

    is_design: bool
    moment_count: int = Field(ge=1)
    first_failure: tuple[int, ...] | None = Field(
        default=None,
        description="First failing exponent tuple, absent for a design.",
    )

    @model_validator(mode="after")
    def require_design_shape(self) -> Self:
        if self.is_design != (self.first_failure is None):
            raise PydanticCustomError(
                "frames.design_shape",
                "design status must agree with the failure ledger",
            )
        if self.first_failure is not None and (
            len(self.first_failure) != self.family.dimension
            or sum(self.first_failure) < 1
            or sum(self.first_failure) > self.strength
        ):
            raise PydanticCustomError(
                "frames.design_failure_shape",
                "a failure witness must be a checked monomial exponent tuple",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        family: VectorFamily,
        weights: tuple[CanonicalRational, ...],
        strength: int,
        is_design: bool,
        moment_count: int,
        first_failure: tuple[int, ...] | None,
    ) -> Self:
        return cls.model_construct(
            family=family,
            weights=weights,
            strength=strength,
            is_design=is_design,
            moment_count=moment_count,
            first_failure=first_failure,
        )


class ProjectiveDesignRequest(StrictModel):
    """A weighted complex family claiming a projective t-design identity."""

    frame: ComplexFrame
    weights: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        description="One nonnegative weight per representative summing to one.",
    )
    strength: int = Field(
        ge=1,
        le=4,
        description="Welch-identity power through which exactness is checked.",
    )

    @model_validator(mode="after")
    def require_weight_axis(self) -> Self:
        if len(self.weights) != len(self.frame.vectors):
            raise PydanticCustomError(
                "frames.design_weight_count",
                "weights must have one entry per design point",
            )
        return self


class ProjectiveDesignResult(ProjectiveDesignRequest):
    """The Welch-identity verdict with both exact sides retained."""

    is_design: bool
    welch_value: CanonicalRational
    welch_target: CanonicalRational

    @classmethod
    def _from_kernel(
        cls,
        *,
        frame: ComplexFrame,
        weights: tuple[CanonicalRational, ...],
        strength: int,
        is_design: bool,
        welch_value: CanonicalRational,
        welch_target: CanonicalRational,
    ) -> Self:
        return cls.model_construct(
            frame=frame,
            weights=weights,
            strength=strength,
            is_design=is_design,
            welch_value=welch_value,
            welch_target=welch_target,
        )


class CyclotomicSicRequest(StrictModel):
    """Cyclotomic projective representatives checked as SIC lines."""

    frame: CyclotomicFrame


class CyclotomicSicResult(CyclotomicSicRequest):
    """A scale-invariant cyclotomic SIC ledger with overlap witnesses."""

    is_sic: bool
    cardinality_residual: ExactInteger = Field(
        description="The supplied line count minus dimension squared."
    )
    equiangular: bool
    squared_overlaps: tuple[tuple[CyclotomicScalar, ...], ...]
    norms: tuple[CyclotomicScalar, ...]

    @model_validator(mode="after")
    def require_sic_shape(self) -> Self:
        count = len(self.frame.vectors)
        if self.cardinality_residual != count - self.frame.dimension**2:
            raise PydanticCustomError(
                "frames.cyclotomic_sic_cardinality",
                "SIC cardinality residual must match the source family",
            )
        if len(self.squared_overlaps) != count or any(
            len(row) != count for row in self.squared_overlaps
        ):
            raise PydanticCustomError(
                "frames.cyclotomic_sic_axes", "overlap ledgers must retain source axes"
            )
        if len(self.norms) != count:
            raise PydanticCustomError(
                "frames.cyclotomic_sic_axes", "norm ledgers must retain source axes"
            )
        if any(scalar.order != self.frame.order for row in self.squared_overlaps for scalar in row):
            raise PydanticCustomError(
                "frames.cyclotomic_sic_order",
                "overlap witnesses must share the frame cyclotomic order",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        frame: CyclotomicFrame,
        is_sic: bool,
        cardinality_residual: ExactInteger,
        equiangular: bool,
        squared_overlaps: tuple[tuple[CyclotomicScalar, ...], ...],
        norms: tuple[CyclotomicScalar, ...],
    ) -> Self:
        return cls.model_construct(
            frame=frame,
            is_sic=is_sic,
            cardinality_residual=cardinality_residual,
            equiangular=equiangular,
            squared_overlaps=squared_overlaps,
            norms=norms,
        )


class CyclotomicSicPovmRequest(StrictModel):
    """An exact cyclotomic SIC frame whose effects resolve the identity."""

    frame: CyclotomicFrame


class CyclotomicSicPovmResult(CyclotomicSicPovmRequest):
    """Exact SIC effects over one common cyclotomic denominator.

    Effect ``i`` is ``numerator[i] / denominator``; the kernel verifies the
    projector identities and the resolution of the identity, so a future
    quantum consumer can accept the retained matrices unchanged.
    """

    effect_denominator: CyclotomicScalar
    effect_numerators: tuple[
        tuple[tuple[CyclotomicScalar, ...], ...], ...
    ]

    @model_validator(mode="after")
    def require_povm_shape(self) -> Self:
        dimension = self.frame.dimension
        if len(self.effect_numerators) != len(self.frame.vectors) or any(
            len(matrix) != dimension or any(len(row) != dimension for row in matrix)
            for matrix in self.effect_numerators
        ):
            raise PydanticCustomError(
                "frames.cyclotomic_povm_axes", "effects must retain source axes"
            )
        if self.effect_denominator.order != self.frame.order or any(
            scalar.order != self.frame.order
            for matrix in self.effect_numerators
            for row in matrix
            for scalar in row
        ):
            raise PydanticCustomError(
                "frames.cyclotomic_povm_order",
                "effects must share the frame cyclotomic order",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        frame: CyclotomicFrame,
        effect_denominator: CyclotomicScalar,
        effect_numerators: tuple[tuple[tuple[CyclotomicScalar, ...], ...], ...],
    ) -> Self:
        return cls.model_construct(
            frame=frame,
            effect_denominator=effect_denominator,
            effect_numerators=effect_numerators,
        )


__all__ = [
    "MAX_DIM",
    "MAX_VECTOR_CELLS",
    "CoherenceResult",
    "ComplexFrameProfileRequest",
    "ComplexFrameProfileResult",
    "CyclotomicSicPovmRequest",
    "CyclotomicSicPovmResult",
    "CyclotomicSicRequest",
    "CyclotomicSicResult",
    "FramePotentialResult",
    "GramResult",
    "MutuallyUnbiasedBasesRequest",
    "MutuallyUnbiasedBasesResult",
    "ProjectiveDesignRequest",
    "ProjectiveDesignResult",
    "SicProfileRequest",
    "SicProfileResult",
    "SphericalDesignRequest",
    "SphericalDesignResult",
    "TightEquiangularProfileResult",
]
