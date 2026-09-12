"""Typed exact contracts for finite vector families and frames."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.topology.frames.values import (
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
                "frames.profile_shape", "non-equiangular frames cannot carry a common value"
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
    dimension: int
    bases: tuple[ComplexFrame, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_basis_axes(self) -> Self:
        if self.dimension < 1 or any(
            basis.dimension != self.dimension or len(basis.vectors) != self.dimension
            for basis in self.bases
        ):
            raise PydanticCustomError(
                "frames.mub_basis_shape",
                "every basis must have exactly dimension vectors of that dimension",
            )
        return self


class MutuallyUnbiasedBasesResult(MutuallyUnbiasedBasesRequest):
    is_mutually_unbiased: bool
    basis_pair_count: int

    @model_validator(mode="after")
    def require_profile_shape(self) -> Self:
        if self.basis_pair_count != len(self.bases) * (len(self.bases) - 1) // 2:
            raise PydanticCustomError(
                "frames.mub_profile_shape", "basis pair count is not canonical"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: MutuallyUnbiasedBasesRequest,
        *,
        is_mutually_unbiased: bool,
    ) -> Self:
        return cls.model_construct(
            dimension=request.dimension,
            bases=request.bases,
            is_mutually_unbiased=is_mutually_unbiased,
            basis_pair_count=len(request.bases) * (len(request.bases) - 1) // 2,
        )


class ComplexFrameDesignProfileRequest(StrictModel):
    frame: ComplexFrame


class ComplexFrameDesignProfileResult(ComplexFrameDesignProfileRequest):
    tight: bool
    equiangular: bool
    common_squared_overlap: CanonicalRational | None

    @model_validator(mode="after")
    def require_profile_shape(self) -> Self:
        if not self.equiangular and self.common_squared_overlap is not None:
            raise PydanticCustomError(
                "frames.design_profile_shape",
                "non-equiangular profile cannot carry a common overlap",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: ComplexFrameDesignProfileRequest,
        *,
        tight: bool,
        equiangular: bool,
        common_squared_overlap: CanonicalRational | None,
    ) -> Self:
        return cls.model_construct(
            frame=request.frame,
            tight=tight,
            equiangular=equiangular,
            common_squared_overlap=common_squared_overlap,
        )


class SicProfileRequest(StrictModel):
    frame: ComplexFrame


class SicProfileResult(SicProfileRequest):
    is_sic: bool
    common_squared_overlap: CanonicalRational | None

    @model_validator(mode="after")
    def require_profile_shape(self) -> Self:
        if not self.is_sic and self.common_squared_overlap is not None:
            raise PydanticCustomError(
                "frames.sic_profile_shape",
                "a non-SIC profile cannot carry a common overlap",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: SicProfileRequest,
        *,
        is_sic: bool,
        common_squared_overlap: CanonicalRational | None,
    ) -> Self:
        return cls.model_construct(
            frame=request.frame,
            is_sic=is_sic,
            common_squared_overlap=common_squared_overlap,
        )


__all__ = [
    "MAX_DIM",
    "MAX_VECTOR_CELLS",
    "CoherenceResult",
    "ComplexFrameDesignProfileRequest",
    "ComplexFrameDesignProfileResult",
    "FramePotentialResult",
    "GramResult",
    "MutuallyUnbiasedBasesRequest",
    "MutuallyUnbiasedBasesResult",
    "SicProfileRequest",
    "SicProfileResult",
    "TightEquiangularProfileResult",
]
