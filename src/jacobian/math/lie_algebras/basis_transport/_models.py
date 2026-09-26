"""Typed contracts for exact Lie-algebra basis transport."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    LieBasisLabel,
)
from jacobian.math.matrices.values import RationalMatrix

MAX_BASIS_CHANGE_DIMENSION = 8
MAX_BASIS_CHANGE_INPUT_DIGITS = 128
MAX_BASIS_CHANGE_WORK = 100_000_000
MAX_BASIS_CHANGE_INTERMEDIATE_DIGITS = 131_072


class LieBasisChangeRequest(StrictModel):
    """Transport an algebra using columns of ``matrix`` as new basis vectors."""

    algebra: FiniteDimensionalLieAlgebra
    basis: tuple[LieBasisLabel, ...] = Field(min_length=1, max_length=8)
    matrix: RationalMatrix = Field(
        description=(
            "Square matrix whose columns are the new basis vectors expressed "
            "in the source algebra's ordered basis."
        )
    )

    @model_validator(mode="after")
    def require_matching_axes(self) -> Self:
        dimension = len(self.algebra.basis)
        if len(self.basis) != dimension or len(set(self.basis)) != dimension:
            raise ValueError(
                "target basis labels must be unique and match the source dimension"
            )
        if self.matrix.row_count != dimension or self.matrix.column_count != dimension:
            raise ValueError(
                "basis-change matrix must be square on the source basis axis"
            )
        return self


class LieBasisChangeResult(StrictModel):
    """The transported algebra and exact coordinate isomorphism."""

    source: FiniteDimensionalLieAlgebra
    target: FiniteDimensionalLieAlgebra
    target_to_source: RationalMatrix = Field(
        description="Columns give target basis vectors in source coordinates."
    )
    source_to_target: RationalMatrix = Field(
        description="The inverse coordinate map from source to target coordinates."
    )

    @model_validator(mode="after")
    def require_map_shapes(self) -> Self:
        dimension = len(self.source.basis)
        if len(self.target.basis) != dimension or any(
            matrix.row_count != dimension or matrix.column_count != dimension
            for matrix in (self.target_to_source, self.source_to_target)
        ):
            raise ValueError("basis isomorphism matrices must match both algebra axes")
        return self

    @classmethod
    def _from_kernel(
        cls,
        source: FiniteDimensionalLieAlgebra,
        target: FiniteDimensionalLieAlgebra,
        target_to_source: RationalMatrix,
        source_to_target: RationalMatrix,
    ) -> LieBasisChangeResult:
        return cls.model_construct(
            source=source,
            target=target,
            target_to_source=target_to_source,
            source_to_target=source_to_target,
        )


__all__ = [
    "MAX_BASIS_CHANGE_DIMENSION",
    "MAX_BASIS_CHANGE_INPUT_DIGITS",
    "MAX_BASIS_CHANGE_INTERMEDIATE_DIGITS",
    "MAX_BASIS_CHANGE_WORK",
    "LieBasisChangeRequest",
    "LieBasisChangeResult",
]
