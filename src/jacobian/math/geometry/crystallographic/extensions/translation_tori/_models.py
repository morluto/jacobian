"""Canonical values for bounded translation torus quotients."""

from __future__ import annotations

from math import comb
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.geometry.crystallographic.extensions._models import (
    CrystallographicFundamentalDomainResult,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
)


class BieberbachTranslationTorusChains(StrictModel):
    """The integral cellular chains of a checked parallelepiped torus.

    The source is a verified fundamental parallelepiped for a pure translation
    group of rank one through four. Its quotient has the product CW structure
    on ``(S^1)^n``; the vectors retain the oriented circle directions.
    """

    source: CrystallographicFundamentalDomainResult
    circle_directions: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1, max_length=4
    )
    quotient_chain_complex: ChainComplexValue

    @model_validator(mode="after")
    def require_product_torus_chain(self) -> Self:
        chain = self.quotient_chain_complex
        dimension = len(self.source.source.affine_realization.source.action_matrices[0])
        expected_basis_sizes = tuple(
            comb(dimension, degree) for degree in range(dimension + 1)
        )
        if (
            not 1 <= dimension <= 4
            or len(self.circle_directions) != dimension
            or any(len(direction) != dimension for direction in self.circle_directions)
            or chain.coefficient_ring != CoefficientRing.INTEGER
            or chain.prime is not None
            or chain.degree_min != 0
            or chain.degree_max != dimension
            or chain.basis_sizes != expected_basis_sizes
            or any(
                matrix
                != tuple(
                    tuple("0" for _ in range(expected_basis_sizes[degree]))
                    for _ in range(expected_basis_sizes[degree - 1])
                )
                for degree, matrix in enumerate(chain.differential_matrices, start=1)
            )
        ):
            raise PydanticCustomError(
                "crystallographic.translation_torus_chain",
                "result must carry the product CW chain complex of a translation torus",
            )
        return self


__all__ = ["BieberbachTranslationTorusChains"]
