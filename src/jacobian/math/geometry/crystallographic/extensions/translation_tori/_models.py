"""Canonical values for the three dimensional translation torus quotient."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
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
    """The integral cellular chains of a checked parallelepiped 3-torus.

    The source is a verified fundamental parallelepiped for a rank-three pure
    translation group. Its quotient is the product CW structure on
    ``(S^1)^3``; the listed vectors retain the three oriented circle directions.
    """

    source: CrystallographicFundamentalDomainResult
    circle_directions: tuple[
        tuple[CanonicalRational, CanonicalRational, CanonicalRational], ...
    ]
    quotient_chain_complex: ChainComplexValue

    @model_validator(mode="after")
    def require_product_torus_chain(self) -> Self:
        chain = self.quotient_chain_complex
        if (
            len(self.circle_directions) != 3
            or chain.coefficient_ring != CoefficientRing.INTEGER
            or chain.prime is not None
            or chain.degree_min != 0
            or chain.degree_max != 3
            or chain.basis_sizes != (1, 3, 3, 1)
            or chain.differential_matrices
            != (
                ((0, 0, 0),),
                ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
                ((0,), (0,), (0,)),
            )
        ):
            raise PydanticCustomError(
                "crystallographic.translation_torus_chain",
                "result must carry the product CW chain complex of a 3-torus",
            )
        return self


__all__ = ["BieberbachTranslationTorusChains"]
