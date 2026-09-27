"""Canonical values for the three dimensional translation torus quotient."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
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


def _source_circle_directions(source: CrystallographicFundamentalDomainResult):
    """Derive the canonical edge directions from the retained eight vertices."""
    vertices = source.source.facet_profile.vertices
    if len(vertices) != 8 or any(len(vertex.coordinates) != 3 for vertex in vertices):
        return None
    points = tuple(
        tuple(value.as_fraction() for value in vertex.coordinates) for vertex in vertices
    )
    point_set = frozenset(points)
    base = min(point_set)
    candidates = tuple(sorted(point for point in point_set if point != base))
    for endpoints in combinations(candidates, 3):
        vectors = tuple(
            tuple(endpoints[index][axis] - base[axis] for axis in range(3))
            for index in range(3)
        )
        sums = frozenset(
            tuple(
                base[axis]
                + sum(
                    (vectors[index][axis] for index in range(3) if mask & (1 << index)),
                    Fraction(0),
                )
                for axis in range(3)
            )
            for mask in range(8)
        )
        if sums == point_set:
            return tuple(sorted(vectors))
    return None


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
        extension = self.source.source.affine_realization.source
        identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        source_directions = _source_circle_directions(self.source)
        directions = tuple(
            tuple(value.as_fraction() for value in vector)
            for vector in self.circle_directions
        )
        pairings = self.source.source.pairings
        signed_directions = {
            tuple(int(value) * sign for value in vector)
            for vector in source_directions or ()
            for sign in (-1, 1)
            if all(value.denominator == 1 for value in vector)
        }
        direction_counts: dict[tuple[int, int, int], int] = {}
        for pairing in pairings:
            vector = pairing.lattice_translation
            direction = min(vector, tuple(-value for value in vector))
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        pairings_by_source = {item.source_facet_index: item for item in pairings}
        source_bound = (
            len(extension.action_matrices[0]) == 3
            and extension.multiplication_table == ((0,),)
            and extension.action_matrices == (identity,)
            and extension.factor_set == (((0, 0, 0),),)
            and source_directions is not None
            and directions == source_directions
            and all(value.denominator == 1 for vector in source_directions for value in vector)
            and len(pairings) == 6
            and all(
                pairing.holonomy_element == 0
                and pairing.source_facet_index != pairing.target_facet_index
                and pairing.lattice_translation in signed_directions
                for pairing in pairings
            )
            and set(direction_counts.values()) == {2}
            and len(direction_counts) == 3
            and len(pairings_by_source) == 6
            and set(pairings_by_source)
            == {pairing.target_facet_index for pairing in pairings}
            and all(
                pairings_by_source[pairing.target_facet_index].target_facet_index
                == pairing.source_facet_index
                and tuple(
                    -value
                    for value in pairings_by_source[
                        pairing.target_facet_index
                    ].lattice_translation
                )
                == pairing.lattice_translation
                for pairing in pairings
            )
        )
        if (
            not self.source.is_fundamental_domain
            or not source_bound
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
