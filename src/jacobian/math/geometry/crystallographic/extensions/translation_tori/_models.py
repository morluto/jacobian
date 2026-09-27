"""Canonical values for bounded translation torus quotients."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import comb
from typing import Annotated, Self

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


def _source_circle_directions(source: CrystallographicFundamentalDomainResult):
    """Derive canonical edge directions from the retained hypercube vertices."""
    vertices = source.source.facet_profile.vertices
    dimension = len(source.source.affine_realization.source.action_matrices[0])
    if (
        not 1 <= dimension <= 4
        or len(vertices) != 1 << dimension
        or any(len(vertex.coordinates) != dimension for vertex in vertices)
    ):
        return None
    points = tuple(
        tuple(value.as_fraction() for value in vertex.coordinates) for vertex in vertices
    )
    point_set = frozenset(points)
    base = min(point_set)
    candidates = tuple(sorted(point for point in point_set if point != base))
    for endpoints in combinations(candidates, dimension):
        vectors = tuple(
            tuple(endpoints[index][axis] - base[axis] for axis in range(dimension))
            for index in range(dimension)
        )
        sums = frozenset(
            tuple(
                base[axis]
                + sum(
                    (vectors[index][axis] for index in range(dimension) if mask & (1 << index)),
                    Fraction(0),
                )
                for axis in range(dimension)
            )
            for mask in range(1 << dimension)
        )
        if sums == point_set:
            return tuple(sorted(vectors))
    return None


class BieberbachTranslationTorusChains(StrictModel):
    """The integral cellular chains of a checked parallelepiped torus.

    The source is a verified fundamental parallelepiped for a pure translation
    group of rank one through four. Its quotient has the product CW structure
    on ``(S^1)^n``; the vectors retain the oriented circle directions.
    """

    source: CrystallographicFundamentalDomainResult
    circle_directions: tuple[
        Annotated[tuple[CanonicalRational, ...], Field(max_length=4)], ...
    ] = Field(min_length=1, max_length=4)
    quotient_chain_complex: ChainComplexValue

    @model_validator(mode="after")
    def require_product_torus_chain(self) -> Self:
        chain = self.quotient_chain_complex
        dimension = len(self.source.source.affine_realization.source.action_matrices[0])
        expected_basis_sizes = tuple(
            comb(dimension, degree) for degree in range(dimension + 1)
        )
        extension = self.source.source.affine_realization.source
        identity = tuple(
            tuple(int(row == column) for column in range(dimension))
            for row in range(dimension)
        )
        source_directions = _source_circle_directions(self.source)
        directions = tuple(
            tuple(value.as_fraction() for value in vector)
            for vector in self.circle_directions
        )
        pairings = self.source.source.pairings
        signed_directions = {
            tuple(value * sign for value in vector)
            for vector in source_directions or ()
            for sign in (-1, 1)
        }
        direction_counts: dict[tuple[int, ...], int] = {}
        for pairing in pairings:
            vector = pairing.lattice_translation
            direction = min(vector, tuple(-value for value in vector))
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        pairings_by_source = {item.source_facet_index: item for item in pairings}
        source_bound = (
            len(extension.action_matrices[0]) == dimension
            and extension.multiplication_table == ((0,),)
            and extension.action_matrices == (identity,)
            and extension.factor_set == ((tuple(0 for _ in range(dimension)),),)
            and source_directions is not None
            and directions == source_directions
            and all(value.denominator == 1 for vector in source_directions for value in vector)
            and len(pairings) == 2 * dimension
            and all(
                pairing.holonomy_element == 0
                and pairing.source_facet_index != pairing.target_facet_index
                and pairing.lattice_translation in signed_directions
                for pairing in pairings
            )
            and set(direction_counts.values()) == {2}
            and len(direction_counts) == dimension
            and len(pairings_by_source) == 2 * dimension
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
            or not 1 <= dimension <= 4
            or len(self.circle_directions) != dimension
            or any(len(direction) != dimension for direction in self.circle_directions)
            or not source_bound
            or chain.coefficient_ring != CoefficientRing.INTEGER
            or chain.prime is not None
            or chain.degree_min != 0
            or chain.degree_max != dimension
            or chain.basis_sizes != expected_basis_sizes
            or any(
                matrix
                != tuple(
                    tuple(0 for _ in range(expected_basis_sizes[degree]))
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
