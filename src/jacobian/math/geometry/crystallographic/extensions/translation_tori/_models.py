"""Canonical values for bounded translation torus quotients."""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from math import comb, prod
from typing import Annotated, Self

from pydantic import ConfigDict, Field, ValidationInfo, field_validator, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.geometry.crystallographic.extensions._models import (
    _bounded_native_source,
    _BoundedChainComplex,
    _BoundedFundamentalDomainSource,
)
from jacobian.math.topology.chain_complexes.values import (
    CoefficientRing,
)


class BieberbachTranslationTorusChains(StrictModel):
    """The integral cellular chains of a checked parallelepiped torus.

    The source is a verified fundamental parallelepiped for a pure translation
    group of rank one through four. Its quotient has the product CW structure
    on ``(S^1)^n``; the vectors retain the oriented circle directions.
    """

    model_config = ConfigDict(revalidate_instances="always")

    source: _BoundedFundamentalDomainSource
    circle_directions: tuple[
        Annotated[tuple[CanonicalRational, ...], Field(max_length=4)], ...
    ] = Field(min_length=1, max_length=4)
    quotient_chain_complex: _BoundedChainComplex

    @field_validator("circle_directions", mode="before")
    @classmethod
    def require_bounded_directions(cls, value: object, info: ValidationInfo) -> object:
        if (type(value) is not tuple and type(value) is not list) or len(value) > 4:
            raise PydanticCustomError(
                "crystallographic.translation_torus_directions",
                "circle directions exceed the rank-four envelope",
            )
        if any(
            (type(row) is not tuple and type(row) is not list) or len(row) > 4
            for row in value
        ):
            raise PydanticCustomError(
                "crystallographic.translation_torus_directions",
                "circle direction rows exceed the rank-four envelope",
            )
        return (
            tuple(tuple(row) for row in value)
            if info.mode == "json"
            else _bounded_native_source(value)
        )

    @model_validator(mode="after")
    def require_product_torus_chain(self) -> Self:
        chain = self.quotient_chain_complex
        pairing = self.source.source
        realization = pairing.affine_realization
        extension = realization.source
        dimension = len(extension.action_matrices[0])
        expected_basis_sizes = tuple(
            comb(dimension, degree) for degree in range(dimension + 1)
        )
        if (
            not self.source.is_fundamental_domain
            or self.source.polytope_volume.as_fraction() != 1
            or self.source.quotient_covolume.as_fraction() != 1
            or not 1 <= dimension <= 4
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
        # The retained source must be a pure translation action, and the
        # retained circle directions must actually generate its vertex set.
        # Checking only the axes above let a checked Klein-bottle source be
        # paired with any all-zero complex, so a caller could receive a value
        # whose claimed torus does not describe its own retained source.
        identity = tuple(
            tuple(int(row == column) for column in range(dimension))
            for row in range(dimension)
        )
        if (
            extension.multiplication_table != ((0,),)
            or extension.action_matrices != (identity,)
            or extension.factor_set != (((0,) * dimension,),)
            or any(
                value.as_fraction()
                for value in realization.section_maps[0].section_shift
            )
        ):
            raise PydanticCustomError(
                "crystallographic.translation_torus_holonomy",
                "a translation torus requires a pure translation action with trivial holonomy",
            )
        coordinates = tuple(
            tuple(value.as_fraction() for value in vertex.coordinates)
            for vertex in pairing.facet_profile.vertices
        )
        points = frozenset(coordinates)
        directions = tuple(
            tuple(value.as_fraction() for value in direction)
            for direction in self.circle_directions
        )
        if len(points) != 1 << dimension or any(
            value.denominator != 1 for direction in directions for value in direction
        ):
            raise PydanticCustomError(
                "crystallographic.translation_torus_directions",
                "circle directions must be integral in the retained translation lattice",
            )
        # This is a bounded intrinsic basis check: at rank <= 4 the determinant
        # has at most 24 products. No geometric search or backend replay is needed.
        determinant = sum(
            (-1)
            ** sum(
                order[left] > order[right]
                for left in range(dimension)
                for right in range(left + 1, dimension)
            )
            * prod(
                directions[row][column].numerator for row, column in enumerate(order)
            )
            for order in permutations(range(dimension))
        )
        if abs(determinant) != 1:
            raise PydanticCustomError(
                "crystallographic.translation_torus_directions",
                "circle directions must be a basis of the retained unit translation lattice",
            )
        base = min(points)
        generated = tuple(
            tuple(
                base[axis]
                + sum(
                    (
                        directions[index][axis]
                        for index in range(dimension)
                        if mask & (1 << index)
                    ),
                    Fraction(0),
                )
                for axis in range(dimension)
            )
            for mask in range(1 << dimension)
        )
        if frozenset(generated) != points:
            raise PydanticCustomError(
                "crystallographic.translation_torus_directions",
                "circle directions must generate the retained parallelepiped domain",
            )
        # The retained facets and translations must describe those opposite
        # faces. Comparing their bounded vertex sets avoids re-deriving facets.
        facet_points = tuple(
            frozenset(coordinates[index] for index in facet.source_vertex_indices)
            for facet in pairing.facet_profile.facets
        )
        expected_facets = {
            frozenset(
                point
                for mask, point in enumerate(generated)
                if bool(mask & (1 << axis)) == side
            )
            for axis in range(dimension)
            for side in (False, True)
        }
        signed_directions = {
            tuple(sign * value for value in direction)
            for direction in directions
            for sign in (-1, 1)
        }
        if (
            len(facet_points) != 2 * dimension
            or set(facet_points) != expected_facets
            or any(
                side.holonomy_element != 0
                or side.lattice_translation not in signed_directions
                or frozenset(
                    tuple(
                        value + shift
                        for value, shift in zip(
                            point, side.lattice_translation, strict=True
                        )
                    )
                    for point in facet_points[side.source_facet_index]
                )
                != facet_points[side.target_facet_index]
                for side in pairing.pairings
            )
        ):
            raise PydanticCustomError(
                "crystallographic.translation_torus_pairings",
                "retained side pairings must identify opposite parallelepiped faces by circle translations",
            )
        return self


__all__ = ["BieberbachTranslationTorusChains"]
