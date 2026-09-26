"""Canonical rational V- and H-representation values."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import AfterValidator, Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_RATIONAL_POLYTOPE_DIMENSION = 7
"""Largest ambient axis supported by the canonical rational values."""

MAX_RATIONAL_POLYHEDRON_INEQUALITIES = 64
"""Maximum H-inequalities accepted by the H-to-V operation."""

MAX_RATIONAL_POLYHEDRON_GENERATORS = 20_000
"""Structural ceiling on points or oriented directions in a V-value."""


def _require_scalar_label(value: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise _validation_error(
            "unicode_scalar_label", "axis labels must be Unicode scalar values"
        )
    return value


PolyhedronAxis = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, strict=True),
    AfterValidator(_require_scalar_label),
]
PolyhedronVector = Annotated[
    tuple[CanonicalRational, ...],
    Field(max_length=MAX_RATIONAL_POLYTOPE_DIMENSION),
]


class RationalPolyhedronSpace(StrictModel):
    """An ordered rational coordinate space, including the zero-dimensional one."""

    axes: tuple[PolyhedronAxis, ...] = Field(max_length=MAX_RATIONAL_POLYTOPE_DIMENSION)

    @model_validator(mode="after")
    def require_distinct_axes(self) -> Self:
        if len(set(self.axes)) != len(self.axes):
            raise _validation_error(
                "coordinate_axes_unique", "coordinate axes must be unique"
            )
        return self


class RationalAffineHalfspace(StrictModel):
    """An exact affine inequality ``normal . x <= bound``."""

    normal: tuple[CanonicalRational, ...] = Field(
        max_length=MAX_RATIONAL_POLYTOPE_DIMENSION
    )
    bound: CanonicalRational


class RationalHPolyhedron(StrictModel):
    """An exact H-presentation; constant rows are allowed and classified exactly."""

    space: RationalPolyhedronSpace
    inequalities: tuple[RationalAffineHalfspace, ...] = Field(
        max_length=MAX_RATIONAL_POLYHEDRON_INEQUALITIES
    )

    @model_validator(mode="after")
    def require_matching_normals(self) -> Self:
        if any(len(row.normal) != len(self.space.axes) for row in self.inequalities):
            raise _validation_error(
                "inequality_dimension", "every normal must match the coordinate space"
            )
        return self


class RationalPolyhedronVPresentation(StrictModel):
    """Finite points plus oriented recession and lineality generators.

    For a nonempty value the represented set is conv(points) + cone(rays) +
    span(lineality). Generator minimality or extremality is not asserted.
    """

    space: RationalPolyhedronSpace
    points: tuple[PolyhedronVector, ...] = Field(
        max_length=MAX_RATIONAL_POLYHEDRON_GENERATORS
    )
    rays: tuple[PolyhedronVector, ...] = Field(
        max_length=MAX_RATIONAL_POLYHEDRON_GENERATORS
    )
    lineality: tuple[PolyhedronVector, ...] = Field(
        max_length=MAX_RATIONAL_POLYTOPE_DIMENSION
    )
    empty: bool
    affine_dimension: int = Field(ge=-1, le=MAX_RATIONAL_POLYTOPE_DIMENSION)

    @model_validator(mode="after")
    def require_well_shaped_generators(self) -> Self:
        dimension = len(self.space.axes)
        if any(
            len(vector) != dimension
            for family in (self.points, self.rays, self.lineality)
            for vector in family
        ):
            raise _validation_error(
                "generator_dimension", "every generator must match the coordinate space"
            )
        if any(
            not any(component.num for component in vector)
            for family in (self.rays, self.lineality)
            for vector in family
        ):
            raise _validation_error(
                "zero_direction", "recession and lineality directions must be nonzero"
            )
        if self.empty:
            if (
                self.points
                or self.rays
                or self.lineality
                or self.affine_dimension != -1
            ):
                raise _validation_error(
                    "empty_v_presentation",
                    "an empty presentation has no generators and dimension -1",
                )
        elif not self.points or not 0 <= self.affine_dimension <= dimension:
            raise _validation_error(
                "nonempty_v_presentation",
                "a nonempty presentation needs a point and a valid affine dimension",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        space: RationalPolyhedronSpace,
        points: tuple[tuple[CanonicalRational, ...], ...],
        rays: tuple[tuple[CanonicalRational, ...], ...],
        lineality: tuple[tuple[CanonicalRational, ...], ...],
        empty: bool,
        affine_dimension: int,
    ) -> Self:
        """Build after the operation has admitted and constructed these arrays."""

        return cls.model_construct(
            space=space,
            points=points,
            rays=rays,
            lineality=lineality,
            empty=empty,
            affine_dimension=affine_dimension,
        )


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polytope.{reason}", message)


class Vertex(StrictModel):
    """One rational vertex of a V-representation."""

    coordinates: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_RATIONAL_POLYTOPE_DIMENSION
    )


class Halfspace(StrictModel):
    """One rational half-space ``<a, x> <= b`` of an H-representation."""

    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_POLYTOPE_DIMENSION,
        description=(
            "Normal vector a of the half-space <a, x> <= b; at least one "
            "entry must be nonzero (all-zero rows are rejected)."
        ),
    )
    offset: CanonicalRational


__all__ = [
    "MAX_RATIONAL_POLYHEDRON_GENERATORS",
    "MAX_RATIONAL_POLYHEDRON_INEQUALITIES",
    "MAX_RATIONAL_POLYTOPE_DIMENSION",
    "Halfspace",
    "RationalAffineHalfspace",
    "RationalHPolyhedron",
    "RationalPolyhedronSpace",
    "RationalPolyhedronVPresentation",
    "Vertex",
]
