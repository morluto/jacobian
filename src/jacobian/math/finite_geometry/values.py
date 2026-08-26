"""Canonical mathematical values for bounded exact finite geometry."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError
from sympy import isprime

from jacobian._models import StrictModel

MAX_DIM = 32
MAX_FIELD_ORDER = 10000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by finite-geometry contracts."""

    return PydanticCustomError(f"finite_geometry.{reason}", message)


class PrimeFieldVectorSpace(StrictModel):
    """An ordered coordinate space over a named prime field."""

    field_order: int = Field(ge=2, le=MAX_FIELD_ORDER)
    axis: tuple[str, ...] = Field(min_length=1, max_length=MAX_DIM)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if not isprime(self.field_order):
            raise _validation_error(
                "field_order_not_prime", "field_order must be prime"
            )
        if any(not label or not label.isidentifier() for label in self.axis):
            raise _validation_error(
                "axis_labels_invalid", "axis labels must be nonempty identifiers"
            )
        if len(set(self.axis)) != len(self.axis):
            raise _validation_error(
                "axis_labels_not_unique", "axis labels must be unique"
            )
        return self


def _validate_vector(vector: tuple[int, ...], space: PrimeFieldVectorSpace) -> None:
    if len(vector) != len(space.axis):
        raise _validation_error(
            "vector_length_mismatch", "vector length must match the ambient axis"
        )
    if any(not 0 <= value < space.field_order for value in vector):
        raise _validation_error(
            "vector_entry_not_canonical",
            "vector entries must be canonical field residues",
        )


def _require_projective_representative(
    coordinates: tuple[int, ...], space: PrimeFieldVectorSpace
) -> None:
    """Require canonical projective coordinates inside ``space``.

    Shared by the point value, point sequences, and enumeration results so
    producers and consumers accept one canonical representative shape:
    canonical residues whose first nonzero entry is one.
    """

    _validate_vector(coordinates, space)
    try:
        first = next(value for value in coordinates if value != 0)
    except StopIteration as exc:
        raise _validation_error(
            "projective_coordinates_zero", "projective coordinates must be nonzero"
        ) from exc
    if first != 1:
        raise _validation_error(
            "projective_coordinates_not_normalized",
            "projective coordinates must have first nonzero entry one",
        )


class ProjectivePoint(StrictModel):
    """A canonical point in a specific prime-field projective space."""

    space: PrimeFieldVectorSpace
    coordinates: tuple[int, ...]

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        _require_projective_representative(self.coordinates, self.space)
        return self


class ProjectivePointSequence(StrictModel):
    """The complete ordered canonical point list of one projective space.

    Native iteration yields typed :class:`ProjectivePoint` items bound to the
    declared parent space, so producers compose directly into consumers such
    as ``embed_projective_point_in_finite_field`` without reconstruction.
    Serialization stores the parent space once and each point as a bare
    canonical coordinate tuple, so complete enumerations stay compact enough
    for the transport envelope instead of repeating parent metadata per
    point.
    """

    space: PrimeFieldVectorSpace
    coordinates: tuple[tuple[int, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_complete_and_canonical(self) -> Self:
        q = self.space.field_order
        n = len(self.space.axis)
        expected = (q**n - 1) // (q - 1)
        if len(self.coordinates) != expected:
            raise _validation_error(
                "point_sequence_count_mismatch",
                "a projective point sequence must list every point of its "
                "declared parent space exactly once",
            )
        for coordinates in self.coordinates:
            _require_projective_representative(coordinates, self.space)
        if len(set(self.coordinates)) != len(self.coordinates):
            raise _validation_error(
                "point_sequence_points_not_unique",
                "projective points must be unique",
            )
        return self

    def __len__(self) -> int:
        return len(self.coordinates)

    def __iter__(self) -> Iterator[ProjectivePoint]:  # type: ignore[override]
        """Iterate as typed projective points bound to ``space``.

        Pydantic's ``BaseModel.__iter__`` yields ``(field, value)`` pairs;
        this canonical value intentionally iterates its mathematical items
        instead.
        """

        return (
            ProjectivePoint(space=self.space, coordinates=coordinates)
            for coordinates in self.coordinates
        )

    @property
    def points(self) -> Iterator[ProjectivePoint]:
        """Iterate the sequence as typed parent-bound projective points."""

        return iter(self)


__all__ = [
    "MAX_DIM",
    "MAX_FIELD_ORDER",
    "PrimeFieldVectorSpace",
    "ProjectivePoint",
    "ProjectivePointSequence",
]
