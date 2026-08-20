"""Provider-independent values for exact finite topological spaces."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_POINTS = 64


class FiniteTopologicalSpace(StrictModel):
    """An immutable finite topological space represented by its specialization
    preorder.

    On a finite set, every topology is Alexandrov, so the topology is
    equivalently represented by a preorder: ``x <= y`` iff x is in the closure
    of {y} (equivalently, every open set containing x also contains y).

    ``points`` are unique labels. ``preorder`` is a tuple of one row per point
    (in the same order), where each row lists the indices of points <= that
    point.
    """

    points: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)
    preorder: tuple[tuple[int, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_well_formed(self) -> Self:
        if len(self.preorder) != len(self.points):
            raise ValueError("preorder must have one row per point")
        for row in self.preorder:
            for idx in row:
                if not 0 <= idx < len(self.points):
                    raise ValueError("preorder index out of range")
        for i in range(len(self.points)):
            if i not in self.preorder[i]:
                raise ValueError("preorder must be reflexive")
        # Transitivity: j in row[i] => row[j] subset of row[i].
        for _i, row in enumerate(self.preorder):
            row_i = set(row)
            for j in row:
                if not set(self.preorder[j]).issubset(row_i):
                    raise ValueError("preorder must be transitive")
        return self


class FiniteTopologicalMap(StrictModel):
    """An immutable continuous map between finite topological spaces."""

    source: FiniteTopologicalSpace
    target: FiniteTopologicalSpace
    point_map: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_map(self) -> Self:
        if len(self.point_map) != len(self.source.points):
            raise ValueError("point_map must have one entry per source point")
        for idx in self.point_map:
            if not 0 <= idx < len(self.target.points):
                raise ValueError("point_map index out of range")
        return self


__all__ = [
    "MAX_POINTS",
    "FiniteTopologicalMap",
    "FiniteTopologicalSpace",
]
