"""Provider-independent values for exact finite topological spaces."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel

MAX_POINTS = 64


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by finite-topology values."""

    return PydanticCustomError(f"finite_topology_space.{reason}", message)


def _require_distinct_points(points: tuple[str, ...]) -> None:
    if len(set(points)) != len(points):
        raise _validation_error(
            "point_labels_not_distinct", "point labels must be distinct"
        )


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

    points: tuple[OpaqueLabel, ...] = Field(min_length=1, max_length=MAX_POINTS)
    preorder: tuple[tuple[int, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_well_formed(self) -> Self:
        _require_distinct_points(self.points)
        if len(self.preorder) != len(self.points):
            raise _validation_error(
                "preorder_row_count_mismatch", "preorder must have one row per point"
            )
        for row in self.preorder:
            for idx in row:
                if not 0 <= idx < len(self.points):
                    raise _validation_error(
                        "preorder_index_out_of_range", "preorder index out of range"
                    )
        for i in range(len(self.points)):
            if i not in self.preorder[i]:
                raise _validation_error(
                    "preorder_not_reflexive", "preorder must be reflexive"
                )
        # Transitivity: j in row[i] => row[j] subset of row[i].
        for _i, row in enumerate(self.preorder):
            row_i = set(row)
            for j in row:
                if not set(self.preorder[j]).issubset(row_i):
                    raise _validation_error(
                        "preorder_not_transitive", "preorder must be transitive"
                    )
        return self


class FiniteTopologicalMap(StrictModel):
    """An immutable continuous map between finite topological spaces."""

    source: FiniteTopologicalSpace
    target: FiniteTopologicalSpace
    point_map: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_map(self) -> Self:
        if len(self.point_map) != len(self.source.points):
            raise _validation_error(
                "point_map_length_mismatch",
                "point_map must have one entry per source point",
            )
        for idx in self.point_map:
            if not 0 <= idx < len(self.target.points):
                raise _validation_error(
                    "point_map_index_out_of_range", "point_map index out of range"
                )
        return self


__all__ = [
    "MAX_POINTS",
    "FiniteTopologicalMap",
    "FiniteTopologicalSpace",
]
