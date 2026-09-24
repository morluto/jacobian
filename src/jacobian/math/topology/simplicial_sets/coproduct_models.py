"""Contracts for finite simplicial-set coproducts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.maps import TruncatedSimplicialMap


class SimplicialSetCoproductRequest(StrictModel):
    """Two complete simplicial-set prefixes with a common degree axis."""

    left: FiniteTruncatedSimplicialSet
    right: FiniteTruncatedSimplicialSet


TaggedSimplex = tuple[Literal["left", "right"], int]


class SimplicialSetCoproductResult(StrictModel):
    """Disjoint-union prefix, explicit factor tags, and canonical inclusions."""

    left: FiniteTruncatedSimplicialSet
    right: FiniteTruncatedSimplicialSet
    simplicial_set: FiniteTruncatedSimplicialSet
    tagged_axes: tuple[tuple[TaggedSimplex, ...], ...]
    left_inclusion: TruncatedSimplicialMap
    right_inclusion: TruncatedSimplicialMap

    @model_validator(mode="after")
    def require_disjoint_union_axes(self) -> Self:
        left, right, coproduct = self.left, self.right, self.simplicial_set
        if not (
            left.max_degree == right.max_degree == coproduct.max_degree
            and len(self.tagged_axes) == len(coproduct.sets)
            and self.left_inclusion.source == left
            and self.right_inclusion.source == right
            and self.left_inclusion.target == coproduct
            and self.right_inclusion.target == coproduct
        ):
            raise ValueError("coproduct factors and retained degree axes must agree")
        for degree, axis in enumerate(self.tagged_axes):
            expected_axis = tuple(
                [("left", i) for i in range(len(left.sets[degree]))]
                + [("right", i) for i in range(len(right.sets[degree]))]
            )
            expected_left = tuple(range(len(left.sets[degree])))
            offset = len(left.sets[degree])
            expected_right = tuple(offset + i for i in range(len(right.sets[degree])))
            if not (
                axis == expected_axis
                and len(coproduct.sets[degree]) == len(expected_axis)
                and self.left_inclusion.maps[degree] == expected_left
                and self.right_inclusion.maps[degree] == expected_right
            ):
                raise ValueError("coproduct simplex tags or inclusion axes are invalid")
        return self


__all__ = [
    "SimplicialSetCoproductRequest",
    "SimplicialSetCoproductResult",
    "TaggedSimplex",
]
