"""Contracts for degreewise products of finite simplicial-set prefixes."""

from __future__ import annotations

from typing import Any, Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.maps import TruncatedSimplicialMap


class SimplicialSetProductRequest(StrictModel):
    """Two complete prefixes with the same retained degree axis."""

    left: FiniteTruncatedSimplicialSet
    right: FiniteTruncatedSimplicialSet


class SimplicialSetProductResult(StrictModel):
    """Product prefix together with its exact factor axes and projections."""

    left: FiniteTruncatedSimplicialSet
    right: FiniteTruncatedSimplicialSet
    simplicial_set: FiniteTruncatedSimplicialSet
    pair_axes: tuple[tuple[tuple[int, int], ...], ...]
    left_projection: TruncatedSimplicialMap
    right_projection: TruncatedSimplicialMap

    @model_validator(mode="after")
    def require_axis_shapes(self) -> Self:
        product, left, right = self.simplicial_set, self.left, self.right
        if not (
            product.max_degree == left.max_degree == right.max_degree
            and len(self.pair_axes) == len(product.sets)
            and self.left_projection.source == product
            and self.right_projection.source == product
            and self.left_projection.target == left
            and self.right_projection.target == right
        ):
            raise ValueError("product factors and retained degree axes must agree")
        for degree, axis in enumerate(self.pair_axes):
            expected_axis = tuple(
                (i, j)
                for i in range(len(left.sets[degree]))
                for j in range(len(right.sets[degree]))
            )
            expected = len(expected_axis)
            if not (
                axis == expected_axis
                and len(product.sets[degree]) == expected
                and self.left_projection.maps[degree] == tuple(i for i, _ in axis)
                and self.right_projection.maps[degree] == tuple(j for _, j in axis)
            ):
                raise ValueError("product simplex axes do not match the factors")
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


__all__ = ["SimplicialSetProductRequest", "SimplicialSetProductResult"]
