"""Provider-independent exact values for finite topological spaces."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"finite_topology.{reason}", message)


class FiniteTopology(StrictModel):
    """A topology on the labelled carrier ``0..point_count-1``."""

    point_count: int = Field(ge=1)
    open_sets: tuple[tuple[int, ...], ...] = Field(min_length=2)

    @model_validator(mode="after")
    def require_topology_axioms(self) -> Self:
        canonical: list[frozenset[int]] = []
        for open_set in self.open_sets:
            if tuple(sorted(set(open_set))) != open_set:
                raise _validation_error(
                    "open_set_not_canonical",
                    "each open set must be sorted with distinct points",
                )
            if any(not 0 <= point < self.point_count for point in open_set):
                raise _validation_error(
                    "open_set_point_out_of_range",
                    "open set point is outside the carrier",
                )
            canonical.append(frozenset(open_set))
        opens = set(canonical)
        if len(opens) != len(canonical):
            raise _validation_error(
                "open_sets_not_distinct", "open sets must be distinct"
            )
        full = frozenset(range(self.point_count))
        if frozenset() not in opens or full not in opens:
            raise _validation_error(
                "missing_extreme_open_sets", "empty and full sets must be open"
            )
        for left_index, left in enumerate(canonical):
            for right in canonical[left_index:]:
                if left | right not in opens:
                    raise _validation_error(
                        "not_closed_under_unions",
                        "open sets must be closed under unions",
                    )
                if left & right not in opens:
                    raise _validation_error(
                        "not_closed_under_intersections",
                        "open sets must be closed under intersections",
                    )
        return self


class PointMap(StrictModel):
    """A total map between labelled finite carriers."""

    domain_point_count: int = Field(ge=1)
    codomain_point_count: int = Field(ge=1)
    values: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_total_bounded_map(self) -> Self:
        if len(self.values) != self.domain_point_count:
            raise _validation_error(
                "map_length_mismatch", "map must have one value per domain point"
            )
        if any(not 0 <= target < self.codomain_point_count for target in self.values):
            raise _validation_error(
                "map_value_out_of_range", "map value is outside the codomain carrier"
            )
        return self


class BeatPointWitness(StrictModel):
    point: int = Field(ge=0)
    witness: int = Field(ge=0)


__all__ = [
    "BeatPointWitness",
    "FiniteTopology",
    "PointMap",
]
