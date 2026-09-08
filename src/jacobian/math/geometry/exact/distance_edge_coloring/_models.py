"""Exact distance palettes on labelled complete graphs."""

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs.colorings import (
    IndexedHyperedgeColoring,
)
from jacobian.math.geometry.exact._models import MAX_PAIRS, PointConfiguration


class DistanceEdgeColoringRequest(StrictModel):
    configuration: PointConfiguration


class DistanceEdgeColoringResult(StrictModel):
    """The complete labelled 2-uniform graph, coloured by squared distance.

    The sole graph carrier is ``coloring.hypergraph``. Its vertex axis is the
    source point order, and edge ``i:j`` joins source positions i<j. Edges occur
    in lexicographic position-pair order. ``squared_distances[color_index]`` is
    the exact colour value. Parsing requires the palette to be strictly
    increasing and duplicate-free; it does not recompute distances.
    """

    configuration: PointConfiguration
    squared_distances: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_PAIRS
    )
    coloring: IndexedHyperedgeColoring

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        labels = tuple(point.label for point in self.configuration.points)
        graph = self.coloring.hypergraph
        if graph.vertices != labels:
            raise ValueError("graph vertices must retain the source point axis")
        expected = tuple(
            (f"{i}:{j}", tuple(sorted((labels[i], labels[j]))))
            for i in range(len(labels))
            for j in range(i + 1, len(labels))
        )
        if graph.edges != expected:
            raise ValueError("graph edges must be the complete canonical source pairs")
        if self.coloring.color_count != len(self.squared_distances):
            raise ValueError(
                "colour count must equal the squared-distance palette size"
            )
        palette = tuple(value.as_fraction() for value in self.squared_distances)
        if palette != tuple(sorted(set(palette))):
            raise ValueError(
                "squared-distance palette must be strictly increasing and duplicate-free"
            )
        return self
