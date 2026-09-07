"""Complete connected colored graph families with positive edge-type costs."""

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.values import ColoredUndirectedGraph, GraphColor


class ColorPairCost(StrictModel):
    colors: tuple[GraphColor, GraphColor]
    cost: int = Field(ge=1, le=1_000_000)


class _EnumerationParameters(StrictModel):
    """Enumerate connected graphs with 2..vertex_bound vertices (no singletons).

    Palette names are distinct and sorted; allowed unordered color pairs are
    sorted and unique. Missing pairs are forbidden. Colors may be absent from
    a graph. The maintained complete graph atlas supports vertex_bound <= 7.
    """

    palette: tuple[GraphColor, ...] = Field(min_length=1, max_length=8)
    edge_costs: tuple[ColorPairCost, ...] = Field(max_length=36)
    vertex_bound: int = Field(ge=2, le=7)
    cost_bound: int = Field(ge=0, le=1_000_000)

    @model_validator(mode="after")
    def require_palette_and_pairs(self) -> Self:
        if self.palette != tuple(sorted(set(self.palette))):
            raise ValueError("palette must be distinct and increasing")
        pairs = tuple(item.colors for item in self.edge_costs)
        if pairs != tuple(sorted(set(pairs))) or any(
            a > b or a not in self.palette or b not in self.palette for a, b in pairs
        ):
            raise ValueError(
                "allowed color pairs must be distinct, sorted unordered pairs from the palette"
            )
        return self


class ConnectedColoredGraphsRequest(_EnumerationParameters):
    """Wire parameters for connected colored graph enumeration."""


class ConnectedColoredGraphFamily(_EnumerationParameters):
    """One representative per color-preserving isomorphism class, without truncation."""

    graphs: tuple[ColoredUndirectedGraph, ...] = Field(max_length=20_000)
