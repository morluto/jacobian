"""Typed wire contracts for structural graph decomposition operations.

All operations in this module act on an undirected simple graph supplied as
a vertex count and a tuple of ``(source, target)`` integer edges.  Vertices
are labelled ``0..vertex_count-1``; the maximum vertex count is 64 and the
maximum edge count is 512, matching the rest of the graph domains.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel


class UndirectedGraph(StrictModel):
    """A simple undirected graph for decomposition operations."""

    vertex_count: int = Field(ge=1, le=64)
    edges: tuple[tuple[int, int], ...] = Field(min_length=0, max_length=512)

    @model_validator(mode="after")
    def require_valid_edges(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for source, target in self.edges:
            if not (
                0 <= source < self.vertex_count and 0 <= target < self.vertex_count
            ):
                raise ValueError("edge vertices must be in 0..vertex_count-1")
            if source == target:
                raise ValueError("self-loops are not allowed")
            endpoint_pair = (source, target)
            canonical = (min(endpoint_pair), max(endpoint_pair))
            if canonical in seen:
                raise ValueError("undirected edges must be unique")
            seen.add(canonical)
        return self


class BlockCutTreeRequest(StrictModel):
    graph: UndirectedGraph


class BlockCutTreeResult(StrictModel):
    """Block-cut tree decomposition of a graph.

    ``blocks`` lists the biconnected components (each a sorted tuple of
    vertices), ``articulation_points`` lists the cut vertices, and ``tree``
    lists the edges of the bipartite block-cut tree joining each block to the
    articulation points it contains.
    """

    blocks: tuple[tuple[int, ...], ...] = Field(default=())
    articulation_points: tuple[int, ...] = Field(default=())
    tree: tuple[tuple[int, int], ...] = Field(default=())
    convention: Literal["NETWORKX_BICONNECTED"] = "NETWORKX_BICONNECTED"


class BridgeBlockRequest(StrictModel):
    graph: UndirectedGraph


class BridgeBlockResult(StrictModel):
    """Bridge-block (2-edge-connected component) decomposition of a graph.

    ``components`` lists each 2-edge-connected component as a sorted tuple of
    vertices, ``bridges`` lists the bridges as normalised ``(u, v)`` pairs, and
    ``tree`` lists the edges of the bridge block tree joining adjacent
    components across each bridge.
    """

    components: tuple[tuple[int, ...], ...] = Field(default=())
    bridges: tuple[tuple[int, int], ...] = Field(default=())
    tree: tuple[tuple[int, int], ...] = Field(default=())
    convention: Literal["NETWORKX_BRIDGES"] = "NETWORKX_BRIDGES"


class EarDecompositionRequest(StrictModel):
    graph: UndirectedGraph


class EarDecompositionResult(StrictModel):
    """Open ear decomposition of a biconnected graph.

    Each ear is a tuple of vertices describing a path whose internal vertex
    is disjoint from all other ears.  The first ear is a cycle.  Graphs with
    fewer than three vertices use the explicit cycle-free convention
    ``biconnected=true, ears=()``.  A graph that is not biconnected is a typed
    ``biconnected=false`` outcome.
    """

    biconnected: bool = True
    ears: tuple[tuple[int, ...], ...] = Field(default=())
    convention: Literal["JACOBIAN_EAR_DECOMPOSITION"] = "JACOBIAN_EAR_DECOMPOSITION"

    @model_validator(mode="after")
    def require_ears_match_biconnectivity(self) -> Self:
        if not self.biconnected and self.ears:
            raise ValueError("a non-biconnected graph must not report ears")
        return self


class BiconnectedComponentsRequest(StrictModel):
    graph: UndirectedGraph


class BiconnectedComponentsResult(StrictModel):
    """All biconnected components of a graph, each a sorted tuple of vertices."""

    components: tuple[tuple[int, ...], ...] = Field(default=())
    convention: Literal["NETWORKX_BICONNECTED"] = "NETWORKX_BICONNECTED"
