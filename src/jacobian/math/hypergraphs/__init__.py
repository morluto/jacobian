"""Supported native finite-hypergraph API."""

from jacobian.math.hypergraphs._models import (
    EdgeIntersectionEntry,
    EdgeIntersectionsResult,
    FiniteHypergraph,
)
from jacobian.math.hypergraphs._operations import edge_intersections

__all__ = [
    "EdgeIntersectionEntry",
    "EdgeIntersectionsResult",
    "FiniteHypergraph",
    "edge_intersections",
]
