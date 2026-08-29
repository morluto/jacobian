"""Typed contracts for the rainbow embedding profile operation."""

from jacobian._models import StrictModel
from jacobian.math.graphs.values import ColoredUndirectedGraph, SimpleUndirectedGraph

MAX_HOST_VERTICES = 16
MAX_PATTERN_VERTICES = 8


class RainbowEmbeddingRequest(StrictModel):
    """Request for the rainbow subgraph embedding profile."""

    pattern: SimpleUndirectedGraph
    host: ColoredUndirectedGraph


class EmbeddingWitness(StrictModel):
    """One rainbow embedding."""

    pattern_to_host: tuple[tuple[str, str], ...]
    edge_color_labels: tuple[str, ...]


class RainbowEmbeddingResult(StrictModel):
    """The complete rainbow subgraph embedding profile."""

    pattern: SimpleUndirectedGraph
    host: ColoredUndirectedGraph
    embeddings: tuple[EmbeddingWitness, ...]
    total_embeddings: int
    rainbow_count: int


__all__ = [
    "MAX_HOST_VERTICES",
    "MAX_PATTERN_VERTICES",
    "EmbeddingWitness",
    "RainbowEmbeddingRequest",
    "RainbowEmbeddingResult",
]
