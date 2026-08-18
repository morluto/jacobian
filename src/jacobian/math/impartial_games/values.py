"""Provider-independent values for exact finite impartial games."""

from __future__ import annotations

from typing import Self

import networkx as nx
from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_POSITIONS = 500
MAX_MOVES = 2_000
MAX_LABEL_LENGTH = 64
MAX_HEAPS = 50
MAX_HEAP_SIZE = 10_000
MAX_NIM_OPTIONS = 5_000
MAX_SUBTRACTION_VALUE = 500
MAX_HEAP_BOUND = 5_000
MAX_SUBTRACTION_WORK = 250_000


class GameMove(StrictModel):
    source: str = Field(min_length=1, max_length=MAX_LABEL_LENGTH)
    target: str = Field(min_length=1, max_length=MAX_LABEL_LENGTH)


class ImpartialGame(StrictModel):
    """A complete finite normal-play impartial game DAG."""

    positions: tuple[str, ...] = Field(min_length=1, max_length=MAX_POSITIONS)
    moves: tuple[GameMove, ...] = Field(max_length=MAX_MOVES)

    @model_validator(mode="after")
    def require_finite_dag(self) -> Self:
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("position labels must be distinct")
        labels = set(self.positions)
        edge_pairs = tuple((move.source, move.target) for move in self.moves)
        if len(set(edge_pairs)) != len(edge_pairs):
            raise ValueError("game moves must be distinct")
        if any(
            source not in labels or target not in labels
            for source, target in edge_pairs
        ):
            raise ValueError("every move endpoint must be a declared position")
        if any(source == target for source, target in edge_pairs):
            raise ValueError("game moves cannot contain self-loops")
        graph: nx.DiGraph[str] = nx.DiGraph()
        graph.add_nodes_from(self.positions)
        graph.add_edges_from(edge_pairs)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("impartial game must be acyclic")
        return self


__all__ = [
    "MAX_HEAPS",
    "MAX_HEAP_BOUND",
    "MAX_HEAP_SIZE",
    "MAX_LABEL_LENGTH",
    "MAX_MOVES",
    "MAX_NIM_OPTIONS",
    "MAX_POSITIONS",
    "MAX_SUBTRACTION_VALUE",
    "MAX_SUBTRACTION_WORK",
    "GameMove",
    "ImpartialGame",
]
