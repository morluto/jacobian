"""Provider-independent values for exact finite impartial games."""

from __future__ import annotations

from collections import deque
from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math._labels import MAX_OPAQUE_LABEL_LENGTH, OpaqueLabel

MAX_POSITIONS = 500
MAX_MOVES = 2_000
MAX_LABEL_LENGTH = MAX_OPAQUE_LABEL_LENGTH
MAX_HEAPS = 50
MAX_HEAP_SIZE = 10_000
MAX_NIM_OPTIONS = 5_000
MAX_SUBTRACTION_VALUE = 500
MAX_HEAP_BOUND = 5_000
MAX_SUBTRACTION_WORK = 250_000


class GameMove(StrictModel):
    source: OpaqueLabel
    target: OpaqueLabel


class ImpartialGame(StrictModel):
    """A complete finite normal-play impartial game DAG."""

    positions: tuple[OpaqueLabel, ...] = Field(min_length=1, max_length=MAX_POSITIONS)
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
        successors: dict[str, list[str]] = {position: [] for position in self.positions}
        indegree = dict.fromkeys(self.positions, 0)
        for source, target in edge_pairs:
            successors[source].append(target)
            indegree[target] += 1
        queue = deque(
            position for position in self.positions if indegree[position] == 0
        )
        visited = 0
        while queue:
            source = queue.popleft()
            visited += 1
            for target in successors[source]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if visited != len(self.positions):
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
