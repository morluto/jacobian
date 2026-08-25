"""Provider-independent values for exact finite impartial games."""

from __future__ import annotations

from collections import deque
from typing import Annotated, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math._labels import MAX_OPAQUE_LABEL_LENGTH, OpaqueLabel

MAX_POSITIONS = 500
MAX_MOVES = 2_000
MAX_LABEL_LENGTH = MAX_OPAQUE_LABEL_LENGTH
MAX_HEAPS = 50
MAX_HEAP_SIZE = 10_000
MAX_NIM_RAW_CANDIDATES = MAX_HEAPS * MAX_HEAP_SIZE
MAX_NIM_DISTINCT_OPTIONS = 50_000
# Keep materialized results below the repository's 10 MiB canonical wire ceiling.
MAX_NIM_OPTION_RESULT_BYTES = 8 * 1024 * 1024
MAX_SUBTRACTION_VALUE = 500
MAX_HEAP_BOUND = 5_000
MAX_SUBTRACTION_WORK = 250_000

NimHeapSize = Annotated[
    StrictInt,
    Field(ge=0, le=MAX_HEAP_SIZE),
]
NimHeapIndex = Annotated[
    StrictInt,
    Field(ge=0, lt=MAX_HEAPS),
]


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
            raise PydanticCustomError(
                "impartial_game.positions_not_unique",
                "position labels must be distinct",
            )
        labels = set(self.positions)
        edge_pairs = tuple((move.source, move.target) for move in self.moves)
        if len(set(edge_pairs)) != len(edge_pairs):
            raise PydanticCustomError(
                "impartial_game.moves_not_unique", "game moves must be distinct"
            )
        if any(
            source not in labels or target not in labels
            for source, target in edge_pairs
        ):
            raise PydanticCustomError(
                "impartial_game.move_endpoint_unknown",
                "every move endpoint must be a declared position",
            )
        if any(source == target for source, target in edge_pairs):
            raise PydanticCustomError(
                "impartial_game.self_loop", "game moves cannot contain self-loops"
            )
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
            raise PydanticCustomError(
                "impartial_game.cyclic", "impartial game must be acyclic"
            )
        return self


class NimPosition(StrictModel):
    """One canonical multiset presentation of finite normal-play Nim heaps."""

    heaps: tuple[NimHeapSize, ...] = Field(
        max_length=MAX_HEAPS,
        description=(
            "Heap sizes in nondecreasing order. Zero heaps are retained as inert "
            "components, and duplicate sizes retain their multiplicity."
        ),
        examples=[[0, 1, 2, 2]],
    )

    @model_validator(mode="after")
    def require_canonical_bounded_heaps(self) -> Self:
        if self.heaps != tuple(sorted(self.heaps)):
            raise PydanticCustomError(
                "impartial_game.heaps_not_sorted",
                "Nim heaps must be in nondecreasing order",
            )
        return self


class NimOption(StrictModel):
    """One distinct one-heap reduction row with every indexed source witness.

    This is a context-dependent row of ``NimOptionsResult``, not a standalone
    canonical value: each row is validated against the request position by
    the result's exact reconstruction binding, so callers obtain rows only
    from that typed result boundary.
    """

    source_heap_indices: tuple[NimHeapIndex, ...] = Field(
        min_length=1,
        max_length=MAX_HEAPS,
        description=(
            "All indices in the retained canonical source position whose move "
            "produces this same multiset option."
        ),
    )
    source_heap_size: NimHeapSize
    replacement_heap_size: NimHeapSize
    resulting_position: NimPosition

    @model_validator(mode="after")
    def require_local_move_shape(self) -> Self:
        if self.source_heap_indices != tuple(sorted(set(self.source_heap_indices))):
            raise PydanticCustomError(
                "impartial_game.source_indices_not_canonical",
                "source heap indices must be distinct and sorted",
            )
        if self.source_heap_size == 0:
            raise PydanticCustomError(
                "impartial_game.zero_heap_move", "a zero heap has no legal Nim move"
            )
        if self.replacement_heap_size >= self.source_heap_size:
            raise PydanticCustomError(
                "impartial_game.move_not_reducing",
                "a Nim move must strictly reduce one heap",
            )
        return self


__all__ = [
    "MAX_HEAPS",
    "MAX_HEAP_BOUND",
    "MAX_HEAP_SIZE",
    "MAX_LABEL_LENGTH",
    "MAX_MOVES",
    "MAX_NIM_DISTINCT_OPTIONS",
    "MAX_NIM_OPTION_RESULT_BYTES",
    "MAX_NIM_RAW_CANDIDATES",
    "MAX_POSITIONS",
    "MAX_SUBTRACTION_VALUE",
    "MAX_SUBTRACTION_WORK",
    "GameMove",
    "ImpartialGame",
    "NimHeapIndex",
    "NimHeapSize",
    "NimPosition",
]
