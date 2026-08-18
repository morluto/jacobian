"""Exact bounded native kernels for finite impartial games."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import xor

import networkx as nx

from jacobian.math.impartial_games.values import (
    MAX_HEAP_BOUND,
    MAX_HEAP_SIZE,
    MAX_HEAPS,
    MAX_MOVES,
    MAX_NIM_OPTIONS,
    MAX_SUBTRACTION_VALUE,
    MAX_SUBTRACTION_WORK,
    GameMove,
    ImpartialGame,
)


@dataclass(frozen=True, slots=True)
class GrundyAnalysis:
    values: tuple[tuple[str, int], ...]
    option_value_sets: tuple[tuple[str, tuple[int, ...]], ...]
    topological_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SubtractionGrundyAnalysis:
    subtraction_set: tuple[int, ...]
    max_heap: int
    grundy_values: tuple[int, ...]
    option_value_sets: tuple[tuple[int, ...], ...]


def mex(values: tuple[int, ...]) -> int:
    """Return the minimum excluded nonnegative integer."""

    if any(value < 0 for value in values):
        raise ValueError("mex values must be nonnegative")
    present = set(values)
    result = 0
    while result in present:
        result += 1
    return result


def grundy_table(game: ImpartialGame) -> GrundyAnalysis:
    """Return the complete exact Grundy analysis of one finite game DAG."""

    graph = _graph(game)
    topological_order = tuple(nx.lexicographical_topological_sort(graph))
    values: dict[str, int] = {}
    option_sets: dict[str, tuple[int, ...]] = {}
    for position in reversed(topological_order):
        option_set = tuple(
            sorted({values[target] for target in graph.successors(position)})
        )
        option_sets[position] = option_set
        values[position] = mex(option_set)
    return GrundyAnalysis(
        values=tuple((position, values[position]) for position in game.positions),
        option_value_sets=tuple(
            (position, option_sets[position]) for position in game.positions
        ),
        topological_order=topological_order,
    )


def birthdays(game: ImpartialGame) -> tuple[tuple[str, int], ...]:
    """Return every position birthday, with terminal positions at zero."""

    graph = _graph(game)
    order = tuple(nx.lexicographical_topological_sort(graph))
    result: dict[str, int] = {}
    for position in reversed(order):
        successor_birthdays = tuple(
            result[target] for target in graph.successors(position)
        )
        result[position] = (
            0 if not successor_birthdays else 1 + max(successor_birthdays)
        )
    return tuple((position, result[position]) for position in game.positions)


def position_grundy(game: ImpartialGame, position: str) -> int:
    if position not in game.positions:
        raise ValueError("position is not in the game")
    return dict(grundy_table(game).values)[position]


def outcome_profile(game: ImpartialGame) -> tuple[tuple[str, ...], tuple[str, ...]]:
    values = dict(grundy_table(game).values)
    return (
        tuple(position for position in game.positions if values[position] == 0),
        tuple(position for position in game.positions if values[position] != 0),
    )


def grundy_classes(game: ImpartialGame) -> tuple[tuple[int, tuple[str, ...]], ...]:
    classes: dict[int, list[str]] = {}
    for position, value in grundy_table(game).values:
        classes.setdefault(value, []).append(position)
    return tuple(
        (value, tuple(positions)) for value, positions in sorted(classes.items())
    )


def nim_sum(heaps: tuple[int, ...]) -> int:
    _validate_heaps(heaps)
    return reduce(xor, heaps, 0)


def nim_options(heaps: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    _validate_heaps(heaps)
    if sum(heaps) > MAX_NIM_OPTIONS:
        raise ValueError("nim option enumeration exceeds the output bound")
    return tuple(
        (*heaps[:index], new_size, *heaps[index + 1 :])
        for index, size in enumerate(heaps)
        for new_size in range(size)
    )


def subtraction_game(subtraction_set: tuple[int, ...], max_heap: int) -> ImpartialGame:
    values = _validate_subtraction_input(subtraction_set, max_heap)
    move_count = sum(
        sum_value <= heap for heap in range(max_heap + 1) for sum_value in values
    )
    if move_count > MAX_MOVES:
        raise ValueError("subtraction game DAG exceeds the move bound")
    positions = tuple(str(heap) for heap in range(max_heap + 1))
    moves = tuple(
        GameMove(source=str(heap), target=str(heap - subtraction))
        for heap in range(max_heap + 1)
        for subtraction in values
        if subtraction <= heap
    )
    return ImpartialGame(positions=positions, moves=moves)


def subtraction_grundy_prefix(
    subtraction_set: tuple[int, ...], max_heap: int
) -> SubtractionGrundyAnalysis:
    values = _validate_subtraction_input(subtraction_set, max_heap)
    if len(values) * (max_heap + 1) > MAX_SUBTRACTION_WORK:
        raise ValueError("subtraction Grundy computation exceeds the work bound")
    grundy = [0] * (max_heap + 1)
    option_sets: list[tuple[int, ...]] = []
    for heap in range(max_heap + 1):
        options = tuple(
            sorted(
                {
                    grundy[heap - subtraction]
                    for subtraction in values
                    if subtraction <= heap
                }
            )
        )
        option_sets.append(options)
        grundy[heap] = mex(options)
    return SubtractionGrundyAnalysis(
        subtraction_set=values,
        max_heap=max_heap,
        grundy_values=tuple(grundy),
        option_value_sets=tuple(option_sets),
    )


def _graph(game: ImpartialGame) -> nx.DiGraph[str]:
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from(game.positions)
    graph.add_edges_from((move.source, move.target) for move in game.moves)
    return graph


def _validate_heaps(heaps: tuple[int, ...]) -> None:
    if not 1 <= len(heaps) <= MAX_HEAPS:
        raise ValueError("heap tuple must contain between 1 and 50 heaps")
    if any(not 0 <= heap <= MAX_HEAP_SIZE for heap in heaps):
        raise ValueError("heap size is outside the supported bound")


def _validate_subtraction_input(
    subtraction_set: tuple[int, ...], max_heap: int
) -> tuple[int, ...]:
    if not 0 <= max_heap <= MAX_HEAP_BOUND:
        raise ValueError("maximum heap is outside the supported bound")
    if not subtraction_set or subtraction_set != tuple(sorted(set(subtraction_set))):
        raise ValueError("subtraction set must be nonempty, distinct, and sorted")
    if any(not 1 <= value <= MAX_SUBTRACTION_VALUE for value in subtraction_set):
        raise ValueError("subtraction value is outside the supported bound")
    return subtraction_set


__all__ = [
    "GrundyAnalysis",
    "SubtractionGrundyAnalysis",
    "birthdays",
    "grundy_classes",
    "grundy_table",
    "mex",
    "nim_options",
    "nim_sum",
    "outcome_profile",
    "position_grundy",
    "subtraction_game",
    "subtraction_grundy_prefix",
]
