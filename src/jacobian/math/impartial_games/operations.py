"""Exact bounded native kernels for finite impartial games."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from heapq import heapify, heappop, heappush
from operator import xor

from jacobian.math.impartial_games._nim_admission import heap_groups, nim_option_plan
from jacobian.math.impartial_games.values import (
    MAX_HEAP_BOUND,
    MAX_MOVES,
    MAX_SUBTRACTION_VALUE,
    MAX_SUBTRACTION_WORK,
    GameMove,
    ImpartialGame,
    NimOption,
    NimPosition,
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

    successors = _successors(game)
    topological_order = _lexicographical_topological_order(game, successors)
    values: dict[str, int] = {}
    option_sets: dict[str, tuple[int, ...]] = {}
    for position in reversed(topological_order):
        option_set = tuple(sorted({values[target] for target in successors[position]}))
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

    successors = _successors(game)
    order = _lexicographical_topological_order(game, successors)
    result: dict[str, int] = {}
    for position in reversed(order):
        successor_birthdays = tuple(result[target] for target in successors[position])
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


def nim_sum(position: NimPosition) -> int:
    """Return the exact normal-play Nim sum of one canonical position."""

    return reduce(xor, position.heaps, 0)


def nim_options(position: NimPosition) -> tuple[NimOption, ...]:
    """Return every distinct canonical one-move option and its source indices."""

    plan = nim_option_plan(position)
    options: list[NimOption] = []
    for source_size, source_indices in heap_groups(position):
        if source_size == 0:
            continue
        for replacement_size in range(source_size):
            resulting_heaps = list(position.heaps)
            resulting_heaps[source_indices[0]] = replacement_size
            options.append(
                NimOption(
                    source_heap_indices=source_indices,
                    source_heap_size=source_size,
                    replacement_heap_size=replacement_size,
                    resulting_position=NimPosition(
                        heaps=tuple(sorted(resulting_heaps))
                    ),
                )
            )
    canonical_options = tuple(
        sorted(options, key=lambda option: option.resulting_position.heaps)
    )
    distinct_results = {option.resulting_position.heaps for option in canonical_options}
    if (
        len(canonical_options) != plan.distinct_option_count
        or len(distinct_results) != plan.distinct_option_count
    ):
        raise RuntimeError("Nim option preflight disagrees with exact enumeration")
    return canonical_options


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


def _successors(game: ImpartialGame) -> dict[str, tuple[str, ...]]:
    successors: dict[str, list[str]] = {position: [] for position in game.positions}
    for move in game.moves:
        successors[move.source].append(move.target)
    return {
        position: tuple(sorted(targets)) for position, targets in successors.items()
    }


def _lexicographical_topological_order(
    game: ImpartialGame, successors: dict[str, tuple[str, ...]]
) -> tuple[str, ...]:
    indegree = dict.fromkeys(game.positions, 0)
    for targets in successors.values():
        for target in targets:
            indegree[target] += 1
    available = [position for position, degree in indegree.items() if degree == 0]
    heapify(available)
    order: list[str] = []
    while available:
        source = heappop(available)
        order.append(source)
        for target in successors[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heappush(available, target)
    if len(order) != len(game.positions):
        raise RuntimeError("validated impartial game unexpectedly contains a cycle")
    return tuple(order)


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
