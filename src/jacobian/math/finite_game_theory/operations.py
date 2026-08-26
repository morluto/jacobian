"""Exact native operations for finite deterministic terminal-payoff games."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.finite_game_theory.values import (
    DeterministicTerminalGame,
    DeterministicTerminalGameSolution,
    PositionOwner,
    StationaryChoice,
    TerminalGameValueClass,
)


@dataclass(frozen=True, slots=True)
class _Arena:
    labels: tuple[str, ...]
    owners: tuple[PositionOwner, ...]
    successors: tuple[tuple[int, ...], ...]
    predecessors: tuple[tuple[int, ...], ...]
    terminal_payoffs: tuple[Fraction | None, ...]
    draw_payoff: Fraction


@dataclass(frozen=True, slots=True)
class _ThresholdRegion:
    max_winning: frozenset[int]
    attractor: frozenset[int]
    ranks: tuple[int | None, ...]


def _arena(game: DeterministicTerminalGame) -> _Arena:
    labels = tuple(position.label for position in game.positions)
    index = {label: offset for offset, label in enumerate(labels)}
    successors: list[list[int]] = [[] for _ in labels]
    predecessors: list[list[int]] = [[] for _ in labels]
    for move in game.moves:
        source = index[move.source]
        target = index[move.target]
        successors[source].append(target)
        predecessors[target].append(source)
    return _Arena(
        labels=labels,
        owners=tuple(position.owner for position in game.positions),
        successors=tuple(tuple(targets) for targets in successors),
        predecessors=tuple(tuple(sources) for sources in predecessors),
        terminal_payoffs=tuple(
            position.payoff.as_fraction() if position.payoff is not None else None
            for position in game.positions
        ),
        draw_payoff=game.draw_payoff.as_fraction(),
    )


def _attractor(
    arena: _Arena,
    targets: tuple[int, ...],
    existential_owner: PositionOwner,
) -> tuple[frozenset[int], tuple[int | None, ...]]:
    """Return one reachability attractor and its strict progress ranks."""

    attracted = set(targets)
    ranks: list[int | None] = [None] * len(arena.labels)
    for target in targets:
        ranks[target] = 0

    remaining = [len(options) for options in arena.successors]
    maximum_seen_rank = [-1] * len(arena.labels)
    pending = deque(targets)
    while pending:
        target = pending.popleft()
        target_rank = ranks[target]
        if target_rank is None:  # pragma: no cover - construction invariant
            raise RuntimeError("attractor queue contains an unranked position")
        for source in arena.predecessors[target]:
            if source in attracted or arena.owners[source] == "TERMINAL":
                continue
            if arena.owners[source] == existential_owner:
                attracted.add(source)
                ranks[source] = target_rank + 1
                pending.append(source)
                continue
            remaining[source] -= 1
            maximum_seen_rank[source] = max(maximum_seen_rank[source], target_rank)
            if remaining[source] == 0:
                attracted.add(source)
                ranks[source] = maximum_seen_rank[source] + 1
                pending.append(source)
    return frozenset(attracted), tuple(ranks)


def _threshold_region(arena: _Arena, threshold: Fraction) -> _ThresholdRegion:
    vertices = frozenset(range(len(arena.labels)))
    if threshold > arena.draw_payoff:
        good_terminals = tuple(
            index
            for index, payoff in enumerate(arena.terminal_payoffs)
            if payoff is not None and payoff >= threshold
        )
        attractor, ranks = _attractor(arena, good_terminals, "MAX")
        return _ThresholdRegion(
            max_winning=attractor,
            attractor=attractor,
            ranks=ranks,
        )

    bad_terminals = tuple(
        index
        for index, payoff in enumerate(arena.terminal_payoffs)
        if payoff is not None and payoff < threshold
    )
    attractor, ranks = _attractor(arena, bad_terminals, "MIN")
    return _ThresholdRegion(
        max_winning=vertices - attractor,
        attractor=attractor,
        ranks=ranks,
    )


def _progress_successor(
    successors: tuple[int, ...],
    ranks: tuple[int | None, ...],
    source: int,
) -> int:
    source_rank = ranks[source]
    if source_rank is None:  # pragma: no cover - construction invariant
        raise RuntimeError("an attracted player position must have a rank")
    candidates: list[int] = []
    for target in successors:
        target_rank = ranks[target]
        if target_rank is not None and target_rank < source_rank:
            candidates.append(target)
    if not candidates:  # pragma: no cover - attractor invariant
        raise RuntimeError("an attracted player position must have a progress move")
    return min(candidates)


def _safe_successor(successors: tuple[int, ...], unsafe: frozenset[int]) -> int:
    candidates = tuple(target for target in successors if target not in unsafe)
    if not candidates:  # pragma: no cover - complement-attractor invariant
        raise RuntimeError("a safety-winning player position must have a safe move")
    return min(candidates)


def _solve_terminal_game_data(
    game: DeterministicTerminalGame,
) -> tuple[
    tuple[TerminalGameValueClass, ...],
    tuple[StationaryChoice, ...],
    tuple[StationaryChoice, ...],
]:
    """Return the canonical all-position value profile and optimal strategies.

    For a threshold above the infinite-play payoff, MAX must reach a terminal
    at or above that threshold. At or below the infinite-play payoff, MAX must
    avoid terminals below the threshold. Standard reachability attractors solve
    both objectives, and the greatest won threshold is the exact game value.
    """

    arena = _arena(game)
    levels = tuple(
        sorted(
            {
                arena.draw_payoff,
                *(payoff for payoff in arena.terminal_payoffs if payoff is not None),
            }
        )
    )
    regions = {level: _threshold_region(arena, level) for level in levels}
    values = [levels[0]] * len(arena.labels)
    for level in levels:
        for vertex in regions[level].max_winning:
            values[vertex] = level

    level_indices = {level: index for index, level in enumerate(levels)}
    max_choices: list[StationaryChoice] = []
    min_choices: list[StationaryChoice] = []
    for vertex, owner in enumerate(arena.owners):
        if owner == "TERMINAL":
            continue
        value = values[vertex]
        if owner == "MAX":
            region = regions[value]
            target = (
                _progress_successor(arena.successors[vertex], region.ranks, vertex)
                if value > arena.draw_payoff
                else _safe_successor(arena.successors[vertex], region.attractor)
            )
            max_choices.append(
                StationaryChoice(
                    position=arena.labels[vertex], target=arena.labels[target]
                )
            )
            continue

        level_index = level_indices[value]
        if value < arena.draw_payoff:
            next_region = regions[levels[level_index + 1]]
            target = _progress_successor(
                arena.successors[vertex], next_region.ranks, vertex
            )
        elif level_index + 1 < len(levels):
            next_region = regions[levels[level_index + 1]]
            target = _safe_successor(arena.successors[vertex], next_region.attractor)
        else:
            target = min(arena.successors[vertex])
        min_choices.append(
            StationaryChoice(position=arena.labels[vertex], target=arena.labels[target])
        )

    class_members: dict[Fraction, list[str]] = {level: [] for level in levels}
    for label, value in zip(arena.labels, values, strict=True):
        class_members[value].append(label)
    value_classes = tuple(
        TerminalGameValueClass(
            payoff=CanonicalRational.from_fraction(level),
            positions=tuple(class_members[level]),
        )
        for level in levels
        if class_members[level]
    )
    return value_classes, tuple(max_choices), tuple(min_choices)


def solve_terminal_game(
    game: DeterministicTerminalGame,
) -> DeterministicTerminalGameSolution:
    """Solve one exact finite deterministic terminal-payoff game."""

    value_classes, max_strategy, min_strategy = _solve_terminal_game_data(game)
    return DeterministicTerminalGameSolution._from_kernel(
        game, value_classes, max_strategy, min_strategy
    )


__all__ = ["solve_terminal_game"]
