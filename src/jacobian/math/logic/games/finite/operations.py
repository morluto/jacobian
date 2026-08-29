"""Exact native operations for finite deterministic terminal-payoff games."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import lcm

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.games.finite._models import (
    MAX_EXACT_EQUILIBRIUM_WORK,
    BestResponseResult,
    NashEquilibriumResult,
    PayoffMatrix,
)
from jacobian.math.logic.games.finite.values import (
    DeterministicTerminalGame,
    DeterministicTerminalGameSolution,
    PositionOwner,
    StationaryChoice,
    TerminalGameValueClass,
    _require_terminal_game_envelope,
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

    _run_admission(lambda: _require_terminal_game_envelope(game))
    value_classes, max_strategy, min_strategy = _solve_terminal_game_data(game)
    return DeterministicTerminalGameSolution._from_kernel(
        game, value_classes, max_strategy, min_strategy
    )


def _wire_rational(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(value)


def _payoff_matrix(matrix: PayoffMatrix) -> list[list[Fraction]]:
    entries = [entry.as_fraction() for entry in matrix.entries]
    return [
        [entries[row * matrix.n_cols + col] for col in range(matrix.n_cols)]
        for row in range(matrix.n_rows)
    ]


def _run_admission(admission: Callable[[], None]) -> None:
    """Translate owner admission failures to the native operation contract."""

    try:
        admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=(), code=exc.type, message=exc.message()
        ) from exc
    except (TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=(), code="finite_game.admission", message=str(exc)
        ) from exc


def best_response(payoff_matrix: PayoffMatrix) -> BestResponseResult:
    """Compute the maximin value and a maximizing row for the row player."""
    matrix = _payoff_matrix(payoff_matrix)
    best_row = 0
    best_value = min(matrix[0])
    for row_index, row in enumerate(matrix[1:], start=1):
        row_min = min(row)
        if row_min > best_value:
            best_value = row_min
            best_row = row_index
    return BestResponseResult._from_kernel(
        value=_wire_rational(best_value), best_row=best_row
    )


def nash_equilibrium(payoff_matrix: PayoffMatrix) -> NashEquilibriumResult:
    """Compute one exact saddle point of a finite 2-player zero-sum game."""

    matrix_value = payoff_matrix
    denominator_digits = sum(len(value.den) for value in matrix_value.entries)
    numerator_digits = max(len(value.num.lstrip("-")) for value in matrix_value.entries)
    elimination_dimension = max(matrix_value.n_rows, matrix_value.n_cols) + 2
    work = elimination_dimension * (denominator_digits + numerator_digits)
    if work > MAX_EXACT_EQUILIBRIUM_WORK:
        raise OperationDomainValidationError(
            location=("payoff_matrix",),
            code="finite_game.exact_equilibrium_budget",
            message="payoffs exceed the published exact-equilibrium work bound",
        )

    import sympy
    from sympy.solvers.simplex import lpmax, lpmin

    matrix = _payoff_matrix(payoff_matrix)
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    denominator_scale = lcm(*(value.denominator for row in matrix for value in row))
    integer_matrix = [
        [int(value * denominator_scale) for value in row] for row in matrix
    ]
    minimum_payoff = min(min(row) for row in integer_matrix)
    shift = max(0, 1 - minimum_payoff)
    shifted_matrix = [[value + shift for value in row] for row in integer_matrix]
    row_symbols = sympy.symbols(f"_row0:{n_rows}")
    column_symbols = sympy.symbols(f"_column0:{n_cols}")

    row_constraints = [symbol >= 0 for symbol in row_symbols]
    row_constraints.extend(
        sum(
            row_symbols[row] * sympy.Rational(shifted_matrix[row][column])
            for row in range(n_rows)
        )
        >= 1
        for column in range(n_cols)
    )
    row_total, row_solution = lpmin(sum(row_symbols), row_constraints)

    column_constraints = [symbol >= 0 for symbol in column_symbols]
    column_constraints.extend(
        sum(
            sympy.Rational(shifted_matrix[row][column]) * column_symbols[column]
            for column in range(n_cols)
        )
        <= 1
        for row in range(n_rows)
    )
    column_total, column_solution = lpmax(sum(column_symbols), column_constraints)
    if row_total != column_total or row_total <= 0:
        raise RuntimeError("exact primal and dual scaled game values disagree")

    row_scale = Fraction(row_total)
    column_scale = Fraction(column_total)
    row_strategy = [
        Fraction(row_solution.get(symbol, 0)) / row_scale for symbol in row_symbols
    ]
    column_strategy = [
        Fraction(column_solution.get(symbol, 0)) / column_scale
        for symbol in column_symbols
    ]
    value = (Fraction(1, 1) / row_scale - shift) / denominator_scale
    if sum(row_strategy) != 1 or any(weight < 0 for weight in row_strategy):
        raise RuntimeError("SymPy returned an invalid row strategy")
    if sum(column_strategy) != 1 or any(weight < 0 for weight in column_strategy):
        raise RuntimeError("SymPy returned an invalid column strategy")
    if any(
        sum(row_strategy[row] * matrix[row][column] for row in range(n_rows)) < value
        for column in range(n_cols)
    ):
        raise RuntimeError("row strategy does not attain the reported game value")
    if any(
        sum(matrix[row][column] * column_strategy[column] for column in range(n_cols))
        > value
        for row in range(n_rows)
    ):
        raise RuntimeError("column strategy does not attain the reported game value")
    return NashEquilibriumResult._from_kernel(
        row_strategy=tuple(_wire_rational(weight) for weight in row_strategy),
        col_strategy=tuple(_wire_rational(weight) for weight in column_strategy),
        value=_wire_rational(value),
    )


__all__ = ["best_response", "nash_equilibrium", "solve_terminal_game"]
