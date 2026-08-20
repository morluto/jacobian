"""Wire adapters for exact bounded impartial-game operations."""

from jacobian.math.impartial_games._models import (
    BirthdayRequest,
    BirthdayResult,
    GrundyEntry,
    GrundyTableRequest,
    GrundyTableResult,
    NimSumRequest,
    NimSumResult,
    OutcomeProfileRequest,
    OutcomeProfileResult,
    SubtractionGrundyPrefixRequest,
    SubtractionGrundyPrefixResult,
)
from jacobian.math.impartial_games.operations import (
    birthdays,
    grundy_table,
    subtraction_grundy_prefix,
)


def compute_grundy_table(request: GrundyTableRequest) -> GrundyTableResult:
    analysis = grundy_table(request.game)
    option_sets = dict(analysis.option_value_sets)
    entries = tuple(
        GrundyEntry(
            position=position,
            grundy=value,
            option_grundy_set=option_sets[position],
        )
        for position, value in analysis.values
    )
    values = tuple(entry.grundy for entry in entries)
    maximum = max(values, default=0)
    return GrundyTableResult(
        game=request.game,
        entries=entries,
        max_grundy=maximum,
        histogram=tuple(values.count(index) for index in range(maximum + 1)),
        topological_order=analysis.topological_order,
    )


def compute_birthday(request: BirthdayRequest) -> BirthdayResult:
    return BirthdayResult(game=request.game, birthdays=birthdays(request.game))


def compute_subtraction_grundy_prefix(
    request: SubtractionGrundyPrefixRequest,
) -> SubtractionGrundyPrefixResult:
    analysis = subtraction_grundy_prefix(request.subtraction_set, request.max_heap)
    return SubtractionGrundyPrefixResult(
        subtraction_set=request.subtraction_set,
        max_heap=request.max_heap,
        grundy_values=analysis.grundy_values,
        option_sets=analysis.option_value_sets,
        p_positions=tuple(
            heap for heap, value in enumerate(analysis.grundy_values) if value == 0
        ),
        n_positions=tuple(
            heap for heap, value in enumerate(analysis.grundy_values) if value != 0
        ),
    )


__all__ = [
    "compute_birthday",
    "compute_grundy_table",
    "compute_subtraction_grundy_prefix",
]


def compute_nim_sum(
    request: NimSumRequest,
) -> NimSumResult:
    """Compute the exact nim sum (bitwise xor) of heap sizes."""

    from functools import reduce
    from operator import xor

    heaps = request.heaps
    nim_sum = 0 if not heaps else reduce(xor, heaps)
    return NimSumResult(
        nim_sum=nim_sum,
        is_p_position=(nim_sum == 0),
        heaps=heaps,
    )


def compute_outcome_profile(
    request: OutcomeProfileRequest,
) -> OutcomeProfileResult:
    """Compute the P/N outcome partition of an impartial game."""

    from jacobian.math.impartial_games.operations import grundy_table

    analysis = grundy_table(request.game)
    p_positions = tuple(pos for pos, g in analysis.values if g == 0)
    n_positions = tuple(pos for pos, g in analysis.values if g > 0)
    terminal_positions = tuple(
        pos
        for pos in request.game.positions
        if not any(m.source == pos for m in request.game.moves)
    )
    return OutcomeProfileResult(
        p_positions=p_positions,
        n_positions=n_positions,
        grundy_values=analysis.values,
        terminal_positions=terminal_positions,
    )
