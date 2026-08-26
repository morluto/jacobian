"""Wire adapters for exact bounded impartial-game operations."""

from jacobian.math.impartial_games._models import (
    BirthdayRequest,
    BirthdayResult,
    DisjunctiveSumRequest,
    DisjunctiveSumResult,
    GrundyEntry,
    GrundyTableRequest,
    GrundyTableResult,
    NimOptionsRequest,
    NimOptionsResult,
    NimSumRequest,
    NimSumResult,
    OutcomeProfileRequest,
    OutcomeProfileResult,
    SubtractionGrundyPrefixRequest,
    SubtractionGrundyPrefixResult,
)
from jacobian.math.impartial_games._nim_admission import nim_option_plan
from jacobian.math.impartial_games.operations import (
    birthdays,
    grundy_table,
    nim_options,
    nim_sum,
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
    return GrundyTableResult._from_kernel(
        request,
        entries,
        maximum,
        tuple(values.count(index) for index in range(maximum + 1)),
        analysis.topological_order,
    )


def compute_birthday(request: BirthdayRequest) -> BirthdayResult:
    return BirthdayResult._from_kernel(request, birthdays(request.game))


def compute_subtraction_grundy_prefix(
    request: SubtractionGrundyPrefixRequest,
) -> SubtractionGrundyPrefixResult:
    analysis = subtraction_grundy_prefix(request.subtraction_set, request.max_heap)
    return SubtractionGrundyPrefixResult._from_kernel(
        request,
        analysis.grundy_values,
        analysis.option_value_sets,
        tuple(heap for heap, value in enumerate(analysis.grundy_values) if value == 0),
        tuple(heap for heap, value in enumerate(analysis.grundy_values) if value != 0),
    )


__all__ = [
    "compute_birthday",
    "compute_disjunctive_sum",
    "compute_grundy_table",
    "compute_nim_options",
    "compute_subtraction_grundy_prefix",
    "verify_birthday_result",
    "verify_disjunctive_sum_result",
    "verify_grundy_table_result",
    "verify_nim_options_result",
    "verify_nim_sum_result",
    "verify_outcome_profile_result",
    "verify_subtraction_grundy_prefix_result",
]


def compute_nim_sum(
    request: NimSumRequest,
) -> NimSumResult:
    """Compute the exact nim sum (bitwise xor) of heap sizes."""

    value = nim_sum(request.position)
    return NimSumResult._from_kernel(request, value)


def compute_nim_options(
    request: NimOptionsRequest,
) -> NimOptionsResult:
    """Enumerate the complete deduplicated option family of a Nim position."""

    options = nim_options(request.position)
    plan = nim_option_plan(request.position)
    return NimOptionsResult._from_kernel(
        request, options, plan.raw_candidate_count, plan.distinct_option_count
    )


def compute_outcome_profile(
    request: OutcomeProfileRequest,
) -> OutcomeProfileResult:
    """Compute the P/N outcome partition of an impartial game."""

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


def compute_disjunctive_sum(
    request: "DisjunctiveSumRequest",
) -> "DisjunctiveSumResult":
    """Compute the Grundy value of a disjunctive sum of impartial games.

    The Grundy value of the disjunctive sum is the bitwise XOR of the
    component Grundy values (the Grundy value of each component's
    start position).
    """
    from functools import reduce
    from operator import xor

    component_grundy_values = []
    for game, start in zip(request.components, request.start_positions, strict=True):
        analysis = grundy_table(game)
        grundy_map = dict(analysis.values)
        component_grundy_values.append(grundy_map[start])
    nim_sum = reduce(xor, component_grundy_values, 0)
    return DisjunctiveSumResult(
        grundy_value=nim_sum,
        component_grundy_values=tuple(component_grundy_values),
        is_p_position=(nim_sum == 0),
        component_count=len(request.components),
    )


def verify_grundy_table_result(result: GrundyTableResult) -> bool:
    """Replay one bounded externally supplied complete Grundy-table claim."""

    analysis = grundy_table(result.game)
    option_sets = dict(analysis.option_value_sets)
    entries = tuple(
        GrundyEntry(
            position=position,
            grundy=value,
            option_grundy_set=option_sets[position],
        )
        for position, value in analysis.values
    )
    values = tuple(value for _, value in analysis.values)
    maximum = max(values, default=0)
    return (
        result.entries == entries
        and result.max_grundy == maximum
        and result.histogram
        == tuple(values.count(index) for index in range(maximum + 1))
        and result.topological_order == analysis.topological_order
    )


def verify_birthday_result(result: BirthdayResult) -> bool:
    """Replay one bounded externally supplied birthday-table claim."""

    return result.birthdays == birthdays(result.game)


def verify_subtraction_grundy_prefix_result(
    result: SubtractionGrundyPrefixResult,
) -> bool:
    """Replay one bounded externally supplied subtraction-prefix claim."""

    analysis = subtraction_grundy_prefix(result.subtraction_set, result.max_heap)
    return (
        result.grundy_values == analysis.grundy_values
        and result.option_sets == analysis.option_value_sets
        and result.p_positions
        == tuple(
            heap for heap, value in enumerate(analysis.grundy_values) if value == 0
        )
        and result.n_positions
        == tuple(
            heap for heap, value in enumerate(analysis.grundy_values) if value != 0
        )
    )


def verify_nim_sum_result(result: NimSumResult) -> bool:
    """Replay one bounded externally supplied Nim-sum claim."""

    return result.nim_sum == nim_sum(result.position) and result.is_p_position == (
        result.nim_sum == 0
    )


def verify_nim_options_result(result: NimOptionsResult) -> bool:
    """Replay one bounded externally supplied complete Nim-options claim."""

    plan = nim_option_plan(result.position)
    return (
        result.options == nim_options(result.position)
        and result.raw_candidate_count == plan.raw_candidate_count
        and result.distinct_option_count == plan.distinct_option_count
    )


def verify_outcome_profile_result(
    request: OutcomeProfileRequest, result: OutcomeProfileResult
) -> bool:
    """Replay one bounded externally supplied impartial-game outcome claim."""

    analysis = grundy_table(request.game)
    return (
        result.p_positions == tuple(pos for pos, value in analysis.values if value == 0)
        and result.n_positions
        == tuple(pos for pos, value in analysis.values if value > 0)
        and result.grundy_values == analysis.values
        and result.terminal_positions
        == tuple(
            pos
            for pos in request.game.positions
            if not any(move.source == pos for move in request.game.moves)
        )
    )


def verify_disjunctive_sum_result(
    request: DisjunctiveSumRequest, result: DisjunctiveSumResult
) -> bool:
    """Replay one bounded externally supplied disjunctive-sum claim."""

    values = tuple(
        dict(grundy_table(game).values)[start]
        for game, start in zip(request.components, request.start_positions, strict=True)
    )
    from functools import reduce
    from operator import xor

    value = reduce(xor, values, 0)
    return (
        result.component_grundy_values == values
        and result.component_count == len(values)
        and result.grundy_value == value
        and result.is_p_position == (value == 0)
    )
