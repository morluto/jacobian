"""Impartial-game operation declarations."""

from typing import Any

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.logic.games.impartial._models import (
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
from jacobian.math.logic.games.impartial.operations import (
    _disjunctive_sum_result,
    _outcome_profile_result,
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


def compute_nim_sum(request: NimSumRequest) -> NimSumResult:
    """Compute the exact nim sum (bitwise xor) of a Nim position."""

    return NimSumResult._from_kernel(request, nim_sum(request.position))


def compute_nim_options(request: NimOptionsRequest) -> NimOptionsResult:
    """Enumerate the complete deduplicated option family of a Nim position."""

    try:
        options = nim_options(request.position)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("position",),
            code="impartial_game.nim_options_not_admitted",
            message=str(exc),
        ) from exc
    return NimOptionsResult._from_kernel(
        request, options, sum(request.position.heaps), len(options)
    )


def compute_outcome_profile(request: OutcomeProfileRequest) -> OutcomeProfileResult:
    """Compute the P/N outcome partition of an impartial game."""

    return _outcome_profile_result(request.game)


def compute_disjunctive_sum(
    request: DisjunctiveSumRequest,
) -> DisjunctiveSumResult:
    """Compute the Grundy value of a disjunctive sum."""

    return _disjunctive_sum_result(request.components, request.start_positions)


_GAME = {
    "positions": ["0", "1", "2", "3"],
    "moves": [
        {"source": "3", "target": "2"},
        {"source": "3", "target": "1"},
        {"source": "2", "target": "1"},
        {"source": "2", "target": "0"},
        {"source": "1", "target": "0"},
    ],
}

_GAME_A = {
    "positions": ["a", "b"],
    "moves": [{"source": "a", "target": "b"}],
}

_GAME_B = {
    "positions": ["c", "d", "e"],
    "moves": [
        {"source": "c", "target": "d"},
        {"source": "d", "target": "e"},
    ],
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="game.impartial.grundy_table.compute",
        title="Compute a complete Grundy table",
        description="Compute every exact Grundy value and canonical option-value set for a "
        "bounded finite normal-play impartial game DAG.",
        request_type=GrundyTableRequest,
        result_type=GrundyTableResult,
        run=compute_grundy_table,
        tags=("game-theory", "impartial", "grundy", "exact"),
        examples=(
            OperationExample(
                name="four_position_game",
                description="Compute the complete Grundy table of a four-position DAG.",
                input={"game": _GAME},
            ),
        ),
    ),
    MathTool(
        operation_id="game.impartial.birthday.compute",
        title="Compute all position birthdays",
        description="Compute the exact DAG height of every position, with terminals at zero.",
        request_type=BirthdayRequest,
        result_type=BirthdayResult,
        run=compute_birthday,
        tags=("game-theory", "impartial", "birthday", "exact"),
        examples=(
            OperationExample(
                name="four_position_birthdays",
                description="Compute every birthday in a four-position game DAG.",
                input={"game": _GAME},
            ),
        ),
    ),
    MathTool(
        operation_id="game.subtraction.grundy_prefix.compute",
        title="Compute a bounded subtraction-game Grundy prefix",
        description="Compute exact Grundy values and canonical option-value sets for every "
        "heap from zero through the explicit maximum; no periodicity is implied.",
        request_type=SubtractionGrundyPrefixRequest,
        result_type=SubtractionGrundyPrefixResult,
        run=compute_subtraction_grundy_prefix,
        tags=("game-theory", "subtraction", "grundy", "exact"),
        examples=(
            OperationExample(
                name="subtract_one_or_three",
                description="Compute heaps zero through five for subtraction set {1,3}.",
                input={"subtraction_set": [1, 3], "max_heap": 5},
            ),
        ),
    ),
    MathTool(
        operation_id="game.nim.nim_sum.compute",
        title="Compute the nim sum of a Nim position",
        description="Compute the exact bitwise xor of a canonical sorted heap multiset, "
        "determining the P/N outcome under normal play.",
        request_type=NimSumRequest,
        result_type=NimSumResult,
        run=compute_nim_sum,
        tags=("game-theory", "nim", "exact"),
        examples=(
            OperationExample(
                name="nim_sum_1_2_3",
                description="Compute the nim sum of heaps (1, 2, 3); "
                "heaps must be nonnegative integers in nondecreasing order.",
                input={"position": {"heaps": [1, 2, 3]}},
            ),
        ),
    ),
    MathTool(
        operation_id="game.nim.options.compute",
        title="Enumerate every distinct legal Nim option",
        description="Return the complete canonical one-move option family of a sorted Nim "
        "heap multiset, retaining every source heap index collapsed by "
        "multiset deduplication.",
        request_type=NimOptionsRequest,
        result_type=NimOptionsResult,
        run=compute_nim_options,
        tags=("game-theory", "nim", "options", "exact"),
        examples=(
            OperationExample(
                name="deduplicated_equal_heaps",
                description="Enumerate every distinct option of Nim heaps (1,2,2); heaps "
                "must be nondecreasing, and zero heaps are retained.",
                input={"position": {"heaps": [1, 2, 2]}},
            ),
        ),
    ),
    MathTool(
        operation_id="game.impartial.outcome_profile.compute",
        title="Compute the P/N outcome partition",
        description="Partition positions into P-positions (Grundy=0, previous player "
        "wins) and N-positions (Grundy>0, next player wins), with the "
        "complete Grundy table and terminal positions.",
        request_type=OutcomeProfileRequest,
        result_type=OutcomeProfileResult,
        run=compute_outcome_profile,
        tags=("game-theory", "impartial", "outcome", "exact"),
        examples=(
            OperationExample(
                name="four_position_outcome",
                description="Compute the P/N outcome partition of a four-position DAG.",
                input={"game": _GAME},
            ),
        ),
    ),
    MathTool(
        operation_id="game.impartial.disjunctive_sum.compute",
        title="Compute the Grundy value of a disjunctive sum",
        description="Compute the exact Grundy value of a disjunctive sum of finite "
        "impartial game components by XOR of their component Grundy values.",
        request_type=DisjunctiveSumRequest,
        result_type=DisjunctiveSumResult,
        run=compute_disjunctive_sum,
        tags=("game-theory", "impartial", "disjunctive-sum", "exact"),
        examples=(
            OperationExample(
                name="two_component_sum",
                description="Compute the disjunctive sum of two game components.",
                input={
                    "components": [_GAME_A, _GAME_B],
                    "start_positions": ["a", "c"],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
