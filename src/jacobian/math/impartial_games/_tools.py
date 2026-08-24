"""Impartial-game operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.impartial_games._models import (
    BirthdayRequest,
    BirthdayResult,
    DisjunctiveSumRequest,
    DisjunctiveSumResult,
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
from jacobian.math.impartial_games._operations import (
    compute_birthday,
    compute_disjunctive_sum,
    compute_grundy_table,
    compute_nim_options,
    compute_nim_sum,
    compute_outcome_profile,
    compute_subtraction_grundy_prefix,
)


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    version: str = "1",
    examples: tuple[OperationExample, ...],
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


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
    _op(
        "game.impartial.grundy_table.compute",
        "Compute a complete Grundy table",
        "Compute every exact Grundy value and canonical option-value set for a "
        "bounded finite normal-play impartial game DAG.",
        GrundyTableRequest,
        GrundyTableResult,
        compute_grundy_table,
        "game-theory",
        "impartial",
        "grundy",
        "exact",
        examples=(
            example(
                "four_position_game",
                "Compute the complete Grundy table of a four-position DAG.",
                {"game": _GAME},
            ),
        ),
    ),
    _op(
        "game.impartial.birthday.compute",
        "Compute all position birthdays",
        "Compute the exact DAG height of every position, with terminals at zero.",
        BirthdayRequest,
        BirthdayResult,
        compute_birthday,
        "game-theory",
        "impartial",
        "birthday",
        "exact",
        examples=(
            example(
                "four_position_birthdays",
                "Compute every birthday in a four-position game DAG.",
                {"game": _GAME},
            ),
        ),
    ),
    _op(
        "game.subtraction.grundy_prefix.compute",
        "Compute a bounded subtraction-game Grundy prefix",
        "Compute exact Grundy values and canonical option-value sets for every "
        "heap from zero through the explicit maximum; no periodicity is implied.",
        SubtractionGrundyPrefixRequest,
        SubtractionGrundyPrefixResult,
        compute_subtraction_grundy_prefix,
        "game-theory",
        "subtraction",
        "grundy",
        "exact",
        examples=(
            example(
                "subtract_one_or_three",
                "Compute heaps zero through five for subtraction set {1,3}.",
                {"subtraction_set": [1, 3], "max_heap": 5},
            ),
        ),
    ),
    _op(
        "game.nim.nim_sum.compute",
        "Compute the nim sum of a Nim position",
        "Compute the exact bitwise xor of a canonical sorted heap multiset, "
        "determining the P/N outcome under normal play.",
        NimSumRequest,
        NimSumResult,
        compute_nim_sum,
        "game-theory",
        "nim",
        "exact",
        version="2",
        examples=(
            example(
                "nim_sum_1_2_3",
                "Compute the nim sum of heaps (1, 2, 3); "
                "heaps must be nonnegative integers in nondecreasing order.",
                {"position": {"heaps": [1, 2, 3]}},
            ),
        ),
    ),
    _op(
        "game.nim.options.compute",
        "Enumerate every distinct legal Nim option",
        "Return the complete canonical one-move option family of a sorted Nim "
        "heap multiset, retaining every source heap index collapsed by "
        "multiset deduplication.",
        NimOptionsRequest,
        NimOptionsResult,
        compute_nim_options,
        "game-theory",
        "nim",
        "options",
        "exact",
        examples=(
            example(
                "deduplicated_equal_heaps",
                "Enumerate every distinct option of Nim heaps (1,2,2); heaps "
                "must be nondecreasing, and zero heaps are retained.",
                {"position": {"heaps": [1, 2, 2]}},
            ),
        ),
    ),
    _op(
        "game.impartial.outcome_profile.compute",
        "Compute the P/N outcome partition",
        "Partition positions into P-positions (Grundy=0, previous player "
        "wins) and N-positions (Grundy>0, next player wins), with the "
        "complete Grundy table and terminal positions.",
        OutcomeProfileRequest,
        OutcomeProfileResult,
        compute_outcome_profile,
        "game-theory",
        "impartial",
        "outcome",
        "exact",
        examples=(
            example(
                "four_position_outcome",
                "Compute the P/N outcome partition of a four-position DAG.",
                {"game": _GAME},
            ),
        ),
    ),
    _op(
        "game.impartial.disjunctive_sum.compute",
        "Compute the Grundy value of a disjunctive sum",
        "Compute the exact Grundy value of a disjunctive sum of finite "
        "impartial game components by XOR of their component Grundy values.",
        DisjunctiveSumRequest,
        DisjunctiveSumResult,
        compute_disjunctive_sum,
        "game-theory",
        "impartial",
        "disjunctive-sum",
        "exact",
        examples=(
            example(
                "two_component_sum",
                "Compute the disjunctive sum of two game components.",
                {
                    "components": [_GAME_A, _GAME_B],
                    "start_positions": ["a", "c"],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
