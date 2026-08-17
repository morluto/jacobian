"""Impartial combinatorial game operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.impartial_games._models import (
    BirthdayRequest,
    BirthdayResult,
    GrundyClassesRequest,
    GrundyClassesResult,
    GrundyTableRequest,
    GrundyTableResult,
    MexRequest,
    MexResult,
    NimEquivalentRequest,
    NimEquivalentResult,
    NimOptionsRequest,
    NimOptionsResult,
    NimSumRequest,
    NimSumResult,
    OutcomeProfileRequest,
    OutcomeProfileResult,
    PositionGrundyRequest,
    PositionGrundyResult,
    SubtractionDAGRequest,
    SubtractionDAGResult,
    SubtractionGrundyPrefixRequest,
    SubtractionGrundyPrefixResult,
)
from jacobian.math.impartial_games._operations import (
    compute_birthday,
    compute_grundy_classes,
    compute_grundy_table,
    compute_mex,
    compute_nim_equivalent,
    compute_nim_options,
    compute_nim_sum,
    compute_outcome_profile,
    compute_position_grundy,
    compute_subtraction_dag,
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
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
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


_SIMPLE_GAME = {
    "game": {
        "positions": ["0", "1", "2", "3"],
        "moves": [
            {"source": "3", "target": "2"},
            {"source": "3", "target": "1"},
            {"source": "2", "target": "1"},
            {"source": "2", "target": "0"},
            {"source": "1", "target": "0"},
        ],
    },
}

IMPARTIAL_GAME_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "game.impartial.grundy_table.compute",
        "Compute Grundy table of an impartial game",
        "Given a finite impartial game DAG, compute the complete Grundy "
        "table with mex per position, max Grundy value, histogram, and "
        "topological evaluation order.",
        GrundyTableRequest,
        GrundyTableResult,
        compute_grundy_table,
        "game-theory",
        "impartial",
        "grundy",
        "exact",
        examples=(
            example(
                "chain_3",
                "Grundy values for a 4-position chain game; the game DAG "
                "must be acyclic.",
                _SIMPLE_GAME,
            ),
        ),
    ),
    _op(
        "game.impartial.position.grundy.compute",
        "Compute Grundy value of one position",
        "Compute the Grundy value of a single position and its reachable "
        "dependency sub-DAG in topological order.",
        PositionGrundyRequest,
        PositionGrundyResult,
        compute_position_grundy,
        "game-theory",
        "impartial",
        "grundy",
        "exact",
        examples=(
            example(
                "chain_3_position",
                "Grundy value of position 3 in a 4-position chain game; "
                "the position must be in the game.",
                {"game": _SIMPLE_GAME["game"], "position": "3"},
            ),
        ),
    ),
    _op(
        "game.impartial.outcome_profile.compute",
        "Compute P/N partition of an impartial game",
        "Partition positions into P-positions (Grundy 0) and N-positions "
        "(Grundy > 0), with terminal positions and the complete Grundy map.",
        OutcomeProfileRequest,
        OutcomeProfileResult,
        compute_outcome_profile,
        "game-theory",
        "impartial",
        "outcome",
        "exact",
        examples=(
            example(
                "chain_3_outcome",
                "P/N partition of a 4-position chain game; the game DAG "
                "must be acyclic.",
                _SIMPLE_GAME,
            ),
        ),
    ),
    _op(
        "game.impartial.nim_equivalent.compute",
        "Compute Nim-heap equivalence of a position",
        "Find the canonical Nim heap size equivalent to one position under "
        "normal-play disjunctive sum, given the complete game DAG.",
        NimEquivalentRequest,
        NimEquivalentResult,
        compute_nim_equivalent,
        "game-theory",
        "impartial",
        "nim",
        "exact",
        examples=(
            example(
                "chain_3_nim_equiv",
                "Nim heap equivalent to position 3 in a 4-position chain; "
                "the position must be in the game.",
                {"game": _SIMPLE_GAME["game"], "position": "3"},
            ),
        ),
    ),
    _op(
        "game.impartial.grundy_classes.compute",
        "Partition positions by Grundy value",
        "Partition positions into equivalence classes by equal Grundy "
        "value, with class histograms.",
        GrundyClassesRequest,
        GrundyClassesResult,
        compute_grundy_classes,
        "game-theory",
        "impartial",
        "grundy",
        "exact",
        examples=(
            example(
                "chain_3_classes",
                "Grundy equivalence classes of a 4-position chain game; "
                "the game DAG must be acyclic.",
                _SIMPLE_GAME,
            ),
        ),
    ),
    _op(
        "game.impartial.birthday.compute",
        "Compute birthdays of all positions",
        "Compute the birthday (DAG height) of every position: 0 for "
        "terminals, 1 + max of successors otherwise.",
        BirthdayRequest,
        BirthdayResult,
        compute_birthday,
        "game-theory",
        "impartial",
        "birthday",
        "exact",
        examples=(
            example(
                "chain_3_birthday",
                "Birthdays of a 4-position chain game; the game DAG must "
                "be acyclic.",
                _SIMPLE_GAME,
            ),
        ),
    ),
    _op(
        "game.nim.nim_sum.compute",
        "Compute nim-sum of heap sizes",
        "Compute the bitwise xor (nim-sum) of a finite tuple of "
        "non-negative heap sizes and determine P/N status.",
        NimSumRequest,
        NimSumResult,
        compute_nim_sum,
        "game-theory",
        "nim",
        "exact",
        examples=(
            example(
                "simple_nim_sum",
                "Nim-sum of heaps [3, 4, 5]; heaps must be non-negative.",
                {"heaps": [3, 4, 5]},
            ),
        ),
    ),
    _op(
        "game.nim.options.compute",
        "Enumerate all legal Nim options",
        "Enumerate every distinct legal option of a Nim position obtained "
        "by reducing exactly one heap.",
        NimOptionsRequest,
        NimOptionsResult,
        compute_nim_options,
        "game-theory",
        "nim",
        "exact",
        examples=(
            example(
                "nim_2_2_options",
                "All options of Nim position [2, 2]; heaps must be "
                "non-negative.",
                {"heaps": [2, 2]},
            ),
        ),
    ),
    _op(
        "game.subtraction.dag.compute",
        "Build game DAG for a subtraction game",
        "Construct the exact game DAG for a bounded subtraction game "
        "with a given subtraction set and maximum heap.",
        SubtractionDAGRequest,
        SubtractionDAGResult,
        compute_subtraction_dag,
        "game-theory",
        "subtraction",
        "exact",
        examples=(
            example(
                "sub_1_3_dag",
                "DAG for subtraction set {1, 3} with max heap 5; "
                "subtraction set values must be positive.",
                {"subtraction_set": [1, 3], "max_heap": 5},
            ),
        ),
    ),
    _op(
        "game.subtraction.grundy_prefix.compute",
        "Compute Grundy prefix of a subtraction game",
        "Compute g(0),...,g(N), option-value sets, and P/N heap sets "
        "for a bounded subtraction game.",
        SubtractionGrundyPrefixRequest,
        SubtractionGrundyPrefixResult,
        compute_subtraction_grundy_prefix,
        "game-theory",
        "subtraction",
        "grundy",
        "exact",
        examples=(
            example(
                "sub_1_3_grundy",
                "Grundy prefix for subtraction set {1, 3} with max heap 5; "
                "subtraction set values must be positive.",
                {"subtraction_set": [1, 3], "max_heap": 5},
            ),
        ),
    ),
    _op(
        "combinatorics.mex.compute",
        "Compute minimum excluded value",
        "Compute the mex (minimum excluded non-negative integer) of a "
        "bounded finite set, with the membership prefix and first gap.",
        MexRequest,
        MexResult,
        compute_mex,
        "combinatorics",
        "mex",
        "exact",
        examples=(
            example(
                "mex_example",
                "Mex of {0, 1, 3, 4}; values must be non-negative.",
                {"values": [0, 1, 3, 4]},
            ),
        ),
    ),
)


TOOLS = IMPARTIAL_GAME_OPERATIONS

__all__ = ["TOOLS"]
