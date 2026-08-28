"""Algebraic combinatorics operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.combinatorics.algebraic import operations as native
from jacobian.math.combinatorics.algebraic._models import (
    ConjugatePartitionRequest,
    ConjugatePartitionResult,
    HookLengthRequest,
    HookLengthResult,
    RSKInverseWordRequest,
    RSKPermutationRequest,
    RSKResult,
    RSKWordRequest,
    StandardYoungTableauCountRequest,
    StandardYoungTableauCountResult,
)
from jacobian.math.combinatorics.algebraic.values import RSKTableauPair
from jacobian.math.logic.languages.words.values import FiniteWord


def hook_lengths(request: HookLengthRequest) -> HookLengthResult:
    hooks = native.hook_lengths(request.partition)
    return HookLengthResult(
        hooks=hooks,
        total_product=format_canonical_integer(native._hook_length_product(hooks)),
    )


def syt_count(
    request: StandardYoungTableauCountRequest,
) -> StandardYoungTableauCountResult:
    count = native.standard_young_tableaux_count(request.partition)
    n = sum(request.partition.parts)
    return StandardYoungTableauCountResult(count=format_canonical_integer(count), n=n)


def conjugate_partition(
    request: ConjugatePartitionRequest,
) -> ConjugatePartitionResult:
    return ConjugatePartitionResult(
        conjugate=native.conjugate_partition(request.partition)
    )


def rsk_permutation(request: RSKPermutationRequest) -> RSKResult:
    try:
        insertion_rows, recording_rows = native._rsk_permutation(request.permutation)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("permutation",),
            code="algebraic_combinatorics.permutation_invalid",
            message=str(exc),
        ) from exc
    return RSKResult._from_kernel(
        request,
        insertion_rows=insertion_rows,
        recording_rows=recording_rows,
    )


def rsk_word(request: RSKWordRequest) -> RSKTableauPair:
    return native.row_insertion_rsk(request.word)


def inverse_rsk_word(request: RSKInverseWordRequest) -> FiniteWord:
    return native.inverse_row_insertion_rsk(request.pair)


def ac_operation[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


_PARTITION_321 = {"partition": {"parts": [3, 2, 1]}}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    ac_operation(
        "combinatorics.hook_length.compute",
        "Compute hook lengths of a Young diagram",
        "Compute the exact hook length of every cell in a Young diagram and "
        "return their product.",
        HookLengthRequest,
        HookLengthResult,
        hook_lengths,
        "combinatorics",
        "young-diagram",
        "hook-length",
        "exact",
        examples=(
            example(
                "partition_321",
                "Compute hook lengths for partition (3,2,1).",
                _PARTITION_321,
            ),
        ),
    ),
    ac_operation(
        "combinatorics.standard_young_tableaux.count",
        "Count standard Young tableaux",
        "Count standard Young tableaux of a partition shape using the exact "
        "hook-length formula.",
        StandardYoungTableauCountRequest,
        StandardYoungTableauCountResult,
        syt_count,
        "combinatorics",
        "young-tableaux",
        "exact",
        examples=(
            example(
                "partition_321",
                "Count tableaux of partition (3,2,1).",
                _PARTITION_321,
            ),
        ),
    ),
    ac_operation(
        "combinatorics.conjugate_partition.compute",
        "Compute a conjugate partition",
        "Transpose the Ferrers diagram and return the exact conjugate partition.",
        ConjugatePartitionRequest,
        ConjugatePartitionResult,
        conjugate_partition,
        "combinatorics",
        "partition",
        "exact",
        examples=(
            example(
                "partition_321",
                "Compute the conjugate of partition (3,2,1).",
                _PARTITION_321,
            ),
        ),
    ),
    ac_operation(
        "combinatorics.rsk.permutation.compute",
        "Compute RSK correspondence for a permutation",
        "Compute the Robinson-Schensted-Knuth correspondence for one "
        "strict permutation of 1..n, returning the exact source, canonical "
        "standard P/Q tableaux, canonical shape, and LIS/LDS lengths under "
        "ROW_INSERTION_RSK_V1.",
        RSKPermutationRequest,
        RSKResult,
        rsk_permutation,
        "combinatorics",
        "rsk",
        "exact",
        examples=(
            example(
                "rsk_permutation_132",
                "Compute RSK of permutation (1, 3, 2); "
                "input must be a permutation of 1..n.",
                {
                    "permutation": [1, 3, 2],
                    "convention": "ROW_INSERTION_RSK_V1",
                },
            ),
        ),
    ),
    ac_operation(
        "tableau.rsk.word.compute",
        "Compute ordinary row-insertion RSK for an ordered word",
        "Compute the compact Robinson-Schensted-Knuth pair for one bounded "
        "word over an explicit ordered alphabet. Letters are inserted from "
        "left to right, bumping the first strictly greater row entry under "
        "ROW_INSERTION_RSK_V1. The result carries the exact alphabet, a "
        "semistandard insertion tableau, a standard recording tableau, and "
        "their validated common shape; no bumping ledger is materialized.",
        RSKWordRequest,
        RSKTableauPair,
        rsk_word,
        "combinatorics",
        "rsk",
        "words",
        "exact",
        examples=(
            example(
                "word_rsk_with_repeated_letters",
                "Insert (c, c, b, d, a) over the ordered alphabet a<b<c<d.",
                {
                    "word": {
                        "alphabet": ["a", "b", "c", "d"],
                        "letters": ["c", "c", "b", "d", "a"],
                    },
                    "convention": "ROW_INSERTION_RSK_V1",
                },
            ),
        ),
    ),
    ac_operation(
        "tableau.rsk.inverse_word.compute",
        "Invert an ordinary word RSK tableau pair",
        "Reconstruct the unique bounded word from a compatible semistandard "
        "insertion tableau and standard recording tableau under "
        "ROW_INSERTION_RSK_V1. Tableau entries are one-based ranks in the "
        "pair's exact ordered alphabet; the result is the canonical finite "
        "word value.",
        RSKInverseWordRequest,
        FiniteWord,
        inverse_rsk_word,
        "combinatorics",
        "rsk",
        "words",
        "inverse",
        "exact",
        examples=(
            example(
                "inverse_word_rsk_with_repeated_letters",
                "Recover (c, c, b, d, a) from its compact RSK pair.",
                {
                    "pair": {
                        "alphabet": ["a", "b", "c", "d"],
                        "insertion_tableau": {"rows": [[1, 3, 4], [2], [3]]},
                        "recording_tableau": {"rows": [[1, 2, 4], [3], [5]]},
                        "shape": {"parts": [3, 1, 1]},
                        "source_kind": "WORD",
                        "convention": "ROW_INSERTION_RSK_V1",
                    },
                    "convention": "ROW_INSERTION_RSK_V1",
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
