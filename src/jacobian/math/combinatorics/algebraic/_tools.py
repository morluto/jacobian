"""Algebraic combinatorics operation declarations."""

from typing import Any

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
        total_product=native._hook_length_product(hooks),
    )


def syt_count(
    request: StandardYoungTableauCountRequest,
) -> StandardYoungTableauCountResult:
    count = native.standard_young_tableaux_count(request.partition)
    n = sum(request.partition.parts)
    return StandardYoungTableauCountResult(count=count, n=n)


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
    try:
        return native.inverse_row_insertion_rsk(request.pair)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_pair_incompatible",
            message=str(exc),
        ) from exc


_PARTITION_321 = {"partition": {"parts": [3, 2, 1]}}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="combinatorics.hook_length.compute",
        title="Compute hook lengths of a Young diagram",
        description="Compute the exact hook length of every cell in a Young diagram and "
        "return their product.",
        request_type=HookLengthRequest,
        result_type=HookLengthResult,
        run=hook_lengths,
        tags=("combinatorics", "young-diagram", "hook-length", "exact"),
        examples=(
            OperationExample(
                name="partition_321",
                description="Compute hook lengths for partition (3,2,1).",
                input=_PARTITION_321,
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.standard_young_tableaux.count",
        title="Count standard Young tableaux",
        description="Count standard Young tableaux of a partition shape using the exact "
        "hook-length formula.",
        request_type=StandardYoungTableauCountRequest,
        result_type=StandardYoungTableauCountResult,
        run=syt_count,
        tags=("combinatorics", "young-tableaux", "exact"),
        examples=(
            OperationExample(
                name="partition_321",
                description="Count tableaux of partition (3,2,1).",
                input=_PARTITION_321,
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.conjugate_partition.compute",
        title="Compute a conjugate partition",
        description="Transpose the Ferrers diagram and return the exact conjugate partition.",
        request_type=ConjugatePartitionRequest,
        result_type=ConjugatePartitionResult,
        run=conjugate_partition,
        tags=("combinatorics", "partition", "exact"),
        examples=(
            OperationExample(
                name="partition_321",
                description="Compute the conjugate of partition (3,2,1).",
                input=_PARTITION_321,
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.rsk.permutation.compute",
        title="Compute RSK correspondence for a permutation",
        description="Compute the Robinson-Schensted-Knuth correspondence for one "
        "strict permutation of 1..n, returning the exact source, canonical "
        "standard P/Q tableaux, canonical shape, and LIS/LDS lengths under "
        "ROW_INSERTION_RSK_V1.",
        request_type=RSKPermutationRequest,
        result_type=RSKResult,
        run=rsk_permutation,
        tags=("combinatorics", "rsk", "exact"),
        examples=(
            OperationExample(
                name="rsk_permutation_132",
                description="Compute RSK of permutation (1, 3, 2); "
                "input must be a permutation of 1..n.",
                input={
                    "permutation": [1, 3, 2],
                    "convention": "ROW_INSERTION_RSK_V1",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tableau.rsk.word.compute",
        title="Compute ordinary row-insertion RSK for an ordered word",
        description="Compute the compact Robinson-Schensted-Knuth pair for one bounded "
        "word over an explicit ordered alphabet. Letters are inserted from "
        "left to right, bumping the first strictly greater row entry under "
        "ROW_INSERTION_RSK_V1. The result carries the exact alphabet, a "
        "semistandard insertion tableau, a standard recording tableau, and "
        "their validated common shape; no bumping ledger is materialized.",
        request_type=RSKWordRequest,
        result_type=RSKTableauPair,
        run=rsk_word,
        tags=("combinatorics", "rsk", "words", "exact"),
        examples=(
            OperationExample(
                name="word_rsk_with_repeated_letters",
                description="Insert (c, c, b, d, a) over the ordered alphabet a<b<c<d.",
                input={
                    "word": {
                        "alphabet": ["a", "b", "c", "d"],
                        "letters": ["c", "c", "b", "d", "a"],
                    },
                    "convention": "ROW_INSERTION_RSK_V1",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tableau.rsk.inverse_word.compute",
        title="Invert an ordinary word RSK tableau pair",
        description="Reconstruct the unique bounded word from a compatible semistandard "
        "insertion tableau and standard recording tableau under "
        "ROW_INSERTION_RSK_V1. Tableau entries are one-based ranks in the "
        "pair's exact ordered alphabet; the result is the canonical finite "
        "word value.",
        request_type=RSKInverseWordRequest,
        result_type=FiniteWord,
        run=inverse_rsk_word,
        tags=("combinatorics", "rsk", "words", "inverse", "exact"),
        examples=(
            OperationExample(
                name="inverse_word_rsk_with_repeated_letters",
                description="Recover (c, c, b, d, a) from its compact RSK pair.",
                input={
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
