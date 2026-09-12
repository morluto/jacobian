"""Algebraic combinatorics operation declarations."""

from typing import Any

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.combinatorics.algebraic import operations as native
from jacobian.math.combinatorics.algebraic._models import (
    ConjugatePartitionRequest,
    ConjugatePartitionResult,
    HookContentCountRequest,
    HookContentCountResult,
    HookLengthRequest,
    HookLengthResult,
    PartitionDominanceRequest,
    PartitionDominanceResult,
    RSKInverseWordRequest,
    RSKPermutationRequest,
    RSKResult,
    RSKWordRequest,
    SemistandardTableauCheckRequest,
    SemistandardTableauCheckResult,
    StandardTableauCheckRequest,
    StandardTableauCheckResult,
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


def hook_content_count(request: HookContentCountRequest) -> HookContentCountResult:
    count = native.hook_content_count(request.partition, request.alphabet_size)
    return HookContentCountResult(
        partition=request.partition,
        alphabet_size=request.alphabet_size,
        count=count,
    )


def partition_dominance(
    request: PartitionDominanceRequest,
) -> PartitionDominanceResult:
    relation, left_sums, right_sums = native.partition_dominance(
        request.left, request.right
    )
    return PartitionDominanceResult(
        left=request.left,
        right=request.right,
        relation=relation,
        left_prefix_sums=left_sums,
        right_prefix_sums=right_sums,
    )


def check_standard_tableau(
    request: StandardTableauCheckRequest,
) -> StandardTableauCheckResult:
    try:
        native.check_standard_tableau(request.tableau)
    except PydanticCustomError:
        return StandardTableauCheckResult(tableau=request.tableau, is_member=False)
    return StandardTableauCheckResult(tableau=request.tableau, is_member=True)


def check_semistandard_tableau(
    request: SemistandardTableauCheckRequest,
) -> SemistandardTableauCheckResult:
    try:
        native.check_semistandard_tableau(request.tableau)
    except PydanticCustomError:
        return SemistandardTableauCheckResult(tableau=request.tableau, is_member=False)
    return SemistandardTableauCheckResult(tableau=request.tableau, is_member=True)


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
    MathTool(
        operation_id="combinatorics.partition.hook_content.count",
        title="Count semistandard tableaux by the hook-content formula",
        description="Return the exact number of semistandard Young tableaux of a "
        "partition shape with entries in 1..m, bound to the source shape and "
        "alphabet.",
        request_type=HookContentCountRequest,
        result_type=HookContentCountResult,
        run=hook_content_count,
        tags=("combinatorics", "young-tableaux", "hook-content", "exact"),
        examples=(
            OperationExample(
                name="shape_21_alphabet_2",
                description="Count SSYTs of shape (2,1) over the alphabet {1,2}.",
                input={"partition": {"parts": [2, 1]}, "alphabet_size": "2"},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.partition.dominance.compare",
        title="Compare partitions in dominance order",
        description="Compare equal-size integer partitions using the complete "
        "leading-row prefix-sum ledger; different sizes are reported as "
        "not comparable.",
        request_type=PartitionDominanceRequest,
        result_type=PartitionDominanceResult,
        run=partition_dominance,
        tags=("combinatorics", "partition", "dominance", "exact"),
        examples=(
            OperationExample(
                name="partition_21_vs_111",
                description="The partition (2,1) dominates (1,1,1).",
                input={
                    "left": {"parts": [2, 1]},
                    "right": {"parts": [1, 1, 1]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.tableau.standard.check",
        title="Check standard Young tableau membership",
        description="Replay the complete shape, row, column, and 1..n entry "
        "conditions and return the source-bound membership decision.",
        request_type=StandardTableauCheckRequest,
        result_type=StandardTableauCheckResult,
        run=check_standard_tableau,
        tags=("combinatorics", "young-tableaux", "validation", "exact"),
        examples=(
            OperationExample(
                name="standard_shape_21",
                description="Check a standard tableau of shape (2,1).",
                input={"tableau": {"rows": [[1, 2], [3]]}},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.tableau.semistandard.check",
        title="Check semistandard Young tableau membership",
        description="Replay the complete shape and weak-row/strict-column "
        "conditions and return the source-bound membership decision.",
        request_type=SemistandardTableauCheckRequest,
        result_type=SemistandardTableauCheckResult,
        run=check_semistandard_tableau,
        tags=("combinatorics", "young-tableaux", "validation", "exact"),
        examples=(
            OperationExample(
                name="semistandard_shape_21",
                description="Check a semistandard tableau of shape (2,1).",
                input={"tableau": {"rows": [[1, 1], [2]]}},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
