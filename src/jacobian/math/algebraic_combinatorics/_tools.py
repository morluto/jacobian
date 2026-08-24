"""Algebraic combinatorics operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.algebraic_combinatorics._models import (
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
from jacobian.math.algebraic_combinatorics._operations import (
    compute_conjugate_partition,
    compute_hook_lengths,
    compute_inverse_rsk_word,
    compute_rsk_permutation,
    compute_rsk_word,
    compute_syt_count,
)
from jacobian.math.algebraic_combinatorics.values import RSKTableauPair
from jacobian.math.words.values import FiniteWord


def ac_operation[RequestT: StrictModel, ResultT: StrictModel](
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


_PARTITION_321 = {"partition": {"parts": [3, 2, 1]}}

ALGEBRAIC_COMBINATORICS_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    ac_operation(
        "combinatorics.hook_length.compute",
        "Compute hook lengths of a Young diagram",
        "Compute the hook length H(i,j) = lambda_i - j + lambda'_j - i + 1 "
        "for each cell (i,j) of the Young diagram of a partition.",
        HookLengthRequest,
        HookLengthResult,
        compute_hook_lengths,
        "combinatorics",
        "hook-length",
        "exact",
        examples=(
            example(
                "partition_321",
                "Hook lengths of partition (3, 2, 1).",
                _PARTITION_321,
            ),
        ),
    ),
    ac_operation(
        "combinatorics.standard_young_tableaux.count",
        "Count standard Young tableaux via the hook length formula",
        "Count the number of standard Young tableaux of a given shape using "
        "the hook length formula: f^lambda = n! / product of hook lengths.",
        StandardYoungTableauCountRequest,
        StandardYoungTableauCountResult,
        compute_syt_count,
        "combinatorics",
        "young-tableaux",
        "exact",
        examples=(
            example(
                "partition_321",
                "Number of SYT for shape (3, 2, 1) is 16.",
                _PARTITION_321,
            ),
        ),
    ),
    ac_operation(
        "combinatorics.conjugate_partition.compute",
        "Compute the conjugate (transpose) partition",
        "Compute the conjugate partition lambda' by transposing the Ferrers "
        "diagram of a partition lambda.",
        ConjugatePartitionRequest,
        ConjugatePartitionResult,
        compute_conjugate_partition,
        "combinatorics",
        "partition",
        "exact",
        examples=(
            example(
                "partition_321",
                "Conjugate of partition (3, 2, 1) is (3, 2, 1).",
                _PARTITION_321,
            ),
        ),
        version="2",
    ),
    ac_operation(
        "combinatorics.rsk.permutation.compute",
        "Compute RSK correspondence for a permutation",
        "Compute the Robinson-Schensted-Knuth correspondence for one "
        "strict permutation of 1..n, returning the exact source, canonical "
        "standard P/Q tableaux, canonical shape, and LIS/LDS lengths under "
        "ROW_INSERTION_RSK_V1. The result independently replays row insertion.",
        RSKPermutationRequest,
        RSKResult,
        compute_rsk_permutation,
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
        version="2",
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
        compute_rsk_word,
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
        version="2",
    ),
    ac_operation(
        "tableau.rsk.inverse_word.compute",
        "Invert an ordinary word RSK tableau pair",
        "Reconstruct the unique bounded word from a compatible semistandard "
        "insertion tableau and standard recording tableau under "
        "ROW_INSERTION_RSK_V1. Tableau entries are one-based ranks in the "
        "pair's exact ordered alphabet, and the reconstructed word is replayed "
        "through forward RSK before it is returned as the canonical finite "
        "word value.",
        RSKInverseWordRequest,
        FiniteWord,
        compute_inverse_rsk_word,
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
        version="2",
    ),
)

TOOLS = ALGEBRAIC_COMBINATORICS_OPERATIONS

__all__ = ["TOOLS"]
