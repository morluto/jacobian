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
    KnuthMovesRequest,
    KnuthMovesResult,
    LongestDecreasingSubsequenceRequest,
    LongestDecreasingSubsequenceResult,
    LongestIncreasingSubsequenceRequest,
    LongestIncreasingSubsequenceResult,
    PartitionDominanceRequest,
    PartitionDominanceResult,
    PlacticEquivalenceRequest,
    PlacticEquivalenceResult,
    PlacticNormalFormRequest,
    PlacticNormalFormResult,
    RSKInversePermutationRequest,
    RSKInverseWordRequest,
    RSKPermutationRequest,
    RSKWordRequest,
    RSKWordTraceRequest,
    RSKWordTraceResult,
    SemistandardTableauCheckRequest,
    SemistandardTableauCheckResult,
    SemistandardYoungTableauCountRequest,
    SemistandardYoungTableauCountResult,
    SkewLittlewoodRichardsonCheckRequest,
    SkewLittlewoodRichardsonCheckResult,
    StandardTableauCheckRequest,
    StandardTableauCheckResult,
    StandardYoungTableauCountRequest,
    StandardYoungTableauCountResult,
)
from jacobian.math.combinatorics.algebraic.biword import (
    Biword,
    BiwordNormalizeRequest,
    BiwordNormalizeResult,
    BiwordRSKPair,
    BiwordRSKRequest,
    GreeneRequest,
    GreeneResult,
    GreeneWitnessRequest,
    GreeneWitnessResult,
    InverseBiwordRSKRequest,
    InverseMatrixRSKRequest,
    MatrixRSKRequest,
    NonnegativeIntegerMatrix,
)
from jacobian.math.combinatorics.algebraic.biword_ops import (
    greene,
    inverse_biword,
    inverse_matrix,
    matrix_biword,
    normalize_biword,
    rsk_biword,
)
from jacobian.math.combinatorics.algebraic.greene_witnesses import (
    compute_greene_witnesses,
)
from jacobian.math.combinatorics.algebraic.subsequences import (
    longest_decreasing_subsequence,
    longest_increasing_subsequence,
)
from jacobian.math.combinatorics.algebraic.values import (
    FinitePermutation,
    PermutationRSKPair,
    RSKTableauPair,
)
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


def semistandard_young_tableaux_count(
    request: SemistandardYoungTableauCountRequest,
) -> SemistandardYoungTableauCountResult:
    return native.semistandard_young_tableaux_count(
        request.partition, request.alphabet_size
    )


def partition_dominance(
    request: PartitionDominanceRequest,
) -> PartitionDominanceResult:
    return native.partition_dominance(request.left, request.right)


def check_standard_tableau(
    request: StandardTableauCheckRequest,
) -> StandardTableauCheckResult:
    return native.check_standard_tableau(request.tableau)


def check_semistandard_tableau(
    request: SemistandardTableauCheckRequest,
) -> SemistandardTableauCheckResult:
    return native.check_semistandard_tableau(request.tableau)


def rsk_permutation(request: RSKPermutationRequest) -> PermutationRSKPair:
    return native.permutation_rsk(request.permutation)


def rsk_word(request: RSKWordRequest) -> RSKTableauPair:
    return native.row_insertion_rsk(request.word)


def rsk_word_trace(request: RSKWordTraceRequest) -> RSKWordTraceResult:
    return native.row_insertion_rsk_trace(request.word)


def inverse_rsk_word(request: RSKInverseWordRequest) -> FiniteWord:
    try:
        return native.inverse_row_insertion_rsk(request.pair)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_pair_incompatible",
            message=str(exc),
        ) from exc


def inverse_rsk_permutation(request: RSKInversePermutationRequest) -> FinitePermutation:
    return native.inverse_permutation_rsk(request.pair)


def knuth_moves(request: KnuthMovesRequest) -> KnuthMovesResult:
    return KnuthMovesResult._from_kernel(
        request, neighbors=native.knuth_moves(request.word)
    )


def plactic_normal_form(request: PlacticNormalFormRequest) -> PlacticNormalFormResult:
    return native.plactic_normal_form(request.word)


def plactic_equivalence(
    request: PlacticEquivalenceRequest,
) -> PlacticEquivalenceResult:
    return native.plactic_equivalence(request.left, request.right)


def check_skew_littlewood_richardson(
    request: SkewLittlewoodRichardsonCheckRequest,
) -> SkewLittlewoodRichardsonCheckResult:
    return native.check_skew_littlewood_richardson(
        request.outer,
        request.inner,
        request.tableau,
        request.content,
        request.convention,
    )


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
        "finite permutation of 1..n, returning its canonical standard P/Q "
        "tableau pair under ROW_INSERTION_RSK_V1.",
        request_type=RSKPermutationRequest,
        result_type=PermutationRSKPair,
        run=rsk_permutation,
        tags=("combinatorics", "rsk", "exact"),
        examples=(
            OperationExample(
                name="rsk_permutation_132",
                description="Compute RSK of permutation (1, 3, 2); "
                "input must be a permutation of 1..n.",
                input={
                    "permutation": {"images": [1, 3, 2]},
                    "convention": "ROW_INSERTION_RSK_V1",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tableau.rsk.inverse_permutation.compute",
        title="Invert a permutation RSK tableau pair",
        description="Reconstruct the unique finite permutation of 1..n from its "
        "compatible pair of standard tableaux by reverse row insertion under "
        "ROW_INSERTION_RSK_V1.",
        request_type=RSKInversePermutationRequest,
        result_type=FinitePermutation,
        run=inverse_rsk_permutation,
        tags=("combinatorics", "rsk", "permutations", "inverse", "exact"),
        examples=(
            OperationExample(
                name="inverse_rsk_permutation_312",
                description="Recover permutation (3, 1, 2) from its standard P/Q pair; both tableaux must be standard and have the same shape.",
                input={
                    "pair": {
                        "p_tableau": {"rows": [[1, 2], [3]]},
                        "q_tableau": {"rows": [[1, 3], [2]]},
                        "convention": "ROW_INSERTION_RSK_V1",
                    },
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
        operation_id="tableau.rsk.word.trace.compute",
        title="Trace ordinary row-insertion RSK for an ordered word",
        description=(
            "Return the final RSK pair and one exact insertion event for every "
            "letter of a bounded ordered word. Each event records the zero-based "
            "cells and entries bumped, plus the terminal added cell and entry "
            "under ROW_INSERTION_RSK_V1. Complete trace work and output are "
            "admitted before insertion."
        ),
        request_type=RSKWordTraceRequest,
        result_type=RSKWordTraceResult,
        run=rsk_word_trace,
        tags=("combinatorics", "rsk", "words", "trace", "exact"),
        discovery_terms=(
            "RSK insertion trace",
            "row bumping path",
            "Robinson-Schensted bumping ledger",
        ),
        examples=(
            OperationExample(
                name="reverse_word_bumping_trace",
                description=(
                    "Trace each insertion for (c,b,a), returning its bump path; "
                    "the alphabet is explicitly ordered a<b<c."
                ),
                input={
                    "word": {
                        "alphabet": ["a", "b", "c"],
                        "letters": ["c", "b", "a"],
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
        operation_id="word.knuth_moves.compute",
        title="Compute every one-step Knuth neighbor of an ordered word",
        description="Materialize the complete finite one-step Knuth neighborhood of a "
        "bounded word over an explicit ordered alphabet: xzy<->zxy when x<=y<z "
        "(K1) and yxz<->yzx when x<y<=z (K2) under ROW_INSERTION_RSK_V1. Every "
        "neighbor carries its window position and the relation used; each "
        "neighbor shares the source insertion tableau. At most one neighbor "
        "per length-three window.",
        request_type=KnuthMovesRequest,
        result_type=KnuthMovesResult,
        run=knuth_moves,
        tags=("combinatorics", "words", "knuth", "plactic", "exact"),
        discovery_terms=("knuth moves", "plactic equivalence", "knuth relations"),
        examples=(
            OperationExample(
                name="k1_neighbor",
                description="The word (a, c, b) over a<b<c admits one K1 move to (c, a, b).",
                input={
                    "word": {
                        "alphabet": ["a", "b", "c"],
                        "letters": ["a", "c", "b"],
                    },
                    "convention": "ROW_INSERTION_RSK_V1",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.tableau.skew_littlewood_richardson.check",
        title="Check skew Littlewood-Richardson tableau membership",
        description="Replay complete skew Littlewood-Richardson membership for one "
        "candidate tableau on a skew shape outer/inner with claimed content nu: "
        "cell coverage of the skew diagram, weak-row/strict-column "
        "semistandardity, exact content multiplicities, and every reading-word "
        "prefix Yamanouchi lattice inequality under "
        "READING_WORD_RL_TOP_V1. Returns the source-bound decision with the "
        "first failed stage and its concrete cell, value, or prefix length.",
        request_type=SkewLittlewoodRichardsonCheckRequest,
        result_type=SkewLittlewoodRichardsonCheckResult,
        run=check_skew_littlewood_richardson,
        tags=("combinatorics", "young-tableaux", "littlewood-richardson", "exact"),
        discovery_terms=(
            "littlewood-richardson tableau",
            "skew shape tableau membership",
            "yamanouchi lattice word",
        ),
        examples=(
            OperationExample(
                name="skew_21_over_1",
                description=(
                    "Check the LR tableau [[1],[2]] of content (1,1) on the skew "
                    "shape (2,1)/(1); rows carry exactly the nonempty skew cells "
                    "in top-to-bottom order."
                ),
                input={
                    "outer": {"parts": [2, 1]},
                    "inner": {"parts": [1]},
                    "tableau": {"rows": [[1], [2]]},
                    "content": {"parts": [1, 1]},
                    "convention": "READING_WORD_RL_TOP_V1",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.semistandard_young_tableaux.count",
        title="Count semistandard Young tableaux",
        description="Return the exact number of semistandard Young tableaux of a "
        "partition shape with entries in 1..m, bound to the source shape and "
        "alphabet.",
        request_type=SemistandardYoungTableauCountRequest,
        result_type=SemistandardYoungTableauCountResult,
        run=semistandard_young_tableaux_count,
        tags=("combinatorics", "young-tableaux", "exact"),
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
        description="Compare equal-size integer partitions in dominance order; "
        "different sizes are reported as not comparable.",
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


def _inverse_biword_run(request: InverseBiwordRSKRequest) -> Biword:
    try:
        return inverse_biword(request.pair)
    except OperationDomainValidationError:
        raise
    except (TypeError, ValueError, IndexError) as exc:
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_inverse",
            message="incompatible biword RSK pair",
        ) from exc


def _inverse_matrix_run(request: InverseMatrixRSKRequest) -> NonnegativeIntegerMatrix:
    try:
        return inverse_matrix(request.pair, request.row_labels, request.column_labels)
    except OperationDomainValidationError:
        raise
    except (TypeError, ValueError, IndexError) as exc:
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_inverse",
            message="incompatible matrix RSK pair",
        ) from exc


def _normalize_run(request: BiwordNormalizeRequest) -> BiwordNormalizeResult:
    try:
        return normalize_biword(request)
    except OperationDomainValidationError:
        raise
    except (TypeError, ValueError, IndexError) as exc:
        raise OperationDomainValidationError(
            location=("biword",),
            code="algebraic_combinatorics.biword_invalid",
            message=str(exc),
        ) from exc


def _rsk_biword_run(request: BiwordRSKRequest) -> BiwordRSKPair:
    try:
        return rsk_biword(request.biword)
    except OperationDomainValidationError:
        raise
    except (TypeError, ValueError, IndexError) as exc:
        raise OperationDomainValidationError(
            location=("biword",),
            code="algebraic_combinatorics.biword_invalid",
            message=str(exc),
        ) from exc


def _matrix_rsk_run(request: MatrixRSKRequest) -> BiwordRSKPair:
    try:
        return matrix_biword(request.matrix)
    except OperationDomainValidationError:
        raise
    except (TypeError, ValueError, IndexError) as exc:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="algebraic_combinatorics.matrix_invalid",
            message=str(exc),
        ) from exc


def _greene_run(request: GreeneRequest) -> GreeneResult:
    try:
        return greene(request.word, request.k)
    except OperationDomainValidationError:
        raise
    except (TypeError, ValueError, IndexError) as exc:
        raise OperationDomainValidationError(
            location=("word",),
            code="algebraic_combinatorics.word_invalid",
            message=str(exc),
        ) from exc


def _greene_witness_run(request: GreeneWitnessRequest) -> GreeneWitnessResult:
    return compute_greene_witnesses(request)


def _longest_increasing_subsequence_run(
    request: LongestIncreasingSubsequenceRequest,
) -> LongestIncreasingSubsequenceResult:
    return longest_increasing_subsequence(request)


def _longest_decreasing_subsequence_run(
    request: LongestDecreasingSubsequenceRequest,
) -> LongestDecreasingSubsequenceResult:
    return longest_decreasing_subsequence(request)


TOOLS = TOOLS + (  # noqa: RUF005
    MathTool(
        operation_id="word.longest_increasing_subsequence.compute",
        title="Compute a strict longest increasing subsequence",
        description=(
            "Return the exact maximum length and one deterministic source-index "
            "witness for a word over its explicit ordered alphabet. Strict means "
            "each selected letter is strictly greater than the previous one, "
            "so duplicate letters cannot both occur. Admission bounds the word "
            "payload cardinality, output size, word length, and the complete "
            "O(n^2) predecessor-pair scan."
        ),
        request_type=LongestIncreasingSubsequenceRequest,
        result_type=LongestIncreasingSubsequenceResult,
        run=_longest_increasing_subsequence_run,
        tags=("combinatorics", "words", "subsequences", "increasing", "exact"),
        discovery_terms=(
            "longest strictly increasing subsequence of a finite word",
            "strict LIS length and source-index witness",
        ),
        examples=(
            OperationExample(
                name="strict_lis_with_duplicates",
                description=(
                    "Find a longest strictly increasing subsequence of (b,a,b,c); "
                    "equal letters cannot both be selected."
                ),
                input={
                    "word": {
                        "alphabet": ["a", "b", "c"],
                        "letters": ["b", "a", "b", "c"],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="word.longest_decreasing_subsequence.compute",
        title="Compute a strict longest decreasing subsequence",
        description=(
            "Return the exact maximum length and one deterministic source-index "
            "witness for a word over its explicit ordered alphabet. Strict means "
            "each selected letter is strictly smaller than the previous one. "
            "Admission bounds payload cardinality, word length, output size, "
            "and the complete O(n^2) predecessor-pair scan."
        ),
        request_type=LongestDecreasingSubsequenceRequest,
        result_type=LongestDecreasingSubsequenceResult,
        run=_longest_decreasing_subsequence_run,
        tags=("combinatorics", "words", "subsequences", "decreasing", "exact"),
        discovery_terms=(
            "longest strictly decreasing subsequence of a finite word",
            "strict LDS length and source-index witness",
        ),
        examples=(
            OperationExample(
                name="strict_lds_with_duplicates",
                description=(
                    "Find a longest strictly decreasing subsequence of (c,a,b,a); "
                    "equal letters cannot both be selected."
                ),
                input={
                    "word": {
                        "alphabet": ["a", "b", "c"],
                        "letters": ["c", "a", "b", "a"],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="word.plactic_normal_form.compute",
        title="Compute the canonical plactic representative of a word",
        description="Insert a bounded ordered word under ROW_INSERTION_RSK_V1 and return its insertion tableau with the canonical representative read left-to-right from the bottom row to the top row.",
        request_type=PlacticNormalFormRequest,
        result_type=PlacticNormalFormResult,
        run=plactic_normal_form,
        tags=("combinatorics", "words", "plactic", "rsk", "exact"),
        discovery_terms=(
            "plactic normal form",
            "canonical row reading word",
            "plactic class",
        ),
        examples=(
            OperationExample(
                name="word_plactic_normal_form",
                description="The word (3,1,2) is already the canonical row reading of its insertion tableau.",
                input={
                    "word": {"alphabet": ["1", "2", "3"], "letters": ["3", "1", "2"]}
                },
            ),
        ),
    ),
    MathTool(
        operation_id="word.plactic_equivalence.compute",
        title="Compare two words in the plactic monoid",
        description=(
            "For words over the same explicit ordered alphabet, return both "
            "ROW_INSERTION_RSK_V1 insertion tableaux and canonical row-reading "
            "forms, together with whether the tableaux agree. Equality of "
            "insertion tableaux is the exact plactic-equivalence criterion."
        ),
        request_type=PlacticEquivalenceRequest,
        result_type=PlacticEquivalenceResult,
        run=plactic_equivalence,
        tags=("combinatorics", "words", "plactic", "rsk", "exact"),
        discovery_terms=(
            "plactic equivalence",
            "Knuth equivalent words",
            "compare plactic classes",
        ),
        examples=(
            OperationExample(
                name="knuth_equivalent_words",
                description="The words (1,3,2) and (3,1,2) have the same insertion tableau.",
                input={
                    "left": {"alphabet": ["1", "2", "3"], "letters": ["1", "3", "2"]},
                    "right": {"alphabet": ["1", "2", "3"], "letters": ["3", "1", "2"]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tableau.biword.normalize.compute",
        title="Normalize a labelled biword",
        description="Sort a finite biword by its explicit top and bottom alphabet order and return the exact stable source permutation.",
        request_type=BiwordNormalizeRequest,
        result_type=BiwordNormalizeResult,
        run=_normalize_run,
        tags=("combinatorics", "biword", "sorting", "exact"),
        examples=(
            OperationExample(
                name="normalize_pairs",
                description="Sort pairs (b,a),(a,b) by the explicit top then bottom alphabet order.",
                input={
                    "top_alphabet": ["a", "b"],
                    "bottom_alphabet": ["a", "b"],
                    "top": ["b", "a"],
                    "bottom": ["a", "b"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tableau.rsk.biword.compute",
        title="Compute row-insertion RSK for a canonical biword",
        description="Insert the bottom row of a lexicographically sorted labelled biword and record the top row, returning semistandard tableaux with exact alphabets, shape, and content context.",
        request_type=BiwordRSKRequest,
        result_type=BiwordRSKPair,
        run=_rsk_biword_run,
        tags=("combinatorics", "rsk", "biword", "exact"),
        examples=(
            OperationExample(
                name="biword_basic",
                description="Compute RSK of the sorted biword (a,b),(b,a); pairs must be sorted by top then bottom.",
                input={
                    "biword": {
                        "top_alphabet": ["a", "b"],
                        "bottom_alphabet": ["a", "b"],
                        "top": ["a", "b"],
                        "bottom": ["b", "a"],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tableau.rsk.matrix.compute",
        title="Compute matrix RSK",
        description="Expand a labelled nonnegative integer matrix into its canonical row/column biword and compute exact row-insertion RSK.",
        request_type=MatrixRSKRequest,
        result_type=BiwordRSKPair,
        run=_matrix_rsk_run,
        tags=("combinatorics", "rsk", "matrix", "exact"),
        examples=(
            OperationExample(
                name="matrix_two_by_two",
                description="Compute matrix RSK for [[1,0],[0,1]]; entries are nonnegative on the labelled axes.",
                input={
                    "matrix": {
                        "row_labels": ["a", "b"],
                        "column_labels": ["x", "y"],
                        "entries": [["1", "0"], ["0", "1"]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tableau.rsk.inverse_biword.compute",
        title="Invert biword RSK",
        description="Reconstruct the unique canonical sorted biword from a compatible semistandard RSK pair.",
        request_type=InverseBiwordRSKRequest,
        result_type=Biword,
        run=_inverse_biword_run,
        tags=("combinatorics", "rsk", "inverse", "biword", "exact"),
        examples=(
            OperationExample(
                name="inverse_empty_biword",
                description="Invert the empty biword RSK pair; both alphabets remain explicit.",
                input={
                    "pair": {
                        "top_alphabet": [],
                        "bottom_alphabet": [],
                        "insertion_tableau": {"rows": []},
                        "recording_tableau": {"rows": []},
                        "shape": {"parts": []},
                        "source_kind": "BIWORD",
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tableau.rsk.inverse_matrix.compute",
        title="Invert matrix RSK",
        description="Reconstruct the unique labelled nonnegative matrix from a compatible matrix-RSK pair and explicit row and column axes.",
        request_type=InverseMatrixRSKRequest,
        result_type=NonnegativeIntegerMatrix,
        run=_inverse_matrix_run,
        tags=("combinatorics", "rsk", "inverse", "matrix", "exact"),
        examples=(
            OperationExample(
                name="inverse_identity_matrix",
                description="Invert matrix RSK on the empty pair over explicit empty axes.",
                input={
                    "pair": {
                        "top_alphabet": [],
                        "bottom_alphabet": [],
                        "insertion_tableau": {"rows": []},
                        "recording_tableau": {"rows": []},
                        "shape": {"parts": []},
                        "source_kind": "MATRIX",
                    },
                    "row_labels": [],
                    "column_labels": [],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="word.greene_invariants.compute",
        title="Compute Greene shape invariants",
        description="Return exact increasing and decreasing Greene totals from row-insertion RSK shape for a bounded ordered word.",
        request_type=GreeneRequest,
        result_type=GreeneResult,
        run=_greene_run,
        tags=("combinatorics", "greene", "rsk", "exact"),
        examples=(
            OperationExample(
                name="greene_word",
                description="Compute Greene invariants of (c,a,b) over a<b<c; the word alphabet supplies the order.",
                input={
                    "word": {"alphabet": ["a", "b", "c"], "letters": ["c", "a", "b"]},
                    "k": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="word.greene_witnesses.compute",
        title="Compute disjoint witnesses for Greene invariants",
        description="For a bounded ordered word, return pairwise-disjoint weakly increasing and strictly decreasing subsequence witnesses attaining the first k Greene invariants under ROW_INSERTION_RSK_V1. Indices are zero-based source positions. This witness operation accepts words of length at most 32 and k at most 8.",
        request_type=GreeneWitnessRequest,
        result_type=GreeneWitnessResult,
        run=_greene_witness_run,
        tags=("combinatorics", "greene", "subsequences", "rsk", "exact"),
        discovery_terms=(
            "Greene subsequence witnesses",
            "disjoint increasing subsequences",
            "disjoint decreasing subsequences",
        ),
        examples=(
            OperationExample(
                name="greene_witnesses_213",
                description="Witness the first two Greene totals of the word (2,1,3).",
                input={
                    "word": {"alphabet": ["1", "2", "3"], "letters": ["2", "1", "3"]},
                    "k": 2,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
