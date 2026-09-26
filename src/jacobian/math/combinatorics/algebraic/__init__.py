"""Algebraic combinatorics operations."""

from jacobian.math.combinatorics.algebraic._models import (
    LongestDecreasingSubsequenceResult,
    LongestIncreasingSubsequenceResult,
    PartitionDominanceResult,
    PlacticNormalFormResult,
    PlacticEquivalenceResult,
    SemistandardTableauCheckResult,
    SemistandardYoungTableauCountResult,
    StandardTableauCheckResult,
)
from jacobian.math.combinatorics.algebraic.biword import (
    Biword,
    GreeneWitnessResult,
    BiwordRSKPair,
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
from jacobian.math.combinatorics.algebraic.operations import (
    check_semistandard_tableau,
    check_standard_tableau,
    conjugate_partition,
    hook_lengths,
    inverse_permutation_rsk,
    inverse_row_insertion_rsk,
    knuth_moves,
    partition_dominance,
    permutation_rsk,
    plactic_equivalence,
    plactic_normal_form,
    row_insertion_rsk,
    semistandard_young_tableaux_count,
    standard_young_tableaux_count,
    tableau_row_reading_word,
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
from jacobian.math.combinatorics.algebraic.weighted_monotone._models import (
    EndpointProfileEntry,
    EndpointProfileResult,
    WeightedOrderedWord,
)
from jacobian.math.combinatorics.algebraic.weighted_monotone.operations import (
    compute_endpoint_profile,
)

__all__ = [
    "Biword",
    "BiwordRSKPair",
    "EndpointProfileEntry",
    "EndpointProfileResult",
    "FinitePermutation",
    "GreeneWitnessResult",
    "LongestDecreasingSubsequenceResult",
    "LongestIncreasingSubsequenceResult",
    "NonnegativeIntegerMatrix",
    "PartitionDominanceResult",
    "PermutationRSKPair",
    "PlacticEquivalenceResult",
    "PlacticNormalFormResult",
    "RSKTableauPair",
    "SemistandardTableauCheckResult",
    "SemistandardYoungTableauCountResult",
    "StandardTableauCheckResult",
    "WeightedOrderedWord",
    "check_semistandard_tableau",
    "check_standard_tableau",
    "compute_endpoint_profile",
    "compute_greene_witnesses",
    "conjugate_partition",
    "greene",
    "hook_lengths",
    "inverse_biword",
    "inverse_matrix",
    "inverse_permutation_rsk",
    "inverse_row_insertion_rsk",
    "knuth_moves",
    "longest_decreasing_subsequence",
    "longest_increasing_subsequence",
    "matrix_biword",
    "normalize_biword",
    "partition_dominance",
    "permutation_rsk",
    "plactic_equivalence",
    "plactic_normal_form",
    "row_insertion_rsk",
    "rsk_biword",
    "semistandard_young_tableaux_count",
    "standard_young_tableaux_count",
    "tableau_row_reading_word",
]
