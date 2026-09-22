"""Algebraic combinatorics operations."""

from jacobian.math.combinatorics.algebraic._models import (
    PartitionDominanceResult,
    SemistandardTableauCheckResult,
    SemistandardYoungTableauCountResult,
    StandardTableauCheckResult,
)
from jacobian.math.combinatorics.algebraic.biword import (
    Biword,
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
from jacobian.math.combinatorics.algebraic.operations import (
    check_semistandard_tableau,
    check_standard_tableau,
    conjugate_partition,
    hook_lengths,
    inverse_row_insertion_rsk,
    knuth_moves,
    partition_dominance,
    row_insertion_rsk,
    semistandard_young_tableaux_count,
    standard_young_tableaux_count,
    verify_rsk,
)
from jacobian.math.combinatorics.algebraic.values import RSKTableauPair
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
    "NonnegativeIntegerMatrix",
    "PartitionDominanceResult",
    "RSKTableauPair",
    "SemistandardTableauCheckResult",
    "SemistandardYoungTableauCountResult",
    "StandardTableauCheckResult",
    "WeightedOrderedWord",
    "check_semistandard_tableau",
    "check_standard_tableau",
    "compute_endpoint_profile",
    "conjugate_partition",
    "greene",
    "hook_lengths",
    "inverse_biword",
    "inverse_matrix",
    "inverse_row_insertion_rsk",
    "knuth_moves",
    "matrix_biword",
    "normalize_biword",
    "partition_dominance",
    "row_insertion_rsk",
    "rsk_biword",
    "semistandard_young_tableaux_count",
    "standard_young_tableaux_count",
    "verify_rsk",
]
