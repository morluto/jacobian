"""Supported native additive-combinatorics API."""

from jacobian.math.combinatorics.additive.operations import (
    additive_energy,
    direct_sum_predicate,
    representation_profile,
    subset_sum_profile,
    sumset_cardinality,
    verify_additive_energy,
    verify_direct_sum_predicate,
    verify_ordered_difference_profile,
    verify_representation_profile,
    verify_sumset_cardinality,
)
from jacobian.math.combinatorics.additive.values import (
    IndexedIntegerSequence,
    IndexSubset,
    SubsetSumProfile,
    SubsetSumProfileEntry,
)

__all__ = [
    "IndexSubset",
    "IndexedIntegerSequence",
    "SubsetSumProfile",
    "SubsetSumProfileEntry",
    "additive_energy",
    "direct_sum_predicate",
    "representation_profile",
    "subset_sum_profile",
    "sumset_cardinality",
    "verify_additive_energy",
    "verify_direct_sum_predicate",
    "verify_ordered_difference_profile",
    "verify_representation_profile",
    "verify_sumset_cardinality",
]
