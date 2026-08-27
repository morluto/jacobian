"""Declarations for 3-term progression hypergraph construction."""

from jacobian.catalog._examples import example
from jacobian.math.combinatorics._progression_hypergraph_models import (
    ProgressionHypergraphRequest,
    ProgressionHypergraphResult,
)
from jacobian.math.combinatorics._progression_hypergraph_operations import (
    construct_3term_progression_hypergraph,
)
from jacobian.math.combinatorics._support import combinatorics_operation

PROGRESSION_HYPERGRAPH_OPERATION = combinatorics_operation(
    "combinatorics.finite_abelian.3term_progression_hypergraph.construct",
    "Construct 3-term progression hypergraph of a finite cyclic group",
    "Construct the 3-uniform hypergraph whose edges are all 3-term arithmetic progressions in Z/nZ.",
    ProgressionHypergraphRequest,
    ProgressionHypergraphResult,
    construct_3term_progression_hypergraph,
    "combinatorics",
    "additive-combinatorics",
    "hypergraph",
    examples=(
        example(
            "three_ap_z5",
            "Construct the 3-AP hypergraph of Z/5Z; the group order must be at least 2.",
            {"group_order": 5},
        ),
    ),
)

__all__ = ["PROGRESSION_HYPERGRAPH_OPERATION"]
