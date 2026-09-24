"""Algebraic topology operations."""

from jacobian.math.topology.edge_paths.operations import (
    concatenate_edge_paths,
    edge_path_word,
    free_reduce,
    fundamental_group_presentation,
    presentation_abelianization,
    verify_edge_path_concatenation,
    verify_edge_path_word,
)
from jacobian.math.topology.edge_paths.presentation_maps import (
    direct_relator_match,
    induced_fundamental_group_map,
)

__all__ = [
    "concatenate_edge_paths",
    "direct_relator_match",
    "edge_path_word",
    "free_reduce",
    "fundamental_group_presentation",
    "induced_fundamental_group_map",
    "presentation_abelianization",
    "verify_edge_path_concatenation",
    "verify_edge_path_word",
]
