"""Algebraic topology operations."""

from jacobian.math.topology.edge_paths.operations import (
    concatenate_edge_paths,
    edge_path_word,
    fundamental_group_presentation,
    verify_edge_path_concatenation,
    verify_edge_path_word,
)
from jacobian.math.topology.edge_paths.presentation_maps import homomorphism

__all__ = [
    "concatenate_edge_paths",
    "edge_path_word",
    "fundamental_group_presentation",
    "homomorphism",
    "verify_edge_path_concatenation",
    "verify_edge_path_word",
]
