"""Algebraic topology operations."""

from jacobian.math.topology.edge_paths.operations import (
    concatenate_edge_paths,
    edge_path_word,
    fundamental_group_presentation,
    verify_edge_path_concatenation,
    verify_edge_path_word,
)

__all__ = [
    "concatenate_edge_paths",
    "edge_path_word",
    "fundamental_group_presentation",
    "verify_edge_path_concatenation",
    "verify_edge_path_word",
]
