"""Finite based chain complexes over exact fields."""

from jacobian.math.chain_complexes.native import (
    chain_map_commutes,
    differential_squares_to_zero,
    homology_groups,
    mapping_cone,
    tensor_product_complex,
)
from jacobian.math.chain_complexes.operations import (
    compute_homology,
    compute_mapping_cone,
    compute_tensor_product,
    construct_chain_complex,
    verify_chain_map,
    verify_differential,
)

# The first five names are the native surface: they accept domain values
# directly. The remainder are the wire-envelope handlers used by MCP.
__all__ = [
    "chain_map_commutes",
    "compute_homology",
    "compute_mapping_cone",
    "compute_tensor_product",
    "construct_chain_complex",
    "differential_squares_to_zero",
    "homology_groups",
    "mapping_cone",
    "tensor_product_complex",
    "verify_chain_map",
    "verify_differential",
]
