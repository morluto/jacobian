"""Finite based chain complexes over exact fields."""

from jacobian.math.chain_complexes.native import (
    chain_map_commutes,
    differential_squares_to_zero,
    homology_groups,
    mapping_cone,
    tensor_product_complex,
)

# The authoritative native surface: every export accepts domain values
# directly. Wire-envelope request handlers live in
# ``jacobian.math.chain_complexes.operations`` and are projected to MCP
# through ``_tools.py``; they are not part of this native API.
__all__ = [
    "chain_map_commutes",
    "differential_squares_to_zero",
    "homology_groups",
    "mapping_cone",
    "tensor_product_complex",
]
