"""Finite based chain complexes over exact coefficient rings."""

from jacobian.math.topology.chain_complexes.operations import (
    chain_map_commutes,
    construct_chain_complex,
    differential_squares_to_zero,
    homology_groups,
    mapping_cone,
    tensor_product_complex,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
    HomologyGroup,
    HomologyGroupValue,
    HomologyResult,
    IntegralFreeGenerator,
    IntegralHomologyGroupValue,
    IntegralTorsionGenerator,
    IntegralVector,
)

# The authoritative native surface: every export accepts domain values
# directly. Wire-envelope request handlers live in ``_tools.py`` and are not
# part of this native API.
__all__ = [
    "ChainComplexValue",
    "CoefficientRing",
    "HomologyGroup",
    "HomologyGroupValue",
    "HomologyResult",
    "IntegralFreeGenerator",
    "IntegralHomologyGroupValue",
    "IntegralTorsionGenerator",
    "IntegralVector",
    "chain_map_commutes",
    "construct_chain_complex",
    "differential_squares_to_zero",
    "homology_groups",
    "mapping_cone",
    "tensor_product_complex",
]
