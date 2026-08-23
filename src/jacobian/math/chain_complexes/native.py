"""Native domain functions accepting chain-complex domain values."""

from __future__ import annotations

from jacobian.math.chain_complexes._models import (
    ComputeHomologyRequest,
    MappingConeRequest,
    TensorProductRequest,
    VerifyChainMapRequest,
    VerifyDifferentialRequest,
)
from jacobian.math.chain_complexes.values import (
    ChainComplexValue,
    HomologyResult,
    MappingConeResult,
    TensorProductResult,
    VerificationResult,
)

__all__ = [
    "chain_map_commutes",
    "differential_squares_to_zero",
    "homology_groups",
    "mapping_cone",
    "tensor_product_complex",
]

MapMatrices = tuple[tuple[tuple[str, ...], ...], ...]


def homology_groups(
    complex_value: ChainComplexValue,
) -> HomologyResult:
    """Exact homology groups bound to their source complex and field.

    The full ``HomologyResult`` carries the coefficient field, prime, and
    degree interval so homology over different fields stays
    distinguishable as a serialized value.
    """
    from jacobian.math.chain_complexes.operations import (
        compute_homology,
    )

    return compute_homology(ComputeHomologyRequest(complex=complex_value))


def differential_squares_to_zero(
    complex_value: ChainComplexValue,
) -> VerificationResult:
    """Verify d^2 = 0 for one chain-complex value."""
    from jacobian.math.chain_complexes.operations import verify_differential

    return verify_differential(VerifyDifferentialRequest(complex=complex_value))


def chain_map_commutes(
    source: ChainComplexValue,
    target: ChainComplexValue,
    map_matrices: MapMatrices,
) -> VerificationResult:
    """Verify that a component-wise chain map commutes with differentials."""
    from jacobian.math.chain_complexes.operations import verify_chain_map

    return verify_chain_map(
        VerifyChainMapRequest(source=source, target=target, map_matrices=map_matrices)
    )


def mapping_cone(
    source: ChainComplexValue, target: ChainComplexValue, map_matrices: MapMatrices
) -> MappingConeResult:
    """Compute the mapping cone of a chain-map value."""
    from jacobian.math.chain_complexes.operations import compute_mapping_cone

    return compute_mapping_cone(
        MappingConeRequest(source=source, target=target, map_matrices=map_matrices)
    )


def tensor_product_complex(
    left: ChainComplexValue, right: ChainComplexValue
) -> TensorProductResult:
    """Compute the tensor product of two chain-complex values."""
    from jacobian.math.chain_complexes.operations import compute_tensor_product

    return compute_tensor_product(TensorProductRequest(left=left, right=right))
