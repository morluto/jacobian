"""Native domain functions accepting chain-complex domain values.

Each wrapper invokes the typed domain kernel directly and assembles its
declared result value, so direct callers compose through the mathematical
kernels without constructing wire request envelopes or repeating transport
admission.
"""

from __future__ import annotations

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
    from jacobian.math.chain_complexes.operations import _compute_homology_groups

    groups = _compute_homology_groups(complex_value)
    return HomologyResult(
        homology_groups=tuple(groups),
        coefficient_field=complex_value.coefficient_field,
        prime=complex_value.prime,
        degree_min=complex_value.degree_min,
        degree_max=complex_value.degree_max,
        complex=complex_value,
    )


def differential_squares_to_zero(
    complex_value: ChainComplexValue,
) -> VerificationResult:
    """Verify d^2 = 0 for one chain-complex value."""
    from jacobian.math.chain_complexes.operations import _differential_verdict

    is_valid, detail = _differential_verdict(complex_value)
    return VerificationResult(is_valid=is_valid, detail=detail, complex=complex_value)


def chain_map_commutes(
    source: ChainComplexValue,
    target: ChainComplexValue,
    map_matrices: MapMatrices,
) -> VerificationResult:
    """Verify that a component-wise chain map commutes with differentials."""
    from jacobian.math.chain_complexes.operations import _chain_map_verdict

    is_valid, detail = _chain_map_verdict(source, target, map_matrices)
    return VerificationResult(
        is_valid=is_valid,
        detail=detail,
        source=source,
        target=target,
        map_matrices=map_matrices,
    )


def mapping_cone(
    source: ChainComplexValue, target: ChainComplexValue, map_matrices: MapMatrices
) -> MappingConeResult:
    """Compute the mapping cone of a chain-map value."""
    from jacobian.math.chain_complexes.operations import _compute_mapping_cone

    cone_basis_sizes, cone_diffs = _compute_mapping_cone(source, target, map_matrices)
    return MappingConeResult(
        cone_basis_sizes=cone_basis_sizes,
        cone_differential_matrices=cone_diffs,
        source_degree_min=source.degree_min,
        target_degree_min=target.degree_min,
        source=source,
        target=target,
        map_matrices=map_matrices,
        value=ChainComplexValue(
            coefficient_field=source.coefficient_field,
            prime=source.prime,
            degree_min=source.degree_min,
            degree_max=source.degree_min + len(cone_basis_sizes) - 1,
            basis_sizes=cone_basis_sizes,
            differential_matrices=cone_diffs,
        ),
    )


def tensor_product_complex(
    left: ChainComplexValue, right: ChainComplexValue
) -> TensorProductResult:
    """Compute the tensor product of two chain-complex values.

    (C ⊗ D)_n = ⊕_{i+j=n} C_i ⊗ D_j with differential d_C ⊗ id + (-1)^i id ⊗ d_D.
    """
    # Native callers bypass the MCP request model, so the shared tensor
    # work admission must run here too: otherwise canonical inputs whose
    # derived group dimensions exceed the budgets reach the dense kernel
    # expansion before any bound rejects them.
    from jacobian.math.chain_complexes._models import (
        _require_admissible_tensor_work,
    )
    from jacobian.math.chain_complexes.operations import _compute_tensor_product

    _require_admissible_tensor_work(left, right)
    tensor_basis_sizes, tensor_diffs = _compute_tensor_product(left, right)
    # Tensor degrees are pairwise sums: the derived complex concentrates
    # on [deg_min, deg_min + group_count - 1].
    group_count = len(tensor_basis_sizes)
    degree_min = left.degree_min + right.degree_min
    degree_max = degree_min + group_count - 1
    return TensorProductResult(
        tensor_basis_sizes=tensor_basis_sizes,
        tensor_differential_matrices=tensor_diffs,
        coefficient_field=left.coefficient_field,
        prime=left.prime,
        degree_min=degree_min,
        degree_max=degree_max,
        left=left,
        right=right,
        value=ChainComplexValue(
            coefficient_field=left.coefficient_field,
            prime=left.prime,
            degree_min=degree_min,
            degree_max=degree_max,
            basis_sizes=tensor_basis_sizes,
            differential_matrices=tensor_diffs,
        ),
    )
