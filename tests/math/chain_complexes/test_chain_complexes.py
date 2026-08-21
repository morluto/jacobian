"""Tests for chain complex operations (#1824)."""

import pytest

from jacobian.math.chain_complexes._models import (
    ComputeHomologyRequest,
    ConstructChainComplexRequest,
    MappingConeRequest,
    TensorProductRequest,
    VerifyChainMapRequest,
    VerifyDifferentialRequest,
)
from jacobian.math.chain_complexes.operations import (
    compute_homology,
    compute_mapping_cone,
    compute_tensor_product,
    construct_chain_complex,
    verify_chain_map,
    verify_differential,
)
from jacobian.math.chain_complexes.values import ChainComplexValue, CoefficientField


def _circle_complex() -> ChainComplexValue:
    return ChainComplexValue(
        coefficient_field=CoefficientField.RATIONAL,
        degree_min=0,
        degree_max=1,
        basis_sizes=(3, 3),
        differential_matrices=(
            (("-1", "1", "0"), ("0", "-1", "1"), ("0", "0", "0")),
        ),
    )


def _point_complex() -> ChainComplexValue:
    """A single point: H_0 = 1, all others 0."""
    return ChainComplexValue(
        coefficient_field=CoefficientField.RATIONAL,
        degree_min=0,
        degree_max=0,
        basis_sizes=(1,),
        differential_matrices=(),
    )


class TestConstructChainComplex:
    def test_construct_circle(self) -> None:
        result = construct_chain_complex(
            ConstructChainComplexRequest(
                basis_sizes=(3, 3),
                differential_matrices=(
                    (("-1", "1", "0"), ("0", "-1", "1"), ("0", "0", "0")),
                ),
            )
        )
        assert result.basis_sizes == (3, 3)
        assert result.degree_max == 1


class TestVerifyDifferential:
    def test_valid_d2_zero(self) -> None:
        result = verify_differential(
            VerifyDifferentialRequest(complex=_circle_complex())
        )
        assert result.is_valid

    def test_point_has_valid_d2(self) -> None:
        result = verify_differential(
            VerifyDifferentialRequest(complex=_point_complex())
        )
        assert result.is_valid


class TestComputeHomology:
    def test_circle_homology(self) -> None:
        result = compute_homology(
            ComputeHomologyRequest(complex=_circle_complex())
        )
        assert result.homology_groups[0].betti_number == 1
        assert result.homology_groups[1].betti_number == 1

    def test_point_homology(self) -> None:
        result = compute_homology(
            ComputeHomologyRequest(complex=_point_complex())
        )
        assert result.homology_groups[0].betti_number == 1


class TestTensorProduct:
    def test_tensor_two_points(self) -> None:
        result = compute_tensor_product(
            TensorProductRequest(left=_point_complex(), right=_point_complex())
        )
        # Tensor of two points is a point: one group of size 1
        assert result.tensor_basis_sizes == (1,)


class TestMappingCone:
    def test_mapping_cone_identity(self) -> None:
        circle = _circle_complex()
        identity_3x3 = (("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1"))
        identity = (identity_3x3,)
        result = compute_mapping_cone(
            MappingConeRequest(
                source=circle,
                target=circle,
                map_matrices=identity,
            )
        )
        assert result.cone_basis_sizes is not None
