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
    verify_differential,
)
from jacobian.math.chain_complexes.values import ChainComplexValue, CoefficientField


def _circle_complex() -> ChainComplexValue:
    return ChainComplexValue(
        coefficient_field=CoefficientField.RATIONAL,
        degree_min=0,
        degree_max=1,
        basis_sizes=(3, 3),
        differential_matrices=((("-1", "1", "0"), ("0", "-1", "1"), ("0", "0", "0")),),
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
        result = compute_homology(ComputeHomologyRequest(complex=_circle_complex()))
        assert result.homology_groups[0].betti_number == 1
        assert result.homology_groups[1].betti_number == 1

    def test_point_homology(self) -> None:
        result = compute_homology(ComputeHomologyRequest(complex=_point_complex()))
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
        identity = (identity_3x3, identity_3x3)
        result = compute_mapping_cone(
            MappingConeRequest(
                source=circle,
                target=circle,
                map_matrices=identity,
            )
        )
        assert result.cone_basis_sizes is not None


class TestChainMapAdmission:
    """Thread fixes: shapes, degree alignment, and defining equations are
    validated at the request boundary."""

    def _two_term(self, degree_min: int, differential: str) -> ChainComplexValue:
        return ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=degree_min,
            degree_max=degree_min + 1,
            basis_sizes=(1, 1),
            differential_matrices=(((differential,),),),
        )

    def test_padded_zero_map_is_rejected(self) -> None:
        """A one-dimensional to two-dimensional map cannot silently pad."""
        source = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=0,
            basis_sizes=(1,),
            differential_matrices=(),
        )
        target = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=0,
            basis_sizes=(2,),
            differential_matrices=(),
        )
        with pytest.raises(ValueError, match="2x1"):
            VerifyChainMapRequest(source=source, target=target, map_matrices=((),))
        with pytest.raises(ValueError, match="2x1"):
            MappingConeRequest(source=source, target=target, map_matrices=((),))

    def test_oversized_matrix_is_rejected(self) -> None:
        circle = _circle_complex()
        oversized = (("1", "0", "0", "0"), ("0", "1", "0", "0"), ("0", "0", "1", "0"))
        with pytest.raises(ValueError, match="3x3"):
            VerifyChainMapRequest(
                source=circle,
                target=circle,
                map_matrices=(oversized, oversized),
            )

    def test_mismatched_degree_intervals_are_rejected(self) -> None:
        shifted = self._two_term(-1, "1")
        circle = _circle_complex()
        ones = (("1",),)
        with pytest.raises(ValueError, match="same degree interval"):
            VerifyChainMapRequest(
                source=circle, target=shifted, map_matrices=(ones, ones)
            )
        with pytest.raises(ValueError, match="same degree interval"):
            MappingConeRequest(source=circle, target=shifted, map_matrices=(ones, ones))

    def test_incomplete_component_count_is_rejected(self) -> None:
        circle = _circle_complex()
        identity = (("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1"))
        with pytest.raises(ValueError, match="per chain degree"):
            VerifyChainMapRequest(
                source=circle, target=circle, map_matrices=(identity,)
            )
        with pytest.raises(ValueError, match="per chain degree"):
            MappingConeRequest(source=circle, target=circle, map_matrices=(identity,))


class TestMappingConeDefiningEquations:
    """A cone is returned only for square-zero complexes and genuine chain maps."""

    def _complex(self, differential: str) -> ChainComplexValue:
        return ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=1,
            basis_sizes=(1, 1),
            differential_matrices=(((differential,),),),
        )

    def test_non_chain_map_cone_is_rejected(self) -> None:
        """Source d=[1], target d=[0], f=[1]: the cone would not be a chain complex."""
        source = self._complex("1")
        target = self._complex("0")
        one = (("1",),)
        with pytest.raises(ValueError, match="commute"):
            compute_mapping_cone(
                MappingConeRequest(
                    source=source, target=target, map_matrices=(one, one)
                )
            )

    def test_non_square_zero_source_cone_is_rejected(self) -> None:
        """A three-term complex violating d^2=0 is rejected before the cone."""
        bad = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=2,
            basis_sizes=(1, 1, 1),
            differential_matrices=((("1",),), (("1",),)),
        )
        one = (("1",),)
        with pytest.raises(ValueError, match="d\\^2=0"):
            compute_mapping_cone(
                MappingConeRequest(source=bad, target=bad, map_matrices=(one, one, one))
            )


class TestTensorProductSignAndBudget:
    def test_koszul_sign_uses_actual_chain_degree(self) -> None:
        """A left factor concentrated in odd degree -1 negates id ⊗ d_D."""
        left = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=-1,
            degree_max=-1,
            basis_sizes=(1,),
            differential_matrices=(),
        )
        right = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=1,
            basis_sizes=(1, 1),
            differential_matrices=((("1",),),),
        )
        result = compute_tensor_product(TensorProductRequest(left=left, right=right))
        assert result.tensor_basis_sizes == (1, 1)
        assert result.tensor_differential_matrices == ((("-1",),),)

    def test_tensor_intermediate_dimensions_are_bounded(self) -> None:
        """Two 64+64 factors meet the input cell limit but their middle
        tensor group has dimension 8192; admission must reject it."""
        zeros_row = tuple("0" for _ in range(64))
        zeros = tuple(zeros_row for _ in range(64))
        big = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=1,
            basis_sizes=(64, 64),
            differential_matrices=(zeros,),
        )
        with pytest.raises(ValueError, match="work bound"):
            TensorProductRequest(left=big, right=big)


class TestPrimeFieldEntries:
    def test_fractional_prime_field_entry_rejected(self) -> None:
        """GF_p entries must be integer residues; "1/2" cannot reach execution."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="integer residue"):
            ChainComplexValue(
                coefficient_field=CoefficientField.PRIME_FIELD,
                prime=5,
                degree_min=0,
                degree_max=1,
                basis_sizes=(1, 1),
                differential_matrices=((("1/2",),),),
            )

    def test_integer_residues_accepted(self) -> None:
        complex_value = ChainComplexValue(
            coefficient_field=CoefficientField.PRIME_FIELD,
            prime=5,
            degree_min=0,
            degree_max=1,
            basis_sizes=(2, 2),
            differential_matrices=((("4", "1"), ("2", "3")),),
        )
        result = verify_differential(VerifyDifferentialRequest(complex=complex_value))
        assert result.is_valid
