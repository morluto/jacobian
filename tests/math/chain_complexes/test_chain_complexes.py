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


class TestChainMapEntryGrammar:
    def test_unparseable_entries_rejected_at_admission(self) -> None:
        """Correctly shaped components with junk entries fail at the
        boundary instead of inside _parse_fraction."""
        circle = _circle_complex()
        identity = (("1", "0", "0"), ("0", "1", "0"), ("0", "x", "1"))
        with pytest.raises(ValueError, match="rational string grammar"):
            VerifyChainMapRequest(
                source=circle, target=circle, map_matrices=(identity, identity)
            )
        zero_den = (("1", "0", "0"), ("0", "1/0", "0"), ("0", "0", "1"))
        with pytest.raises(ValueError):
            MappingConeRequest(
                source=circle, target=circle, map_matrices=(zero_den, zero_den)
            )


class TestTensorProductPreconditions:
    def test_non_square_zero_factor_rejected(self) -> None:
        """Tensoring complexes violating d^2=0 is rejected before building."""
        bad = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=2,
            basis_sizes=(1, 1, 1),
            differential_matrices=((("1",),), (("1",),)),
        )
        point = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=0,
            basis_sizes=(1,),
            differential_matrices=(),
        )
        with pytest.raises(ValueError, match="d\\^2=0"):
            compute_tensor_product(TensorProductRequest(left=bad, right=point))
        with pytest.raises(ValueError, match="d\\^2=0"):
            compute_tensor_product(TensorProductRequest(left=point, right=bad))

    def test_allocated_differential_cells_are_bounded(self) -> None:
        """A singleton 64-dim left factor against a 65-degree right factor
        passes group-dimension checks but allocates ~4M dense cells; the
        cell-count work bound rejects it."""
        left = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=0,
            basis_sizes=(64,),
            differential_matrices=(),
        )
        right = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=32,
            basis_sizes=(4,) * 33,
            differential_matrices=((("0", "0", "0", "0"),) * 4,) * 32,
        )
        with pytest.raises(ValueError, match="work bound"):
            TensorProductRequest(left=left, right=right)


class TestHomologySourceBinding:
    def test_result_retains_and_replays_source(self) -> None:
        from jacobian.math.chain_complexes.values import HomologyResult

        complex_value = _circle_complex()
        request = ComputeHomologyRequest(complex=complex_value)
        result = compute_homology(request)
        assert result.complex == complex_value
        revalidated = HomologyResult.model_validate(result.model_dump())
        assert revalidated.homology_groups == result.homology_groups

    def test_forged_profile_is_rejected(self) -> None:
        from jacobian.math.chain_complexes.values import (
            HomologyGroupValue,
            HomologyResult,
        )

        payload_groups = (
            HomologyGroupValue(
                degree=0, cycle_rank=5, boundary_rank=0, betti_number=100
            ),
        )
        with pytest.raises(ValueError, match="betti_number"):
            HomologyResult(
                homology_groups=payload_groups,
                coefficient_field=CoefficientField.RATIONAL,
                degree_min=0,
                degree_max=0,
                complex=_point_complex(),
            )


class TestAggregateChainMapWork:
    def test_aggregate_component_cells_bounded(self) -> None:
        """Alternating (64, 0) sizes pass per-complex cell checks but would
        admit dozens of dense 64x64 map components; the aggregate budget
        rejects them."""
        alternating = tuple(64 if i % 2 == 0 else 0 for i in range(33))
        complex_value = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=32,
            basis_sizes=alternating,
            differential_matrices=tuple(
                ((),) * 64 if i % 2 == 0 else () for i in range(32)
            ),
        )
        identity_64 = tuple(
            tuple("1" if i == j else "0" for j in range(64)) for i in range(64)
        )
        components = tuple(identity_64 if i % 2 == 0 else () for i in range(33))
        with pytest.raises(ValueError, match="aggregate"):
            VerifyChainMapRequest(
                source=complex_value, target=complex_value, map_matrices=components
            )


class TestHomologyParentMatch:
    def test_parent_mismatch_rejected(self) -> None:
        from jacobian.math.chain_complexes.values import (
            CoefficientField,
            HomologyGroupValue,
            HomologyResult,
        )

        with pytest.raises(ValueError, match="field and prime must match"):
            HomologyResult(
                homology_groups=(
                    HomologyGroupValue(
                        degree=0, cycle_rank=1, boundary_rank=0, betti_number=1
                    ),
                ),
                coefficient_field=CoefficientField.PRIME_FIELD,
                prime=2,
                degree_min=0,
                degree_max=0,
                complex=_point_complex(),
            )


class TestNativeSurface:
    def test_domain_value_functions(self) -> None:
        """Native exports accept domain values, not request envelopes."""
        from jacobian.math.chain_complexes import (
            chain_map_commutes,
            homology_groups,
            tensor_product_complex,
        )

        circle = _circle_complex()
        groups = homology_groups(circle)
        assert groups[0].betti_number == 1 and groups[1].betti_number == 1

        identity = (("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1"))
        verdict = chain_map_commutes(circle, circle, (identity, identity))
        assert verdict.is_valid

        product = tensor_product_complex(_point_complex(), _point_complex())
        assert product.tensor_basis_sizes == (1,)


class TestChainMapEndpointPrecondition:
    def test_non_square_zero_endpoints_fail_verification(self) -> None:
        """Endpoints violating d^2=0 admit no chain map; the identity
        components must not validate as commuting."""
        from jacobian.math.chain_complexes.operations import verify_chain_map

        bad = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=2,
            basis_sizes=(1, 1, 1),
            differential_matrices=((("1",),), (("1",),)),
        )
        one = (("1",),)
        result = verify_chain_map(
            VerifyChainMapRequest(source=bad, target=bad, map_matrices=(one, one, one))
        )
        assert result.is_valid is False
        assert "d^2=0" in result.detail


class TestZeroWidthProducts:
    def test_zero_row_operand_preserves_columns(self) -> None:
        from jacobian.math.chain_complexes.operations import verify_chain_map

        source = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=1,
            basis_sizes=(1, 1),
            differential_matrices=((("0",),),),
        )
        target = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=1,
            basis_sizes=(1, 0),
            differential_matrices=(((),),),
        )
        # Component 0: 1x1 zero; component 1: target has no degree-1 group.
        map_matrices = ((("0",),), ())
        result = verify_chain_map(
            VerifyChainMapRequest(
                source=source, target=target, map_matrices=map_matrices
            )
        )
        assert result.is_valid


class TestTensorContextAndShapes:
    def test_tensor_carries_canonical_context(self) -> None:
        result = compute_tensor_product(
            TensorProductRequest(left=_point_complex(), right=_point_complex())
        )
        assert result.coefficient_field == CoefficientField.RATIONAL
        assert result.prime is None
        assert (result.degree_min, result.degree_max) == (0, 0)

    def test_shifted_tensor_degree_interval(self) -> None:
        left = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=-1,
            degree_max=-1,
            basis_sizes=(1,),
            differential_matrices=(),
        )
        right = _point_complex()
        result = compute_tensor_product(TensorProductRequest(left=left, right=right))
        assert (result.degree_min, result.degree_max) == (-1, -1)

    def test_zero_width_differential_keeps_rows(self) -> None:
        """A (1,0)-by-point tensor must represent its zero differential as a
        one-row zero-width matrix, not an empty matrix."""
        left = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=1,
            basis_sizes=(1, 0),
            differential_matrices=(((),),),
        )
        right = _point_complex()
        result = compute_tensor_product(TensorProductRequest(left=left, right=right))
        assert result.tensor_basis_sizes == (1, 0)
        assert result.tensor_differential_matrices == (((),),)

    def test_every_endpoint_product_checked(self) -> None:
        """A four-term endpoint with differentials (0, 1, 1) has d0*d1 == 0
        but d1*d2 != 0; the identity map must fail verification."""
        from jacobian.math.chain_complexes.operations import verify_chain_map

        bad = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=3,
            basis_sizes=(1, 1, 1, 1),
            differential_matrices=((("0",),), (("1",),), (("1",),)),
        )
        one = (("1",),)
        result = verify_chain_map(
            VerifyChainMapRequest(
                source=bad, target=bad, map_matrices=(one, one, one, one)
            )
        )
        assert result.is_valid is False
        assert "d^2=0" in result.detail


class TestTensorValueComposition:
    def test_tensor_result_exposes_chain_complex_value(self) -> None:
        """The derived complex composes as a first-class value."""
        from jacobian.math.chain_complexes.values import ChainComplexValue

        result = compute_tensor_product(
            TensorProductRequest(left=_point_complex(), right=_point_complex())
        )
        assert isinstance(result.value, ChainComplexValue)
        assert result.value.basis_sizes == (1,)
        homology_groups(result.value)


def homology_groups(complex_value):
    from jacobian.math.chain_complexes import homology_groups as native

    return native(complex_value)
