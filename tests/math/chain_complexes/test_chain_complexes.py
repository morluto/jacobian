"""Tests for chain complex operations (#1824)."""

from typing import Any

import pytest
from pydantic import ValidationError

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
    verify_mapping_cone_result,
    verify_tensor_product_result,
    verify_verification_result,
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


class TestConstructAdmitsOnlyChainComplexes:
    def test_non_square_zero_differentials_rejected_at_admission(self) -> None:
        """Identity differentials on 1-dim groups compose to the identity,
        not zero: the public construct operation must refuse them instead
        of labelling arbitrary matrices an exact chain complex."""
        with pytest.raises(ValidationError) as exc_info:
            ConstructChainComplexRequest(
                coefficient_field=CoefficientField.RATIONAL,
                basis_sizes=(1, 1, 1),
                differential_matrices=((("1",),), (("1",),)),
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "chain_complex.differential_not_square_zero"
        )

    def test_square_zero_differentials_admitted(self) -> None:
        from jacobian.math.chain_complexes.values import ChainComplexValue

        request = ConstructChainComplexRequest(
            coefficient_field=CoefficientField.RATIONAL,
            basis_sizes=(1, 1, 1),
            differential_matrices=((("0",),), (("0",),)),
        )
        value = ChainComplexValue(
            coefficient_field=request.coefficient_field,
            prime=request.prime,
            degree_min=0,
            degree_max=len(request.basis_sizes) - 1,
            basis_sizes=request.basis_sizes,
            differential_matrices=request.differential_matrices,
        )
        assert value.basis_sizes == (1, 1, 1)


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

    def test_result_requires_canonical_value(self) -> None:
        """A tensor result without its canonical value cannot validate, and
        projections that disagree with the retained value are rejected."""
        from pydantic import ValidationError

        from jacobian.math.chain_complexes.values import TensorProductResult

        with pytest.raises(ValidationError):
            TensorProductResult(
                tensor_basis_sizes=(1,), tensor_differential_matrices=()
            )
        result = compute_tensor_product(
            TensorProductRequest(left=_point_complex(), right=_point_complex())
        )
        payload = result.model_dump()
        payload["tensor_basis_sizes"] = (5,)
        with pytest.raises(ValidationError):
            TensorProductResult.model_validate(payload)


class TestCanonicalCoefficientSpellings:
    def test_noncanonical_spellings_rejected(self) -> None:
        """One rational has one spelling: no leading zeros, reduced
        fractions, integer denominators spelled as integers."""
        from pydantic import ValidationError

        for bad in ("01", "2/4", "3/1", "-0", "0/2", "007"):
            with pytest.raises(ValidationError):
                ChainComplexValue(
                    coefficient_field=CoefficientField.RATIONAL,
                    degree_min=0,
                    degree_max=1,
                    basis_sizes=(1, 1),
                    differential_matrices=(((bad,),),),
                )

    def test_prime_field_residues_bounded(self) -> None:
        """GF_p entries are residues in [0, p), canonically spelled."""
        from pydantic import ValidationError

        for bad in ("7", "-1"):
            with pytest.raises(ValidationError):
                ChainComplexValue(
                    coefficient_field=CoefficientField.PRIME_FIELD,
                    prime=5,
                    degree_min=0,
                    degree_max=1,
                    basis_sizes=(1, 1),
                    differential_matrices=(((bad,),),),
                )


class TestAggregateEntryWorkBound:
    def test_dense_high_height_matrix_rejected(self) -> None:
        """A dense 32x32 matrix of 500-digit entries is sub-megabyte but
        exceeds the aggregate digit-work budget coupled to elimination."""
        from pydantic import ValidationError

        big = tuple(
            tuple(str(10**500 + r * 33 + c) for c in range(32)) for r in range(32)
        )
        with pytest.raises(ValidationError):
            ChainComplexValue(
                coefficient_field=CoefficientField.RATIONAL,
                degree_min=0,
                degree_max=1,
                basis_sizes=(32, 32),
                differential_matrices=(big,),
            )


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

    def test_noncommuting_map_rejected_at_admission(self) -> None:
        """Correctly shaped square-zero endpoints with a noncommuting map
        are out of the operation's domain: admission rejects the request
        instead of letting execution die inside the cone construction."""
        zero = self._complex("0")
        one = self._complex("1")
        with pytest.raises(ValidationError):
            MappingConeRequest(
                source=zero,
                target=one,
                # f_0 = 0, f_1 = 1: d_target * f_1 = 1 != 0 = f_0 * d_source
                map_matrices=((("0",),), (("1",),)),
            )

    def test_retained_map_components_must_match_request_contract(self) -> None:
        """A serialized cone cannot revalidate against an undersized
        retained map that replay padding would silently accept."""
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
            basis_sizes=(1,),
            differential_matrices=(),
        )
        result = compute_mapping_cone(
            MappingConeRequest(source=source, target=target, map_matrices=((("0",),),))
        )
        payload = result.model_dump()
        payload["map_matrices"] = [()]
        from jacobian.math.chain_complexes.values import MappingConeResult

        assert not verify_mapping_cone_result(MappingConeResult.model_validate(payload))


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

    def test_expanded_coefficients_rejected_at_admission(self) -> None:
        """A (7,7) dense 512-digit differential by an 8-dimensional point
        repeats 49 coefficients eight times; admission must reject the
        expansion instead of failing inside result construction."""
        big_row = tuple(str(10**511 + i) for i in range(7))
        big = tuple(big_row for _ in range(7))
        left = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=1,
            basis_sizes=(7, 7),
            differential_matrices=(big,),
        )
        point = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=0,
            basis_sizes=(8,),
            differential_matrices=(),
        )
        with pytest.raises(ValueError, match="serialization exceeds"):
            TensorProductRequest(left=left, right=point)

    def test_small_coefficient_expansion_still_accepted(self) -> None:
        """The same shapes with single-digit coefficients stay inside the
        expanded-character budget and return the typed tensor value."""
        left = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=1,
            basis_sizes=(7, 7),
            differential_matrices=((("1",) * 7,) * 7,),
        )
        point = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=0,
            basis_sizes=(8,),
            differential_matrices=(),
        )
        result = compute_tensor_product(TensorProductRequest(left=left, right=point))
        assert result.tensor_basis_sizes == (56, 56)
        assert verify_differential(
            VerifyDifferentialRequest(complex=result.value)
        ).is_valid

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

        with pytest.raises(ValidationError):
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
        result = homology_groups(circle)
        assert result.homology_groups[0].betti_number == 1
        assert result.homology_groups[1].betti_number == 1
        # The native value carries the source's field context so GF(p) and
        # QQ homology stay distinguishable.
        assert result.coefficient_field == circle.coefficient_field
        assert result.prime == circle.prime

        identity = (("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1"))
        verdict = chain_map_commutes(circle, circle, (identity, identity))
        assert verdict.is_valid

        product = tensor_product_complex(_point_complex(), _point_complex())
        assert product.tensor_basis_sizes == (1,)

    def test_native_tensor_applies_work_admission_before_expansion(self) -> None:
        """Native callers cannot reach the dense kernel with canonical
        inputs whose derived group dimensions exceed the work bounds."""
        from jacobian.math.chain_complexes import tensor_product_complex

        factor = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            prime=None,
            degree_min=0,
            degree_max=1,
            basis_sizes=(64, 64),
            differential_matrices=([["0"] * 64 for _ in range(64)],),
        )
        # The derived tensor group sizes would be (4096, 8192, 4096); the
        # admission bound must reject before any dense expansion allocates.
        with pytest.raises(ValueError, match="work bound"):
            tensor_product_complex(factor, factor)

    def test_native_surface_excludes_wire_envelope_handlers(self) -> None:
        """The authoritative package __all__ advertises only value-based
        functions; request handlers stay private to operations/_tools."""
        import jacobian.math.chain_complexes as chain_complexes_package

        assert set(chain_complexes_package.__all__) == {
            "chain_map_commutes",
            "differential_squares_to_zero",
            "homology_groups",
            "mapping_cone",
            "tensor_product_complex",
        }
        from jacobian.math.chain_complexes.operations import compute_homology

        request = compute_homology.__annotations__.get("request")
        assert request is not None


class TestMappingConeCanonicalValue:
    """The cone is returned as a first-class chain-complex value whose
    derived bounds are admitted at the request boundary."""

    def test_identity_cone_composes_into_homology_unchanged(self) -> None:
        """The identity cone of a point feeds ComputeHomologyRequest
        directly through its canonical value."""
        from jacobian.math.chain_complexes._models import ComputeHomologyRequest
        from jacobian.math.chain_complexes.operations import compute_homology

        point = _point_complex()
        one = (("1",),)
        result = compute_mapping_cone(
            MappingConeRequest(source=point, target=point, map_matrices=(one,))
        )
        homology = compute_homology(ComputeHomologyRequest(complex=result.value))
        assert [group.betti_number for group in homology.homology_groups] == [
            0,
            0,
        ]
        assert result.value.degree_min == point.degree_min

    def test_derived_group_dimension_rejected_at_admission(self) -> None:
        """A cone group of 64 + 64 faces exceeds the canonical basis bound
        and fails the request instead of dying inside execution."""
        source = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=1,
            basis_sizes=(64, 0),
            differential_matrices=(((),) * 64,),
        )
        target = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=1,
            basis_sizes=(0, 64),
            differential_matrices=((),),
        )
        with pytest.raises(ValueError, match="MAX_BASIS_SIZE"):
            MappingConeRequest(
                source=source,
                target=target,
                map_matrices=((), ((),) * 64),
            )

    def test_derived_degree_interval_rejected_at_admission(self) -> None:
        """A 33-group source forces a 34-group cone whose degree interval
        leaves the canonical chain-degree range."""
        sizes = (1,) * 33
        zeros = ((("0",),),) * 32
        complex_value = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=32,
            basis_sizes=sizes,
            differential_matrices=zeros,
        )
        one = (("1",),)
        with pytest.raises(ValueError, match="less than or equal"):
            MappingConeRequest(
                source=complex_value,
                target=complex_value,
                map_matrices=(one,) * 33,
            )

    def test_derived_serialization_envelope_rejected_at_admission(self) -> None:
        """Copied coefficients push the cone past the canonical entry-char
        ceiling; admission rejects it before any dense allocation."""
        digits = 200
        row = tuple(str(10**digits + i) for i in range(16))
        big_zero = tuple(row for _ in range(16))
        source = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=1,
            basis_sizes=(16, 16),
            differential_matrices=(big_zero,),
        )
        target = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=1,
            basis_sizes=(16, 16),
            differential_matrices=(big_zero,),
        )
        identity_16 = tuple(
            tuple("1" if i == j else "0" for j in range(16)) for i in range(16)
        )
        with pytest.raises(ValueError, match="serialization exceeds"):
            MappingConeRequest(
                source=source,
                target=target,
                map_matrices=(identity_16, identity_16),
            )

    def test_result_round_trips_and_rejects_tampered_value(self) -> None:
        from pydantic import ValidationError

        from jacobian.math.chain_complexes.values import MappingConeResult

        point = _point_complex()
        one = (("1",),)
        result = compute_mapping_cone(
            MappingConeRequest(source=point, target=point, map_matrices=(one,))
        )
        revalidated = MappingConeResult.model_validate(result.model_dump())
        assert revalidated == result

        payload = result.model_dump()
        payload["value"]["basis_sizes"] = (2, *payload["value"]["basis_sizes"][1:])
        with pytest.raises(ValidationError):
            MappingConeResult.model_validate(payload)


class TestSchemaVisibleCoefficientGrammar:
    def test_construct_schema_documents_entry_grammar(self) -> None:
        description = ConstructChainComplexRequest.model_json_schema()["properties"][
            "differential_matrices"
        ]["description"]
        assert "residues in [0, p)" in description
        assert "never evaluates" in description

    def test_chain_map_schemas_document_map_matrix_grammar(self) -> None:
        for model in (VerifyChainMapRequest, MappingConeRequest):
            description = model.model_json_schema()["properties"]["map_matrices"][
                "description"
            ]
            assert "canonical coefficient grammar" in description
            assert "residues in [0, p)" in description


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


class TestEmptyRowWidthChainMaps:
    """A zero-row differential or component keeps its declared inner
    width in the chain-map equation instead of raising an
    inner-dimension error on an admitted request."""

    def _source(self) -> ChainComplexValue:
        return ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=1,
            basis_sizes=(1, 1),
            differential_matrices=((("0",),),),
        )

    def _target(self) -> ChainComplexValue:
        return ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=1,
            basis_sizes=(0, 1),
            differential_matrices=((),),
        )

    def _map(self) -> tuple[tuple[str, ...], ...]:
        return ((), (("1",),))

    def test_zero_row_target_differential_verifies(self) -> None:
        from jacobian.math.chain_complexes.operations import verify_chain_map

        request = VerifyChainMapRequest(
            source=self._source(),
            target=self._target(),
            map_matrices=self._map(),
        )
        result = verify_chain_map(request)
        assert result.is_valid
        assert type(result).model_validate(result.model_dump()).is_valid

    def test_corresponding_cone_is_admitted_and_returned(self) -> None:
        from jacobian.math.chain_complexes.operations import compute_mapping_cone

        request = MappingConeRequest(
            source=self._source(),
            target=self._target(),
            map_matrices=self._map(),
        )
        result = compute_mapping_cone(request)
        # cone_n = target_n + source_{n-1}: (0, 1+1, 0+1).
        assert result.cone_basis_sizes == (0, 2, 1)

    def test_zero_row_homology_request_returns_typed_result(self) -> None:
        """The homology kernel's square-zero replay keeps declared widths
        for a complex whose first group is empty."""
        from jacobian.math.chain_complexes.operations import compute_homology

        shifted = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=2,
            basis_sizes=(0, 1, 1),
            differential_matrices=((), (("1",),)),
        )
        result = compute_homology(ComputeHomologyRequest(complex=shifted))
        assert [group.betti_number for group in result.homology_groups] == [0, 0, 0]

    def test_empty_middle_cone_group_round_trips(self) -> None:
        """A cone whose zeroth group is empty keeps its widths through
        construction and validation."""
        from jacobian.math.chain_complexes.operations import compute_mapping_cone
        from jacobian.math.chain_complexes.values import MappingConeResult

        endpoint = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=1,
            basis_sizes=(0, 1),
            differential_matrices=((),),
        )
        result = compute_mapping_cone(
            MappingConeRequest(
                source=endpoint,
                target=endpoint,
                map_matrices=((), (("1",),)),
            )
        )
        revalidated = MappingConeResult.model_validate(result.model_dump())
        assert revalidated.cone_basis_sizes == (0, 1, 1)


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


class TestTensorProductFactorBinding:
    """A tensor result replays its defining construction against retained
    factors so detached or forged products cannot validate."""

    def test_forged_unrelated_value_rejected(self) -> None:
        """Matching duplicate projections do not make an authored value
        an exact tensor product."""
        from jacobian.math.chain_complexes.values import TensorProductResult

        unrelated = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=0,
            basis_sizes=(7,),
            differential_matrices=(),
        )
        claim = TensorProductResult(
            tensor_basis_sizes=(7,),
            tensor_differential_matrices=(),
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=0,
            left=_point_complex(),
            right=_point_complex(),
            value=unrelated,
        )
        assert not verify_tensor_product_result(claim)

    def test_non_square_zero_factor_rejected_by_replay(self) -> None:
        """A retained factor violating d^2=0 fails the construction
        replay instead of surviving under the tensor-product claim."""
        from jacobian.math.chain_complexes.values import TensorProductResult

        bad = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=2,
            basis_sizes=(1, 1, 1),
            differential_matrices=((("1",),), (("1",),)),
        )
        claim = TensorProductResult(
            tensor_basis_sizes=(1, 1, 1),
            tensor_differential_matrices=((("1",),), (("1",),)),
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=2,
            left=bad,
            right=_point_complex(),
            value=bad,
        )
        assert not verify_tensor_product_result(claim)

    def test_genuine_result_round_trips(self) -> None:
        from jacobian.math.chain_complexes.values import TensorProductResult

        circle = _circle_complex()
        point = _point_complex()
        result = compute_tensor_product(TensorProductRequest(left=circle, right=point))
        revalidated = TensorProductResult.model_validate(result.model_dump())
        assert revalidated.value.basis_sizes == (3, 3)

    def test_tampered_factor_rejected(self) -> None:
        from jacobian.math.chain_complexes.values import TensorProductResult

        result = compute_tensor_product(
            TensorProductRequest(left=_point_complex(), right=_point_complex())
        )
        payload = result.model_dump()
        payload["right"] = payload["right"] | {"basis_sizes": (2,)}
        assert not verify_tensor_product_result(
            TensorProductResult.model_validate(payload)
        )

    def test_tampered_degree_provenance_rejected(self) -> None:
        from jacobian.math.chain_complexes.values import TensorProductResult

        shifted = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=-1,
            degree_max=-1,
            basis_sizes=(1,),
            differential_matrices=(),
        )
        result = compute_tensor_product(
            TensorProductRequest(left=shifted, right=_point_complex())
        )
        payload = result.model_dump()
        payload["degree_min"] = 0
        with pytest.raises(ValidationError):
            TensorProductResult.model_validate(payload)


def homology_groups(complex_value):
    from jacobian.math.chain_complexes import homology_groups as native

    return native(complex_value)


class TestChainDegreeDiagnostics:
    def test_diagnostics_report_declared_degree(self) -> None:
        """A shifted complex reports its declared chain degree, not the
        tuple index, in d^2 failures."""
        from jacobian.math.chain_complexes._models import (
            ComputeHomologyRequest,
        )

        bad = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=-2,
            degree_max=0,
            basis_sizes=(1, 1, 1),
            differential_matrices=((("1",),), (("1",),)),
        )
        # The failing pair is (d_{-2}, d_{-1}); its composition is reported
        # at the middle declared degree -1, matching verify_differential.
        with pytest.raises(ValueError, match="chain degree -1"):
            compute_homology(ComputeHomologyRequest(complex=bad))


class TestTensorPrimeFieldResidues:
    def test_signed_tensor_coefficients_reduced_modulo_p(self) -> None:
        """A GF(3) tensor with a sign negation serializes the canonical
        residue 2, not -1, so the derived value validates."""
        left = ChainComplexValue(
            coefficient_field=CoefficientField.PRIME_FIELD,
            prime=3,
            degree_min=1,
            degree_max=1,
            basis_sizes=(1,),
            differential_matrices=(),
        )
        right = ChainComplexValue(
            coefficient_field=CoefficientField.PRIME_FIELD,
            prime=3,
            degree_min=0,
            degree_max=1,
            basis_sizes=(1, 1),
            differential_matrices=((("1",),),),
        )
        result = compute_tensor_product(TensorProductRequest(left=left, right=right))
        entries = [
            entry
            for matrix in result.tensor_differential_matrices
            for row in matrix
            for entry in row
        ]
        assert entries
        assert all(not entry.startswith("-") for entry in entries)
        assert all(entry in {"0", "1", "2"} for entry in entries)

    def test_signed_tensor_coefficient_exact_residue(self) -> None:
        """The reviewer's counterexample returns exactly the residue 2,
        both as the serialized projection and the retained complex."""
        left = ChainComplexValue(
            coefficient_field=CoefficientField.PRIME_FIELD,
            prime=3,
            degree_min=1,
            degree_max=1,
            basis_sizes=(1,),
            differential_matrices=(),
        )
        right = ChainComplexValue(
            coefficient_field=CoefficientField.PRIME_FIELD,
            prime=3,
            degree_min=0,
            degree_max=1,
            basis_sizes=(1, 1),
            differential_matrices=((("1",),),),
        )
        result = compute_tensor_product(TensorProductRequest(left=left, right=right))
        assert result.tensor_differential_matrices == ((("2",),),)
        assert result.value.differential_matrices == ((("2",),),)
        assert verify_differential(
            VerifyDifferentialRequest(complex=result.value)
        ).is_valid

    def test_koszul_sign_boundary_mod_two(self) -> None:
        """The p = 2 boundary: the Koszul sign -1 is the residue 1."""
        left = ChainComplexValue(
            coefficient_field=CoefficientField.PRIME_FIELD,
            prime=2,
            degree_min=1,
            degree_max=1,
            basis_sizes=(1,),
            differential_matrices=(),
        )
        right = ChainComplexValue(
            coefficient_field=CoefficientField.PRIME_FIELD,
            prime=2,
            degree_min=0,
            degree_max=1,
            basis_sizes=(1, 1),
            differential_matrices=((("1",),),),
        )
        result = compute_tensor_product(TensorProductRequest(left=left, right=right))
        assert result.tensor_differential_matrices == ((("1",),),)

    def test_rational_tensor_keeps_signed_spelling(self) -> None:
        """QQ tensor products keep their exact signed coefficients, so the
        reduction is specific to prime-field serialization."""
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
        assert result.tensor_differential_matrices == ((("-1",),),)

    def test_out_of_range_chain_map_entry_rejected(self) -> None:
        """GF(p) chain maps enforce canonical residues like complexes do."""
        source = ChainComplexValue(
            coefficient_field=CoefficientField.PRIME_FIELD,
            prime=3,
            degree_min=0,
            degree_max=1,
            basis_sizes=(1, 1),
            differential_matrices=((("2",),),),
        )
        with pytest.raises(ValueError, match="residue"):
            VerifyChainMapRequest(
                source=source,
                target=source,
                map_matrices=((("4",),), (("4",),)),
            )

    def _gf_point(self, prime: int) -> ChainComplexValue:
        return ChainComplexValue(
            coefficient_field=CoefficientField.PRIME_FIELD,
            prime=prime,
            degree_min=0,
            degree_max=0,
            basis_sizes=(1,),
            differential_matrices=(),
        )

    def test_map_entry_modulus_boundaries(self) -> None:
        """Every spelling outside [0, p) fails admission for both chain-map
        requests; an in-range residue still verifies."""
        point = self._gf_point(3)
        for bad in ("3", "4", "-1"):
            with pytest.raises(ValueError, match="residue"):
                VerifyChainMapRequest(
                    source=point, target=point, map_matrices=(((bad,),),)
                )
            with pytest.raises(ValueError, match="residue"):
                MappingConeRequest(
                    source=point, target=point, map_matrices=(((bad,),),)
                )
        from jacobian.math.chain_complexes.operations import verify_chain_map

        request = VerifyChainMapRequest(
            source=point, target=point, map_matrices=((("2",),),)
        )
        assert verify_chain_map(request).is_valid

    def test_rational_map_entries_keep_integer_grammar(self) -> None:
        """The modulus check applies only to prime-field components: "4"
        remains admissible over QQ."""
        rational_point = _point_complex()
        request = VerifyChainMapRequest(
            source=rational_point,
            target=rational_point,
            map_matrices=((("4",),),),
        )
        assert request.map_matrices == ((("4",),),)


class TestPrimeFieldDerivedSerialization:
    def test_mapping_cone_serializes_prime_field_residues(self) -> None:
        """The cone's -d_C block becomes the canonical GF(p) residue and
        the decomposition composes as a square-zero chain complex."""
        gf3_two_term = ChainComplexValue(
            coefficient_field=CoefficientField.PRIME_FIELD,
            prime=3,
            degree_min=0,
            degree_max=1,
            basis_sizes=(1, 1),
            differential_matrices=((("1",),),),
        )
        one = (("1",),)
        result = compute_mapping_cone(
            MappingConeRequest(
                source=gf3_two_term, target=gf3_two_term, map_matrices=(one, one)
            )
        )
        assert result.cone_differential_matrices == (
            (("1", "1"),),
            (("2",), ("1",)),
        )
        cone = ChainComplexValue(
            coefficient_field=CoefficientField.PRIME_FIELD,
            prime=3,
            degree_min=0,
            degree_max=2,
            basis_sizes=result.cone_basis_sizes,
            differential_matrices=result.cone_differential_matrices,
        )
        assert verify_differential(VerifyDifferentialRequest(complex=cone)).is_valid


class TestNativeHomologyFieldBinding:
    """The native homology wrapper returns the source-bound result, so
    homology over different coefficient fields stays distinct."""

    @staticmethod
    def _native(complex_value: ChainComplexValue):
        from jacobian.math.chain_complexes import (
            homology_groups as native_homology_groups,
        )

        return native_homology_groups(complex_value)

    def _gf_point(self, prime: int) -> ChainComplexValue:
        return ChainComplexValue(
            coefficient_field=CoefficientField.PRIME_FIELD,
            prime=prime,
            degree_min=0,
            degree_max=0,
            basis_sizes=(1,),
            differential_matrices=(),
        )

    def test_result_carries_coefficient_field(self) -> None:
        from jacobian.math.chain_complexes.values import HomologyResult

        point = self._gf_point(2)
        result = self._native(point)
        assert isinstance(result, HomologyResult)
        assert result.coefficient_field == CoefficientField.PRIME_FIELD
        assert result.prime == 2
        assert result.complex == point

    def test_different_fields_yield_distinct_results(self) -> None:
        over_two = self._native(self._gf_point(2))
        over_three = self._native(self._gf_point(3))
        assert over_two.prime == 2
        assert over_three.prime == 3
        assert over_two != over_three

    def test_matches_wire_operation(self) -> None:
        from jacobian.math.chain_complexes._models import ComputeHomologyRequest
        from jacobian.math.chain_complexes.operations import compute_homology

        point = self._gf_point(3)
        native_result = self._native(point)
        wire_result = compute_homology(ComputeHomologyRequest(complex=point))
        assert native_result == wire_result


class TestZeroWidthGroupComposition:
    def test_zero_row_left_operand_preserves_declared_width(self) -> None:
        """A (0 x 1) differential followed by (1 x 1) composes to a valid
        zero product instead of raising an inner-dimension error."""
        shifted = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=2,
            basis_sizes=(0, 1, 1),
            differential_matrices=((), (("1",),)),
        )
        result = verify_differential(VerifyDifferentialRequest(complex=shifted))
        assert result.is_valid

    def test_nonsquare_zero_homology_rejected_at_request(self) -> None:
        """Homology requests whose d^2 != 0 fail at the typed boundary."""
        bad = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=2,
            basis_sizes=(1, 1, 1),
            differential_matrices=((("1",),), (("1",),)),
        )
        with pytest.raises(ValueError, match="d\\^2"):
            ComputeHomologyRequest(complex=bad)
        with pytest.raises(ValueError, match="d\\^2"):
            TensorProductRequest(left=bad, right=_point_complex())

    def test_differential_failure_reports_declared_degree(self) -> None:
        """A complex concentrated in degrees -5..-3 reports degree -4, not
        the tuple index."""
        neg = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=-5,
            degree_max=-3,
            basis_sizes=(1, 1, 1),
            differential_matrices=((("1",),), (("1",),)),
        )
        result = verify_differential(VerifyDifferentialRequest(complex=neg))
        assert not result.is_valid
        assert "-4" in result.detail


class TestVerificationVerdictBinding:
    def test_forged_verdict_rejected(self) -> None:
        """A successful verdict cannot validate against a complex whose
        d^2 is nonzero, and a detached verdict cannot validate at all."""
        from jacobian.math.chain_complexes.values import VerificationResult

        bad = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=2,
            basis_sizes=(1, 1, 1),
            differential_matrices=((("1",),), (("1",),)),
        )
        assert not verify_verification_result(
            VerificationResult(is_valid=True, detail="d^2 = 0", complex=bad)
        )
        with pytest.raises(ValidationError):
            VerificationResult(is_valid=True, detail="d^2 = 0")

    def test_contradictory_detail_rejected(self) -> None:
        """The detail is authoritative: a square-zero point cannot carry
        an explanation contradicting its replayed verdict."""
        from jacobian.math.chain_complexes.values import VerificationResult

        point = _point_complex()
        assert not verify_verification_result(
            VerificationResult(
                is_valid=True,
                detail="d^2 != 0",
                complex=point,
            )
        )
        genuine = verify_differential(VerifyDifferentialRequest(complex=point))
        revalidated = VerificationResult.model_validate(genuine.model_dump())
        assert revalidated.detail == "d^2 = 0 for all degrees"

    def test_true_verdict_retains_complex(self) -> None:
        """Successful differential verification retains its input."""
        result = verify_differential(
            VerifyDifferentialRequest(complex=_circle_complex())
        )
        assert result.is_valid
        assert result.complex is not None
        revalidated = type(result).model_validate(result.model_dump())
        assert revalidated.is_valid


class TestVerificationReplayParentChecks:
    """A replayed chain-map verdict applies the request model's complete
    endpoint checks, not the source modulus alone."""

    def _point(self, field: CoefficientField, prime: int | None) -> ChainComplexValue:
        return ChainComplexValue(
            coefficient_field=field,
            prime=prime,
            degree_min=0,
            degree_max=0,
            basis_sizes=(1,),
            differential_matrices=(),
        )

    def test_cross_field_endpoints_rejected(self) -> None:
        from jacobian.math.chain_complexes.values import VerificationResult

        claim = VerificationResult(
            is_valid=True,
            detail="commutes",
            source=self._point(CoefficientField.RATIONAL, None),
            target=self._point(CoefficientField.PRIME_FIELD, 2),
            map_matrices=((("1",),),),
        )
        assert not verify_verification_result(claim)

    def test_mismatched_primes_rejected(self) -> None:
        from jacobian.math.chain_complexes.values import VerificationResult

        claim = VerificationResult(
            is_valid=True,
            detail="commutes",
            source=self._point(CoefficientField.PRIME_FIELD, 2),
            target=self._point(CoefficientField.PRIME_FIELD, 3),
            map_matrices=((("1",),),),
        )
        assert not verify_verification_result(claim)

    def test_shifted_degree_intervals_rejected(self) -> None:
        from jacobian.math.chain_complexes.values import VerificationResult

        shifted = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=-1,
            degree_max=-1,
            basis_sizes=(1,),
            differential_matrices=(),
        )
        claim = VerificationResult(
            is_valid=True,
            detail="commutes",
            source=_point_complex(),
            target=shifted,
            map_matrices=((("1",),),),
        )
        assert not verify_verification_result(claim)

    def test_out_of_range_map_residue_rejected(self) -> None:
        """GF(3) map entries are replayed under p=3, so '4' cannot pass."""
        from jacobian.math.chain_complexes.values import VerificationResult

        gf3 = self._point(CoefficientField.PRIME_FIELD, 3)
        claim = VerificationResult(
            is_valid=True,
            detail="commutes",
            source=gf3,
            target=gf3,
            map_matrices=((("4",),),),
        )
        assert not verify_verification_result(claim)

    def test_genuine_verdict_round_trips(self) -> None:
        from jacobian.math.chain_complexes.values import VerificationResult

        point = _point_complex()
        verdict = VerificationResult(
            is_valid=True,
            detail="chain map commutes with differentials",
            source=point,
            target=point,
            map_matrices=((("1",),),),
        )
        assert type(verdict).model_validate(verdict.model_dump()).is_valid


class TestMappingConeSourceBinding:
    """A cone result replays its defining construction against retained
    endpoints so detached or forged cones cannot validate."""

    def _circle(self) -> ChainComplexValue:
        return _circle_complex()

    def _identity(self) -> tuple[tuple[str, ...], ...]:
        return (("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1"))

    def _cone_payload(self) -> dict[str, Any]:
        circle = self._circle()
        identity = self._identity()
        result = compute_mapping_cone(
            MappingConeRequest(
                source=circle, target=circle, map_matrices=(identity,) * 2
            )
        )
        return result.model_dump()

    def test_genuine_cone_round_trips(self) -> None:
        from jacobian.math.chain_complexes.values import MappingConeResult

        revalidated = MappingConeResult.model_validate(self._cone_payload())
        assert revalidated.cone_basis_sizes == (3, 6, 3)

    def test_detached_cone_rejected(self) -> None:
        """Without retained endpoints no cone payload validates at all."""
        from jacobian.math.chain_complexes.values import MappingConeResult

        with pytest.raises(ValidationError):
            MappingConeResult.model_validate(
                {
                    "cone_basis_sizes": (1, 1, 1),
                    "cone_differential_matrices": ((("1",),), (("1",),)),
                    "source_degree_min": 0,
                    "target_degree_min": 0,
                }
            )

    def test_non_chain_complex_endpoints_rejected(self) -> None:
        """Endpoints violating d^2=0 admit no cone, even if the authored
        matrices happen to be square-zero themselves."""
        from jacobian.math.chain_complexes.values import MappingConeResult

        bad = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=2,
            basis_sizes=(1, 1, 1),
            differential_matrices=((("1",),), (("1",),)),
        )
        one = (("1",),)
        with pytest.raises(ValidationError):
            MappingConeResult(
                cone_basis_sizes=(1, 1, 1),
                cone_differential_matrices=((("1",),), (("1",),)),
                source_degree_min=0,
                target_degree_min=0,
                source=bad,
                target=bad,
                map_matrices=(one, one, one),
            )

    def test_tampered_cone_differentials_rejected(self) -> None:
        from jacobian.math.chain_complexes.values import MappingConeResult

        payload = self._cone_payload()
        payload["cone_differential_matrices"] = (
            (("1",) * 6,) * 6,
            (("0",) * 6,) * 6,
        )
        with pytest.raises(ValidationError):
            MappingConeResult.model_validate(payload)

    def test_tampered_degree_provenance_rejected(self) -> None:
        from jacobian.math.chain_complexes.values import MappingConeResult

        payload = self._cone_payload()
        payload["source_degree_min"] = 5
        with pytest.raises(ValidationError):
            MappingConeResult.model_validate(payload)

    def test_shifted_genuine_cone_round_trips(self) -> None:
        """Cone provenance follows the declared interval of a shifted pair."""
        from jacobian.math.chain_complexes.values import MappingConeResult

        shifted = ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=-1,
            degree_max=0,
            basis_sizes=(3, 3),
            differential_matrices=(
                (("-1", "1", "0"), ("0", "-1", "1"), ("0", "0", "0")),
            ),
        )
        identity = self._identity()
        result = compute_mapping_cone(
            MappingConeRequest(
                source=shifted, target=shifted, map_matrices=(identity,) * 2
            )
        )
        revalidated = MappingConeResult.model_validate(result.model_dump())
        assert revalidated.source_degree_min == -1
        assert revalidated.cone_basis_sizes == (3, 6, 3)


class TestReviewRegressions2236:
    def test_empty_width_construct_replay_preserves_declared_widths(self):
        """A valid declared 0x1 map survives the construct-time d^2 replay."""
        from jacobian.math.chain_complexes._models import (
            CoefficientField,
            ConstructChainComplexRequest,
        )

        request = ConstructChainComplexRequest(
            coefficient_field=CoefficientField.PRIME_FIELD,
            prime=5,
            basis_sizes=(0, 1, 1),
            differential_matrices=((), (("1",),)),
        )
        assert request.basis_sizes == (0, 1, 1)

    def test_endpoint_replay_reports_shifted_chain_degrees(self):
        """The shared endpoint replay names declared chain degrees, not 0."""

        from jacobian.math.chain_complexes.operations import (
            _matrix_to_fractions,
            _require_square_zero,
        )

        def broken_diffs():
            # Two 2x2 identity differentials over GF(5): d^2 != 0 by design.
            identity = (("1", "0"), ("0", "1"))
            return [
                _matrix_to_fractions(identity, 2, 2, 5),
                _matrix_to_fractions(identity, 2, 2, 5),
            ]

        with pytest.raises(ValueError) as shifted_error:
            _require_square_zero(
                broken_diffs(),
                5,
                label="source",
                group_columns=[2, 2, 2],
                degree_min=-5,
            )
        detail = str(shifted_error.value)
        # The composed pair's middle declared degree, matching the
        # verification operation's verdict for the same complex.
        assert "at chain degree -4" in detail, detail


class TestNativeWrappersCallKernelsDirectly:
    """Native wrappers invoke typed kernels without the wire envelope."""

    @staticmethod
    def _circle() -> ChainComplexValue:
        return ChainComplexValue(
            coefficient_field=CoefficientField.RATIONAL,
            degree_min=0,
            degree_max=1,
            basis_sizes=(1, 1),
            differential_matrices=((("0",),),),
        )

    def test_native_paths_survive_disabled_wire_handlers(self, monkeypatch):
        """Blocking every wire handler still yields correct native results."""
        import jacobian.math.chain_complexes.operations as ops

        def blocked(*args, **kwargs):
            raise AssertionError("wire handler reached from the native path")

        for name in (
            "compute_homology",
            "verify_differential",
            "verify_chain_map",
            "compute_mapping_cone",
            "compute_tensor_product",
        ):
            monkeypatch.setattr(ops, name, blocked)

        from jacobian.math.chain_complexes import native

        circle = self._circle()
        homology = native.homology_groups(circle)
        assert homology.homology_groups[0].betti_number == 1
        assert native.differential_squares_to_zero(circle).is_valid is True
        identity_map = ((("1",),),)
        assert len(identity_map) == 1
        identity_map = ((("1",),), (("1",),))
        cone = native.mapping_cone(circle, circle, identity_map)
        assert cone.source_degree_min == 0
        tensor = native.tensor_product_complex(circle, circle)
        assert tensor.value.degree_max == 2

    def test_native_chain_map_verdict_matches_wire_semantics(self, monkeypatch):
        import jacobian.math.chain_complexes.operations as ops

        def blocked(*args, **kwargs):
            raise AssertionError("wire handler reached from the native path")

        monkeypatch.setattr(ops, "verify_chain_map", blocked)
        from jacobian.math.chain_complexes import native

        circle = self._circle()
        # One component per chain group: the identity chain map.
        identity = ((("1",),), (("1",),))
        verdict = native.chain_map_commutes(circle, circle, identity)
        assert verdict.is_valid is True
