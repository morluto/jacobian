"""Tests for chain complex operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.chain_complexes._models import (
    ChainComplex,
    HomologyRequest,
    HomologyResult,
    MappingConeRequest,
    MatrixEntry,
)
from jacobian.math.chain_complexes._operations import (
    compute_homology,
    compute_mapping_cone,
)
from jacobian.math.chain_complexes._tools import TOOLS


class TestHomology:
    """Test chain complex homology computation."""

    def test_zero_differentials(self):
        """With zero differentials, homology equals the chain groups."""
        cx = ChainComplex(
            prime=2,
            min_degree=0,
            max_degree=2,
            dimensions=(1, 2, 1),
            differentials=(),
        )
        result = compute_homology(HomologyRequest(complex=cx))
        assert result.groups[0].betti == 1
        assert result.groups[1].betti == 2
        assert result.groups[2].betti == 1

    def test_identity_differential(self):
        """A complex with an identity-like differential reduces homology."""
        cx = ChainComplex(
            prime=2,
            min_degree=0,
            max_degree=1,
            dimensions=(1, 2),
            differentials=(
                (
                    MatrixEntry(row=0, col=0, value="1"),
                    MatrixEntry(row=0, col=1, value="1"),
                ),
            ),
        )
        result = compute_homology(HomologyRequest(complex=cx))
        assert result.groups[0].betti == 0
        assert result.groups[1].betti == 1

    def test_exact_sequence(self):
        """An exact sequence has zero homology everywhere."""
        # Valid chain complex with d^2=0: C0=1, C1=2, C2=1 over GF(2)
        # d1: C1 -> C0 is 1x2 [[1,1]] rank1, d2: C2 -> C1 is 2x1 [[1],[1]] rank1
        # d1*d2 = [[2]] = 0 mod2, so d^2=0.
        cx = ChainComplex(
            prime=2,
            min_degree=0,
            max_degree=2,
            dimensions=(1, 2, 1),
            differentials=(
                (
                    MatrixEntry(row=0, col=0, value="1"),
                    MatrixEntry(row=0, col=1, value="1"),
                ),
                (
                    MatrixEntry(row=0, col=0, value="1"),
                    MatrixEntry(row=1, col=0, value="1"),
                ),
            ),
        )
        result = compute_homology(HomologyRequest(complex=cx))
        # H0 = 1 -1 =0, H1 = (2-1)-1=0, H2=1-1=0
        assert result.groups[0].betti == 0
        assert result.groups[1].betti == 0
        assert result.groups[2].betti == 0

    def test_single_degree(self):
        """A complex with one degree has homology equal to that degree."""
        cx = ChainComplex(
            prime=5,
            min_degree=0,
            max_degree=0,
            dimensions=(3,),
            differentials=(),
        )
        result = compute_homology(HomologyRequest(complex=cx))
        assert result.groups[0].betti == 3

    def test_prime_field(self):
        """Test with a different prime field."""
        cx = ChainComplex(
            prime=7,
            min_degree=0,
            max_degree=1,
            dimensions=(2, 2),
            differentials=(),
        )
        result = compute_homology(HomologyRequest(complex=cx))
        assert result.groups[0].betti == 2
        assert result.groups[1].betti == 2


class TestPublishedExample:
    @pytest.mark.parametrize(
        "tool",
        TOOLS,
        ids=[t.operation_id for t in TOOLS],
    )
    def test_published_example_validates_and_runs(self, tool):
        assert tool.examples
        for published in tool.examples:
            request = tool.request_type.model_validate(published.input)
            assert tool.run(request) is not None

    def test_published_homology_example_betti_numbers(self):
        homology = next(
            tool for tool in TOOLS if tool.operation_id.endswith("homology.compute")
        )
        request = homology.request_type.model_validate(homology.examples[0].input)
        result = compute_homology(request)
        assert [group.betti for group in result.groups] == [0, 0, 1]

    def test_published_mapping_cone_example_is_acyclic(self):
        cone_tool = next(
            tool for tool in TOOLS if tool.operation_id.endswith("mapping_cone.compute")
        )
        request = cone_tool.request_type.model_validate(cone_tool.examples[0].input)
        result = compute_mapping_cone(request)
        assert result.cone.dimensions == (1, 1)
        assert len(result.cone.differentials) == 1


class TestAdmissionRegressions:
    def test_oversized_group_dimensions_rejected(self):
        with pytest.raises(ValidationError, match="dense-work bound"):
            ChainComplex(
                prime=2,
                min_degree=0,
                max_degree=1,
                dimensions=(10**9, 10**9),
                differentials=((),),
            )

    def test_degree_eleven_homology_roundtrips(self):
        request = HomologyRequest(
            complex=ChainComplex(
                prime=3,
                min_degree=11,
                max_degree=11,
                dimensions=(1,),
                differentials=(),
            )
        )
        result = compute_homology(request)
        assert result.max_degree == 11
        assert result.groups[0].degree == 11
        assert HomologyResult.model_validate(result.model_dump()) == result

    def test_chain_map_into_out_of_range_target_degree_rejected(self):
        """f_0 must index the target group at degree 0, not tuple position."""
        source = ChainComplex(
            prime=2, min_degree=0, max_degree=0, dimensions=(1,), differentials=()
        )
        target = ChainComplex(
            prime=2, min_degree=1, max_degree=1, dimensions=(1,), differentials=()
        )
        with pytest.raises(ValidationError, match="exceeds target dimension"):
            MappingConeRequest(
                source=source,
                target=target,
                chain_map=((MatrixEntry(row=0, col=0, value="1"),),),
            )

    def test_chain_map_to_matching_degree_accepted(self):
        source = ChainComplex(
            prime=2, min_degree=0, max_degree=0, dimensions=(1,), differentials=()
        )
        target = ChainComplex(
            prime=2, min_degree=0, max_degree=0, dimensions=(1,), differentials=()
        )
        request = MappingConeRequest(
            source=source,
            target=target,
            chain_map=((MatrixEntry(row=0, col=0, value="1"),),),
        )
        cone = compute_mapping_cone(request)
        assert cone.cone.dimensions == (1, 1)


class TestMappingConeValidationRegression:
    """Regressions for shaped zero maps, boundary equations, and cone bounds."""

    def test_zero_differential_identity_chain_map_accepted(self):
        # Both complexes use the admitted differentials=() zero-map shape with
        # an identity chain map: the commutator products are shaped zero
        # matrices that must compare equal instead of [] vs [[]].
        source = ChainComplex(
            prime=2, min_degree=0, max_degree=1, dimensions=(1, 1), differentials=()
        )
        target = ChainComplex(
            prime=2, min_degree=0, max_degree=1, dimensions=(1, 1), differentials=()
        )
        request = MappingConeRequest(
            source=source,
            target=target,
            chain_map=(
                (MatrixEntry(row=0, col=0, value="1"),),
                (MatrixEntry(row=0, col=0, value="1"),),
            ),
        )
        cone = compute_mapping_cone(request)
        assert cone.cone.dimensions == (1, 2, 1)

    def test_boundary_equation_crossing_target_top_degree_rejected(self):
        # Source C_1 -> C_0 with identity differential, target concentrated in
        # degree 0, nonzero f_0: the degree-1 equation is 0 = f_0 d^C_1 != 0,
        # so the request must be rejected even though D_1 does not exist.
        source = ChainComplex(
            prime=2,
            min_degree=0,
            max_degree=1,
            dimensions=(1, 1),
            differentials=((MatrixEntry(row=0, col=0, value="1"),),),
        )
        target = ChainComplex(
            prime=2, min_degree=0, max_degree=0, dimensions=(1,), differentials=()
        )
        with pytest.raises(ValidationError, match="commute"):
            MappingConeRequest(
                source=source,
                target=target,
                chain_map=(
                    (MatrixEntry(row=0, col=0, value="1"),),
                    (),
                ),
            )

    def test_overlapping_cone_group_dimension_rejected(self):
        # Two independently admissible 512-dimensional groups overlap in the
        # cone: Cone_1 = C_0 + D_1 has dimension 1024 and cannot be built.
        source = ChainComplex(
            prime=2, min_degree=0, max_degree=0, dimensions=(512,), differentials=()
        )
        target = ChainComplex(
            prime=2, min_degree=1, max_degree=1, dimensions=(512,), differentials=()
        )
        with pytest.raises(ValidationError, match="dense-work"):
            MappingConeRequest(source=source, target=target, chain_map=((),))

    def test_shifted_source_top_degree_without_cone_degree_rejected(self):
        # A source concentrated at the top supported degree shifts the cone to
        # degree 12, which no ChainComplex can represent.
        source = ChainComplex(
            prime=2, min_degree=11, max_degree=11, dimensions=(1,), differentials=()
        )
        target = ChainComplex(
            prime=2, min_degree=11, max_degree=11, dimensions=(1,), differentials=()
        )
        with pytest.raises(ValidationError, match=r"\[-10, 11\]|degree"):
            MappingConeRequest(
                source=source,
                target=target,
                chain_map=((MatrixEntry(row=0, col=0, value="1"),),),
            )


class TestTriageRegressions:
    """Regression tests for the PR 2246 triage fixes."""

    def test_bottom_degree_chain_map_equation_checked(self):
        # Source concentrated in degree 1 (zero source differential), target
        # with identity d_1: D_1 -> D_0, identity f_1: the bottom equation
        # d^D_1 * f_1 = 0 fails, so the map is not a chain map.
        source = ChainComplex(
            prime=2, min_degree=1, max_degree=1, dimensions=(1,), differentials=()
        )
        target = ChainComplex(
            prime=2,
            min_degree=0,
            max_degree=1,
            dimensions=(1, 1),
            differentials=((MatrixEntry(row=0, col=0, value="1"),),),
        )
        with pytest.raises(ValidationError, match="commute at degree 1"):
            MappingConeRequest(
                source=source,
                target=target,
                chain_map=((MatrixEntry(row=0, col=0, value="1"),),),
            )

    def test_homology_result_rejects_forged_groups(self):
        cx = ChainComplex(
            prime=2,
            min_degree=0,
            max_degree=1,
            dimensions=(1, 2),
            differentials=(
                (
                    MatrixEntry(row=0, col=0, value="1"),
                    MatrixEntry(row=0, col=1, value="1"),
                ),
            ),
        )
        result = compute_homology(HomologyRequest(complex=cx))
        forged = result.model_copy(
            update={
                "groups": (
                    type(result.groups[0])(
                        degree=0,
                        betti=1,
                        dimension=0,
                        boundary_rank=0,
                        cycle_rank=0,
                    ),
                    result.groups[1],
                )
            }
        )
        with pytest.raises(ValidationError, match="exact homology"):
            HomologyResult.model_validate(forged.model_dump())


class TestReviewRegressions:
    def test_duplicate_differential_coordinates_rejected(self):
        with pytest.raises(ValidationError, match="duplicate matrix entry"):
            ChainComplex(
                prime=2,
                min_degree=0,
                max_degree=1,
                dimensions=(2, 2),
                differentials=(
                    (
                        MatrixEntry(row=0, col=0, value="1"),
                        MatrixEntry(row=0, col=0, value="1"),
                    ),
                ),
            )

    def test_entry_list_longer_than_cells_rejected(self):
        source = ChainComplex(
            prime=2, min_degree=0, max_degree=0, dimensions=(1,), differentials=()
        )
        target = ChainComplex(
            prime=2, min_degree=0, max_degree=0, dimensions=(1,), differentials=()
        )
        # chain_map component for a 1x1 map admits one cell; three entries
        # cannot carry distinct in-range coordinates.
        with pytest.raises(ValidationError, match="rows x columns cells"):
            MappingConeRequest(
                source=source,
                target=target,
                chain_map=(
                    (
                        MatrixEntry(row=0, col=0, value="1"),
                        MatrixEntry(row=0, col=0, value="1"),
                        MatrixEntry(row=0, col=0, value="1"),
                    ),
                ),
            )

    def test_mapping_cone_result_replays_retained_source_map(self):
        from pydantic import ValidationError

        from jacobian.math.chain_complexes._models import MappingConeResult

        source = ChainComplex(
            prime=5, min_degree=0, max_degree=1, dimensions=(1, 1), differentials=((),)
        )
        target = ChainComplex(
            prime=5, min_degree=0, max_degree=0, dimensions=(1,), differentials=()
        )
        request = MappingConeRequest(
            source=source,
            target=target,
            chain_map=((MatrixEntry(row=0, col=0, value="1"),), ()),
        )
        result = compute_mapping_cone(request)
        assert MappingConeResult.model_validate(result.model_dump()) == result
        relayed = result.model_dump()
        relayed["cone"]["dimensions"] = (2, 9, 9)
        with pytest.raises(ValidationError, match="exact mapping cone"):
            MappingConeResult.model_validate(relayed)
