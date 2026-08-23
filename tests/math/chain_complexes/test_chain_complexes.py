"""Tests for chain complex operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.chain_complexes._models import (
    MAX_AGGREGATE_CELLS,
    MAX_AGGREGATE_ELIMINATION_WORK,
    MAX_AGGREGATE_MULTIPLICATION_WORK,
    ChainComplex,
    HomologyRequest,
    HomologyResult,
    MappingConeRequest,
    MappingConeResult,
)
from jacobian.math.chain_complexes._operations import (
    compute_homology,
    compute_mapping_cone,
)
from jacobian.math.chain_complexes._tools import TOOLS


def mat(prime: int, rows: list[list[int]]) -> dict:
    return {"prime": prime, "entries": rows, "columns": len(rows[0]) if rows else 0}


def zeros_dict(prime: int, rows: int, cols: int) -> dict:
    return mat(prime, [[0] * cols for _ in range(rows)])


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
            differentials=({"prime": 2, "entries": [[1, 1]], "columns": 2},),
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
                {"prime": 2, "entries": [[1, 1]], "columns": 2},
                {"prime": 2, "entries": [[1], [1]], "columns": 1},
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

    def test_ranks_come_from_the_shared_kernel(self):
        """Nonunit residues over an odd prime rank exactly like the kernel."""

        from jacobian.math.prime_field_linear_algebra import (
            PrimeFieldMatrix,
            rank,
        )

        cx = ChainComplex(
            prime=7,
            min_degree=0,
            max_degree=1,
            dimensions=(2, 1),
            differentials=({"prime": 7, "entries": [[3], [5]], "columns": 1},),
        )
        result = compute_homology(HomologyRequest(complex=cx))
        kernel_rank = rank(
            PrimeFieldMatrix(prime=7, entries=((3,), (5,)), columns=1)
        )
        assert kernel_rank == 1
        # d_1: C_1 -> C_0 arrives at degree 0 with the kernel's exact rank.
        assert result.groups[0].boundary_rank == kernel_rank
        assert result.groups[0].betti == 1
        assert result.groups[1].cycle_rank == 0


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
                differentials=(),
            )

    def test_group_dimensions_beyond_the_kernel_cap_rejected(self):
        """Differentials are canonical kernel values, capped at 256."""
        with pytest.raises(ValidationError, match="dense-work bound"):
            ChainComplex(
                prime=2,
                min_degree=0,
                max_degree=1,
                dimensions=(300, 300),
                differentials=(),
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
        with pytest.raises(ValidationError, match="must have 0 rows"):
            MappingConeRequest(
                source=source,
                target=target,
                chain_map=({"prime": 2, "entries": [[1]], "columns": 1},),
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
            chain_map=({"prime": 2, "entries": [[1]], "columns": 1},),
        )
        cone = compute_mapping_cone(request)
        assert cone.cone.dimensions == (1, 1)


class TestMappingConeValidationRegression:
    """Regressions for shaped zero maps, boundary equations, and cone bounds."""

    def test_zero_differential_identity_chain_map_accepted(self):
        # Both complexes use the admitted differentials=() zero-map shape with
        # an identity chain map: the commutator products compare equal.
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
                {"prime": 2, "entries": [[1]], "columns": 1},
                {"prime": 2, "entries": [[1]], "columns": 1},
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
            differentials=({"prime": 2, "entries": [[1]], "columns": 1},),
        )
        target = ChainComplex(
            prime=2, min_degree=0, max_degree=0, dimensions=(1,), differentials=()
        )
        with pytest.raises(ValidationError, match="commute"):
            MappingConeRequest(
                source=source,
                target=target,
                chain_map=(
                    {"prime": 2, "entries": [[1]], "columns": 1},
                    {"prime": 2, "entries": [], "columns": 1},
                ),
            )

    def test_overlapping_cone_group_dimension_rejected(self):
        # Two independently admissible 200-dimensional groups overlap in the
        # cone: Cone_1 = C_0 + D_1 has dimension 400 and cannot be built as
        # canonical prime-field matrices.
        source = ChainComplex(
            prime=2, min_degree=0, max_degree=0, dimensions=(200,), differentials=()
        )
        target = ChainComplex(
            prime=2, min_degree=1, max_degree=1, dimensions=(200,), differentials=()
        )
        with pytest.raises(ValidationError, match="dense-work"):
            MappingConeRequest(
                source=source,
                target=target,
                # f_0 lands in the absent group D_0: a zero-row matrix over
                # all 200 source basis vectors.
                chain_map=({"prime": 2, "entries": [], "columns": 200},),
            )

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
                chain_map=({"prime": 2, "entries": [[1]], "columns": 1},),
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
            differentials=({"prime": 2, "entries": [[1]], "columns": 1},),
        )
        with pytest.raises(ValidationError, match="commute at degree 1"):
            MappingConeRequest(
                source=source,
                target=target,
                chain_map=({"prime": 2, "entries": [[1]], "columns": 1},),
            )

    def test_homology_result_rejects_forged_groups(self):
        cx = ChainComplex(
            prime=2,
            min_degree=0,
            max_degree=1,
            dimensions=(1, 2),
            differentials=({"prime": 2, "entries": [[1, 1]], "columns": 2},),
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


class TestCanonicalMatrixValues:
    """Differentials and chain maps are the one canonical matrix value."""

    def test_noncanonical_residue_rejected_before_execution(self):
        with pytest.raises(ValidationError, match="canonical"):
            ChainComplex(
                prime=2,
                min_degree=0,
                max_degree=1,
                dimensions=(1, 1),
                differentials=({"prime": 2, "entries": [[3]], "columns": 1},),
            )

    def test_multiple_encodings_of_one_entry_are_impossible(self):
        """The sparse encodings '1' vs '3' mod 2 cannot both represent 1."""
        canonical = ChainComplex(
            prime=2,
            min_degree=0,
            max_degree=1,
            dimensions=(1, 1),
            differentials=({"prime": 2, "entries": [[1]], "columns": 1},),
        ).differentials[0]
        assert canonical.entries == ((1,),)

    def test_differential_prime_mismatch_rejected(self):
        with pytest.raises(ValidationError, match="match the complex prime"):
            ChainComplex(
                prime=3,
                min_degree=0,
                max_degree=1,
                dimensions=(1, 1),
                differentials=({"prime": 2, "entries": [[1]], "columns": 1},),
            )

    def test_wrong_differential_shape_rejected(self):
        with pytest.raises(ValidationError, match="must have 2 rows"):
            ChainComplex(
                prime=2,
                min_degree=0,
                max_degree=1,
                dimensions=(2, 2),
                differentials=({"prime": 2, "entries": [[1, 0]], "columns": 2},),
            )

    def test_nonzero_composition_rejected(self):
        """d^2 != 0 is not a chain complex and must fail admission."""
        with pytest.raises(ValidationError, match="must satisfy d"):
            ChainComplex(
                prime=2,
                min_degree=0,
                max_degree=2,
                dimensions=(1, 1, 1),
                differentials=(
                    {"prime": 2, "entries": [[1]], "columns": 1},
                    {"prime": 2, "entries": [[1]], "columns": 1},
                ),
            )

    def test_all_ones_over_gf2_composes_to_zero(self):
        """The review's all-ones example is admitted and exact over GF(2)."""
        cx = ChainComplex(
            prime=2,
            min_degree=0,
            max_degree=2,
            dimensions=(1, 2, 1),
            differentials=(
                {"prime": 2, "entries": [[1, 1]], "columns": 2},
                {"prime": 2, "entries": [[1], [1]], "columns": 1},
            ),
        )
        result = compute_homology(HomologyRequest(complex=cx))
        assert all(group.betti == 0 for group in result.groups)


class TestAggregateBudgets:
    """Aggregate budgets bound one request's total validation work."""

    def test_admits_useful_request_near_the_boundary(self):
        size = 220
        cx = ChainComplex(
            prime=2,
            min_degree=0,
            max_degree=3,
            dimensions=(size, size, size, size),
            differentials=tuple(zeros_dict(2, size, size) for _ in range(3)),
        )
        cells = 3 * size * size
        work = 2 * size**3
        assert cells <= MAX_AGGREGATE_CELLS
        assert work <= MAX_AGGREGATE_MULTIPLICATION_WORK
        assert work <= MAX_AGGREGATE_ELIMINATION_WORK
        result = compute_homology(HomologyRequest(complex=cx))
        assert all(group.betti == size for group in result.groups)

    def test_accepts_requests_exactly_at_both_work_boundaries(self):
        size = 256
        cx = ChainComplex(
            prime=2,
            min_degree=0,
            max_degree=2,
            dimensions=(size, size, size),
            differentials=(zeros_dict(2, size, size), zeros_dict(2, size, size)),
        )
        assert 2 * size**3 == MAX_AGGREGATE_MULTIPLICATION_WORK
        assert 2 * size**3 == MAX_AGGREGATE_ELIMINATION_WORK
        result = compute_homology(HomologyRequest(complex=cx))
        assert all(group.betti == size for group in result.groups)

    def test_cell_budget_rejects_many_full_differentials(self):
        """Many admissible matrices together exceed the aggregate cell budget."""
        size = 210
        count = 6
        assert count * size * size > MAX_AGGREGATE_CELLS
        with pytest.raises(ValidationError, match="matrix cells"):
            ChainComplex(
                prime=2,
                min_degree=0,
                max_degree=count,
                dimensions=(size,) * (count + 1),
                differentials=tuple(zeros_dict(2, size, size) for _ in range(count)),
            )

    def test_multiplication_budget_rejects_dense_composition(self):
        """Cells stay under budget while predicted composition work does not."""
        size = 256
        count = 4
        assert count * size * size <= MAX_AGGREGATE_CELLS
        assert (count - 1) * size**3 > MAX_AGGREGATE_MULTIPLICATION_WORK
        with pytest.raises(ValidationError, match="field multiplications"):
            ChainComplex(
                prime=2,
                min_degree=0,
                max_degree=count,
                dimensions=(size,) * (count + 1),
                differentials=tuple(zeros_dict(2, size, size) for _ in range(count)),
            )

    def test_elimination_budget_rejects_dense_rank_profile(self):
        """Composition stays under budget while predicted rank work does not."""
        size = 200
        count = 5
        assert count * size * size <= MAX_AGGREGATE_CELLS
        assert (count - 1) * size**3 <= MAX_AGGREGATE_MULTIPLICATION_WORK
        assert count * size**3 > MAX_AGGREGATE_ELIMINATION_WORK
        cx = ChainComplex(
            prime=2,
            min_degree=0,
            max_degree=count,
            dimensions=(size,) * (count + 1),
            differentials=tuple(zeros_dict(2, size, size) for _ in range(count)),
        )
        with pytest.raises(ValidationError, match="elimination operations"):
            HomologyRequest(complex=cx)

    def test_mapping_cone_budget_counts_both_complexes_and_predicted_cone(self):
        """The gate includes source, target, chain map, and the derived cone."""
        size = 80
        degrees = 11
        base = {
            "prime": 2,
            "min_degree": 0,
            "max_degree": degrees - 1,
            "dimensions": [size] * degrees,
            "differentials": [
                zeros_dict(2, size, size) for _ in range(degrees - 1)
            ],
        }
        payload = {
            "source": base,
            "target": base,
            "chain_map": [zeros_dict(2, size, size) for _ in range(degrees)],
        }
        input_cells = 2 * (degrees - 1) * size * size + degrees * size * size
        assert input_cells <= MAX_AGGREGATE_CELLS
        with pytest.raises(ValidationError, match="matrix cells"):
            MappingConeRequest.model_validate(payload)


class TestResultReplay:
    """Results retain their inputs and replay against the exact kernels."""

    def test_mapping_cone_result_replays_retained_source_map(self):
        source = ChainComplex(
            prime=5,
            min_degree=0,
            max_degree=1,
            dimensions=(1, 1),
            differentials=({"prime": 5, "entries": [[0]], "columns": 1},),
        )
        target = ChainComplex(
            prime=5, min_degree=0, max_degree=0, dimensions=(1,), differentials=()
        )
        request = MappingConeRequest(
            source=source,
            target=target,
            chain_map=(
                {"prime": 5, "entries": [[1]], "columns": 1},
                {"prime": 5, "entries": [], "columns": 1},
            ),
        )
        result = compute_mapping_cone(request)
        assert MappingConeResult.model_validate(result.model_dump()) == result
        # A relayed payload whose retained request differs cannot revalidate:
        # the replayed cone no longer matches the attached one.
        relayed = result.model_dump()
        relayed["request"]["chain_map"][0]["entries"] = [[0]]
        with pytest.raises(ValidationError, match="exact mapping cone"):
            MappingConeResult.model_validate(relayed)

    def test_cone_span_beyond_21_groups_rejected_at_admission(self):
        # Disjoint endpoints derive Cone(f)_n = C_{n-1} + D_n over
        # [-10, 11]: 22 consecutive groups, one more than any ChainComplex
        # represents. Admission must reject the request before construction.
        source = ChainComplex(
            prime=2, min_degree=10, max_degree=10, dimensions=(1,), differentials=()
        )
        target = ChainComplex(
            prime=2, min_degree=-10, max_degree=-10, dimensions=(1,), differentials=()
        )
        with pytest.raises(ValidationError, match="22 consecutive degrees"):
            MappingConeRequest(
                source=source,
                target=target,
                # D_10 does not exist: f_10 is the zero-row shaped zero map.
                chain_map=({"prime": 2, "entries": [], "columns": 1},),
            )

    def test_full_span_21_group_cone_accepted_and_roundtrips(self):
        # A derived cone of exactly 21 groups stays representable: source
        # degrees [-10, 9] shifted to [-9, 10] plus a target group at -10
        # spans [-10, 10], so execution must return a typed result.
        source = ChainComplex(
            prime=3,
            min_degree=-10,
            max_degree=9,
            dimensions=tuple(1 for _ in range(20)),
            differentials=(),
        )
        target = ChainComplex(
            prime=3, min_degree=-10, max_degree=-10, dimensions=(1,), differentials=()
        )
        request = MappingConeRequest(
            source=source,
            target=target,
            chain_map=tuple(
                # f_deg is the shaped zero map: 1x1 into D_-10 at degree -10,
                # zero-row elsewhere because the target has no group there.
                zeros_dict(3, 1, 1) if deg == -10 else {"prime": 3, "entries": [], "columns": 1}
                for deg in range(-10, 10)
            ),
        )
        result = compute_mapping_cone(request)
        assert len(result.cone.dimensions) == 21
        assert result.cone.min_degree == -10
        assert result.cone.max_degree == 10
        assert MappingConeResult.model_validate(result.model_dump()) == result

