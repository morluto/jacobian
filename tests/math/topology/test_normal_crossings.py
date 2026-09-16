"""Tests for normal crossings dual complexes and nearby-cycle lattices."""

from __future__ import annotations

from itertools import combinations, pairwise
from typing import NamedTuple

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.normal_crossings._models import (
    MAX_NC_BRANCH_MULTIPLICITY,
    MAX_NC_COMPONENTS,
    MAX_NC_STRATA,
    DualComplexResult,
    NearbyCycleLatticesResult,
    NormalCrossingsPresentation,
    NormalCrossingsPresentationRequest,
    NormalCrossingsStratum,
    SpecializationMap,
)
from jacobian.math.topology.normal_crossings._tools import TOOLS
from jacobian.math.topology.normal_crossings.operations import (
    dual_complex,
    nearby_cycle_lattices,
    specialization_matrix,
    verify_dual_complex_claim,
    verify_nearby_cycle_lattices_claim,
)

_DUAL_ID = "topology.normal_crossings.dual_complex.compute"
_LATTICE_ID = "topology.normal_crossings.nearby_cycle_lattices.compute"


def _tool(operation_id: str):
    return next(tool for tool in TOOLS if tool.operation_id == operation_id)


def _strata(
    strata: tuple[tuple[tuple[str, ...], int], ...],
) -> tuple[NormalCrossingsStratum, ...]:
    return tuple(
        NormalCrossingsStratum(components=key, dimension=dimension)
        for key, dimension in strata
    )


class _Args(NamedTuple):
    components: tuple[str, ...]
    strata: tuple[NormalCrossingsStratum, ...]


def _request(
    components: tuple[str, ...],
    strata: tuple[tuple[tuple[str, ...], int], ...],
) -> _Args:
    return _Args(components=components, strata=_strata(strata))


def _wire_request(
    components: tuple[str, ...],
    strata: tuple[tuple[tuple[str, ...], int], ...],
) -> NormalCrossingsPresentationRequest:
    return NormalCrossingsPresentationRequest(
        components=components, strata=_strata(strata)
    )


def _all_subsets(
    key: tuple[str, ...], component_dimension: int
) -> list[tuple[tuple[str, ...], int]]:
    return [
        (subset, component_dimension - (len(subset) - 1))
        for size in range(len(key), 0, -1)
        for subset in combinations(key, size)
    ]


_SMOOTH = _request(("D0",), ((("D0",), 2),))
_NODE = _request(
    ("D0", "D1"),
    tuple(_all_subsets(("D0", "D1"), 1)),
)
_TRIPLE = _request(
    ("D0", "D1", "D2"),
    tuple(_all_subsets(("D0", "D1", "D2"), 2)),
)
_DISJOINT = _request(
    ("D0", "D1"),
    ((("D0",), 1), (("D1",), 1)),
)
_TWO_NODES = _request(
    ("D0", "D1", "D2", "D3"),
    (
        (("D0",), 1),
        (("D1",), 1),
        (("D2",), 1),
        (("D3",), 1),
        (("D0", "D1"), 0),
        (("D2", "D3"), 0),
    ),
)


def _matmul(
    left: tuple[tuple[int, ...], ...],
    right: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    left_columns = len(left[0]) if left else 0
    right_columns = len(right[0]) if right else 0
    return (
        tuple(
            tuple(
                sum(row[k] * matrix_row[column] for k, matrix_row in enumerate(right))
                for column in range(right_columns)
            )
            for row in left
        )
        if left_columns == len(right)
        else pytest.fail("incomposable matrices")
    )


def _basis_rows(r: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(1 if column == k else -1 if column == k + 1 else 0 for column in range(r))
        for k in range(r - 1)
    )


def _specialization(
    result: NearbyCycleLatticesResult,
    target: tuple[str, ...],
    source: tuple[str, ...],
) -> SpecializationMap:
    return next(
        item
        for item in result.specializations
        if item.target_components == target and item.source_components == source
    )


class TestKnownAnswers:
    def test_smooth_local_model_dual_point(self) -> None:
        result = dual_complex(_SMOOTH.components, _SMOOTH.strata)
        assert result.dual_complex.vertices == ("D0",)
        assert result.dual_complex.maximal_simplices == (("D0",),)
        assert result.dual_complex.dimension == 0
        assert result.dual_complex.f_vector == (1,)

    def test_node_local_model_dual_interval(self) -> None:
        result = dual_complex(_NODE.components, _NODE.strata)
        assert result.dual_complex.vertices == ("D0", "D1")
        assert result.dual_complex.maximal_simplices == (("D0", "D1"),)
        assert result.dual_complex.dimension == 1
        assert result.dual_complex.f_vector == (2, 1)

    def test_triple_point_local_model_dual_full_simplex(self) -> None:
        result = dual_complex(_TRIPLE.components, _TRIPLE.strata)
        assert result.dual_complex.maximal_simplices == (("D0", "D1", "D2"),)
        assert result.dual_complex.dimension == 2
        assert result.dual_complex.f_vector == (3, 3, 1)

    def test_dual_face_counts_match_strata_by_cardinality(self) -> None:
        result = dual_complex(_TRIPLE.components, _TRIPLE.strata)
        assert result.dual_complex.f_vector == tuple(
            len(group.strata) for group in result.strata_by_cardinality
        )
        assert tuple(group.cardinality for group in result.strata_by_cardinality) == (
            1,
            2,
            3,
        )

    def test_branch_multiplicity_transport(self) -> None:
        result = dual_complex(_TRIPLE.components, _TRIPLE.strata)
        assert tuple(
            (record.components, record.branch_multiplicity) for record in result.strata
        ) == tuple(
            (stratum.components, len(stratum.components))
            for stratum in result.presentation.strata
        )

    def test_cech_signed_incidence_matrices(self) -> None:
        result = dual_complex(_TRIPLE.components, _TRIPLE.strata)
        assert result.cech_value.basis_sizes == (3, 3, 1)
        assert result.cech_value.coefficient_ring.value == "ZZ"
        assert result.cech_value.differential_matrices[0] == (
            ("-1", "-1", "0"),
            ("1", "0", "-1"),
            ("0", "1", "1"),
        )
        assert result.cech_value.differential_matrices[1] == (
            ("1",),
            ("-1",),
            ("1",),
        )
        assert tuple(
            entry.upper_dimension for entry in result.differential_squared_zero
        ) == (1, 2)

    def test_triple_point_stalk_lattice(self) -> None:
        result = nearby_cycle_lattices(_TRIPLE.components, _TRIPLE.strata)
        stalk = next(
            lattice
            for lattice in result.lattices
            if lattice.components == ("D0", "D1", "D2")
        )
        assert stalk.lattice_rank == 2
        assert stalk.saturated_basis.entries == ((1, -1, 0), (0, 1, -1))
        assert stalk.milnor_fiber_cohomology_ranks == (1, 2, 1)

    def test_node_stalk_and_zero_specialization(self) -> None:
        result = nearby_cycle_lattices(_NODE.components, _NODE.strata)
        stalk = next(
            lattice for lattice in result.lattices if lattice.components == ("D0", "D1")
        )
        assert stalk.lattice_rank == 1
        assert stalk.saturated_basis.entries == ((1, -1),)
        assert stalk.milnor_fiber_cohomology_ranks == (1, 1)
        for target in (("D0",), ("D1",)):
            map_ = _specialization(result, target, ("D0", "D1"))
            assert (map_.matrix.row_count, map_.matrix.column_count) == (1, 0)
            assert map_.matrix.entries == ((),)

    def test_triple_point_sum_fold_specializations(self) -> None:
        result = nearby_cycle_lattices(_TRIPLE.components, _TRIPLE.strata)
        assert _specialization(
            result, ("D0", "D1"), ("D0", "D1", "D2")
        ).matrix.entries == ((1,), (0,))
        assert _specialization(
            result, ("D0", "D2"), ("D0", "D1", "D2")
        ).matrix.entries == ((1,), (0,))
        assert _specialization(
            result, ("D1", "D2"), ("D0", "D1", "D2")
        ).matrix.entries == ((-1,), (1,))
        assert len(result.specializations) == 9


class TestBoundaryDegenerate:
    def test_single_component_rank_zero_lattice(self) -> None:
        result = nearby_cycle_lattices(_SMOOTH.components, _SMOOTH.strata)
        assert len(result.lattices) == 1
        stalk = result.lattices[0]
        assert stalk.lattice_rank == 0
        assert (
            stalk.saturated_basis.row_count,
            stalk.saturated_basis.column_count,
        ) == (
            0,
            1,
        )
        assert stalk.saturated_basis.entries == ()
        assert stalk.milnor_fiber_cohomology_ranks == (1,)
        assert result.specializations == ()

    def test_disconnected_components_dual_and_cech(self) -> None:
        result = dual_complex(_DISJOINT.components, _DISJOINT.strata)
        assert result.dual_complex.maximal_simplices == (("D0",), ("D1",))
        assert result.cech_value.basis_sizes == (2,)
        assert result.cech_value.differential_matrices == ()
        assert result.differential_squared_zero == ()

    def test_two_disjoint_nodes(self) -> None:
        result = dual_complex(_TWO_NODES.components, _TWO_NODES.strata)
        assert result.dual_complex.maximal_simplices == (("D0", "D1"), ("D2", "D3"))
        assert result.cech_value.basis_sizes == (4, 2)
        lattices = nearby_cycle_lattices(_TWO_NODES.components, _TWO_NODES.strata)
        assert len(lattices.specializations) == 4
        assert all(
            lattice.lattice_rank == len(lattice.components) - 1
            for lattice in lattices.lattices
        )

    def test_unsorted_input_canonicalizes(self) -> None:
        shuffled = _request(
            ("D2", "D0", "D1"),
            (
                (("D2",), 2),
                (("D0", "D1", "D2"), 0),
                (("D0",), 2),
                (("D1", "D2"), 1),
                (("D1",), 2),
                (("D0", "D2"), 1),
                (("D0", "D1"), 1),
            ),
        )
        result = dual_complex(shuffled.components, shuffled.strata)
        assert result.presentation.components == ("D0", "D1", "D2")
        assert result == dual_complex(_TRIPLE.components, _TRIPLE.strata)


class TestAdversarial:
    def test_non_downward_closed_incidence_rejected(self) -> None:
        broken = _request(
            ("D0", "D1", "D2"),
            (
                (("D0",), 2),
                (("D1",), 2),
                (("D2",), 2),
                (("D0", "D1"), 1),
                (("D0", "D2"), 1),
                (("D0", "D1", "D2"), 0),
            ),
        )
        with pytest.raises(OperationDomainValidationError) as excinfo:
            dual_complex(broken.components, broken.strata)
        assert (
            excinfo.value.errors()[0]["type"]
            == "topology.normal_crossings.not_downward_closed"
        )

    def test_missing_component_stratum_rejected(self) -> None:
        broken = _request(
            ("D0", "D1"),
            ((("D0",), 1), (("D0", "D1"), 0)),
        )
        with pytest.raises(OperationDomainValidationError) as excinfo:
            nearby_cycle_lattices(broken.components, broken.strata)
        assert (
            excinfo.value.errors()[0]["type"]
            == "topology.normal_crossings.missing_component_stratum"
        )

    def test_duplicate_stratum_component_sets_rejected(self) -> None:
        broken = _request(
            ("D0", "D1"),
            (
                (("D0",), 1),
                (("D1",), 1),
                (("D0", "D1"), 0),
                (("D0", "D1"), 0),
            ),
        )
        with pytest.raises(OperationDomainValidationError) as excinfo:
            dual_complex(broken.components, broken.strata)
        assert (
            excinfo.value.errors()[0]["type"]
            == "topology.normal_crossings.duplicate_stratum"
        )

    def test_undeclared_component_rejected(self) -> None:
        broken = _request(("D0",), ((("D0", "D1"), 0), (("D0",), 1), (("D1",), 1)))
        with pytest.raises(OperationDomainValidationError) as excinfo:
            dual_complex(broken.components, broken.strata)
        assert (
            excinfo.value.errors()[0]["type"]
            == "topology.normal_crossings.undeclared_component"
        )

    def test_dimension_identity_violation_rejected(self) -> None:
        broken = _request(
            ("D0", "D1"),
            ((("D0",), 1), (("D1",), 2), (("D0", "D1"), 0)),
        )
        with pytest.raises(OperationDomainValidationError) as excinfo:
            dual_complex(broken.components, broken.strata)
        assert (
            excinfo.value.errors()[0]["type"]
            == "topology.normal_crossings.dimension_identity_violated"
        )

    def test_forged_specialization_matrix_fails_verification(self) -> None:
        claim = nearby_cycle_lattices(_TRIPLE.components, _TRIPLE.strata)
        assert verify_nearby_cycle_lattices_claim(claim)
        honest = _specialization(claim, ("D0", "D1"), ("D0", "D1", "D2"))
        forged_map = SpecializationMap(
            source_components=honest.source_components,
            target_components=honest.target_components,
            matrix=honest.matrix.model_copy(update={"entries": ((2,), (0,))}),
        )
        forged = tuple(
            forged_map if item is honest else item for item in claim.specializations
        )
        assert not verify_nearby_cycle_lattices_claim(
            claim.model_copy(update={"specializations": forged})
        )

    def test_forged_dual_complex_claim_fails_verification(self) -> None:
        claim = dual_complex(_TRIPLE.components, _TRIPLE.strata)
        assert verify_dual_complex_claim(claim)
        assert not verify_dual_complex_claim(
            claim.model_copy(update={"differential_squared_zero": ()})
        )

    def test_forged_presentation_fails_verification(self) -> None:
        claim = dual_complex(_TRIPLE.components, _TRIPLE.strata)
        forged_strata = tuple(
            stratum.model_copy(update={"dimension": stratum.dimension + 1})
            for stratum in claim.presentation.strata
        )
        assert not verify_dual_complex_claim(
            claim.model_copy(
                update={
                    "presentation": claim.presentation.model_copy(
                        update={"strata": forged_strata}
                    )
                }
            )
        )


class TestDefiningInvariants:
    @pytest.mark.parametrize(
        "presentation",
        [_SMOOTH, _NODE, _TRIPLE, _DISJOINT, _TWO_NODES],
    )
    def test_cech_differential_squares_to_zero(
        self, presentation: NormalCrossingsPresentation
    ) -> None:
        result = dual_complex(presentation.components, presentation.strata)
        matrices = tuple(
            tuple(tuple(int(entry) for entry in row) for row in matrix)
            for matrix in result.cech_value.differential_matrices
        )
        for lower, upper in pairwise(matrices):
            product = _matmul(lower, upper)
            assert all(value == 0 for row in product for value in row)

    @pytest.mark.parametrize("presentation", [_SMOOTH, _NODE, _TRIPLE, _TWO_NODES])
    def test_saturated_basis_sum_zero_replay(
        self, presentation: NormalCrossingsPresentation
    ) -> None:
        result = nearby_cycle_lattices(presentation.components, presentation.strata)
        for lattice in result.lattices:
            r = lattice.branch_multiplicity
            assert lattice.saturated_basis.entries == _basis_rows(r)
            assert all(sum(row) == 0 for row in lattice.saturated_basis.entries)

    @pytest.mark.parametrize("presentation", [_NODE, _TRIPLE, _TWO_NODES])
    def test_specializations_land_in_target_kernel(
        self, presentation: NormalCrossingsPresentation
    ) -> None:
        """Row-sum-zero replay: ambient images of source basis vectors."""

        result = nearby_cycle_lattices(presentation.components, presentation.strata)
        for map_ in result.specializations:
            target_rows = _basis_rows(len(map_.target_components))
            positions = {
                label: index for index, label in enumerate(map_.target_components)
            }
            fold_at = positions[map_.target_components[-1]]
            for k, row in enumerate(map_.matrix.entries):
                ambient = [0] * len(map_.target_components)
                for a, coefficient in enumerate(row):
                    for column, value in enumerate(target_rows[a]):
                        ambient[column] += coefficient * value
                expected = [0] * len(map_.target_components)
                for position, label in enumerate(map_.source_components):
                    delta = 1 if position == k else -1 if position == k + 1 else 0
                    expected[positions.get(label, fold_at)] += delta
                assert ambient == expected
                assert sum(ambient) == 0

    def test_composition_equals_direct_map(self) -> None:
        result = nearby_cycle_lattices(_TRIPLE.components, _TRIPLE.strata)
        first = _specialization(result, ("D0", "D1"), ("D0", "D1", "D2"))
        second = _specialization(result, ("D0",), ("D0", "D1"))
        direct = specialization_matrix(
            _TRIPLE.components, _TRIPLE.strata, ("D0", "D1", "D2"), ("D0",)
        )
        assert (
            _matmul(first.matrix.entries, second.matrix.entries)
            == direct.matrix.entries
        )

    def test_direct_two_step_composition(self) -> None:
        result = nearby_cycle_lattices(_TRIPLE.components, _TRIPLE.strata)
        middle = _specialization(result, ("D1", "D2"), ("D0", "D1", "D2"))
        second = _specialization(result, ("D1",), ("D1", "D2"))
        direct = specialization_matrix(
            _TRIPLE.components, _TRIPLE.strata, ("D0", "D1", "D2"), ("D1",)
        )
        assert (
            _matmul(middle.matrix.entries, second.matrix.entries)
            == direct.matrix.entries
        )

    def test_exterior_rank_table_matches_binomials(self) -> None:
        result = nearby_cycle_lattices(_TRIPLE.components, _TRIPLE.strata)
        for lattice in result.lattices:
            r = lattice.branch_multiplicity
            assert lattice.milnor_fiber_cohomology_ranks == tuple(
                _binomial(r - 1, q) for q in range(r)
            )


def _binomial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    value = 1
    for i in range(k):
        value = value * (n - i) // (i + 1)
    return value


class TestEnvelope:
    def test_component_budget(self) -> None:
        components = tuple(f"D{index}" for index in range(MAX_NC_COMPONENTS + 1))
        with pytest.raises(ValidationError):
            _wire_request(
                components, tuple(((component,), 0) for component in components)
            )

    def test_strata_budget(self) -> None:
        components = tuple(f"D{index}" for index in range(MAX_NC_COMPONENTS))
        strata: list[tuple[tuple[str, ...], int]] = [
            ((component,), 0) for component in components
        ]
        strata.extend(
            (pair, 0)
            for pair in combinations(components, 2)
            if len(strata) < MAX_NC_STRATA + 1
        )
        with pytest.raises(ValidationError):
            _wire_request(components, tuple(strata))

    def test_branch_multiplicity_budget(self) -> None:
        components = tuple(
            f"D{index}" for index in range(MAX_NC_BRANCH_MULTIPLICITY + 1)
        )
        strata = tuple(((component,), 0) for component in components)
        with pytest.raises(ValidationError):
            _wire_request(components, (*strata, (components, 0)))

    def _full_local_model(self, count: int) -> _Args:
        components = tuple(f"D{index}" for index in range(count))
        return _request(
            components,
            tuple(
                (subset, count - len(subset))
                for size in range(1, count + 1)
                for subset in combinations(components, size)
            ),
        )

    def test_envelope_maximum_lattice_model_accepts(self) -> None:
        request = self._full_local_model(MAX_NC_BRANCH_MULTIPLICITY)
        result = nearby_cycle_lattices(request.components, request.strata)
        top = next(
            lattice
            for lattice in result.lattices
            if lattice.branch_multiplicity == MAX_NC_BRANCH_MULTIPLICITY
        )
        assert top.milnor_fiber_cohomology_ranks == (1, 7, 21, 35, 35, 21, 7, 1)

    def test_envelope_maximum_dual_model_accepts(self) -> None:
        request = self._full_local_model(6)
        dual = dual_complex(request.components, request.strata)
        assert dual.dual_complex.maximal_simplices == (
            ("D0", "D1", "D2", "D3", "D4", "D5"),
        )

    def test_full_eight_branch_cech_rejected_by_preflight(self) -> None:
        request = self._full_local_model(MAX_NC_BRANCH_MULTIPLICITY)
        with pytest.raises(OperationResourceAdmissionError) as excinfo:
            dual_complex(request.components, request.strata)
        assert (
            excinfo.value.errors()[0]["type"]
            == "topology.normal_crossings.cech_group_budget"
        )


class TestCatalogParity:
    def test_native_matches_catalog_dual_complex(self) -> None:
        tool = _tool(_DUAL_ID)
        request = tool.request_type.model_validate(
            {
                "components": ["D0", "D1", "D2"],
                "strata": [
                    {"components": ["D0"], "dimension": 2},
                    {"components": ["D1"], "dimension": 2},
                    {"components": ["D2"], "dimension": 2},
                    {"components": ["D0", "D1"], "dimension": 1},
                    {"components": ["D0", "D2"], "dimension": 1},
                    {"components": ["D1", "D2"], "dimension": 1},
                    {"components": ["D0", "D1", "D2"], "dimension": 0},
                ],
            }
        )
        presentation = NormalCrossingsPresentation(
            components=request.components, strata=request.strata
        )
        assert tool.run(request) == dual_complex(
            presentation.components, presentation.strata
        )

    def test_native_matches_catalog_lattices(self) -> None:
        tool = _tool(_LATTICE_ID)
        request = tool.request_type.model_validate(
            {
                "components": ["D0", "D1"],
                "strata": [
                    {"components": ["D0"], "dimension": 1},
                    {"components": ["D1"], "dimension": 1},
                    {"components": ["D0", "D1"], "dimension": 0},
                ],
            }
        )
        presentation = NormalCrossingsPresentation(
            components=request.components, strata=request.strata
        )
        assert tool.run(request) == nearby_cycle_lattices(
            presentation.components, presentation.strata
        )

    def test_examples_execute(self) -> None:
        for operation_id in (_DUAL_ID, _LATTICE_ID):
            tool = _tool(operation_id)
            assert tool.examples
            for example in tool.examples:
                request = tool.request_type.model_validate(example.input)
                assert tool.run(request) is not None

    def test_serialization_round_trips(self) -> None:
        dual = dual_complex(_TRIPLE.components, _TRIPLE.strata)
        assert DualComplexResult.model_validate_json(dual.model_dump_json()) == dual
        lattices = nearby_cycle_lattices(_TRIPLE.components, _TRIPLE.strata)
        assert (
            NearbyCycleLatticesResult.model_validate_json(lattices.model_dump_json())
            == lattices
        )
