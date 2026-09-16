"""Tests for topology.discrete_morse.matching.construct (#1809)."""

from __future__ import annotations

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology._models import (
    canonical_complex,
    simplicial_complex_request_from_value,
)
from jacobian.math.topology.discrete_morse import (
    CriticalCellProfile,
    DiscreteMorseMatchingRequest,
    DiscreteMorseMatchingResult,
    MorseMatchingFault,
    MorseMatchingOutcome,
    construct_matching,
)

OPERATION_ID = "topology.discrete_morse.matching.construct"


def _request(
    vertices: list[str],
    facets: list[list[str]],
    pairs: list[tuple[list[str], list[str]]],
) -> DiscreteMorseMatchingRequest:
    return DiscreteMorseMatchingRequest.model_validate(
        {
            "complex": {"vertices": vertices, "facets": facets},
            "pairs": [{"face": face, "coface": coface} for face, coface in pairs],
        }
    )


def _circle_request(
    pairs: list[tuple[list[str], list[str]]],
) -> DiscreteMorseMatchingRequest:
    return _request(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]], pairs)


class TestKnownAnswer:
    def test_circle_matching_leaves_one_vertex_and_one_edge(self) -> None:
        result = construct_matching(
            _circle_request([(["a"], ["a", "b"]), (["c"], ["a", "c"])])
        )
        assert result.outcome is MorseMatchingOutcome.ACYCLIC_MATCHING
        profile = result.critical_profile
        assert profile is not None
        assert profile.counts_by_dimension == (1, 1)
        assert profile.critical_cells == (("b",), ("b", "c"))
        assert profile.euler_characteristic == 0
        assert profile.closure_euler_characteristic == 0
        assert result.hasse_edges == 6

    def test_single_two_simplex_collapses_to_one_vertex(self) -> None:
        result = construct_matching(
            _request(
                ["a", "b", "c"],
                [["a", "b", "c"]],
                [
                    (["b"], ["a", "b"]),
                    (["c"], ["b", "c"]),
                    (["a", "c"], ["a", "b", "c"]),
                ],
            )
        )
        assert result.outcome is MorseMatchingOutcome.ACYCLIC_MATCHING
        profile = result.critical_profile
        assert profile is not None
        assert profile.counts_by_dimension == (1, 0, 0)
        assert profile.critical_cells == (("a",),)
        assert profile.euler_characteristic == 1

    def test_topological_order_respects_directed_hasse_edges(self) -> None:
        result = construct_matching(
            _circle_request([(["a"], ["a", "b"]), (["c"], ["a", "c"])])
        )
        assert result.outcome is MorseMatchingOutcome.ACYCLIC_MATCHING
        order = result.topological_order
        position = {cell: index for index, cell in enumerate(order)}
        matched = {(pair.face, pair.coface) for pair in result.pairs}
        cells = set(order)
        for coface in cells:
            for i in range(len(coface)):
                face = coface[:i] + coface[i + 1 :]
                if face not in cells:
                    continue
                if (face, coface) in matched:
                    assert position[face] < position[coface]
                else:
                    assert position[coface] < position[face]


class TestBoundaryDegenerate:
    def test_empty_matching_makes_every_cell_critical(self) -> None:
        result = construct_matching(_circle_request([]))
        assert result.outcome is MorseMatchingOutcome.ACYCLIC_MATCHING
        profile = result.critical_profile
        assert profile is not None
        assert profile.counts_by_dimension == (3, 3)
        assert len(profile.critical_cells) == 6
        assert result.pairs == ()

    def test_single_vertex_complex(self) -> None:
        result = construct_matching(_request(["a"], [["a"]], []))
        assert result.outcome is MorseMatchingOutcome.ACYCLIC_MATCHING
        profile = result.critical_profile
        assert profile is not None
        assert profile.counts_by_dimension == (1,)
        assert profile.critical_cells == (("a",),)
        assert result.hasse_edges == 0
        assert result.topological_order == (("a",),)

    def test_interval_full_collapse_has_one_critical_vertex(self) -> None:
        result = construct_matching(
            _request(
                ["a", "b", "c", "d"],
                [["a", "b"], ["b", "c"], ["c", "d"]],
                [
                    (["b"], ["a", "b"]),
                    (["c"], ["b", "c"]),
                    (["d"], ["c", "d"]),
                ],
            )
        )
        assert result.outcome is MorseMatchingOutcome.ACYCLIC_MATCHING
        profile = result.critical_profile
        assert profile is not None
        assert profile.counts_by_dimension == (1, 0)
        assert profile.euler_characteristic == 1

    def test_two_isolated_points_leave_two_critical_vertices(self) -> None:
        result = construct_matching(_request(["a", "b"], [["a"], ["b"]], []))
        assert result.outcome is MorseMatchingOutcome.ACYCLIC_MATCHING
        profile = result.critical_profile
        assert profile is not None
        assert profile.counts_by_dimension == (2,)
        assert profile.euler_characteristic == 2


class TestAdversarial:
    def test_cyclic_matching_returns_concrete_closed_v_path(self) -> None:
        result = construct_matching(
            _circle_request(
                [
                    (["a"], ["a", "b"]),
                    (["b"], ["b", "c"]),
                    (["c"], ["a", "c"]),
                ]
            )
        )
        assert result.outcome is MorseMatchingOutcome.CYCLIC_MATCHING
        path = result.closed_v_path
        assert len(path) >= 2
        assert len(set(path)) == len(path)
        assert result.critical_profile is None
        assert result.topological_order == ()

    def test_closed_v_path_replays_on_the_directed_hasse_graph(self) -> None:
        result = construct_matching(
            _circle_request(
                [
                    (["a"], ["a", "b"]),
                    (["b"], ["b", "c"]),
                    (["c"], ["a", "c"]),
                ]
            )
        )
        assert result.outcome is MorseMatchingOutcome.CYCLIC_MATCHING
        matched = {(pair.face, pair.coface) for pair in result.pairs}
        path = result.closed_v_path
        for source, target in zip(path, (*path[1:], path[0]), strict=True):
            if (source, target) in matched:
                continue
            assert set(target) < set(source)
            assert (target, source) not in matched

    def test_non_cover_pair_is_rejected(self) -> None:
        result = construct_matching(
            _request(
                ["a", "b", "c"],
                [["a", "b", "c"]],
                [(["a"], ["a", "b", "c"])],
            )
        )
        assert result.outcome is MorseMatchingOutcome.INVALID_MATCHING
        assert result.fault is MorseMatchingFault.NOT_A_COVER_PAIR
        assert result.fault_pair_index == 0

    def test_unknown_cell_is_rejected(self) -> None:
        result = construct_matching(_circle_request([(["a"], ["a", "z"])]))
        assert result.outcome is MorseMatchingOutcome.INVALID_MATCHING
        assert result.fault is MorseMatchingFault.UNKNOWN_CELL
        assert result.fault_pair_index == 0

    def test_duplicate_cell_across_pairs_is_rejected(self) -> None:
        result = construct_matching(
            _circle_request([(["a"], ["a", "b"]), (["a"], ["a", "c"])])
        )
        assert result.outcome is MorseMatchingOutcome.INVALID_MATCHING
        assert result.fault is MorseMatchingFault.DUPLICATE_CELL
        assert result.fault_pair_index == 1

    def test_source_substitution_from_equal_shaped_complex_is_rejected(
        self,
    ) -> None:
        result = construct_matching(_circle_request([(["x"], ["x", "y"])]))
        assert result.outcome is MorseMatchingOutcome.INVALID_MATCHING
        assert result.fault is MorseMatchingFault.UNKNOWN_CELL


class TestDefiningInvariant:
    @pytest.mark.parametrize(
        "pairs",
        [
            [],
            [(["a"], ["a", "b"])],
            [(["a"], ["a", "b"]), (["c"], ["a", "c"])],
        ],
    )
    def test_euler_identity_between_critical_and_closure_counts(
        self, pairs: list[tuple[list[str], list[str]]]
    ) -> None:
        result = construct_matching(_circle_request(pairs))
        assert result.outcome is MorseMatchingOutcome.ACYCLIC_MATCHING
        profile = result.critical_profile
        assert profile is not None
        critical_euler = sum(
            (-1) ** dimension * count
            for dimension, count in enumerate(profile.counts_by_dimension)
        )
        closure_euler = sum(
            (-1) ** dimension * count
            for dimension, count in enumerate(result.complex.f_vector)
        )
        assert profile.euler_characteristic == critical_euler
        assert profile.closure_euler_characteristic == closure_euler
        assert critical_euler == closure_euler

    def test_critical_and_matched_cells_partition_the_closure(self) -> None:
        result = construct_matching(
            _circle_request([(["a"], ["a", "b"]), (["c"], ["a", "c"])])
        )
        assert result.outcome is MorseMatchingOutcome.ACYCLIC_MATCHING
        profile = result.critical_profile
        assert profile is not None
        matched = {cell for pair in result.pairs for cell in (pair.face, pair.coface)}
        closure = {
            face for group in result.complex.faces_by_dimension for face in group.faces
        }
        assert matched | set(profile.critical_cells) == closure
        assert matched & set(profile.critical_cells) == set()


class TestNativeCatalogParity:
    def test_catalog_tool_runs_the_same_kernel(self) -> None:
        tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == OPERATION_ID)
        request = _circle_request([(["a"], ["a", "b"]), (["c"], ["a", "c"])])
        assert tool.run(request) == construct_matching(request)

    def test_published_examples_execute(self) -> None:
        tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == OPERATION_ID)
        outcomes = []
        for example in tool.examples:
            request = DiscreteMorseMatchingRequest.model_validate(example.input)
            outcomes.append(tool.run(request).outcome)
        assert outcomes == [
            MorseMatchingOutcome.ACYCLIC_MATCHING,
            MorseMatchingOutcome.CYCLIC_MATCHING,
        ]


class TestSerialization:
    def test_acyclic_result_round_trips(self) -> None:
        result = construct_matching(
            _circle_request([(["a"], ["a", "b"]), (["c"], ["a", "c"])])
        )
        restored = DiscreteMorseMatchingResult.model_validate_json(
            result.model_dump_json()
        )
        assert restored == result

    def test_cyclic_result_round_trips(self) -> None:
        result = construct_matching(
            _circle_request(
                [
                    (["a"], ["a", "b"]),
                    (["b"], ["b", "c"]),
                    (["c"], ["a", "c"]),
                ]
            )
        )
        restored = DiscreteMorseMatchingResult.model_validate_json(
            result.model_dump_json()
        )
        assert restored == result

    def test_forged_critical_profile_fails_validation(self) -> None:
        result = construct_matching(
            _circle_request([(["a"], ["a", "b"]), (["c"], ["a", "c"])])
        )
        payload = result.model_dump(mode="json")
        payload["critical_profile"]["euler_characteristic"] = 7
        with pytest.raises(ValueError):
            DiscreteMorseMatchingResult.model_validate(payload)

    def test_forged_partition_fails_validation(self) -> None:
        result = construct_matching(_circle_request([]))
        payload = result.model_dump(mode="json")
        payload["critical_profile"]["critical_cells"] = [["a"]]
        with pytest.raises(ValueError):
            DiscreteMorseMatchingResult.model_validate(payload)


class TestEnvelope:
    def test_pair_count_above_the_envelope_is_a_resource_rejection(self) -> None:
        canonical = canonical_complex(("a", "b"), (("a", "b"),))
        from jacobian.math.topology.discrete_morse._models import (
            MAX_MORSE_PAIRS,
            MatchingPair,
        )

        request = DiscreteMorseMatchingRequest.model_construct(
            complex=simplicial_complex_request_from_value(canonical),
            pairs=tuple(
                MatchingPair(face=("a",), coface=("a", "b"))
                for _ in range(MAX_MORSE_PAIRS + 1)
            ),
        )
        with pytest.raises(OperationResourceAdmissionError) as excinfo:
            construct_matching(request)
        assert excinfo.value.errors()[0]["type"] == (
            "topology.discrete_morse.admission.pairs"
        )

    def test_pair_count_at_the_envelope_is_admitted(self) -> None:
        canonical = canonical_complex(("a", "b"), (("a", "b"),))
        from jacobian.math.topology.discrete_morse._models import (
            MAX_MORSE_PAIRS,
            MatchingPair,
        )

        request = DiscreteMorseMatchingRequest.model_construct(
            complex=simplicial_complex_request_from_value(canonical),
            pairs=tuple(
                MatchingPair(face=("a",), coface=("a", "b"))
                for _ in range(MAX_MORSE_PAIRS)
            ),
        )
        result = construct_matching(request)
        assert result.outcome is MorseMatchingOutcome.INVALID_MATCHING
        assert result.fault is MorseMatchingFault.DUPLICATE_CELL

    def test_cell_envelope_constant_matches_the_published_bound(self) -> None:
        from jacobian.math.topology.discrete_morse._models import MAX_MORSE_CELLS

        assert MAX_MORSE_CELLS == 4096


class TestProfileModel:
    def test_negative_critical_counts_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            CriticalCellProfile(
                counts_by_dimension=(-1, 2),
                critical_cells=(("a",), ("b",), ("a", "b")),
                euler_characteristic=1,
                closure_euler_characteristic=1,
            )
