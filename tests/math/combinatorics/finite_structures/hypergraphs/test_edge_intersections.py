"""Defining-invariant and boundary tests for edge-intersection profiles."""

from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGE_INTERSECTION_CELLS,
    MAX_EDGE_INTERSECTION_RESULT_BYTES,
    MAX_EDGE_PAIR_COUNT,
    EdgeIntersectionEntry,
    EdgeIntersectionsRequest,
    EdgeIntersectionsResult,
    FiniteHypergraph,
    _edge_intersection_preflight_data,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._operations import (
    compute_edge_intersections,
    edge_intersections,
)

NONLINEAR = {
    "vertices": ["a", "b", "c", "d"],
    "edges": [
        ["e1", ["a", "b", "c"]],
        ["e2", ["b", "c", "d"]],
        ["e3", ["a", "d"]],
    ],
}


def _profile(source: object) -> EdgeIntersectionsResult:
    return compute_edge_intersections(
        EdgeIntersectionsRequest.model_validate({"hypergraph": source})
    )


class TestEdgeIntersections:
    def test_complete_nonlinear_profile_and_first_violation(self) -> None:
        result = _profile(NONLINEAR)

        assert tuple(
            (
                entry.left_edge_id,
                entry.right_edge_id,
                entry.intersection,
                entry.intersection_size,
            )
            for entry in result.pair_intersections
        ) == (
            ("e1", "e2", ("b", "c"), 2),
            ("e1", "e3", ("a",), 1),
            ("e2", "e3", ("d",), 1),
        )
        assert result.pair_count == 3
        assert result.histogram == ((1, 2), (2, 1))
        assert result.maximum_intersection_size == 2
        assert not result.is_linear
        assert result.first_linearity_violation == result.pair_intersections[0]

    def test_linear_profile_includes_disjoint_pairs(self) -> None:
        result = _profile(
            {
                "vertices": ["a", "b", "c", "d"],
                "edges": [
                    ["left", ["a", "b"]],
                    ["middle", ["b", "c"]],
                    ["isolated", ["d"]],
                ],
            }
        )

        assert result.histogram == ((0, 2), (1, 1))
        assert result.maximum_intersection_size == 1
        assert result.is_linear
        assert result.first_linearity_violation is None

    @pytest.mark.parametrize(
        "hypergraph",
        (
            {"vertices": [], "edges": []},
            {"vertices": ["a", "b"], "edges": [["only", ["a", "b"]]]},
        ),
    )
    def test_zero_or_one_edge_has_empty_linear_profile(
        self, hypergraph: object
    ) -> None:
        result = _profile(hypergraph)

        assert result.pair_intersections == ()
        assert result.pair_count == 0
        assert result.histogram == ()
        assert result.maximum_intersection_size == 0
        assert result.is_linear
        assert result.first_linearity_violation is None

    def test_duplicate_member_sets_remain_distinct_indexed_edges(self) -> None:
        result = _profile(
            {
                "vertices": ["a", "b"],
                "edges": [
                    ["copy-1", ["a", "b"]],
                    ["copy-2", ["a", "b"]],
                ],
            }
        )

        assert result.pair_count == 1
        assert result.pair_intersections[0] == EdgeIntersectionEntry(
            left_edge_id="copy-1",
            right_edge_id="copy-2",
            intersection=("a", "b"),
            intersection_size=2,
        )
        assert not result.is_linear

    def test_ledger_covers_every_declared_edge_pair_once(self) -> None:
        result = _profile(
            {
                "vertices": ["a", "b", "c"],
                "edges": [
                    ["z", ["a", "b"]],
                    ["a", ["b", "c"]],
                    ["m", ["a", "c"]],
                    ["q", []],
                ],
            }
        )
        edge_ids = tuple(edge_id for edge_id, _ in result.hypergraph.edges)

        assert tuple(
            (entry.left_edge_id, entry.right_edge_id)
            for entry in result.pair_intersections
        ) == tuple(combinations(edge_ids, 2))
        assert result.pair_count == len(result.pair_intersections)
        assert sum(count for _, count in result.histogram) == result.pair_count

    def test_native_function_accepts_canonical_hypergraph(self) -> None:
        hypergraph = FiniteHypergraph.model_validate(NONLINEAR)

        assert edge_intersections(hypergraph) == _profile(NONLINEAR)

    def test_result_round_trip_revalidates_source_relation(self) -> None:
        result = _profile(NONLINEAR)

        assert EdgeIntersectionsResult.model_validate(result.model_dump()) == result


class TestEdgeIntersectionBinding:
    def test_rejects_forged_pair_intersection(self) -> None:
        payload = _profile(NONLINEAR).model_dump(mode="json")
        payload["pair_intersections"][0]["intersection"] = ["b"]
        payload["pair_intersections"][0]["intersection_size"] = 1

        with pytest.raises(ValidationError):
            EdgeIntersectionsResult.model_validate(payload)

    def test_rejects_forged_histogram(self) -> None:
        payload = _profile(NONLINEAR).model_dump(mode="json")
        payload["histogram"] = [[1, 3]]

        with pytest.raises(ValidationError):
            EdgeIntersectionsResult.model_validate(payload)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        (
            (
                "maximum_intersection_size",
                1,
                "must be derived from pair_intersections",
            ),
            ("is_linear", True, "must match the exact pair intersections"),
            (
                "first_linearity_violation",
                None,
                "must be the first canonical pair",
            ),
        ),
    )
    def test_rejects_forged_derived_values(
        self, field: str, value: object, message: str
    ) -> None:
        payload = _profile(NONLINEAR).model_dump(mode="json")
        payload[field] = value

        with pytest.raises(ValidationError):
            EdgeIntersectionsResult.model_validate(payload)

    def test_rejects_aggregate_authored_intersections_before_replay(self) -> None:
        vertices = tuple(f"v{i:02}" for i in range(14))
        entry = EdgeIntersectionEntry(
            left_edge_id="left",
            right_edge_id="right",
            intersection=vertices,
            intersection_size=len(vertices),
        )
        entry_count = MAX_EDGE_INTERSECTION_CELLS // len(vertices) + 1

        with pytest.raises(ValidationError):
            EdgeIntersectionsResult(
                hypergraph=FiniteHypergraph(vertices=vertices, edges=()),
                pair_intersections=(entry,) * entry_count,
                pair_count=0,
                histogram=(),
                maximum_intersection_size=0,
                is_linear=True,
            )


class TestEdgeIntersectionPreflight:
    def test_near_intersection_cell_boundary_is_accepted(self) -> None:
        vertices = tuple(f"v{i:02}" for i in range(13))
        hypergraph = FiniteHypergraph(
            vertices=vertices,
            edges=tuple((f"e{i:03}", vertices) for i in range(100)),
        )
        request = EdgeIntersectionsRequest(hypergraph=hypergraph)
        pair_count, incidences, cells, estimated_bytes = (
            _edge_intersection_preflight_data(hypergraph)
        )

        assert pair_count == 4_950
        assert incidences == 1_300
        assert cells == 64_350 <= MAX_EDGE_INTERSECTION_CELLS
        result = compute_edge_intersections(request)
        assert result.pair_count == pair_count
        assert (
            sum(entry.intersection_size for entry in result.pair_intersections) == cells
        )
        actual_bytes = len(encode_strict_json(result.model_dump(mode="json")))
        assert actual_bytes <= estimated_bytes <= MAX_EDGE_INTERSECTION_RESULT_BYTES

    def test_immediately_larger_intersection_cell_family_is_rejected(self) -> None:
        vertices = tuple(f"v{i:02}" for i in range(14))

        request = EdgeIntersectionsRequest.model_validate(
            {
                "hypergraph": {
                    "vertices": vertices,
                    "edges": tuple((f"e{i:03}", vertices) for i in range(100)),
                }
            }
        )
        with pytest.raises(ValueError, match="intersection-cell"):
            compute_edge_intersections(request)

    def test_sparse_overlap_family_uses_exact_incidence_cell_count(self) -> None:
        vertices = tuple(f"v{i:03}" for i in range(100))
        edges = tuple(
            (
                f"e{edge_index:03}",
                tuple(
                    vertices[(edge_index + offset) % len(vertices)]
                    for offset in range(14)
                ),
            )
            for edge_index in range(100)
        )
        hypergraph = FiniteHypergraph(vertices=vertices, edges=edges)

        pair_count, incidences, cells, estimated_bytes = (
            _edge_intersection_preflight_data(hypergraph)
        )
        request = EdgeIntersectionsRequest(hypergraph=hypergraph)
        result = compute_edge_intersections(request)

        assert pair_count == 4_950
        assert incidences == 1_400
        assert cells == 100 * (14 * 13 // 2) == 9_100
        assert (
            sum(entry.intersection_size for entry in result.pair_intersections) == cells
        )
        actual_bytes = len(encode_strict_json(result.model_dump(mode="json")))
        assert actual_bytes <= estimated_bytes <= MAX_EDGE_INTERSECTION_RESULT_BYTES

    def test_more_than_one_hundred_indexed_edges_is_admitted(self) -> None:
        request = EdgeIntersectionsRequest.model_validate(
            {
                "hypergraph": {
                    "vertices": [],
                    "edges": tuple((f"e{i:03}", ()) for i in range(101)),
                }
            }
        )

        assert len(request.hypergraph.edges) == 101

    def test_serialized_output_bound_is_rejected_before_pair_materialization(
        self,
    ) -> None:
        # Control characters are one UTF-8 byte each but require six canonical
        # JSON bytes.  This stays well inside the source label-byte bound while
        # making the complete 64,350-membership ledger exceed 10 MiB.
        vertices = tuple(chr(codepoint) * 64 for codepoint in range(1, 14))

        request = EdgeIntersectionsRequest.model_validate(
            {
                "hypergraph": {
                    "vertices": vertices,
                    "edges": tuple((f"e{i:03}", vertices) for i in range(100)),
                }
            }
        )
        with pytest.raises(ValueError, match="canonical output limit"):
            compute_edge_intersections(request)

    def test_output_bound_preserves_exact_non_normalized_label_bytes(self) -> None:
        # Each 63-code-point label is 189 UTF-8 bytes, but NFC would compose
        # every three Jamo into one 3-byte Hangul syllable. Dispatch preserves
        # the exact source strings, so a normalized estimate would undercount
        # this complete ledger by more than eight MiB and admit an oversized
        # result.
        vertices = tuple(
            (chr(0x1100 + index) + "\u1161\u11a8") * 21 for index in range(13)
        )

        request = EdgeIntersectionsRequest.model_validate(
            {
                "hypergraph": {
                    "vertices": vertices,
                    "edges": tuple((f"e{i:03}", vertices) for i in range(100)),
                }
            }
        )
        with pytest.raises(ValueError, match="canonical output limit"):
            compute_edge_intersections(request)

    def test_schema_exposes_complete_profile_bounds(self) -> None:
        request_schema = EdgeIntersectionsRequest.model_json_schema()
        metadata = request_schema["properties"]["hypergraph"]
        result_schema = EdgeIntersectionsResult.model_json_schema()

        assert metadata["edge_pair_bound"] == MAX_EDGE_PAIR_COUNT
        assert metadata["intersection_cells_bound"] == MAX_EDGE_INTERSECTION_CELLS
        assert (
            metadata["canonical_result_bytes_bound"]
            == MAX_EDGE_INTERSECTION_RESULT_BYTES
        )
        assert (
            result_schema["properties"]["pair_intersections"]["maxItems"]
            == MAX_EDGE_PAIR_COUNT
        )
