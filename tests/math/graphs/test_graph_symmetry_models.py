from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
from jacobian.math.graphs.isomorphism import (
    ColoredUndirectedGraph,
    canonicalize_colored_graph,
)
from jacobian.math.graphs.symmetry._models import (
    GraphEdgeOrbit,
    GraphSymmetryOrbitRequest,
    GraphSymmetryOrbitResult,
    GraphVertexOrbit,
)
from jacobian.math.graphs.symmetry._operations import _generator_orbits
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _path_request() -> dict[str, object]:
    return {
        "graph": {
            "graph": {
                "vertices": ["a", "b", "c"],
                "edges": [["a", "b"], ["b", "c"]],
            },
            "vertex_colors": ["endpoint", "middle", "endpoint"],
        },
        "generators": [
            {
                "generator_id": "reflection",
                "mapping": [["a", "c"], ["b", "b"], ["c", "a"]],
            }
        ],
    }


def test_graph_symmetry_request_binds_total_color_preserving_generators() -> None:
    request = GraphSymmetryOrbitRequest.model_validate(_path_request())

    assert request.generators[0].mapping == (("a", "c"), ("b", "b"), ("c", "a"))
    assert request.graph.vertex_colors[1] == "middle"


def test_graph_symmetry_request_rejects_object_shaped_mapping() -> None:
    payload = _path_request()
    payload["generators"] = [
        {
            "generator_id": "reflection",
            "mapping": {"a": "c", "b": "b", "c": "a"},
        }
    ]

    with pytest.raises(ValidationError, match="mapping"):
        GraphSymmetryOrbitRequest.model_validate(payload)


def test_graph_symmetry_request_rejects_incomplete_permutation() -> None:
    payload = _path_request()
    del payload["generators"][0]["mapping"][2]  # type: ignore[index]

    with pytest.raises(ValidationError, match="total vertex permutation"):
        GraphSymmetryOrbitRequest.model_validate(payload)


def test_graph_symmetry_request_requires_declared_vertex_order_mapping() -> None:
    payload = _path_request()
    payload["generators"][0]["mapping"] = [  # type: ignore[index]
        ["b", "b"],
        ["a", "c"],
        ["c", "a"],
    ]

    with pytest.raises(ValidationError, match="total vertex permutation"):
        GraphSymmetryOrbitRequest.model_validate(payload)


def test_graph_symmetry_request_rejects_color_breaking_generator() -> None:
    payload = _path_request()
    payload["graph"]["vertex_colors"][2] = "distinguished"  # type: ignore[index]

    with pytest.raises(ValidationError, match="preserve declared vertex colors"):
        GraphSymmetryOrbitRequest.model_validate(payload)


def test_graph_symmetry_request_rejects_labels_outside_artifact_budget() -> None:
    payload = {
        "graph": {"graph": {"vertices": ["a" * 65], "edges": []}},
        "generators": [],
    }

    with pytest.raises(ValidationError, match="at most 64 UTF-8 bytes"):
        GraphSymmetryOrbitRequest.model_validate(payload)


def test_graph_symmetry_request_requires_nfc_generator_identifiers() -> None:
    payload = _path_request()
    payload["generators"][0]["generator_id"] = "refle\u0301ction"  # type: ignore[index]

    with pytest.raises(ValidationError, match="must use Unicode NFC"):
        GraphSymmetryOrbitRequest.model_validate(payload)


def test_graph_symmetry_schema_publishes_nfc_requirement() -> None:
    """math.find callers must see the NFC requirement before math.run rejects."""

    from jacobian.math.graphs.symmetry._models import GraphAutomorphismGenerator

    generator_schema = GraphAutomorphismGenerator.model_json_schema()
    request_schema = GraphSymmetryOrbitRequest.model_json_schema()

    assert (
        "NFC"
        in generator_schema["properties"]["generator_id"]["description"]
        == request_schema["$defs"]["GraphAutomorphismGenerator"]["properties"][
            "generator_id"
        ]["description"]
    )
    assert (
        "NFC"
        in request_schema["$defs"]["GraphAutomorphismGenerator"]["properties"][
            "mapping"
        ]["description"]
    )
    assert (
        "NFC"
        in request_schema["$defs"]["ColoredUndirectedGraph"]["properties"][
            "vertex_colors"
        ]["description"]
    )


def test_canonicalization_result_passes_unchanged_into_symmetry_request() -> None:
    source = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("a", "b", "c"),
            edges=(("a", "b"), ("b", "c")),
        ),
        vertex_colors=("endpoint", "middle", "endpoint"),
        edge_colors=("outer", "outer"),
    )
    canonical = canonicalize_colored_graph(source).canonical_graph

    request = GraphSymmetryOrbitRequest(
        graph=canonical,
        generators=(
            {
                "generator_id": "reflection",
                "mapping": (("v00", "v01"), ("v01", "v00"), ("v02", "v02")),
            },
        ),
    )

    assert request.graph is canonical
    assert request.action == "DECLARED_AUTOMORPHISM_GENERATORS"

    result = _generator_orbits(request)
    assert result.source is request
    assert tuple(orbit.members for orbit in result.vertex_orbits) == (
        ("v00", "v01"),
        ("v02",),
    )
    assert len(result.edge_orbits) == 1


def _reflection_path_source() -> GraphSymmetryOrbitRequest:
    return GraphSymmetryOrbitRequest.model_validate(_path_request())


def _reflection_path_result_payload() -> dict[str, object]:
    return {
        "source": _path_request(),
        "vertices": ["a", "b", "c"],
        "edges": [["a", "b"], ["b", "c"]],
        "generator_ids": ["reflection"],
        "generator_count": 1,
        "vertex_orbits": [
            {"orbit_index": 0, "representative": "a", "members": ["a", "c"]},
            {"orbit_index": 1, "representative": "b", "members": ["b"]},
        ],
        "edge_orbits": [
            {
                "orbit_index": 0,
                "representative": ["a", "b"],
                "members": [["a", "b"], ["b", "c"]],
            },
        ],
        "vertex_orbit_count": 2,
        "edge_orbit_count": 1,
        "vertex_color_mode": "DECLARED",
        "edge_color_mode": "UNCOLORED",
    }


def test_graph_symmetry_result_retains_declared_source_action() -> None:
    result = GraphSymmetryOrbitResult.model_validate(_reflection_path_result_payload())

    assert result.source.generators[0].mapping == (
        ("a", "c"),
        ("b", "b"),
        ("c", "a"),
    )
    assert result.source.graph.vertex_colors[1] == "middle"
    assert result.action == "DECLARED_GENERATED_SUBGROUP"
    assert (
        result.generator_validation
        == "ALL_DECLARED_GENERATORS_PRESERVE_GRAPH_AND_COLORS"
    )
    assert result.orbit_completeness == "COMPLETE_FOR_DECLARED_GENERATORS"
    assert (
        result.automorphism_group_completeness == "FULL_AUTOMORPHISM_GROUP_NOT_CLAIMED"
    )


def test_graph_symmetry_operation_produces_source_bound_result() -> None:
    request = _reflection_path_source()
    result = _generator_orbits(request)

    assert result.source is request
    assert tuple(orbit.members for orbit in result.vertex_orbits) == (
        ("a", "c"),
        ("b",),
    )
    assert tuple(orbit.members for orbit in result.edge_orbits) == (
        (("a", "b"), ("b", "c")),
    )


def test_graph_symmetry_retained_source_action_is_deeply_immutable() -> None:
    """A validated result must stay bound to its declared action forever."""

    request = _reflection_path_source()
    result = _generator_orbits(request)

    mapping = result.source.generators[0].mapping
    assert isinstance(mapping, tuple)
    assert all(
        isinstance(pair, tuple) and all(isinstance(item, str) for item in pair)
        for pair in mapping
    )
    with pytest.raises(ValidationError, match="frozen"):
        result.source.generators[0].mapping = ()  # type: ignore[assignment]


def test_graph_symmetry_den_num_identity_generator_canonicalizes() -> None:
    """Labels spelling the canonical rational keys must not collide."""

    payload = {
        "graph": {"graph": {"vertices": ["den", "num"], "edges": [["den", "num"]]}},
        "generators": [
            {
                "generator_id": "identity",
                "mapping": [["den", "den"], ["num", "num"]],
            }
        ],
    }
    request = GraphSymmetryOrbitRequest.model_validate(payload)
    result = _generator_orbits(request)

    encoded = canonicalize_json(result.model_dump(mode="json"))
    assert b"vertex_orbit_count" in encoded


def test_graph_symmetry_result_rejects_incomplete_orbit_partition() -> None:
    with pytest.raises(ValidationError, match="complete canonical vertex partition"):
        GraphSymmetryOrbitResult(
            source=GraphSymmetryOrbitRequest.model_validate(
                {
                    "graph": {"graph": {"vertices": ["a", "b"], "edges": [["a", "b"]]}},
                    "generators": [],
                }
            ),
            vertices=("a", "b"),
            edges=(("a", "b"),),
            generator_ids=(),
            generator_count=0,
            vertex_orbits=(
                GraphVertexOrbit(
                    orbit_index=0,
                    representative="a",
                    members=("a",),
                ),
            ),
            edge_orbits=(
                GraphEdgeOrbit(
                    orbit_index=0,
                    representative=("a", "b"),
                    members=(("a", "b"),),
                ),
            ),
            vertex_orbit_count=1,
            edge_orbit_count=1,
            vertex_color_mode="UNCOLORED",
            edge_color_mode="UNCOLORED",
        )


def test_graph_symmetry_result_rejects_singletons_contradicting_reflection() -> None:
    """Singleton orbits must not claim completeness for a swapping generator."""

    payload = _reflection_path_result_payload()
    payload["vertex_orbits"] = [
        {"orbit_index": index, "representative": vertex, "members": [vertex]}
        for index, vertex in enumerate(("a", "b", "c"))
    ]
    payload["vertex_orbit_count"] = 3

    with pytest.raises(
        ValidationError, match="exact orbits of the declared generators"
    ):
        GraphSymmetryOrbitResult.model_validate(payload)


def test_graph_symmetry_result_rejects_edge_split_contradicting_generators() -> None:
    payload = _reflection_path_result_payload()
    payload["edge_orbits"] = [
        {
            "orbit_index": 0,
            "representative": ["a", "b"],
            "members": [["a", "b"]],
        },
        {
            "orbit_index": 1,
            "representative": ["b", "c"],
            "members": [["b", "c"]],
        },
    ]
    payload["edge_orbit_count"] = 2

    with pytest.raises(
        ValidationError, match="exact orbits of the declared generators"
    ):
        GraphSymmetryOrbitResult.model_validate(payload)


def test_graph_symmetry_result_rejects_generator_ids_not_matching_source() -> None:
    payload = _reflection_path_result_payload()
    payload["generator_ids"] = []
    payload["generator_count"] = 0

    with pytest.raises(ValidationError, match="retained source action"):
        GraphSymmetryOrbitResult.model_validate(payload)


def test_graph_symmetry_result_rejects_vertices_not_matching_source() -> None:
    payload = _reflection_path_result_payload()
    payload["vertices"] = ["a", "b"]
    payload["edges"] = [["a", "b"]]
    payload["edge_orbits"] = [
        {
            "orbit_index": 0,
            "representative": ["a", "b"],
            "members": [["a", "b"]],
        },
    ]

    with pytest.raises(ValidationError, match="retained source action"):
        GraphSymmetryOrbitResult.model_validate(payload)


def test_graph_symmetry_result_rejects_color_modes_contradicting_source() -> None:
    payload = _reflection_path_result_payload()
    payload["vertex_color_mode"] = "UNCOLORED"

    with pytest.raises(ValidationError, match="retained source vertex colors"):
        GraphSymmetryOrbitResult.model_validate(payload)

    uncolored_payload = _reflection_path_result_payload()
    uncolored_payload["source"] = {
        "graph": {
            "graph": {"vertices": ["a", "b", "c"], "edges": [["a", "b"], ["b", "c"]]}
        },
        "generators": [
            {
                "generator_id": "reflection",
                "mapping": [["a", "c"], ["b", "b"], ["c", "a"]],
            }
        ],
    }
    uncolored_payload["vertex_color_mode"] = "DECLARED"

    with pytest.raises(ValidationError, match="retained source vertex colors"):
        GraphSymmetryOrbitResult.model_validate(uncolored_payload)
