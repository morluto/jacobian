from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.canonical import (
    CanonicalLimits,
    canonicalize_json,
    encode_strict_json,
)
from jacobian.math.graphs.symmetry._models import (
    GraphSymmetryOrbitRequest,
    GraphSymmetryOrbitResult,
    _estimate_orbit_result_wire_bytes,
)


def _path_request() -> dict[str, object]:
    return {
        "graph": {
            "vertices": ["a", "b", "c"],
            "edges": [["a", "b"], ["b", "c"]],
        },
        "generators": [
            {
                "generator_id": "reflection",
                "mapping": {"a": "c", "b": "b", "c": "a"},
            }
        ],
        "vertex_colors": [
            {"vertex": "a", "color": "endpoint"},
            {"vertex": "b", "color": "middle"},
            {"vertex": "c", "color": "endpoint"},
        ],
    }


def test_graph_symmetry_request_binds_total_color_preserving_generators() -> None:
    request = GraphSymmetryOrbitRequest.model_validate(_path_request())

    assert request.generators[0].mapping["a"] == "c"
    assert request.vertex_colors[1].color == "middle"


def test_graph_symmetry_request_rejects_incomplete_permutation() -> None:
    payload = _path_request()
    payload["generators"][0]["mapping"].pop("c")  # type: ignore[index]

    with pytest.raises(ValidationError, match="total vertex permutation"):
        GraphSymmetryOrbitRequest.model_validate(payload)


def test_graph_symmetry_request_rejects_color_breaking_generator() -> None:
    payload = _path_request()
    payload["vertex_colors"][2]["color"] = "distinguished"  # type: ignore[index]

    with pytest.raises(ValidationError, match="preserve declared vertex colors"):
        GraphSymmetryOrbitRequest.model_validate(payload)


def test_graph_symmetry_request_rejects_labels_outside_artifact_budget() -> None:
    payload = {
        "graph": {"vertices": ["a" * 65], "edges": []},
        "generators": [],
    }

    with pytest.raises(ValidationError, match="1-64 characters"):
        GraphSymmetryOrbitRequest.model_validate(payload)


def _wide_orbit_payload(generators: int) -> dict[str, object]:
    """A 256-vertex, 4096-edge graph with maximal 64-character labels."""

    prefix = "\U0001d552" * 60
    vertices = [f"{prefix}{index:04d}" for index in range(256)]
    edges: list[list[str]] = []
    for left in range(256):
        for right in range(left + 1, 256):
            if len(edges) == 4096:
                break
            edges.append([vertices[left], vertices[right]])
        if len(edges) == 4096:
            break
    return {
        "graph": {"vertices": vertices, "edges": edges},
        "generators": [
            {
                "generator_id": f"{'g' * 62}{index:02d}",
                "mapping": {vertex: vertex for vertex in vertices},
            }
            for index in range(generators)
        ],
    }


def test_graph_symmetry_request_admission_bounds_retained_source_output() -> None:
    payload = _wide_orbit_payload(30)
    assert len(encode_strict_json(payload)) <= CanonicalLimits().max_input_bytes

    with pytest.raises(ValidationError, match="canonical output limit"):
        GraphSymmetryOrbitRequest.model_validate(payload)


def test_graph_symmetry_admitted_request_result_fits_canonical_output() -> None:
    from jacobian.math.graphs.symmetry._operations import _generator_orbits

    request = GraphSymmetryOrbitRequest.model_validate(_wide_orbit_payload(8))
    result = _generator_orbits(request)

    encoded = canonicalize_json(result.model_dump(mode="json"))
    assert len(encoded) <= CanonicalLimits().max_output_bytes
    assert len(encoded) <= _estimate_orbit_result_wire_bytes(request)


def test_graph_symmetry_admission_estimate_bounds_actual_result_wire() -> None:
    from jacobian.math.graphs.symmetry._operations import _generator_orbits

    request = GraphSymmetryOrbitRequest.model_validate(_path_request())
    result = _generator_orbits(request)

    actual = len(canonicalize_json(result.model_dump(mode="json")))
    assert actual > 0
    assert _estimate_orbit_result_wire_bytes(request) >= actual


def test_graph_symmetry_result_rejects_incomplete_orbit_partition() -> None:
    with pytest.raises(ValidationError, match="complete canonical vertex partition"):
        GraphSymmetryOrbitResult(
            source=GraphSymmetryOrbitRequest.model_validate(
                {
                    "graph": {
                        "vertices": ["a", "b"],
                        "edges": [["a", "b"]],
                    },
                    "generators": [],
                }
            ),
            vertices=("a", "b"),
            edges=(("a", "b"),),
            generator_ids=(),
            generator_count=0,
            vertex_orbits=(
                {
                    "orbit_index": 0,
                    "representative": "a",
                    "members": ["a"],
                },
            ),
            edge_orbits=(
                {
                    "orbit_index": 0,
                    "representative": ["a", "b"],
                    "members": [["a", "b"]],
                },
            ),
            vertex_orbit_count=1,
            edge_orbit_count=1,
            vertex_color_mode="UNCOLORED",
            edge_color_mode="UNCOLORED",
        )


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

    assert result.source.generators[0].mapping == {"a": "c", "b": "b", "c": "a"}
    assert result.source.vertex_colors[1].color == "middle"
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
    from jacobian.math.graphs.symmetry._operations import _generator_orbits

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


def test_graph_symmetry_result_rejects_singletons_contradicting_reflection() -> None:
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
        "graph": {"vertices": ["a", "b", "c"], "edges": [["a", "b"], ["b", "c"]]},
        "generators": [
            {
                "generator_id": "reflection",
                "mapping": {"a": "c", "b": "b", "c": "a"},
            }
        ],
    }
    uncolored_payload["vertex_color_mode"] = "DECLARED"

    with pytest.raises(ValidationError, match="retained source vertex colors"):
        GraphSymmetryOrbitResult.model_validate(uncolored_payload)
