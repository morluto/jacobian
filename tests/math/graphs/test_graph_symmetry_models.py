from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.isomorphism import (
    ColoredUndirectedGraph,
    canonicalize_colored_graph,
)
from jacobian.math.graphs.symmetry._models import (
    GraphSymmetryOrbitRequest,
    GraphSymmetryOrbitResult,
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
                "mapping": {"a": "c", "b": "b", "c": "a"},
            }
        ],
    }


def test_graph_symmetry_request_binds_total_color_preserving_generators() -> None:
    request = GraphSymmetryOrbitRequest.model_validate(_path_request())

    assert request.generators[0].mapping["a"] == "c"
    assert request.graph.vertex_colors[1] == "middle"


def test_graph_symmetry_request_rejects_incomplete_permutation() -> None:
    payload = _path_request()
    payload["generators"][0]["mapping"].pop("c")  # type: ignore[index]

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


def test_graph_symmetry_request_rejects_non_nfc_color_names() -> None:
    payload = _path_request()
    payload["graph"]["vertex_colors"] = ["endpo\u0069\u0301nt", "middle", "endpoint"]  # type: ignore[index]

    with pytest.raises(ValidationError, match="Unicode NFC"):
        GraphSymmetryOrbitRequest.model_validate(payload)


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
                "mapping": {"v00": "v01", "v01": "v00", "v02": "v02"},
            },
        ),
    )

    assert request.graph is canonical
    assert request.action == "DECLARED_AUTOMORPHISM_GENERATORS"

    result = _generator_orbits(request)
    assert tuple(orbit.members for orbit in result.vertex_orbits) == (
        ("v00", "v01"),
        ("v02",),
    )
    assert len(result.edge_orbits) == 1


def test_graph_symmetry_result_rejects_incomplete_orbit_partition() -> None:
    with pytest.raises(ValidationError, match="complete canonical vertex partition"):
        GraphSymmetryOrbitResult(
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
