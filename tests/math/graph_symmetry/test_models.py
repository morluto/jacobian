from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graph_symmetry._models import (
    GraphSymmetryOrbitRequest,
    GraphSymmetryOrbitResult,
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
