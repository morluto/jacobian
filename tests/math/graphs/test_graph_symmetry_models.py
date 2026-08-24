from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.canonical import (
    CanonicalLimits,
    canonicalize_json,
    encode_strict_json,
)
from jacobian.math.graphs.isomorphism import (
    ColoredUndirectedGraph,
    canonicalize_colored_graph,
)
from jacobian.math.graphs.symmetry._models import (
    MAX_GRAPH_SYMMETRY_GENERATORS,
    GraphEdgeOrbit,
    GraphSymmetryOrbitRequest,
    GraphSymmetryOrbitResult,
    GraphVertexOrbit,
    _orbit_result_canonical_wire_bytes,
)
from jacobian.math.graphs.symmetry._operations import _generator_orbits
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _path_request() -> dict[str, Any]:
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
    del payload["generators"][0]["mapping"][2]

    with pytest.raises(ValidationError, match="total vertex permutation"):
        GraphSymmetryOrbitRequest.model_validate(payload)


def test_graph_symmetry_request_requires_declared_vertex_order_mapping() -> None:
    payload = _path_request()
    payload["generators"][0]["mapping"] = [["b", "b"], ["a", "c"], ["c", "a"]]

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


def _wide_orbit_payload(generators: int) -> dict[str, object]:
    """A 256-vertex, 4096-edge graph with maximal 64-byte labels."""

    prefix = "a" * 60
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
        "graph": {"graph": {"vertices": vertices, "edges": edges}},
        "generators": [
            {
                "generator_id": f"{'g' * 62}{index:02d}",
                "mapping": [[vertex, vertex] for vertex in vertices],
            }
            for index in range(generators)
        ],
    }


def test_graph_symmetry_request_admission_bounds_retained_source_output() -> None:
    """The maximal admissible envelope keeps its retained source inside budget.

    The canonical colored-graph value caps every retained string at 64
    UTF-8 bytes and the operation admits at most 64 generators, so even the
    maximal request's echoed source plus priced orbit result stays inside
    the canonical output limit while the wire-size function equals the
    actual canonicalized result size.
    """

    from jacobian.math.graphs.symmetry._operations import _generator_orbits

    payload = _wide_orbit_payload(MAX_GRAPH_SYMMETRY_GENERATORS)
    assert len(encode_strict_json(payload)) <= CanonicalLimits().max_input_bytes

    request = GraphSymmetryOrbitRequest.model_validate(payload)
    result = _generator_orbits(request)
    encoded = canonicalize_json(result.model_dump(mode="json"))
    assert _orbit_result_canonical_wire_bytes(request) == len(encoded)
    assert (
        _orbit_result_canonical_wire_bytes(request)
        <= CanonicalLimits().max_output_bytes
    )


def test_graph_symmetry_admitted_request_result_fits_canonical_output() -> None:
    from jacobian.math.graphs.symmetry._operations import _generator_orbits

    request = GraphSymmetryOrbitRequest.model_validate(_wide_orbit_payload(8))
    result = _generator_orbits(request)

    encoded = canonicalize_json(result.model_dump(mode="json"))
    assert len(encoded) <= CanonicalLimits().max_output_bytes
    assert len(encoded) == _orbit_result_canonical_wire_bytes(request)


def test_graph_symmetry_admission_estimate_bounds_actual_result_wire() -> None:
    from jacobian.math.graphs.symmetry._operations import _generator_orbits

    request = GraphSymmetryOrbitRequest.model_validate(_path_request())
    result = _generator_orbits(request)

    actual = len(canonicalize_json(result.model_dump(mode="json")))
    assert actual > 0
    assert _orbit_result_canonical_wire_bytes(request) == actual


def test_graph_symmetry_request_admits_result_near_output_limit() -> None:
    """A maximal graph whose exact representative totals fit must be admitted.

    Charging every orbit representative with the largest label or edge pair
    rejected this request even though its canonical result fits the output
    limit, so the boundary case pins the sums-based bound.
    """

    from jacobian.math.graphs.symmetry._operations import _generator_orbits

    request = GraphSymmetryOrbitRequest.model_validate(_wide_orbit_payload(14))
    result = _generator_orbits(request)

    encoded = canonicalize_json(result.model_dump(mode="json"))
    assert len(encoded) <= CanonicalLimits().max_output_bytes
    assert len(encoded) == _orbit_result_canonical_wire_bytes(request)


def test_transitive_action_charges_only_possible_representatives() -> None:
    """A transitive rotation proves most elements cannot be representatives.

    A 256-vertex circulant graph with edges at distances 1-16 and one
    rotation generator plus fourteen identity generators has exactly one
    vertex orbit and sixteen edge orbits, so charging every declared vertex
    and edge as a representative inflated the estimate above the canonical
    output limit even though the exact result keeps over 2 MiB of headroom.
    """

    from jacobian.math.graphs.symmetry._operations import _generator_orbits

    prefix = "a" * 60
    vertices = [f"{prefix}{index:04d}" for index in range(256)]
    edges: list[list[str]] = []
    for distance in range(1, 17):
        for index in range(256):
            left, right = (
                vertices[index],
                vertices[(index + distance) % 256],
            )
            edge = [left, right] if left < right else [right, left]
            if edge not in edges:
                edges.append(edge)
        if len(edges) > 4096:
            break
    edges = sorted(edges)[:4096]
    payload = {
        "graph": {"graph": {"vertices": vertices, "edges": edges}},
        "generators": [
            {
                "generator_id": "rotation",
                "mapping": [
                    [vertex, vertices[(index + 1) % 256]]
                    for index, vertex in enumerate(vertices)
                ],
            },
            *(
                {
                    "generator_id": f"{'i' * 62}{index:02d}",
                    "mapping": [[vertex, vertex] for vertex in vertices],
                }
                for index in range(14)
            ),
        ],
    }
    request = GraphSymmetryOrbitRequest.model_validate(payload)
    result = _generator_orbits(request)

    assert result.vertex_orbit_count == 1
    assert result.edge_orbit_count == 16
    encoded = canonicalize_json(result.model_dump(mode="json"))
    estimate = _orbit_result_canonical_wire_bytes(request)
    limit = CanonicalLimits().max_output_bytes
    assert estimate == len(encoded) <= limit
    assert limit - len(encoded) > 2 * 1024 * 1024


def test_colored_singleton_orbits_price_exact_fixed_structure() -> None:
    """One-character colors on every maximal vertex must stay admitted.

    Charging a flat 64-byte structure per singleton orbit priced 256 vertex
    plus 4,096 edge singleton orbits far above their exact wire structure,
    inflating the estimate past the canonical output limit even though the
    exact result keeps over 40 KiB of headroom, so this boundary case pins
    per-orbit pricing at each index's digit width and separators.
    """

    from jacobian.math.graphs.symmetry._operations import _generator_orbits

    payload = _wide_orbit_payload(14)
    payload["graph"]["vertex_colors"] = [  # type: ignore[index]
        "c"
        for vertex in payload["graph"]["graph"]["vertices"]  # type: ignore[index]
    ]
    request = GraphSymmetryOrbitRequest.model_validate(payload)
    result = _generator_orbits(request)

    assert result.vertex_orbit_count == 256
    assert result.edge_orbit_count == 4096
    encoded = canonicalize_json(result.model_dump(mode="json"))
    estimate = _orbit_result_canonical_wire_bytes(request)
    limit = CanonicalLimits().max_output_bytes
    assert estimate == len(encoded) <= limit
    assert limit - len(encoded) >= 40_000


def test_singleton_orbit_separators_price_from_computed_blocks() -> None:
    """Edge colors on the maximal singleton payload must stay admitted.

    Charging two member separators per declared vertex and edge priced one
    nonexistent comma into every singleton orbit block, so adding a
    six-character edge color to every edge pushed this fitting canonical
    result past the output limit; the separator term must come from the
    computed partition's block sizes instead.
    """

    from jacobian.math.graphs.symmetry._operations import _generator_orbits

    payload = _wide_orbit_payload(14)
    payload["graph"]["vertex_colors"] = [  # type: ignore[index]
        "c"
        for vertex in payload["graph"]["graph"]["vertices"]  # type: ignore[index]
    ]
    payload["graph"]["edge_colors"] = [  # type: ignore[index]
        "colors"
        for edge in payload["graph"]["graph"]["edges"]  # type: ignore[index]
    ]
    request = GraphSymmetryOrbitRequest.model_validate(payload)
    result = _generator_orbits(request)

    assert result.vertex_orbit_count == 256
    assert result.edge_orbit_count == 4096
    encoded = canonicalize_json(result.model_dump(mode="json"))
    estimate = _orbit_result_canonical_wire_bytes(request)
    limit = CanonicalLimits().max_output_bytes
    assert estimate == len(encoded) <= limit


def test_graph_symmetry_estimate_bounds_heterogeneous_representatives() -> None:
    """One large label among tiny ones must not inflate the estimate by a maximum."""

    from jacobian.math.graphs.symmetry._operations import _generator_orbits

    vertices = ["h" * 64] + [f"t{index:03d}" for index in range(255)]
    ring = vertices[1:]
    successor = dict(zip(ring, [*ring[1:], ring[0]], strict=True))
    edges = sorted([a, b] if a < b else [b, a] for a, b in successor.items())
    payload = {
        "graph": {"graph": {"vertices": vertices, "edges": edges}},
        "generators": [
            {
                "generator_id": "rotation",
                "mapping": [
                    [vertex, successor.get(vertex, vertex)] for vertex in vertices
                ],
            }
        ],
    }
    request = GraphSymmetryOrbitRequest.model_validate(payload)
    result = _generator_orbits(request)

    actual = len(canonicalize_json(result.model_dump(mode="json")))
    estimate = _orbit_result_canonical_wire_bytes(request)
    assert actual == estimate <= CanonicalLimits().max_output_bytes


def test_graph_symmetry_request_requires_nfc_generator_identifiers() -> None:
    payload = _path_request()
    payload["generators"][0]["generator_id"] = "refle\u0301ction"

    with pytest.raises(ValidationError, match="must use Unicode NFC"):
        GraphSymmetryOrbitRequest.model_validate(payload)


def test_graph_symmetry_request_requires_nfc_vertex_colors() -> None:
    payload = _path_request()
    payload["graph"]["vertex_colors"][0] = "\u0344endpoint"  # type: ignore[index]

    with pytest.raises(ValidationError, match="must use Unicode NFC"):
        GraphSymmetryOrbitRequest.model_validate(payload)


def test_graph_symmetry_request_requires_nfc_edge_colors() -> None:
    payload = _path_request()
    payload["graph"]["edge_colors"] = ["\u0344" * 64, "\u0344" * 64]  # type: ignore[index]

    with pytest.raises(ValidationError, match="must use Unicode NFC"):
        GraphSymmetryOrbitRequest.model_validate(payload)


def test_graph_symmetry_schema_publishes_nfc_requirement() -> None:
    """math.find callers must see the NFC requirement before math.run rejects."""

    from jacobian.math.graphs.symmetry._models import (
        GraphAutomorphismGenerator,
        GraphSymmetryOrbitRequest,
    )

    generator_schema = GraphAutomorphismGenerator.model_json_schema()
    request_schema = GraphSymmetryOrbitRequest.model_json_schema()

    assert (
        "NFC"
        in request_schema["$defs"]["GraphAutomorphismGenerator"]["properties"][
            "generator_id"
        ]["description"]
    )
    assert (
        "NFC"
        in generator_schema["properties"]["generator_id"]["description"]
        == request_schema["$defs"]["GraphAutomorphismGenerator"]["properties"][
            "generator_id"
        ]["description"]
    )
    assert (
        "NFC"
        in request_schema["$defs"]["ColoredUndirectedGraph"]["properties"][
            "vertex_colors"
        ]["description"]
    )
    assert (
        "NFC"
        in request_schema["$defs"]["ColoredUndirectedGraph"]["properties"][
            "edge_colors"
        ]["description"]
    )


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
                "mapping": (("v00", "v01"), ("v01", "v00"), ("v02", "v02")),
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


def test_graph_symmetry_nfc_request_wire_matches_canonicalized_result() -> None:
    """NFC-retained strings make the strict measurements canonical-exact."""

    from jacobian.math.graphs.symmetry._operations import _generator_orbits

    request = GraphSymmetryOrbitRequest.model_validate(_path_request())
    result = _generator_orbits(request)

    dumped = result.model_dump(mode="json")
    assert len(encode_strict_json(dumped)) == len(canonicalize_json(dumped))
    assert _orbit_result_canonical_wire_bytes(request) == len(canonicalize_json(dumped))


def test_graph_symmetry_result_rejects_incomplete_orbit_partition() -> None:
    with pytest.raises(ValidationError, match="complete canonical vertex partition"):
        GraphSymmetryOrbitResult(
            source=GraphSymmetryOrbitRequest.model_validate(
                {
                    "graph": {
                        "graph": {
                            "vertices": ["a", "b"],
                            "edges": [["a", "b"]],
                        },
                    },
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


def test_graph_symmetry_retained_source_action_is_deeply_immutable() -> None:
    """A validated result must stay bound to its declared action forever."""

    from jacobian.math.graphs.symmetry._operations import _generator_orbits

    request = _reflection_path_source()
    result = _generator_orbits(request)

    mapping = result.source.generators[0].mapping
    assert isinstance(mapping, tuple)
    assert all(
        isinstance(pair, tuple) and all(isinstance(item, str) for item in pair)
        for pair in mapping
    )
    with pytest.raises(ValidationError, match="frozen"):
        result.source.generators[0].mapping = ()


def test_graph_symmetry_den_num_identity_generator_canonicalizes() -> None:
    """Labels spelling the canonical rational keys must not collide."""

    from jacobian.math.graphs.symmetry._operations import _generator_orbits

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
        "graph": {
            "graph": {"vertices": ["a", "b", "c"], "edges": [["a", "b"], ["b", "c"]]},
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


def test_orbit_result_wire_size_is_exact_across_partition_shapes() -> None:
    """The wire-size function equals the canonicalized result byte for byte.

    Admission prices every fixed and variable component of the retained
    result exactly - no reserve, no per-orbit padding - so this invariant
    must hold across singleton partitions, transitive actions, rational-key
    labels, colored axes, and the empty graph alike.
    """

    from jacobian.math.graphs.symmetry._operations import _generator_orbits

    def den_num() -> dict[str, Any]:
        return {
            "graph": {"graph": {"vertices": ["den", "num"], "edges": [["den", "num"]]}},
            "generators": [
                {
                    "generator_id": "identity",
                    "mapping": [["den", "den"], ["num", "num"]],
                }
            ],
        }

    def empty() -> dict[str, Any]:
        return {"graph": {"graph": {"vertices": [], "edges": []}}, "generators": []}

    def mixed_orbits() -> dict[str, Any]:
        payload = _path_request()
        payload["generators"] = [
            payload["generators"][0],
            {
                "generator_id": "identity",
                "mapping": [["a", "a"], ["b", "b"], ["c", "c"]],
            },
        ]
        return payload

    def colored_singletons() -> dict[str, Any]:
        payload = _wide_orbit_payload(14)
        payload["graph"]["vertex_colors"] = [  # type: ignore[index]
            "c"
            for vertex in payload["graph"]["graph"]["vertices"]  # type: ignore[index]
        ]
        payload["graph"]["edge_colors"] = [  # type: ignore[index]
            "colors"
            for edge in payload["graph"]["graph"]["edges"]  # type: ignore[index]
        ]
        return payload

    payloads = [
        ("path_reflection", _path_request()),
        ("den_num_identity", den_num()),
        ("empty_graph", empty()),
        ("reflection_plus_identity", mixed_orbits()),
        ("maximal_generators", _wide_orbit_payload(MAX_GRAPH_SYMMETRY_GENERATORS)),
        ("colored_singletons", colored_singletons()),
    ]
    for name, payload in payloads:
        request = GraphSymmetryOrbitRequest.model_validate(payload)
        result = _generator_orbits(request)
        encoded = canonicalize_json(result.model_dump(mode="json"))
        assert _orbit_result_canonical_wire_bytes(request) == len(encoded), name
        assert len(encoded) <= CanonicalLimits().max_output_bytes


def test_graph_symmetry_admits_fifteen_character_color_boundary() -> None:
    """The stacked reviewer shape must be priced without reserve padding.

    The reported boundary replaces one-character vertex colors with valid
    15-character NFC colors on the maximal singleton payload with six-
    character edge colors; a flat envelope reserve on top of otherwise
    exact pricing rejected such fitting requests.
    """

    from jacobian.math.graphs.symmetry._operations import _generator_orbits

    payload = _wide_orbit_payload(14)
    payload["graph"]["vertex_colors"] = [  # type: ignore[index]
        "c" * 15
        for vertex in payload["graph"]["graph"]["vertices"]  # type: ignore[index]
    ]
    payload["graph"]["edge_colors"] = [  # type: ignore[index]
        "colors"
        for edge in payload["graph"]["graph"]["edges"]  # type: ignore[index]
    ]
    request = GraphSymmetryOrbitRequest.model_validate(payload)
    result = _generator_orbits(request)

    assert result.vertex_orbit_count == 256
    assert result.edge_orbit_count == 4096
    encoded = canonicalize_json(result.model_dump(mode="json"))
    assert _orbit_result_canonical_wire_bytes(request) == len(encoded)
    assert len(encoded) <= CanonicalLimits().max_output_bytes


def test_graph_symmetry_admission_flips_exactly_at_the_output_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Headroom admission must reject precisely when the true result overflows.

    A flat reserve rejected requests whose exact result still fit the
    transport envelope. With pricing exact, admission accepts when the
    configured limit equals the complete canonical serialization and
    rejects one byte below it.
    """

    import jacobian.math.graphs.symmetry._models as symmetry_models

    request = GraphSymmetryOrbitRequest.model_validate(_wide_orbit_payload(14))
    actual = _orbit_result_canonical_wire_bytes(request)

    monkeypatch.setattr(
        symmetry_models,
        "CanonicalLimits",
        lambda **kwargs: CanonicalLimits(max_output_bytes=actual),
    )
    assert GraphSymmetryOrbitRequest.model_validate(_wide_orbit_payload(14))

    monkeypatch.setattr(
        symmetry_models,
        "CanonicalLimits",
        lambda **kwargs: CanonicalLimits(max_output_bytes=actual - 1),
    )
    with pytest.raises(ValidationError, match="canonical output limit"):
        GraphSymmetryOrbitRequest.model_validate(_wide_orbit_payload(14))


def test_graph_symmetry_operation_declares_version_seven() -> None:
    """The source-bound wire contract is not compatible with advertised v6.

    v6 clients saw an object-shaped generator mapping and no required
    ``source`` field on results; this contract requires ordered pairs and
    binds every result to its declared source, so the declaration must
    advertise a new version.
    """

    from jacobian.catalog.models import MathTool
    from jacobian.math.graphs.symmetry._operations import GRAPH_SYMMETRY_OPERATIONS

    (declaration,) = GRAPH_SYMMETRY_OPERATIONS
    assert isinstance(declaration, MathTool)
    assert declaration.version == "7"


def test_graph_symmetry_schema_publishes_aggregate_output_envelope() -> None:
    """math.find readers must see the aggregate retained-result bound."""

    from jacobian.math.graphs.symmetry._operations import GRAPH_SYMMETRY_OPERATIONS

    schema = GraphSymmetryOrbitRequest.model_json_schema()
    assert "canonical output limit" in schema["description"]
    assert "retains" in schema["description"]

    (declaration,) = GRAPH_SYMMETRY_OPERATIONS
    assert "canonical" in declaration.description
    assert "output limit" in declaration.description
