from jacobian.math.topology.release import (
    GraphCliqueRequest,
    OneSkeletonRequest,
    OneSkeletonResult,
    graph_clique_complex,
    one_skeleton,
)
from jacobian.math.topology.release_tools import TOOLS


def test_one_skeleton_matches_independent_face_oracle_and_retains_isolates() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "topology.simplicial_complex.one_skeleton.compute"
    )
    result = tool.run(
        OneSkeletonRequest(
            complex={
                "vertices": ["d", "c", "b", "a"],
                "facets": [["a", "b"], ["b", "c"], ["d"]],
            }
        )
    )

    # Independent definition: an edge is exactly a two-element face contained
    # in at least one maximal simplex; isolated vertices remain graph vertices.
    source_facets = ({"a", "b"}, {"b", "c"}, {"d"})
    expected_label_edges = {
        tuple(sorted(pair))
        for facet in source_facets
        for pair in ((left, right) for left in facet for right in facet if left < right)
    }
    assert result.source.vertices == ("a", "b", "c", "d")
    assert result.vertex_labels == result.source.vertices
    assert result.graph.vertex_count == 4
    assert {
        tuple(result.vertex_labels[index] for index in edge)
        for edge in result.graph.edges
    } == expected_label_edges
    assert result.edge_faces == tuple(
        tuple(result.vertex_labels[index] for index in edge)
        for edge in result.graph.edges
    )
    assert OneSkeletonResult.model_validate_json(result.model_dump_json()) == result


def test_one_skeleton_graph_value_composes_unchanged_with_graph_clique() -> None:
    source = {
        "vertices": ["a", "b", "c", "d"],
        "facets": [["a", "b"], ["b", "c"], ["c", "d"], ["a", "d"]],
    }
    projected = one_skeleton(OneSkeletonRequest(complex=source))
    reconstructed = graph_clique_complex(GraphCliqueRequest(graph=projected.graph))

    assert reconstructed.clique_complex.maximal_simplices == (
        ("v0", "v1"),
        ("v0", "v3"),
        ("v1", "v2"),
        ("v2", "v3"),
    )
    assert reconstructed.graph_edges == (
        ("v0", "v1"),
        ("v0", "v3"),
        ("v1", "v2"),
        ("v2", "v3"),
    )


def test_decoded_one_skeleton_rejects_forged_provenance() -> None:
    result = one_skeleton(
        OneSkeletonRequest(complex={"vertices": ["a", "b"], "facets": [["a", "b"]]})
    )

    payload = result.model_dump(mode="json")
    payload["graph"]["edges"] = []
    payload["edge_faces"] = []

    try:
        OneSkeletonResult.model_validate(payload)
    except ValueError as error:
        assert "graph edges must correspond exactly to source 1-faces" in str(error)
    else:
        raise AssertionError("forged one-skeleton provenance was accepted")
