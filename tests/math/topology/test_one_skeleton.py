from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
from jacobian.math.topology import canonicalize
from jacobian.math.topology.release import (
    OneSkeletonRequest,
    OneSkeletonResult,
    SimplicialComplexRequest,
    graph_clique_complex,
    one_skeleton,
)


def test_one_skeleton_matches_independent_face_oracle_and_retains_isolates() -> None:
    result = one_skeleton(
        OneSkeletonRequest.model_validate(
            {
                "complex": {
                    "vertices": ["d", "c", "b", "a"],
                    "facets": [["a", "b"], ["b", "c"], ["d"]],
                }
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


def test_one_skeleton_accepts_public_canonical_complex_value() -> None:
    source = canonicalize(("b", "a"), (("a", "b"),)).complex
    result = one_skeleton(source)
    assert result.source == source
    assert result.edge_faces == (("a", "b"),)


def test_one_skeleton_revalidates_forged_nested_request() -> None:
    forged_complex = SimplicialComplexRequest.model_construct(
        vertices=None, facets=()
    )
    forged_request = OneSkeletonRequest.model_construct(complex=forged_complex)

    try:
        one_skeleton(forged_request)
    except Exception as error:
        from jacobian.catalog.models import OperationDomainValidationError

        assert isinstance(error, OperationDomainValidationError)
        assert error.errors()[0]["type"] == "topology.one_skeleton.invalid_complex"
    else:
        raise AssertionError("a forged nested request was accepted")


def test_one_skeleton_graph_value_composes_unchanged_with_graph_clique() -> None:
    source = {
        "vertices": ["a", "b", "c", "d"],
        "facets": [["a", "b"], ["b", "c"], ["c", "d"], ["a", "d"]],
    }
    projected = one_skeleton(OneSkeletonRequest.model_validate({"complex": source}))
    reconstructed = graph_clique_complex(projected.graph)

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
        OneSkeletonRequest.model_validate(
            {"complex": {"vertices": ["a", "b"], "facets": [["a", "b"]]}}
        )
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


def test_decoded_one_skeleton_rejects_forged_source_face_axis() -> None:
    result = one_skeleton(
        OneSkeletonRequest.model_validate(
            {"complex": {"vertices": ["a", "b"], "facets": [["a", "b"]]}}
        )
    )
    payload = result.model_dump(mode="json")
    payload["source"]["maximal_simplices"] = [["a"], ["b"]]

    try:
        OneSkeletonResult.model_validate(payload)
    except ValueError as error:
        assert "source 1-face axis must match its maximal facets" in str(error)
    else:
        raise AssertionError("a forged source face axis was accepted")


def test_one_skeleton_provenance_check_rejects_oversized_facets_before_pairs() -> None:
    labels = tuple("abcdefghi")
    result = one_skeleton(
        OneSkeletonRequest.model_validate(
            {"complex": {"vertices": labels[:-1], "facets": [labels[:-1]]}}
        )
    )
    forged_source = result.source.model_copy(
        update={"vertices": labels, "maximal_simplices": (labels,)}
    )
    forged_result = OneSkeletonResult.model_construct(
        source=forged_source,
        graph=IndexedSimpleUndirectedGraph(vertex_count=len(labels), edges=()),
        vertex_labels=labels,
        edge_faces=(),
    )

    validator_name = "require_source_axes"
    validator = getattr(forged_result, validator_name)
    try:
        validator()
    except ValueError as error:
        assert "source facets exceed the admitted shape bounds" in str(error)
    else:
        raise AssertionError("an oversized source facet was accepted")


def test_one_skeleton_provenance_check_caps_forged_facet_count_before_pairs() -> None:
    forged_source = canonicalize(
        tuple(f"v{i}" for i in range(20)),
        tuple((f"v{i}",) for i in range(20)),
    ).complex.model_copy(
        update={
            "maximal_simplices": tuple(
                (f"v{i}", f"v{j}")
                for i in range(20)
                for j in range(i + 1, 20)
            )[:129]
        }
    )
    forged_result = OneSkeletonResult.model_construct(
        source=forged_source,
        graph=IndexedSimpleUndirectedGraph(vertex_count=20, edges=()),
        vertex_labels=tuple(sorted(f"v{i}" for i in range(20))),
        edge_faces=(),
    )

    validator_name = "require_source_axes"
    validator = getattr(forged_result, validator_name)
    try:
        validator()
    except ValueError as error:
        assert "source facets exceed the admitted shape bounds" in str(error)
    else:
        raise AssertionError("an oversized forged facet tuple was accepted")
