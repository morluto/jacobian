"""Exact contracts for color-preserving graph canonicalization."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from collections.abc import Iterable

import networkx as nx
import pytest
from pydantic import ValidationError
from pydantic_core import PydanticCustomError

import jacobian.math.graphs.isomorphism._canonicalization as isomorphism_canonicalization
import jacobian.math.graphs.isomorphism._canonicalization_bounds as isomorphism_bounds
from jacobian.math.graphs import explicit_graph
from jacobian.math.graphs.isomorphism import (
    ColoredGraphCanonicalizationResult,
    ColoredUndirectedGraph,
    canonicalize_colored_graph,
)
from jacobian.math.graphs.isomorphism._models import ColoredGraphCanonicalizationRequest
from jacobian.math.graphs.isomorphism._operations import (
    compute_colored_graph_canonicalization,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(
    vertices: Iterable[str],
    edges: Iterable[tuple[str, str]],
    *,
    vertex_colors: tuple[str, ...] = (),
    edge_colors: tuple[str, ...] = (),
) -> ColoredUndirectedGraph:
    return ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=tuple(vertices), edges=tuple(edges)),
        vertex_colors=vertex_colors,
        edge_colors=edge_colors,
    )


def _canonicalize(graph: ColoredUndirectedGraph) -> ColoredGraphCanonicalizationResult:
    return canonicalize_colored_graph(graph)


def _independent_relabel(
    graph: ColoredUndirectedGraph,
    mapping: dict[str, str],
) -> ColoredUndirectedGraph:
    vertex_colors = (
        {
            mapping[vertex]: graph.vertex_colors[index]
            for index, vertex in enumerate(graph.graph.vertices)
        }
        if graph.vertex_colors
        else {}
    )
    transformed_edges = []
    for edge_index, (left, right) in enumerate(graph.graph.edges):
        mapped = tuple(sorted((mapping[left], mapping[right])))
        transformed_edges.append(
            (
                mapped,
                graph.edge_colors[edge_index] if graph.edge_colors else None,
            )
        )
    transformed_edges.sort(key=lambda item: item[0])
    vertices = tuple(sorted(mapping.values()))
    return _graph(
        vertices,
        (edge for edge, _color in transformed_edges),
        vertex_colors=(
            tuple(vertex_colors[vertex] for vertex in vertices)
            if graph.vertex_colors
            else ()
        ),
        edge_colors=(
            tuple(color for _edge, color in transformed_edges if color is not None)
            if graph.edge_colors
            else ()
        ),
    )


def test_isomorphic_colored_graphs_have_identical_canonical_values() -> None:
    source = _graph(
        ("a", "b", "c", "d"),
        (("a", "b"), ("b", "c"), ("c", "d")),
        vertex_colors=("endpoint", "middle", "middle", "endpoint"),
        edge_colors=("outer", "middle", "outer"),
    )
    relabeling = {"a": "w", "b": "z", "c": "x", "d": "y"}
    relabeled = _independent_relabel(source, relabeling)

    first = _canonicalize(source)
    second = _canonicalize(relabeled)

    assert first.canonical_graph == second.canonical_graph
    first_mapping = {
        item.source_vertex: item.canonical_vertex for item in first.relabeling
    }
    assert _independent_relabel(source, first_mapping) == first.canonical_graph


def test_vertex_and_edge_colors_are_part_of_the_equivalence_relation() -> None:
    path_edges = (("a", "b"), ("b", "c"), ("c", "d"))
    red_end_edge = _graph(
        ("a", "b", "c", "d"),
        path_edges,
        edge_colors=("red", "plain", "plain"),
    )
    red_middle_edge = _graph(
        ("a", "b", "c", "d"),
        path_edges,
        edge_colors=("plain", "red", "plain"),
    )
    red_endpoint = _graph(
        ("a", "b", "c", "d"),
        path_edges,
        vertex_colors=("red", "plain", "plain", "plain"),
    )
    red_internal = _graph(
        ("a", "b", "c", "d"),
        path_edges,
        vertex_colors=("plain", "red", "plain", "plain"),
    )

    assert _canonicalize(red_end_edge).canonical_graph != (
        _canonicalize(red_middle_edge).canonical_graph
    )
    assert _canonicalize(red_endpoint).canonical_graph != (
        _canonicalize(red_internal).canonical_graph
    )


def test_empty_graph_retains_its_uncolored_ambient_value() -> None:
    result = _canonicalize(_graph((), ()))

    assert result.canonical_graph == _graph((), ())
    assert result.relabeling == ()


def test_existing_graph_value_composes_and_canonical_output_is_idempotent() -> None:
    base_graph = explicit_graph(
        ("c", "a", "b"),
        (("c", "b"), ("b", "a")),
    )
    colored_graph = ColoredUndirectedGraph(graph=base_graph)

    first = canonicalize_colored_graph(colored_graph)
    second = canonicalize_colored_graph(first.canonical_graph)

    assert first.source_graph.graph == base_graph
    assert second.canonical_graph == first.canonical_graph


def test_bicentral_unequal_half_tree_is_relabeling_invariant() -> None:
    # The central edge a-b has height-two halves of sizes three and four.
    # This is the size-tie-break shape highlighted by the Erdős 993 fixture.
    tree = _graph(
        ("a", "b", "c", "d", "e", "f", "g"),
        (
            ("a", "b"),
            ("a", "c"),
            ("b", "e"),
            ("c", "d"),
            ("e", "f"),
            ("e", "g"),
        ),
    )
    relabeled = _independent_relabel(
        tree,
        {"a": "q", "b": "t", "c": "p", "d": "s", "e": "n", "f": "r", "g": "u"},
    )

    assert (
        _canonicalize(tree).canonical_graph == _canonicalize(relabeled).canonical_graph
    )


def _pruefer_tree(sequence: tuple[int, ...]) -> tuple[tuple[str, str], ...]:
    vertex_count = len(sequence) + 2
    degree = [1] * vertex_count
    for vertex in sequence:
        degree[vertex] += 1
    edges: list[tuple[int, int]] = []
    for vertex in sequence:
        leaf = next(index for index, value in enumerate(degree) if value == 1)
        edges.append((leaf, vertex))
        degree[leaf] -= 1
        degree[vertex] -= 1
    remaining = [index for index, value in enumerate(degree) if value == 1]
    edges.append((remaining[0], remaining[1]))
    labels = ("a", "b", "c", "d")
    return tuple(
        sorted(tuple(sorted((labels[left], labels[right]))) for left, right in edges)
    )


def test_all_labeled_four_vertex_trees_give_the_exact_two_representatives() -> None:
    actual = {
        _canonicalize(
            _graph(("a", "b", "c", "d"), _pruefer_tree(sequence))
        ).canonical_graph
        for sequence in itertools.product(range(4), repeat=2)
    }
    expected = {
        _graph(
            ("v00", "v01", "v02", "v03"),
            (("v00", "v01"), ("v00", "v02"), ("v00", "v03")),
        ),
        _graph(
            ("v00", "v01", "v02", "v03"),
            (("v00", "v01"), ("v00", "v02"), ("v01", "v03")),
        ),
    }

    # Compare the representative set, not only its cardinality: a duplicate
    # plus an omission can preserve the headline count.
    assert actual == expected


def test_canonical_equality_agrees_with_networkx_on_all_order_four_graphs() -> None:
    vertices = ("a", "b", "c", "d")
    possible_edges = tuple(itertools.combinations(vertices, 2))
    graphs = tuple(
        _graph(
            vertices,
            tuple(
                edge
                for edge_index, edge in enumerate(possible_edges)
                if mask & (1 << edge_index)
            ),
        )
        for mask in range(1 << len(possible_edges))
    )
    canonical = tuple(_canonicalize(graph).canonical_graph for graph in graphs)

    for left_index, left in enumerate(graphs):
        nx_left = nx.Graph()
        nx_left.add_nodes_from(left.graph.vertices)
        nx_left.add_edges_from(left.graph.edges)
        for right_index in range(left_index, len(graphs)):
            right = graphs[right_index]
            nx_right = nx.Graph()
            nx_right.add_nodes_from(right.graph.vertices)
            nx_right.add_edges_from(right.graph.edges)
            assert (canonical[left_index] == canonical[right_index]) == (
                nx.is_isomorphic(nx_left, nx_right)
            )


def test_result_rejects_source_conclusion_and_tie_break_mutations() -> None:
    source = _graph(
        ("a", "b", "c", "d"),
        (("a", "b"), ("b", "c"), ("c", "d")),
    )
    result = _canonicalize(source)

    changed_source = result.model_dump(mode="json")
    changed_source["source_graph"]["graph"]["edges"] = [
        ["a", "b"],
        ["c", "d"],
    ]
    with pytest.raises(ValidationError):
        ColoredGraphCanonicalizationResult.model_validate(changed_source)

    changed_graph = result.model_dump(mode="json")
    changed_graph["canonical_graph"]["graph"]["edges"] = [
        ["v00", "v01"],
        ["v00", "v02"],
        ["v00", "v03"],
    ]
    with pytest.raises(ValidationError):
        ColoredGraphCanonicalizationResult.model_validate(changed_graph)

    changed_tie_break = result.model_dump(mode="json")
    relabeling = changed_tie_break["relabeling"]
    index_by_source = {
        item["source_vertex"]: index for index, item in enumerate(relabeling)
    }
    for left, right in (("a", "d"), ("b", "c")):
        left_index = index_by_source[left]
        right_index = index_by_source[right]
        (
            relabeling[left_index]["canonical_vertex"],
            relabeling[right_index]["canonical_vertex"],
        ) = (
            relabeling[right_index]["canonical_vertex"],
            relabeling[left_index]["canonical_vertex"],
        )
    with pytest.raises(ValidationError):
        ColoredGraphCanonicalizationResult.model_validate(changed_tie_break)


def test_request_admits_by_color_class_size_not_only_vertex_count() -> None:
    nine = tuple(f"v{index:02d}" for index in range(9))
    ten = tuple(f"v{index:02d}" for index in range(10))
    sixty_four = tuple(f"v{index:02d}" for index in range(64))

    ColoredGraphCanonicalizationRequest(colored_graph=_graph(nine, ()))
    ColoredGraphCanonicalizationRequest(
        colored_graph=_graph(
            sixty_four,
            (),
            vertex_colors=tuple(f"color-{index:02d}" for index in range(64)),
        )
    )
    with pytest.raises(ValidationError):
        ColoredGraphCanonicalizationRequest(colored_graph=_graph(ten, ()))


def test_edgeless_distinguished_carrier_admits_past_the_fixed_order_cap() -> None:
    """A 65-vertex edgeless graph with distinct vertex colors must be admitted.

    Distinct colors leave one candidate labeling, only ``2 * 65`` replay-work
    units, and a small exact result, so the derived work and result admission
    accepts what a fixed 64-vertex carrier cap would have rejected.
    """
    count = 65
    vertices = tuple(f"w{index:03d}" for index in range(count))
    colors = tuple(f"c{index:03d}" for index in range(count))

    result = _canonicalize(_graph(vertices, (), vertex_colors=colors))

    assert result.canonical_graph.graph.vertices == tuple(
        f"v{index:02d}" for index in range(count)
    )
    assert result.canonical_graph.vertex_colors == tuple(sorted(colors))
    assert tuple(
        (item.source_vertex, item.canonical_vertex) for item in result.relabeling
    ) == tuple((source, f"v{index:02d}") for index, source in enumerate(vertices))


def test_canonical_labels_stay_sorted_across_the_three_digit_boundary() -> None:
    count = 105
    vertices = tuple(f"w{index:03d}" for index in range(count))
    colors = tuple(sorted((f"c{index:03d}" for index in range(count)), reverse=True))

    result = _canonicalize(_graph(vertices, (), vertex_colors=colors))

    expected_labels = tuple(f"v{index:03d}" for index in range(count))
    assert tuple(sorted(expected_labels)) == expected_labels
    assert result.canonical_graph.graph.vertices == expected_labels
    assert result.canonical_graph.vertex_colors == tuple(sorted(colors))

    ColoredGraphCanonicalizationResult.model_validate(result.model_dump(mode="json"))


def test_request_rejects_edge_key_work_before_enumeration() -> None:
    eight_vertices = tuple(f"v{index:02d}" for index in range(8))
    nine_vertices = tuple(f"v{index:02d}" for index in range(9))

    ColoredGraphCanonicalizationRequest(
        colored_graph=_graph(
            eight_vertices,
            tuple(itertools.combinations(eight_vertices, 2)),
        )
    )
    with pytest.raises(ValidationError):
        ColoredGraphCanonicalizationRequest(
            colored_graph=_graph(
                nine_vertices,
                tuple(itertools.combinations(nine_vertices, 2)),
            )
        )


@pytest.mark.parametrize(
    "graph",
    [
        _graph(tuple(f"v{index:02d}" for index in range(10)), ()),
        _graph(
            tuple(f"v{index:02d}" for index in range(9)),
            tuple(
                itertools.combinations(tuple(f"v{index:02d}" for index in range(9)), 2)
            ),
        ),
    ],
    ids=["permutation-bound", "replay-work-bound"],
)
def test_native_admission_failure_raises_wire_validation_error(
    graph: ColoredUndirectedGraph,
) -> None:
    """Native callers see the public ValidationError the wire path raises.

    A valid graph over the execution bound must not leak the core
    ``PydanticCustomError`` from the shared admission function; the native
    entry point translates it without rebuilding a wire request, keeping the
    advertised native/wire typed-outcome parity.
    """

    with pytest.raises(ValidationError) as native:
        canonicalize_colored_graph(graph)

    assert isinstance(native.value.__cause__, PydanticCustomError)
    with pytest.raises(ValidationError) as wire:
        ColoredGraphCanonicalizationRequest(colored_graph=graph)

    assert [item | {"input": None} for item in native.value.errors()] == [
        item | {"input": None} for item in wire.value.errors()
    ]


def _dense_complete_colored_graph(label_bytes: int) -> ColoredUndirectedGraph:
    labels = tuple(f"{index:02d}-" + "x" * (label_bytes - 3) for index in range(64))
    complete_edges = tuple(itertools.combinations(labels, 2))
    return _graph(
        labels,
        complete_edges,
        vertex_colors=tuple(f"color-{index:02d}" for index in range(64)),
        edge_colors=("e" * 64,) * len(complete_edges),
    )


def test_request_admits_transport_bounded_dense_results() -> None:
    """A 64-vertex complete graph with 49-byte labels fits the derived budget.

    Distinct vertex colors leave one candidate labeling, the replay work stays
    below its bound, and the exact result sits far below the repository's
    10 MiB canonical JSON output limit, so only the superseded 512 KiB ceiling
    rejected it.
    """
    dense_graph = _dense_complete_colored_graph(49)

    assert (
        isomorphism_canonicalization.canonicalization_result_wire_bytes(dense_graph)
        > 512 * 1024
    )
    result = compute_colored_graph_canonicalization(
        ColoredGraphCanonicalizationRequest(colored_graph=dense_graph)
    )
    assert canonicalize_colored_graph(dense_graph) == result

    labels = tuple(f"{index:02d}-" + "x" * (49 - 3) for index in range(64))
    mapping = {item.source_vertex: item.canonical_vertex for item in result.relabeling}
    assert mapping == {label: f"v{index:02d}" for index, label in enumerate(labels)}


def test_request_enforces_source_bound_result_byte_boundary(monkeypatch) -> None:
    max_shape = _dense_complete_colored_graph(64)
    assert (
        isomorphism_canonicalization.canonicalization_result_wire_bytes(max_shape)
        <= isomorphism_canonicalization.MAX_CANONICALIZATION_RESULT_BYTES
    )

    monkeypatch.setattr(
        isomorphism_bounds,
        "MAX_CANONICALIZATION_RESULT_BYTES",
        512 * 1024,
    )
    with pytest.raises(ValidationError):
        ColoredGraphCanonicalizationRequest(colored_graph=max_shape)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"graph": {"vertices": ["a", "a"], "edges": []}},
            "unique",
        ),
        (
            {
                "graph": {
                    "vertices": ["a", "b"],
                    "edges": [["b", "a"]],
                }
            },
            "in order",
        ),
        (
            {
                "graph": {
                    "vertices": ["a", "b"],
                    "edges": [["a", "b"]],
                },
                "vertex_colors": ["only-one"],
            },
            "align one color with every vertex",
        ),
        (
            {"graph": {"vertices": ["é" * 33], "edges": []}},
            "64 UTF-8 bytes",
        ),
        (
            {"graph": {"vertices": [""], "edges": []}},
            "must not be empty",
        ),
        (
            {
                "graph": {"vertices": ["a"], "edges": []},
                "vertex_colors": ["e\u0301"],
            },
            "Unicode NFC",
        ),
        (
            {
                "graph": {"vertices": ["a", "b"], "edges": [["a", "b"]]},
                "edge_colors": ["\U0001f384" * 17],
            },
            "64 UTF-8 bytes",
        ),
    ],
)
def test_colored_graph_rejects_noncanonical_presentations(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError):
        ColoredUndirectedGraph.model_validate(payload)


def test_schema_explains_alignment_and_work_admission() -> None:
    schema = ColoredGraphCanonicalizationRequest.model_json_schema()
    graph_schema = schema["$defs"]["ColoredUndirectedGraph"]

    assert "aligned" in graph_schema["description"]
    assert "execution-plus-validation replay work" in schema["description"]


def test_schema_documents_nfc_and_byte_limits_for_color_names() -> None:
    """The published schema must reveal the byte and normalization constraints.

    ``maxLength`` counts characters while the validator binds UTF-8 bytes, so a
    64-character emoji color passes the schema length but not the value check;
    the field descriptions must state both requirements.
    """
    schema = ColoredUndirectedGraph.model_json_schema()

    for field in ("vertex_colors", "edge_colors", "graph"):
        description = schema["properties"][field]["description"]
        assert "NFC" in description, field
        assert "UTF-8 bytes" in description, field


def test_canonical_value_is_hash_seed_independent() -> None:
    code = """
import json
from jacobian.math.graphs.isomorphism import (
    ColoredUndirectedGraph,
    canonicalize_colored_graph,
)
graph = ColoredUndirectedGraph(
    graph={
        "vertices": ("a", "b", "c", "d"),
        "edges": (("a", "b"), ("a", "d"), ("b", "c"), ("c", "d")),
    },
)
result = canonicalize_colored_graph(
    graph
)
print(json.dumps(result.canonical_graph.model_dump(mode="json"), sort_keys=True))
"""
    outputs = []
    for seed in ("1", "8675309"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        outputs.append(json.loads(completed.stdout))

    assert outputs[0] == outputs[1]


def test_native_api_exports_typed_canonicalization() -> None:
    from jacobian.math.graphs import isomorphism

    assert "ColoredUndirectedGraph" in isomorphism.__all__
    assert "canonicalize_colored_graph" in isomorphism.__all__


def test_catalog_path_and_native_api_agree() -> None:
    graph = _graph(
        ("a", "b", "c", "d"),
        (("a", "b"), ("b", "c"), ("c", "d")),
        vertex_colors=("endpoint", "middle", "middle", "endpoint"),
        edge_colors=("outer", "middle", "outer"),
    )
    request = ColoredGraphCanonicalizationRequest(colored_graph=graph)

    assert compute_colored_graph_canonicalization(request) == (
        canonicalize_colored_graph(graph)
    )


def test_catalog_execution_admits_the_parsed_request_once(monkeypatch) -> None:
    """``math.run`` parses once; the adapter must not readmit the request.

    The result-size preflight runs once per request admission. For an
    already-parsed request, exactly one further admission may occur inside
    the catalog run: the result replay's own source-bound revalidation. A
    second one means the adapter detoured through the validating native
    wrapper instead of the shared request-accepting implementation.
    """
    parsed = ColoredGraphCanonicalizationRequest(
        colored_graph=_graph(("a", "b"), (("a", "b"),))
    )
    expected = canonicalize_colored_graph(parsed.colored_graph)

    admissions: list[ColoredUndirectedGraph] = []
    real_preflight = isomorphism_bounds.canonicalization_result_wire_bytes

    def counted_preflight(graph: ColoredUndirectedGraph) -> int:
        admissions.append(graph)
        return real_preflight(graph)

    monkeypatch.setattr(
        isomorphism_bounds,
        "canonicalization_result_wire_bytes",
        counted_preflight,
    )

    result = compute_colored_graph_canonicalization(parsed)

    assert result == expected
    assert len(admissions) == 1
    assert admissions[0] == parsed.colored_graph
