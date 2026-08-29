from __future__ import annotations

from typing import cast

import networkx as nx
import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.rooted_trees import (
    RootedTreeFinePartition,
    RootedTreeFinePartitionConstructed,
    RootedTreeNotATree,
    construct_fine_partition,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph_value(graph: nx.Graph[int]) -> SimpleUndirectedGraph:
    labels = {vertex: f"v{vertex:02d}" for vertex in graph.nodes}
    return SimpleUndirectedGraph(
        vertices=tuple(sorted(labels.values())),
        edges=tuple(
            sorted(
                (labels[left], labels[right])
                if labels[left] < labels[right]
                else (labels[right], labels[left])
                for left, right in graph.edges
            )
        ),
    )


def _backend(graph: SimpleUndirectedGraph) -> nx.Graph[str]:
    backend: nx.Graph[str] = nx.Graph()
    backend.add_nodes_from(graph.vertices)
    backend.add_edges_from(graph.edges)
    return backend


def _assert_fine_partition(result: RootedTreeFinePartition) -> None:
    outcome = result.outcome
    assert isinstance(outcome, RootedTreeFinePartitionConstructed)
    graph = _backend(result.graph)
    depths = nx.single_source_shortest_path_length(graph, result.root)
    seeds_x = set(outcome.seeds_x)
    seeds_y = set(outcome.seeds_y)
    seeds = seeds_x | seeds_y

    assert result.root in seeds_x
    assert all(depths[seed] % 2 == 0 for seed in seeds_x)
    assert all(depths[seed] % 2 == 1 for seed in seeds_y)
    assert len(seeds_x) * result.component_size_limit <= 12 * (len(graph) - 1)
    assert len(seeds_y) * result.component_size_limit <= 12 * (len(graph) - 1)

    expected_components = {
        frozenset(component)
        for component in nx.connected_components(graph.subgraph(set(graph) - seeds))
    }
    assert {
        frozenset(shrub.vertices) for shrub in outcome.shrubs
    } == expected_components

    reported_edges = set(outcome.seed_edges)
    row_count = len(outcome.seed_edges)
    for shrub in outcome.shrubs:
        vertices = set(shrub.vertices)
        boundary = set(shrub.boundary_seeds)
        assert 1 <= len(vertices) <= result.component_size_limit
        assert boundary
        assert boundary <= seeds_x or boundary <= seeds_y
        assert shrub.route_side == ("X" if boundary <= seeds_x else "Y")
        assert set(shrub.edges) == {
            tuple(sorted(edge)) for edge in graph.subgraph(vertices).edges
        }
        assert set(shrub.boundary_edges) == {
            tuple(sorted((vertex, neighbor)))
            for vertex in vertices
            for neighbor in graph.neighbors(vertex)
            if neighbor in seeds
        }
        route = nx.shortest_path(graph, shrub.root_vertex, result.root)
        assert route[1] == shrub.upper_seed
        reported_edges.update(shrub.edges)
        reported_edges.update(shrub.boundary_edges)
        row_count += len(shrub.edges) + len(shrub.boundary_edges)

    assert reported_edges == set(result.graph.edges)
    assert row_count == len(result.graph.edges)
    assert (
        RootedTreeFinePartition.model_validate_json(result.model_dump_json()) == result
    )


@pytest.fixture
def path_five() -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d", "e"),
        edges=(("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")),
    )


def test_path_partition_has_deterministic_source_bound_rows(
    path_five: SimpleUndirectedGraph,
) -> None:
    result = construct_fine_partition(path_five, "a", 2)

    assert isinstance(result.outcome, RootedTreeFinePartitionConstructed)
    assert result.outcome.seeds_x == ("a", "c")
    assert result.outcome.seeds_y == ()
    assert result.outcome.seed_edges == ()
    assert tuple(shrub.vertices for shrub in result.outcome.shrubs) == (
        ("b",),
        ("d", "e"),
    )
    assert result.outcome.shrubs[0].boundary_seeds == ("a", "c")
    assert result.outcome.shrubs[0].upper_seed == "a"
    assert result.outcome.shrubs[1].upper_seed == "c"
    _assert_fine_partition(result)


@pytest.mark.parametrize(("root", "size_limit"), [("c", 1), ("e", 2)])
def test_path_partition_supports_different_roots_and_limits(
    path_five: SimpleUndirectedGraph, root: str, size_limit: int
) -> None:
    _assert_fine_partition(construct_fine_partition(path_five, root, size_limit))


def test_full_parity_repair_cuts_both_sides_of_an_internal_component() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("v00", "v01", "v02", "v03", "v04"),
        edges=(
            ("v00", "v01"),
            ("v00", "v03"),
            ("v01", "v02"),
            ("v03", "v04"),
        ),
    )

    result = construct_fine_partition(graph, "v02", 1)

    assert isinstance(result.outcome, RootedTreeFinePartitionConstructed)
    # v00 is the W3 repair seed: omitting it leaves opposite-parity boundary seeds.
    assert result.outcome.seeds_x == ("v00", "v02")
    assert result.outcome.seeds_y == ("v01", "v03")
    assert tuple(shrub.boundary_seeds for shrub in result.outcome.shrubs) == (("v03",),)
    _assert_fine_partition(result)


@pytest.mark.parametrize("root", ["center", "leaf-a"])
def test_star_partition_is_valid_for_different_declared_roots(root: str) -> None:
    graph = SimpleUndirectedGraph(
        vertices=("center", "leaf-a", "leaf-b", "leaf-c", "leaf-d"),
        edges=(
            ("center", "leaf-a"),
            ("center", "leaf-b"),
            ("center", "leaf-c"),
            ("center", "leaf-d"),
        ),
    )

    _assert_fine_partition(construct_fine_partition(graph, root, 1))


def test_largest_component_limit_leaves_one_root_seed(
    path_five: SimpleUndirectedGraph,
) -> None:
    result = construct_fine_partition(path_five, "a", len(path_five.vertices) - 1)

    assert isinstance(result.outcome, RootedTreeFinePartitionConstructed)
    assert result.outcome.seeds_x == ("a",)
    assert result.outcome.seeds_y == ()
    assert tuple(shrub.vertices for shrub in result.outcome.shrubs) == (
        ("b", "c", "d", "e"),
    )
    _assert_fine_partition(result)


def test_representation_order_does_not_change_the_constructed_outcome() -> None:
    ordered = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d", "e"),
        edges=(("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")),
    )
    permuted = SimpleUndirectedGraph(
        vertices=("e", "c", "a", "d", "b"),
        edges=(("d", "e"), ("b", "c"), ("a", "b"), ("c", "d")),
    )

    left = construct_fine_partition(ordered, "a", 2)
    right = construct_fine_partition(permuted, "a", 2)

    assert left.outcome == right.outcome


def test_relabelling_preserves_the_contract_without_promising_the_same_cut() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("ant", "bee", "cat", "dog", "elk"),
        edges=(("ant", "dog"), ("bee", "elk"), ("cat", "dog"), ("cat", "elk")),
    )

    _assert_fine_partition(construct_fine_partition(graph, "bee", 2))


@pytest.mark.parametrize(
    ("graph", "connected", "has_cycle", "component_count"),
    [
        (
            SimpleUndirectedGraph(
                vertices=("a", "b", "c"),
                edges=(("a", "b"), ("a", "c"), ("b", "c")),
            ),
            True,
            True,
            1,
        ),
        (
            SimpleUndirectedGraph(
                vertices=("a", "b", "c", "d"),
                edges=(("a", "b"), ("c", "d")),
            ),
            False,
            False,
            2,
        ),
    ],
)
def test_non_tree_inputs_return_source_bound_diagnostics(
    graph: SimpleUndirectedGraph,
    connected: bool,
    has_cycle: bool,
    component_count: int,
) -> None:
    result = construct_fine_partition(graph, graph.vertices[0], 1)

    assert result.outcome == RootedTreeNotATree(
        connected=connected,
        has_cycle=has_cycle,
        component_count=component_count,
    )
    assert (
        RootedTreeFinePartition.model_validate_json(result.model_dump_json()) == result
    )


def test_result_parsing_rejects_constructed_rows_not_bound_to_source() -> None:
    payload = {
        "graph": {"vertices": ["a", "b"], "edges": [["a", "b"]]},
        "root": "a",
        "component_size_limit": 1,
        "outcome": {
            "status": "CONSTRUCTED",
            "seeds_x": ["a"],
            "seeds_y": [],
            "seed_edges": [],
            "shrubs": [],
        },
    }

    with pytest.raises(ValidationError, match="partition all graph vertices"):
        RootedTreeFinePartition.model_validate(payload)


def test_result_parsing_rejects_contradictory_non_tree_diagnostic() -> None:
    payload = {
        "graph": {"vertices": ["a", "b"], "edges": [["a", "b"]]},
        "root": "a",
        "component_size_limit": 1,
        "outcome": {
            "status": "NOT_A_TREE",
            "connected": False,
            "has_cycle": True,
            "component_count": 2,
        },
    }

    with pytest.raises(ValidationError, match="diagnostic must match"):
        RootedTreeFinePartition.model_validate(payload)


def test_result_parsing_rejects_non_tree_status_for_a_tree() -> None:
    payload = {
        "graph": {"vertices": ["a", "b"], "edges": [["a", "b"]]},
        "root": "a",
        "component_size_limit": 1,
        "outcome": {
            "status": "NOT_A_TREE",
            "connected": True,
            "has_cycle": False,
            "component_count": 1,
        },
    }

    with pytest.raises(ValidationError, match="disconnected or cyclic"):
        RootedTreeFinePartition.model_validate(payload)


def test_result_parsing_rejects_undeclared_upper_seed_without_key_error(
    path_five: SimpleUndirectedGraph,
) -> None:
    payload = construct_fine_partition(path_five, "a", 2).model_dump(mode="json")
    payload["outcome"]["shrubs"][0]["upper_seed"] = "missing"
    payload["outcome"]["shrubs"][0]["boundary_seeds"] = ["c", "missing"]

    with pytest.raises(ValidationError, match="upper_seed must be a retained seed"):
        RootedTreeFinePartition.model_validate(payload)


@pytest.mark.parametrize(
    ("root", "size_limit", "code"),
    [
        ("missing", 1, "root_membership"),
        ("a", 0, "component_size_limit"),
        ("a", 5, "component_size_limit"),
    ],
)
def test_semantic_admission_rejects_invalid_root_or_limit(
    path_five: SimpleUndirectedGraph,
    root: str,
    size_limit: int,
    code: str,
) -> None:
    with pytest.raises(OperationDomainValidationError) as caught:
        construct_fine_partition(path_five, root, size_limit)

    assert (
        caught.value.errors()[0]["type"] == f"graph.rooted_tree.fine_partition.{code}"
    )


def test_empty_graph_is_rejected_before_backend_execution() -> None:
    graph = SimpleUndirectedGraph(vertices=(), edges=())

    with pytest.raises(OperationDomainValidationError) as caught:
        construct_fine_partition(graph, "root", 1)

    assert caught.value.errors()[0]["type"].endswith("nonempty_graph")


def test_utf8_label_byte_boundary_is_admitted_before_construction() -> None:
    accepted_label = "é" * 32
    accepted = SimpleUndirectedGraph(
        vertices=("root", accepted_label), edges=(("root", accepted_label),)
    )
    rejected_label = "é" * 33
    rejected = SimpleUndirectedGraph(
        vertices=("root", rejected_label), edges=(("root", rejected_label),)
    )

    _assert_fine_partition(construct_fine_partition(accepted, "root", 1))
    with pytest.raises(OperationDomainValidationError) as caught:
        construct_fine_partition(rejected, "root", 1)
    assert caught.value.errors()[0]["type"].endswith("label_bytes")


def test_unencodable_label_is_rejected_by_native_admission() -> None:
    surrogate = "\ud800"
    graph = SimpleUndirectedGraph(
        vertices=("root", surrogate),
        edges=(("root", surrogate),),
    )

    with pytest.raises(OperationDomainValidationError) as caught:
        construct_fine_partition(graph, "root", 1)

    assert caught.value.errors()[0]["type"].endswith("label_utf8")


def test_empty_source_label_is_rejected_before_result_construction() -> None:
    graph = SimpleUndirectedGraph(vertices=("", "root"), edges=(("", "root"),))

    with pytest.raises(OperationDomainValidationError) as caught:
        construct_fine_partition(graph, "root", 1)

    assert caught.value.errors()[0]["type"].endswith("empty_label")


def test_structural_result_parsing_rejects_an_empty_retained_source_label() -> None:
    payload = {
        "graph": {
            "vertices": ["", "a", "b"],
            "edges": [["", "a"], ["", "b"], ["a", "b"]],
        },
        "root": "a",
        "component_size_limit": 1,
        "outcome": {
            "status": "NOT_A_TREE",
            "connected": True,
            "has_cycle": True,
            "component_count": 1,
        },
    }

    with pytest.raises(ValidationError, match="graph vertex labels must not be empty"):
        RootedTreeFinePartition.model_validate(payload)


def test_shared_maximum_order_path_is_admitted() -> None:
    vertices = tuple(f"v{index:03d}" for index in range(256))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple((vertices[index], vertices[index + 1]) for index in range(255)),
    )

    result = construct_fine_partition(graph, vertices[0], 255)

    assert isinstance(result.outcome, RootedTreeFinePartitionConstructed)
    assert result.outcome.seeds_x == (vertices[0],)
    assert len(result.outcome.shrubs) == 1
    assert len(result.outcome.shrubs[0].vertices) == 255
    _assert_fine_partition(result)


def test_retained_non_tree_must_fit_the_canonical_output_budget() -> None:
    # NUL consumes one UTF-8 byte but six canonical JSON bytes, exercising the
    # distinction between label-byte admission and the retained wire bound.
    vertices = tuple(f"{index:03d}" + "\x00" * 61 for index in range(180))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            (vertices[left], vertices[right])
            for left in range(len(vertices))
            for right in range(left + 1, len(vertices))
        ),
    )

    with pytest.raises(OperationDomainValidationError) as caught:
        construct_fine_partition(graph, vertices[0], 1)

    assert caught.value.errors()[0]["type"].endswith("output_budget")


def test_every_nonisomorphic_tree_through_order_seven_satisfies_contract() -> None:
    checked = 0
    for order in range(2, 8):
        for backend in nx.generators.nonisomorphic_trees(order):
            graph = _graph_value(cast("nx.Graph[int]", backend))
            for root in graph.vertices:
                for size_limit in range(1, order):
                    _assert_fine_partition(
                        construct_fine_partition(graph, root, size_limit)
                    )
                    checked += 1

    assert checked == 734


def test_catalog_example_executes_and_round_trips() -> None:
    operation = Catalog.open().operation("graph.rooted_tree.fine_partition.construct")
    assert operation is not None
    assert len(operation.examples) == 1
    request = operation.request_type.model_validate(operation.examples[0].input)

    result = operation.run(request)

    assert isinstance(result.outcome, RootedTreeFinePartitionConstructed)
    assert operation.result_type.model_validate_json(result.model_dump_json()) == result
