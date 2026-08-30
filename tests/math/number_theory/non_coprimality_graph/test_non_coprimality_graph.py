"""Tests for the non-coprimality conflict-graph operation."""

from collections.abc import Sequence

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.number_theory.non_coprimality_graph._models import (
    NonCoprimalityGraphRequest,
    NonCoprimalityGraphResult,
)
from jacobian.math.number_theory.non_coprimality_graph._tools import TOOLS
from jacobian.math.number_theory.non_coprimality_graph.operations import (
    non_coprimality_graph,
)


def _graph(vertices: Sequence[str]) -> SimpleUndirectedGraph:
    """Run the operation on a list of integer-string vertices."""
    request = NonCoprimalityGraphRequest.model_validate(
        {"elements": {"elements": vertices}}
    )
    result = non_coprimality_graph(request.elements.elements)
    return result.graph


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "number_theory.integer_set.non_coprimality_graph.compute"
    }


def test_result_type_is_returned() -> None:
    request = NonCoprimalityGraphRequest.model_validate(
        {"elements": {"elements": ["2", "3", "5"]}}
    )
    assert isinstance(
        non_coprimality_graph(request.elements.elements), NonCoprimalityGraphResult
    )


def test_empty_set_gives_empty_graph() -> None:
    graph = _graph([])
    assert graph.vertices == ()
    assert graph.edges == ()


def test_single_element_has_no_edges() -> None:
    graph = _graph(["7"])
    assert graph.vertices == ("7",)
    assert graph.edges == ()


def test_two_coprime_elements_have_no_edge() -> None:
    graph = _graph(["2", "3"])
    assert graph.vertices == ("2", "3")
    assert graph.edges == ()


def test_two_non_coprime_elements_have_one_edge() -> None:
    graph = _graph(["6", "10"])
    assert graph.vertices == ("6", "10")
    assert graph.edges == (("10", "6"),)


def test_pairwise_coprime_gives_edgeless_graph() -> None:
    graph = _graph(["2", "3", "5", "7"])
    assert graph.vertices == ("2", "3", "5", "7")
    assert graph.edges == ()


def test_all_share_factor_two_is_k3() -> None:
    graph = _graph(["2", "4", "6"])
    assert graph.vertices == ("2", "4", "6")
    assert len(graph.edges) == 3
    assert {tuple(e) for e in graph.edges} == {("2", "4"), ("2", "6"), ("4", "6")}


def test_mixed_pair_shares_factor() -> None:
    # gcd(2,6)=2, gcd(2,10)=2, gcd(3,6)=3, gcd(6,10)=2
    # gcd(2,3)=1, gcd(3,10)=1
    graph = _graph(["2", "3", "6", "10"])
    assert graph.vertices == ("2", "3", "6", "10")
    assert {tuple(e) for e in graph.edges} == {
        ("2", "6"),
        ("10", "2"),
        ("3", "6"),
        ("10", "6"),
    }


def test_primes_are_isolated() -> None:
    graph = _graph(["11", "13", "17", "19"])
    assert graph.edges == ()


def test_duplicate_value_powers_share_factors() -> None:
    # 2 and all powers of 2 share factor 2
    graph = _graph(["2", "4", "8", "16", "32"])
    # All pairs share gcd 2, so complete graph K5
    assert len(graph.vertices) == 5
    assert len(graph.edges) == 10  # C(5,2) = 10


def test_large_coprime_pair() -> None:
    # Two large distinct primes are coprime
    graph = _graph(["998244353", "1000000007"])
    assert graph.edges == ()


def test_large_non_coprime_pair() -> None:
    # Two numbers sharing a large factor
    graph = _graph(["1000000007", "2000000014"])  # both divisible by 1000000007
    assert graph.edges == (("1000000007", "2000000014"),)


def test_canonical_integer_above_python_digit_limit_is_supported() -> None:
    value = "9" * 5_000

    graph = _graph([value])

    assert graph.vertices == (value,)


def test_vertex_order_is_sorted_by_value() -> None:
    # Input is given out of numeric order
    graph = _graph(["10", "2", "6", "3"])
    # Sorted by numeric value: 2, 3, 6, 10
    assert graph.vertices == ("2", "3", "6", "10")


def test_edges_use_canonical_string_order() -> None:
    # Ensure edges always have left < right as strings
    graph = _graph(["6", "10"])
    for left, right in graph.edges:
        assert left < right


def test_request_model_validates_finite_integer_set() -> None:
    request = NonCoprimalityGraphRequest.model_validate(
        {"elements": {"elements": ["2", "4", "6"]}}
    )
    assert request.elements.elements == ("2", "4", "6")


def test_request_rejects_duplicate_elements() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        NonCoprimalityGraphRequest.model_validate(
            {"elements": {"elements": ["2", "2", "3"]}}
        )


def test_operation_through_tools_run() -> None:
    tool = TOOLS[0]
    request = tool.request_type.model_validate(
        {"elements": {"elements": ["2", "3", "6"]}}
    )
    result = tool.run(request)
    assert result.graph.vertices == ("2", "3", "6")
    assert len(result.graph.edges) == 2  # (2,6) and (3,6)


def test_examples_are_schema_valid() -> None:
    tool = TOOLS[0]
    for example in tool.examples:
        request = tool.request_type.model_validate(example.input)
        result = tool.run(request)
        assert isinstance(result, NonCoprimalityGraphResult)


def test_graph_is_simple_undirected_graph() -> None:
    graph = _graph(["2", "3", "6"])
    assert isinstance(graph, SimpleUndirectedGraph)


def test_max_256_vertices_allowed() -> None:
    vertices = [str(i) for i in range(1, 257)]
    graph = _graph(vertices)
    assert len(graph.vertices) == 256


def test_max_257_vertices_rejected() -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    vertices = tuple(str(i) for i in range(1, 258))
    with pytest.raises(OperationDomainValidationError):
        non_coprimality_graph(vertices)


def test_non_positive_integer_rejected() -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    # Zero is non-positive
    with pytest.raises(OperationDomainValidationError):
        non_coprimality_graph(("0", "2", "3"))


def test_negative_integer_rejected() -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    # FiniteIntegerSet allows negatives, but our operation rejects them
    with pytest.raises(OperationDomainValidationError):
        non_coprimality_graph(("-5", "2", "3"))


def test_complete_graph_when_all_pairwise_non_coprime() -> None:
    # All even numbers
    graph = _graph(["2", "4", "6", "8"])
    # Every pair shares factor 2 -> complete graph K4
    assert len(graph.edges) == 6  # C(4,2)


def test_disconnected_components() -> None:
    # {2,4} connected, {3,9} connected, no cross edges
    graph = _graph(["2", "4", "3", "9"])
    edges = {tuple(e) for e in graph.edges}
    assert edges == {("2", "4"), ("3", "9")}


def test_complete_graph_output_is_admitted_before_gcd_work() -> None:
    base = 2 * 10**999
    vertices = tuple(str(base + 2 * index) for index in range(120))

    with pytest.raises(OperationDomainValidationError, match="canonical output"):
        non_coprimality_graph(vertices)
