"""Defining evidence for exact independence polynomials of bounded trees."""

from __future__ import annotations

import math
from typing import cast

import networkx as nx
import pytest
from pydantic import ValidationError

from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
)
from jacobian.math.graphs.operations import explicit_graph
from jacobian.math.graphs.polynomials import (
    independence_polynomial,
    independence_polynomial_coefficients,
)
from jacobian.math.graphs.polynomials._models import (
    MAX_INDEPENDENCE_POLYNOMIAL_COEFFICIENT_DIGITS,
    TreeIndependencePolynomialRequest,
    TreeIndependencePolynomialResult,
)
from jacobian.math.graphs.polynomials._operations import (
    compute_independence_polynomial,
    verify_tree_independence_polynomial_result,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.polynomials._elementary_operations import (
    rational_polynomial_evaluate,
)
from jacobian.math.polynomials._models import RationalPolynomialEvaluationRequest
from jacobian.math.polynomials.values import RationalPolynomial


def _path(order: int) -> SimpleUndirectedGraph:
    vertices = tuple(f"v{index:03d}" for index in range(order))
    edges = tuple((vertices[index], vertices[index + 1]) for index in range(order - 1))
    return explicit_graph(vertices, edges)


def _star(leaves: int) -> SimpleUndirectedGraph:
    vertices = ("center", *(f"leaf-{index:03d}" for index in range(leaves)))
    edges = tuple(("center", leaf) for leaf in vertices[1:])
    return explicit_graph(vertices, edges)


def _from_networkx(graph: nx.Graph[int]) -> SimpleUndirectedGraph:
    labels = {vertex: f"v{vertex:03d}" for vertex in graph.nodes}
    return explicit_graph(
        tuple(labels.values()),
        tuple((labels[left], labels[right]) for left, right in graph.edges),
    )


def _brute_force_coefficients(graph: SimpleUndirectedGraph) -> tuple[int, ...]:
    indices = {vertex: index for index, vertex in enumerate(graph.vertices)}
    edge_masks = tuple(
        (1 << indices[left]) | (1 << indices[right]) for left, right in graph.edges
    )
    counts = [0] * (len(graph.vertices) + 1)
    for mask in range(1 << len(graph.vertices)):
        if all(mask & edge_mask != edge_mask for edge_mask in edge_masks):
            counts[mask.bit_count()] += 1
    while len(counts) > 1 and counts[-1] == 0:
        counts.pop()
    return tuple(counts)


def _dense_coefficients(polynomial: RationalPolynomial) -> tuple[int, ...]:
    degree = max(
        (term.exponents[0] for term in polynomial.polynomial.terms),
        default=0,
    )
    coefficients = [0] * (degree + 1)
    for term in polynomial.polynomial.terms:
        numerator, denominator = term.coefficient.as_integer_ratio()
        assert denominator == 1
        coefficients[term.exponents[0]] = numerator
    return tuple(coefficients)


@pytest.mark.parametrize(
    ("graph", "expected"),
    [
        (explicit_graph(("v",), ()), (1, 1)),
        (_path(2), (1, 2)),
        (_path(5), (1, 5, 6, 1)),
        (_star(4), (1, 5, 6, 4, 1)),
    ],
)
def test_known_independence_polynomials(
    graph: SimpleUndirectedGraph,
    expected: tuple[int, ...],
) -> None:
    assert independence_polynomial_coefficients(graph) == expected
    assert _dense_coefficients(independence_polynomial(graph)) == expected

    result = compute_independence_polynomial(
        TreeIndependencePolynomialRequest(graph=graph)
    )
    assert result.coefficients == tuple(
        format_canonical_integer(coefficient) for coefficient in expected
    )
    assert result.independence_number == len(expected) - 1
    assert result.independent_set_count == format_canonical_integer(sum(expected))


def test_all_free_trees_through_order_eight_match_subset_enumeration() -> None:
    trees = [explicit_graph(("v",), ())]
    for order in range(2, 9):
        trees.extend(
            _from_networkx(cast("nx.Graph[int]", tree))
            for tree in nx.nonisomorphic_trees(order)
        )

    assert len(trees) == 48
    for graph in trees:
        expected = _brute_force_coefficients(graph)
        coefficients = independence_polynomial_coefficients(graph)
        polynomial = independence_polynomial(graph)

        assert coefficients == expected
        assert _dense_coefficients(polynomial) == expected
        assert coefficients[0] == 1
        assert coefficients[1] == len(graph.vertices)
        assert all(coefficient > 0 for coefficient in coefficients)
        assert len(coefficients) - 1 == len(expected) - 1
        assert sum(coefficients) == sum(expected)


def test_root_choice_and_vertex_relabeling_do_not_change_polynomial() -> None:
    graph = _path(8)
    different_root = SimpleUndirectedGraph(
        vertices=tuple(reversed(graph.vertices)),
        edges=graph.edges,
    )
    relabeling = {
        vertex: f"renamed-{len(graph.vertices) - index:03d}"
        for index, vertex in enumerate(graph.vertices)
    }
    relabeled = explicit_graph(
        tuple(relabeling.values()),
        tuple((relabeling[left], relabeling[right]) for left, right in graph.edges),
    )

    expected = independence_polynomial(graph)
    assert independence_polynomial(different_root) == expected
    assert independence_polynomial(relabeled) == expected


@pytest.mark.parametrize(
    "graph",
    [
        explicit_graph(("v",), ()),
        _path(4),
    ],
)
def test_serialized_polynomial_feeds_exact_evaluation_unchanged(
    graph: SimpleUndirectedGraph,
) -> None:
    result = compute_independence_polynomial(
        TreeIndependencePolynomialRequest(graph=graph)
    )
    serialized = result.model_dump(mode="json")["polynomial"]
    evaluation_request = RationalPolynomialEvaluationRequest.model_validate(
        {
            "polynomial": serialized,
            "point": {"num": "1", "den": "1"},
        }
    )

    evaluation = rational_polynomial_evaluate(evaluation_request)

    assert evaluation.value.num == result.independent_set_count
    assert evaluation.value.den == "1"


@pytest.mark.parametrize(
    "graph",
    [
        explicit_graph((), ()),
        explicit_graph(("a", "b", "c"), (("a", "b"),)),
        explicit_graph(
            ("a", "b", "c"),
            (("a", "b"), ("a", "c"), ("b", "c")),
        ),
    ],
)
def test_request_rejects_empty_disconnected_and_cyclic_graphs(
    graph: SimpleUndirectedGraph,
) -> None:
    with pytest.raises(ValidationError):
        TreeIndependencePolynomialRequest(graph=graph)


def test_star_beyond_the_old_consumer_degree_cap_is_admitted_exactly() -> None:
    leaves = 128
    star = _star(leaves)
    binomials = tuple(math.comb(leaves, k) for k in range(leaves + 1))
    expected = (binomials[0], binomials[1] + 1, *binomials[2:])

    request = TreeIndependencePolynomialRequest(graph=star)
    result = compute_independence_polynomial(request)

    assert result.coefficients == tuple(
        format_canonical_integer(coefficient) for coefficient in expected
    )
    assert result.independence_number == leaves
    assert result.independent_set_count == format_canonical_integer(sum(expected))
    assert result.independent_set_count == format_canonical_integer((1 << leaves) + 1)
    assert len(result.polynomial.polynomial.terms) == leaves + 1
    assert _dense_coefficients(independence_polynomial(star)) == expected


def test_full_vertex_envelope_path_is_admitted_and_over_envelope_is_rejected() -> None:
    admitted = _path(256)
    request = TreeIndependencePolynomialRequest(graph=admitted)

    result = compute_independence_polynomial(request)

    assert result.independence_number == 128
    assert len(result.coefficients) == 129
    assert result.coefficients[-1] == format_canonical_integer(
        math.comb(256 - 128 + 1, 128)
    )
    with pytest.raises(ValidationError):
        TreeIndependencePolynomialRequest(graph=_path(257))


def test_request_reserves_output_headroom_for_the_retained_source() -> None:
    output_limit = CanonicalLimits().max_output_bytes
    graph = explicit_graph(("v" * (output_limit - 300),), ())
    encoded_request = encode_strict_json({"graph": graph.model_dump(mode="json")})

    assert len(encoded_request) <= output_limit
    with pytest.raises(ValidationError):
        TreeIndependencePolynomialRequest(graph=graph)


def test_request_schema_exposes_tree_and_work_preconditions() -> None:
    schema = TreeIndependencePolynomialRequest.model_json_schema()

    assert "nonempty" in schema["description"]
    assert "acyclic" in schema["description"]
    assert "convolution-work" in schema["properties"]["graph"]["description"]
    graph_schema = schema["$defs"]["SimpleUndirectedGraph"]
    assert graph_schema["properties"]["vertices"]["maxItems"] == 256


def test_result_rejects_a_weaker_polynomial_and_a_changed_source() -> None:
    graph = _path(4)
    valid = compute_independence_polynomial(
        TreeIndependencePolynomialRequest(graph=graph)
    ).model_dump(mode="json")
    weaker = valid.copy()
    weaker["polynomial"] = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": "4", "den": "1"},
                    "exponents": [1],
                },
                {
                    "coefficient": {"num": "1", "den": "1"},
                    "exponents": [0],
                },
            ]
        },
    }
    changed_source = valid.copy()
    changed_source["graph"] = _path(5).model_dump(mode="json")

    with pytest.raises(ValidationError):
        TreeIndependencePolynomialResult.model_validate(weaker)
    with pytest.raises(ValidationError):
        TreeIndependencePolynomialResult.model_validate(changed_source)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("coefficients", ["1", "4", "2"], "coefficients do not match"),
        ("independence_number", 1, "independence number does not match"),
        ("independent_set_count", "7", "independent-set count does not match"),
    ],
)
def test_result_rejects_mutated_derived_values(
    field: str,
    replacement: object,
    message: str,
) -> None:
    valid = compute_independence_polynomial(
        TreeIndependencePolynomialRequest(graph=_path(4))
    ).model_dump(mode="json")
    valid[field] = replacement

    with pytest.raises(ValidationError):
        TreeIndependencePolynomialResult.model_validate(valid)


def test_result_rejects_a_polynomial_outside_qq_x() -> None:
    valid = compute_independence_polynomial(
        TreeIndependencePolynomialRequest(graph=_path(4))
    ).model_dump(mode="json")
    valid["polynomial"]["variables"] = ["y"]

    with pytest.raises(ValidationError):
        TreeIndependencePolynomialResult.model_validate(valid)


def test_result_rejects_overbudget_coefficients() -> None:
    valid = compute_independence_polynomial(
        TreeIndependencePolynomialRequest(graph=_path(4))
    ).model_dump(mode="json")
    digits = MAX_INDEPENDENCE_POLYNOMIAL_COEFFICIENT_DIGITS
    valid["polynomial"]["polynomial"]["terms"][0]["coefficient"]["num"] = "9" * (
        digits + 1
    )

    with pytest.raises(ValidationError):
        TreeIndependencePolynomialResult.model_validate(valid)


_OVERBUDGET_DIGITS = MAX_INDEPENDENCE_POLYNOMIAL_COEFFICIENT_DIGITS + 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("coefficients", ["9" * _OVERBUDGET_DIGITS, "1", "3"]),
        ("independent_set_count", "9" * _OVERBUDGET_DIGITS),
    ],
)
def test_result_rejects_overbudget_derived_values(
    field: str,
    replacement: object,
) -> None:
    valid = compute_independence_polynomial(
        TreeIndependencePolynomialRequest(graph=_path(4))
    ).model_dump(mode="json")
    valid[field] = replacement

    with pytest.raises(ValidationError):
        TreeIndependencePolynomialResult.model_validate(valid)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("coefficients", ["1", "-4", "3"]),
        ("independent_set_count", "-8"),
    ],
)
def test_result_rejects_negative_derived_values(
    field: str,
    replacement: object,
) -> None:
    valid = compute_independence_polynomial(
        TreeIndependencePolynomialRequest(graph=_path(4))
    ).model_dump(mode="json")
    valid[field] = replacement

    with pytest.raises(ValidationError):
        TreeIndependencePolynomialResult.model_validate(valid)


def test_explicit_verifier_rejects_a_structurally_consistent_forged_claim() -> None:
    graph = _path(4)
    forged = TreeIndependencePolynomialResult(
        graph=graph,
        coefficients=("1", "4", "2"),
        polynomial=RationalPolynomial.model_validate(
            {
                "domain": "QQ",
                "variables": ["x"],
                "polynomial": {
                    "terms": [
                        {
                            "coefficient": {"num": "2", "den": "1"},
                            "exponents": [2],
                        },
                        {
                            "coefficient": {"num": "4", "den": "1"},
                            "exponents": [1],
                        },
                        {
                            "coefficient": {"num": "1", "den": "1"},
                            "exponents": [0],
                        },
                    ]
                },
            }
        ),
        independence_number=2,
        independent_set_count="7",
    )

    assert not verify_tree_independence_polynomial_result(forged)


def test_native_module_exports_canonical_value_and_dense_projection() -> None:
    from jacobian.math.graphs import polynomials

    assert polynomials.__all__ == [
        "independence_polynomial",
        "independence_polynomial_coefficients",
    ]
    assert all(hasattr(polynomials, name) for name in polynomials.__all__)
    assert type(independence_polynomial(_path(2))) is RationalPolynomial
    assert independence_polynomial_coefficients(_path(2)) == (1, 2)
