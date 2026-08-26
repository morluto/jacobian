"""Exact checks for source-bound chromatic-number certificates."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations, product

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.graphs.coloring import _operations
from jacobian.math.graphs.coloring._chromatic_number_models import (
    MAX_CHROMATIC_CERTIFICATE_DERIVED_RATIONAL_DIGITS,
    MAX_CHROMATIC_CERTIFICATE_EDGES,
    MAX_CHROMATIC_CERTIFICATE_RATIONAL_DIGITS,
    MAX_CHROMATIC_CERTIFICATE_VERTICES,
    ChromaticNumberCertificateCheckRequest,
    ChromaticNumberCertificateCheckResult,
)
from jacobian.math.graphs.coloring._operations import (
    compute_chromatic_number_certificate_check,
    verify_chromatic_number_certificate_check_result,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _rational(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(numerator, denominator)


def _graph(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(vertices=vertices, edges=tuple(sorted(edges)))


def _check(
    graph: SimpleUndirectedGraph,
    claimed_chromatic_number: int,
    coloring: tuple[int, ...],
    weights: tuple[CanonicalRational, ...],
) -> ChromaticNumberCertificateCheckResult:
    return compute_chromatic_number_certificate_check(
        ChromaticNumberCertificateCheckRequest(
            graph=graph,
            claimed_chromatic_number=claimed_chromatic_number,
            coloring=coloring,
            weights=weights,
        )
    )


def _small_chromatic_oracle(
    order: int,
    edges: tuple[tuple[int, int], ...],
) -> tuple[int, tuple[int, ...]]:
    """Independent exhaustive oracle used only for graphs of order at most four."""
    if order == 0:
        return 0, ()
    for colors in range(1, order + 1):
        for coloring in product(range(colors), repeat=order):
            if all(coloring[left] != coloring[right] for left, right in edges):
                return colors, coloring
    raise AssertionError("every finite graph is colorable with one color per vertex")


def _maximum_clique_oracle(
    order: int,
    edges: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    edge_set = {frozenset(edge) for edge in edges}
    for size in range(order, -1, -1):
        for candidate in combinations(range(order), size):
            if all(
                frozenset((left, right)) in edge_set
                for left, right in combinations(candidate, 2)
            ):
                return candidate
    raise AssertionError("the empty set is a clique")


def test_k23_exact_certificate() -> None:
    left = ("a0", "a1")
    right = ("b0", "b1", "b2")
    vertices = left + right
    graph = _graph(
        vertices,
        tuple((a, b) for a in left for b in right),
    )

    result = _check(
        graph,
        2,
        (0, 0, 1, 1, 1),
        (_rational(1, 2), _rational(1, 2)) + (_rational(1, 3),) * 3,
    )

    assert result.verdict == "ACCEPTED"
    assert result.reason == "ACCEPTED"
    assert result.weight_sum == _rational(2)
    assert result.certified_lower_bound == 2


def test_campbell_sixteen_face_conflict_graph_certificate() -> None:
    # Campbell, arXiv:2608.06863v1, Theorem 2 gives these facets and proves
    # that face 236 meets every other face while no independent class has
    # more than two faces: https://arxiv.org/abs/2608.06863v1
    faces = (
        "123",
        "124",
        "135",
        "146",
        "156",
        "236",
        "245",
        "257",
        "268",
        "278",
        "345",
        "348",
        "367",
        "378",
        "468",
        "567",
    )
    graph = _graph(
        faces,
        tuple(
            (left, right) if left < right else (right, left)
            for left, right in combinations(faces, 2)
            if set(left) & set(right)
        ),
    )
    color_by_face = {
        "123": 0,
        "567": 0,
        "124": 1,
        "378": 1,
        "135": 2,
        "468": 2,
        "146": 3,
        "278": 3,
        "156": 4,
        "348": 4,
        "236": 5,
        "245": 6,
        "367": 6,
        "257": 7,
        "268": 8,
        "345": 8,
    }

    result = _check(
        graph,
        9,
        tuple(color_by_face[face] for face in faces),
        tuple(_rational(1) if face == "236" else _rational(1, 2) for face in faces),
    )

    assert result.verdict == "ACCEPTED"
    assert result.weight_sum == _rational(17, 2)
    assert result.certified_lower_bound == 9


def test_all_graphs_through_order_four_match_independent_oracle() -> None:
    for order in range(5):
        integer_edges = tuple(combinations(range(order), 2))
        for edge_mask in range(1 << len(integer_edges)):
            selected_edges = tuple(
                edge
                for index, edge in enumerate(integer_edges)
                if edge_mask & (1 << index)
            )
            optimum, coloring = _small_chromatic_oracle(order, selected_edges)
            clique = _maximum_clique_oracle(order, selected_edges)
            # Every graph through order four is perfect; assert that the
            # independent oracle really supplied a matching lower witness.
            assert len(clique) == optimum
            vertices = tuple(str(index) for index in range(order))
            graph = _graph(
                vertices,
                tuple((str(left), str(right)) for left, right in selected_edges),
            )
            weights = tuple(
                _rational(1 if vertex in clique else 0) for vertex in range(order)
            )

            result = _check(graph, optimum, coloring, weights)

            assert result.verdict == "ACCEPTED"
            assert result.certified_lower_bound == optimum


def test_coloring_alone_cannot_establish_optimality() -> None:
    vertices = tuple(str(index) for index in range(5))
    graph = _graph(
        vertices,
        (("0", "1"), ("0", "4"), ("1", "2"), ("2", "3"), ("3", "4")),
    )

    result = _check(graph, 3, (0, 1, 0, 1, 2), (_rational(0),) * 5)

    assert result.verdict == "REJECTED"
    assert result.reason == "LOWER_BOUND_BELOW_CLAIM"
    assert result.certified_lower_bound == 0

    certified = _check(
        graph,
        3,
        (0, 1, 0, 1, 2),
        (_rational(1, 2),) * 5,
    )
    assert certified.verdict == "ACCEPTED"
    assert certified.weight_sum == _rational(5, 2)
    assert certified.certified_lower_bound == 3


def test_overweight_independent_set_returns_first_gray_code_witness() -> None:
    graph = _graph(("c", "a", "b"), ())

    result = _check(
        graph,
        1,
        (0, 0, 0),
        (_rational(1), _rational(1), _rational(1)),
    )

    assert result.verdict == "REJECTED"
    assert result.reason == "INDEPENDENT_SET_OVERWEIGHT"
    assert result.blocking_independent_set == ("c", "a")
    assert result.blocking_independent_set_weight == _rational(2)


@pytest.mark.parametrize(
    ("claimed", "coloring", "weights", "reason", "blocking_vertex"),
    [
        (0, (0,), (_rational(1),), "CLAIM_OUT_OF_RANGE", None),
        (2, (0,), (_rational(1),), "CLAIM_OUT_OF_RANGE", None),
        (1, (1,), (_rational(1),), "COLOR_OUT_OF_PALETTE", "v"),
        (1, (0,), (_rational(-1),), "NEGATIVE_WEIGHT", "v"),
    ],
)
def test_invalid_mathematical_evidence_returns_typed_rejection(
    claimed: int,
    coloring: tuple[int, ...],
    weights: tuple[CanonicalRational, ...],
    reason: str,
    blocking_vertex: str | None,
) -> None:
    result = _check(_graph(("v",), ()), claimed, coloring, weights)

    assert result.verdict == "REJECTED"
    assert result.reason == reason
    assert result.blocking_vertex == blocking_vertex


def test_monochromatic_edge_returns_canonical_edge() -> None:
    graph = _graph(("c", "a", "b"), (("a", "b"), ("b", "c")))

    result = _check(
        graph,
        2,
        (0, 1, 1),
        (_rational(0), _rational(1), _rational(1)),
    )

    assert result.verdict == "REJECTED"
    assert result.reason == "MONOCHROMATIC_EDGE"
    assert result.blocking_edge == ("a", "b")


def test_empty_and_edgeless_semantics() -> None:
    empty = _check(_graph((), ()), 0, (), ())
    assert empty.verdict == "ACCEPTED"
    assert empty.certified_lower_bound == 0

    edgeless = _check(
        _graph(("a", "b", "c"), ()),
        1,
        (0, 0, 0),
        (_rational(1, 3),) * 3,
    )
    assert edgeless.verdict == "ACCEPTED"
    assert edgeless.certified_lower_bound == 1


def test_vertex_axis_controls_coloring_and_weight_alignment() -> None:
    graph = _graph(("c", "a", "b"), (("a", "b"), ("b", "c")))

    result = _check(
        graph,
        2,
        (0, 0, 1),
        (_rational(0), _rational(1), _rational(1)),
    )

    assert result.verdict == "ACCEPTED"
    assert result.graph.vertices == ("c", "a", "b")
    assert result.coloring == (0, 0, 1)
    assert result.weights == (_rational(0), _rational(1), _rational(1))


def _accepted_edge_result() -> ChromaticNumberCertificateCheckResult:
    return _check(
        _graph(("a", "b"), (("a", "b"),)),
        2,
        (0, 1),
        (_rational(1), _rational(1)),
    )


@pytest.mark.parametrize(
    "field", ["graph", "claimed_chromatic_number", "coloring", "weights"]
)
def test_explicit_verifier_rejects_forged_sources(field: str) -> None:
    payload = deepcopy(_accepted_edge_result().model_dump(mode="json"))
    if field == "graph":
        payload["graph"]["edges"] = []
    elif field == "claimed_chromatic_number":
        payload[field] = 1
    elif field == "coloring":
        payload[field] = [0, 0]
    else:
        payload[field] = [
            {"num": "0", "den": "1"},
            {"num": "0", "den": "1"},
        ]

    assert not verify_chromatic_number_certificate_check_result(
        ChromaticNumberCertificateCheckResult.model_validate(payload)
    )


def test_explicit_verifier_rejects_forged_witness_and_conclusion() -> None:
    rejected = _check(
        _graph(("a", "b"), ()),
        1,
        (0, 0),
        (_rational(1), _rational(1)),
    )
    payload = deepcopy(rejected.model_dump(mode="json"))
    payload["blocking_independent_set"] = ["b"]
    assert not verify_chromatic_number_certificate_check_result(
        ChromaticNumberCertificateCheckResult.model_validate(payload)
    )

    payload = deepcopy(rejected.model_dump(mode="json"))
    payload["verdict"] = "ACCEPTED"
    payload["reason"] = "ACCEPTED"
    assert not verify_chromatic_number_certificate_check_result(
        ChromaticNumberCertificateCheckResult.model_validate(payload)
    )


def test_producer_evaluates_the_certificate_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ChromaticNumberCertificateCheckRequest(
        graph=_graph(("a", "b"), (("a", "b"),)),
        claimed_chromatic_number=2,
        coloring=(0, 1),
        weights=(_rational(1), _rational(1)),
    )
    evaluator_name = "_evaluate_chromatic_number_certificate"
    original = getattr(_operations, evaluator_name)
    calls = 0

    def counted(*args: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(_operations, "_evaluate_chromatic_number_certificate", counted)
    result = compute_chromatic_number_certificate_check(request)

    assert result.verdict == "ACCEPTED"
    assert calls == 1


def test_result_preflights_oversized_forged_derived_rationals() -> None:
    oversized = {
        "num": "1" * (MAX_CHROMATIC_CERTIFICATE_DERIVED_RATIONAL_DIGITS + 1),
        "den": "1",
    }

    payload = deepcopy(_accepted_edge_result().model_dump(mode="json"))
    payload["weight_sum"] = oversized
    with pytest.raises(ValidationError):
        ChromaticNumberCertificateCheckResult.model_validate(payload)

    rejected = _check(
        _graph(("a", "b"), ()),
        1,
        (0, 0),
        (_rational(1), _rational(1)),
    )
    payload = deepcopy(rejected.model_dump(mode="json"))
    payload["blocking_independent_set_weight"] = oversized
    with pytest.raises(ValidationError):
        ChromaticNumberCertificateCheckResult.model_validate(payload)


def test_vertex_and_subset_enumeration_boundaries() -> None:
    order = MAX_CHROMATIC_CERTIFICATE_VERTICES
    vertices = tuple(f"v{index:02d}" for index in range(order))
    graph = _graph(vertices, tuple(combinations(vertices, 2)))
    assert len(graph.edges) == MAX_CHROMATIC_CERTIFICATE_EDGES
    result = _check(
        graph,
        order,
        tuple(range(order)),
        (_rational(1),) * order,
    )
    assert result.verdict == "ACCEPTED"

    oversized_graph = _graph(
        tuple(f"v{index:02d}" for index in range(order + 1)),
        (),
    )
    with pytest.raises(ValidationError) as error:
        ChromaticNumberCertificateCheckRequest(
            graph=oversized_graph,
            claimed_chromatic_number=1,
            coloring=(0,) * order,
            weights=(_rational(1, order),) * order,
        )
    assert (
        error.value.errors()[0]["type"]
        == "graph.chromatic_number_certificate_checking_supports_at_most"
    )


def test_rational_digit_and_total_work_boundaries() -> None:
    maximum_denominator = 10 ** (MAX_CHROMATIC_CERTIFICATE_RATIONAL_DIGITS - 1)
    accepted = _check(
        _graph(("v",), ()),
        1,
        (0,),
        (_rational(1, maximum_denominator),),
    )
    assert accepted.verdict == "ACCEPTED"

    with pytest.raises(ValidationError):
        ChromaticNumberCertificateCheckRequest.model_validate(
            {
                "graph": {"vertices": ["v"], "edges": []},
                "claimed_chromatic_number": 1,
                "coloring": [0],
                "weights": [
                    {
                        "num": "1",
                        "den": str(maximum_denominator * 10),
                    }
                ],
            }
        )

    denominator_base = 10**39
    accepted_order = MAX_CHROMATIC_CERTIFICATE_VERTICES - 1
    accepted_work = _check(
        _graph(tuple(f"a{index:02d}" for index in range(accepted_order)), ()),
        1,
        (0,) * accepted_order,
        tuple(
            _rational(1, denominator_base + index) for index in range(accepted_order)
        ),
    )
    assert accepted_work.verdict == "ACCEPTED"

    rejected_order = MAX_CHROMATIC_CERTIFICATE_VERTICES
    with pytest.raises(ValidationError) as error:
        ChromaticNumberCertificateCheckRequest(
            graph=_graph(tuple(f"r{index:02d}" for index in range(rejected_order)), ()),
            claimed_chromatic_number=1,
            coloring=(0,) * rejected_order,
            weights=tuple(
                _rational(1, denominator_base + index)
                for index in range(rejected_order)
            ),
        )
    assert (
        error.value.errors()[0]["type"]
        == "graph.chromatic_number_certificate_exact_replay_work_exceeds"
    )


def test_retained_source_output_headroom_boundary() -> None:
    limit = CanonicalLimits().max_output_bytes
    admitted_label = "a" * (limit // 2 - 8192)
    admitted = _check(
        _graph((admitted_label,), ()),
        1,
        (0,),
        (_rational(2),),
    )
    assert admitted.reason == "INDEPENDENT_SET_OVERWEIGHT"
    assert len(encode_strict_json(admitted.model_dump(mode="json"))) <= limit

    rejected_label = "b" * (limit // 2)
    with pytest.raises(ValidationError):
        ChromaticNumberCertificateCheckRequest(
            graph=_graph((rejected_label,), ()),
            claimed_chromatic_number=1,
            coloring=(0,),
            weights=(_rational(2),),
        )


def test_schema_and_tool_expose_bounds_axis_and_example() -> None:
    from jacobian.math.graphs.coloring._tools import TOOLS

    schema = ChromaticNumberCertificateCheckRequest.model_json_schema()
    graph_schema = schema["properties"]["graph"]
    assert (
        graph_schema["properties"]["vertices"]["maxItems"]
        == MAX_CHROMATIC_CERTIFICATE_VERTICES
    )
    assert (
        graph_schema["properties"]["edges"]["maxItems"]
        == MAX_CHROMATIC_CERTIFICATE_EDGES
    )
    assert (
        schema["properties"]["coloring"]["maxItems"]
        == MAX_CHROMATIC_CERTIFICATE_VERTICES
    )
    assert (
        schema["properties"]["weights"]["maxItems"]
        == MAX_CHROMATIC_CERTIFICATE_VERTICES
    )
    assert "graph.vertices order" in schema["properties"]["coloring"]["description"]
    assert "independent set" in schema["properties"]["weights"]["description"]
    result_schema = ChromaticNumberCertificateCheckResult.model_json_schema()
    assert (
        result_schema["properties"]["graph"]["properties"]["vertices"]["maxItems"]
        == MAX_CHROMATIC_CERTIFICATE_VERTICES
    )
    assert (
        "does not by itself refute"
        in result_schema["properties"]["verdict"]["description"]
    )

    tool = next(
        operation
        for operation in TOOLS
        if operation.operation_id == "graph.coloring.chromatic_number.check"
    )
    assert len(tool.description) <= 512
    assert tool.examples
    request = tool.request_type.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert result.verdict == "ACCEPTED"


def test_float_weight_is_not_part_of_the_exact_contract() -> None:
    with pytest.raises(ValidationError):
        ChromaticNumberCertificateCheckRequest.model_validate(
            {
                "graph": {"vertices": ["a"], "edges": []},
                "claimed_chromatic_number": 1,
                "coloring": [0],
                "weights": [0.5],
            }
        )
