from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
from tests.component.checkers.exact_domain_checker_support import (
    _GRAPH_CASES,
    _request,
)
from tests.support.artifacts import canonical_digest as _digest
from tests.support.rationals import rational_payload as _q

from jacobian_checkers.graph_exact_operations import (
    check_graph_diameter,
    check_graph_distance_matrix,
    check_graph_induced_tree_maximum,
    check_graph_minimum_spanning_tree,
    check_graph_radius,
)


def _minimum_spanning_tree_checker_request() -> dict[str, Any]:
    return copy.deepcopy(
        next(
            checker_request
            for checker, checker_request in _GRAPH_CASES
            if checker is check_graph_minimum_spanning_tree
        )
    )


def _distance_matrix_checker_request(
    *,
    vertices: list[str],
    edges: list[list[str]],
    result_vertices: list[str],
    distances: list[list[int | None]],
    connected: bool,
) -> dict[str, Any]:
    return _request(
        "graph.distance_matrix.compute",
        "graph.distance-matrix.all-sources-bfs-v1",
        {
            "graph": {
                "graph_schema_version": "1",
                "vertices": vertices,
                "edges": edges,
            }
        },
        {
            "semantics_version": "unweighted-shortest-path-distance-matrix.v1",
            "vertex_ordering": "LEXICOGRAPHIC_ASCENDING",
            "pair_coverage": "ALL_ORDERED_VERTEX_PAIRS",
            "unreachable_representation": "JSON_NULL",
            "vertices": result_vertices,
            "distances": distances,
            "connected": connected,
        },
    )


def test_graph_checker_reports_its_actual_exhaustive_replay_method() -> None:
    checker, checker_request = next(
        case for case in _GRAPH_CASES if case[0] is check_graph_induced_tree_maximum
    )

    decision = checker(checker_request)

    assert decision["accepted"] is True
    assert decision["detail"] == (
        "independent finite-subset exhaustive replay accepted "
        "graph.induced_tree.maximum.compute"
    )
    assert "FLINT" not in decision["detail"]


def test_minimum_spanning_tree_checker_rejects_incomplete_cycle_coverage() -> None:
    checker_request = _minimum_spanning_tree_checker_request()
    checker_request["candidate"]["payload"]["optimality_certificate"]["checks"] = []
    checker_request["candidate"]["payload_digest"] = _digest(
        checker_request["candidate"]["payload"]
    )

    decision = check_graph_minimum_spanning_tree(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_minimum_spanning_tree_checker_rejects_a_nonminimum_tree() -> None:
    checker_request = _minimum_spanning_tree_checker_request()
    result = checker_request["candidate"]["payload"]
    result["tree_edges"] = [
        {"endpoints": ["a", "b"], "weight": _q(1)},
        {"endpoints": ["a", "c"], "weight": _q(4)},
    ]
    result["total_weight"] = _q(5)
    result["optimality_certificate"]["checks"] = [
        {
            "non_tree_edge": ["b", "c"],
            "edge_weight": _q(2),
            "tree_path_vertices": ["b", "a", "c"],
            "maximum_tree_path_weight": _q(4),
            "condition": "EDGE_WEIGHT_GTE_MAXIMUM_TREE_PATH_WEIGHT",
        }
    ]
    checker_request["candidate"]["payload_digest"] = _digest(result)

    decision = check_graph_minimum_spanning_tree(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_minimum_spanning_tree_checker_rejects_oversized_source_weight() -> None:
    checker_request = _minimum_spanning_tree_checker_request()
    checker_request["claim"]["payload"]["graph"]["edges"][0]["weight"] = {
        "num": "1" * 257,
        "den": "1",
    }
    checker_request["claim"]["payload_digest"] = _digest(
        checker_request["claim"]["payload"]
    )

    decision = check_graph_minimum_spanning_tree(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


@pytest.mark.parametrize(
    ("checker", "operation_id", "witness_format", "result"),
    (
        (
            check_graph_diameter,
            "graph.invariant.diameter.compute",
            "graph.diameter.all-sources-bfs-v1",
            {
                "status": "NOT_APPLICABLE",
                "diameter": None,
                "connected": False,
                "exactness": "NOT_APPLICABLE",
                "detail": "diameter requires a nonempty connected graph",
            },
        ),
        (
            check_graph_radius,
            "graph.invariant.radius.compute",
            "graph.radius.all-sources-bfs-v1",
            {
                "status": "NOT_APPLICABLE",
                "radius": None,
                "connected": False,
                "exactness": "NOT_APPLICABLE",
                "detail": "radius requires a nonempty connected graph",
            },
        ),
    ),
)
@pytest.mark.parametrize(
    "graph",
    (
        {
            "graph_schema_version": "1",
            "vertices": [],
            "edges": [],
        },
        {
            "graph_schema_version": "1",
            "vertices": ["a", "b", "c"],
            "edges": [["a", "b"]],
        },
    ),
    ids=("empty", "disconnected"),
)
def test_graph_metric_checker_accepts_exact_inapplicable_boundary(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    operation_id: str,
    witness_format: str,
    result: dict[str, Any],
    graph: dict[str, Any],
) -> None:
    checker_request = _request(
        operation_id,
        witness_format,
        {"graph": graph},
        result,
    )

    decision = checker(checker_request)

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


@pytest.mark.parametrize(
    ("checker", "operation_id", "witness_format", "field"),
    (
        (
            check_graph_diameter,
            "graph.invariant.diameter.compute",
            "graph.diameter.all-sources-bfs-v1",
            "diameter",
        ),
        (
            check_graph_radius,
            "graph.invariant.radius.compute",
            "graph.radius.all-sources-bfs-v1",
            "radius",
        ),
    ),
)
def test_graph_metric_checker_accepts_singleton_zero(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    operation_id: str,
    witness_format: str,
    field: str,
) -> None:
    checker_request = _request(
        operation_id,
        witness_format,
        {
            "graph": {
                "graph_schema_version": "1",
                "vertices": ["only"],
                "edges": [],
            }
        },
        {
            "status": "COMPUTED",
            field: 0,
            "connected": True,
            "exactness": "EXACT",
            "detail": None,
        },
    )

    assert checker(checker_request)["accepted"] is True


@pytest.mark.parametrize(
    "checker_request",
    (
        _distance_matrix_checker_request(
            vertices=[],
            edges=[],
            result_vertices=[],
            distances=[],
            connected=False,
        ),
        _distance_matrix_checker_request(
            vertices=["only"],
            edges=[],
            result_vertices=["only"],
            distances=[[0]],
            connected=True,
        ),
        _distance_matrix_checker_request(
            vertices=["c", "a", "b"],
            edges=[["a", "b"]],
            result_vertices=["a", "b", "c"],
            distances=[[0, 1, None], [1, 0, None], [None, None, 0]],
            connected=False,
        ),
    ),
    ids=("empty", "singleton", "disconnected"),
)
def test_distance_matrix_checker_accepts_exact_boundary_claims(
    checker_request: dict[str, Any],
) -> None:
    decision = check_graph_distance_matrix(checker_request)

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"
    assert decision["method"] == "EXHAUSTIVE_FINITE"
    assert decision["coverage"] == "EXHAUSTIVE"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda result: result.update(vertices=["b", "a", "c"]),
        lambda result: result.update(
            distances=[[0, 1], [1, 0]],
        ),
        lambda result: result["distances"][0].__setitem__(0, 1),
        lambda result: result["distances"][0].__setitem__(1, 0),
        lambda result: result["distances"][0].__setitem__(2, 1),
        lambda result: result["distances"][2].__setitem__(0, 1),
        lambda result: result.update(
            distances=[[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        ),
        lambda result: result.update(connected=False),
        lambda result: result.update(
            semantics_version="unweighted-shortest-path-distance-matrix.v2"
        ),
        lambda result: result.update(extra="forged"),
        lambda result: result["distances"][0].__setitem__(1, True),
    ),
    ids=(
        "wrong-order",
        "wrong-shape",
        "diagonal",
        "off-diagonal-zero",
        "asymmetric-left",
        "asymmetric-right",
        "wrong-shortest-paths",
        "wrong-connectedness",
        "wrong-semantics",
        "extra-field",
        "boolean-distance",
    ),
)
def test_distance_matrix_checker_rejects_false_certification_paths(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    checker_request = _distance_matrix_checker_request(
        vertices=["c", "a", "b"],
        edges=[["a", "b"], ["b", "c"]],
        result_vertices=["a", "b", "c"],
        distances=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        connected=True,
    )
    result = checker_request["candidate"]["payload"]
    mutate(result)
    checker_request["candidate"]["payload_digest"] = _digest(result)

    decision = check_graph_distance_matrix(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


@pytest.mark.parametrize(
    ("checker", "operation_id", "witness_format", "field"),
    (
        (
            check_graph_diameter,
            "graph.invariant.diameter.compute",
            "graph.diameter.all-sources-bfs-v1",
            "diameter",
        ),
        (
            check_graph_radius,
            "graph.invariant.radius.compute",
            "graph.radius.all-sources-bfs-v1",
            "radius",
        ),
    ),
)
@pytest.mark.parametrize(
    "mutation",
    (
        lambda result, field: result.update({field: 0}),
        lambda result, field: result.update(status="NOT_APPLICABLE"),
        lambda result, field: result.update(connected=False),
        lambda result, field: result.update(exactness="NOT_APPLICABLE"),
        lambda result, field: result.update(detail="forged"),
    ),
    ids=(
        "wrong-value",
        "wrong-status",
        "wrong-connectivity",
        "wrong-exactness",
        "detail",
    ),
)
def test_graph_metric_checker_rejects_forged_connected_result(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    operation_id: str,
    witness_format: str,
    field: str,
    mutation: Callable[[dict[str, Any], str], object],
) -> None:
    checker_request = _request(
        operation_id,
        witness_format,
        {
            "graph": {
                "graph_schema_version": "1",
                "vertices": ["a", "b", "c", "d"],
                "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
            }
        },
        {
            "status": "COMPUTED",
            field: 3 if field == "diameter" else 2,
            "connected": True,
            "exactness": "EXACT",
            "detail": None,
        },
    )
    mutation(checker_request["candidate"]["payload"], field)
    checker_request["candidate"]["payload_digest"] = _digest(
        checker_request["candidate"]["payload"]
    )

    decision = checker(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda result: result.update(
            {
                "maximum_matching_cardinality": 0,
                "witness_edges": [],
                "certificate": {
                    **result["certificate"],
                    "upper_bound": 0,
                },
            }
        ),
        lambda result: result.update(witness_edges=[["x", "y"]]),
        lambda result: result["certificate"].update(barrier_vertices=["outside"]),
        lambda result: result["certificate"].update(odd_component_count=1),
        lambda result: result["certificate"].update(upper_bound=0),
    ),
)
def test_maximum_matching_checker_rejects_false_or_rebound_certificates(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    checker, checker_request = _GRAPH_CASES[-1]
    adversarial = copy.deepcopy(checker_request)
    mutate(adversarial["candidate"]["payload"])
    adversarial["candidate"]["payload_digest"] = _digest(
        adversarial["candidate"]["payload"]
    )

    decision = checker(adversarial)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_maximum_matching_checker_reports_tutte_berge_replay() -> None:
    checker, checker_request = _GRAPH_CASES[-1]

    decision = checker(checker_request)

    assert decision["accepted"] is True
    assert decision["detail"] == (
        "independent Tutte-Berge barrier replay accepted "
        "graph.invariant.maximum_matching.compute"
    )
    assert decision["arithmetic"] == "EXACT_INTEGER"
