"""Clean-room verifier shared by the multi-tool-coordination-v1 pilot."""

from __future__ import annotations

import json
from collections import deque
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    false_verified_claim,
    load_submission,
    load_submission_raw,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
MAX_EVIDENCE_BYTES = 256 * 1024


def _frozen() -> dict:
    try:
        value = json.loads((TESTS / "input.json").read_text())
    except (OSError, ValueError, RecursionError, MemoryError):
        return {}
    return value if isinstance(value, dict) else {}


def _rational(value: object) -> Fraction | None:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        return None
    numerator, denominator = value["num"], value["den"]
    if (
        not isinstance(numerator, str)
        or not isinstance(denominator, str)
        or not numerator.lstrip("-").isdigit()
        or not denominator.isdigit()
    ):
        return None
    try:
        parsed = Fraction(int(numerator), int(denominator))
    except (ValueError, ZeroDivisionError):
        return None
    if parsed.denominator <= 0:
        return None
    if str(parsed.numerator) != numerator or str(parsed.denominator) != denominator:
        return None
    return parsed


def _determinant(values: list[list[Fraction | int]]) -> Fraction:
    if not values or any(len(row) != len(values) for row in values):
        return Fraction()
    matrix = [[Fraction(item) for item in row] for row in values]
    result = Fraction(1)
    for column in range(len(matrix)):
        pivot = next(
            (row for row in range(column, len(matrix)) if matrix[row][column]),
            None,
        )
        if pivot is None:
            return Fraction()
        if pivot != column:
            matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
            result = -result
        pivot_value = matrix[column][column]
        result *= pivot_value
        for row in range(column + 1, len(matrix)):
            scale = matrix[row][column] / pivot_value
            for item in range(column, len(matrix)):
                matrix[row][item] -= scale * matrix[column][item]
    return result


def _graph_result(result: object, frozen: dict) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "maximum_degree_vertices",
        "distance_to_set",
        "maximum_distance_to_set",
        "maximizing_vertices",
    }:
        return False
    vertices = frozen.get("vertices")
    edges = frozen.get("edges")
    if (
        not isinstance(vertices, list)
        or len(vertices) != len(set(vertices))
        or not all(isinstance(vertex, str) for vertex in vertices)
        or not isinstance(edges, list)
    ):
        return False
    adjacency = {vertex: set() for vertex in vertices}
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or edge[0] not in adjacency
            or edge[1] not in adjacency
            or edge[0] == edge[1]
        ):
            return False
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])
    maximum_degree = max(map(len, adjacency.values()))
    maximum_vertices = sorted(
        vertex for vertex in vertices if len(adjacency[vertex]) == maximum_degree
    )
    distances = dict.fromkeys(maximum_vertices, 0)
    queue = deque(maximum_vertices)
    while queue:
        vertex = queue.popleft()
        for neighbor in adjacency[vertex]:
            if neighbor not in distances:
                distances[neighbor] = distances[vertex] + 1
                queue.append(neighbor)
    if len(distances) != len(vertices):
        return False
    maximum_distance = max(distances.values())
    expected = {
        "maximum_degree_vertices": maximum_vertices,
        "distance_to_set": [
            {"vertex": vertex, "distance": distances[vertex]}
            for vertex in sorted(vertices)
        ],
        "maximum_distance_to_set": maximum_distance,
        "maximizing_vertices": sorted(
            vertex for vertex in vertices if distances[vertex] == maximum_distance
        ),
    }
    return result == expected


def _edge(value: object) -> tuple[int, int] | None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int for item in value)
        or value[0] >= value[1]
    ):
        return None
    return value[0], value[1]


def _tree_is_valid(tree: list[tuple[int, int]], vertices: set[int]) -> bool:
    if len(tree) != len(vertices) - 1 or len(set(tree)) != len(tree):
        return False
    parent = {vertex: vertex for vertex in vertices}

    def root(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for left, right in tree:
        a, b = root(left), root(right)
        if a == b:
            return False
        parent[a] = b
    return len({root(vertex) for vertex in vertices}) == 1


def _cycle_complex(
    frozen: dict,
) -> tuple[set[int], set[tuple[int, int, int]], set[tuple[int, int]]] | None:
    vertices_raw, facets_raw = frozen.get("vertices"), frozen.get("facets")
    if (
        not isinstance(vertices_raw, list)
        or not all(type(vertex) is int for vertex in vertices_raw)
        or not isinstance(facets_raw, list)
        or any(
            not isinstance(facet, list)
            or len(facet) != 3
            or any(type(vertex) is not int for vertex in facet)
            for facet in facets_raw
        )
    ):
        return None
    vertices = set(vertices_raw)
    facets = {tuple(facet) for facet in facets_raw}
    if len(facets) != 10:
        return None
    edges = {edge for facet in facets for edge in combinations(sorted(facet), 2)}
    return vertices, facets, edges


def _cycle_witness_edges(
    result: dict, vertices: set[int], edges: set[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    tree_raw = result["spanning_tree"]
    non_tree_raw = result["non_tree_edges"]
    if not isinstance(tree_raw, list) or not isinstance(non_tree_raw, list):
        return None
    parsed = [_edge(item) for item in tree_raw + non_tree_raw]
    if any(item is None for item in parsed):
        return None
    tree = [item for item in parsed[: len(tree_raw)] if item is not None]
    non_tree = [item for item in parsed[len(tree_raw) :] if item is not None]
    if (
        not _tree_is_valid(tree, vertices)
        or set(tree) | set(non_tree) != edges
        or set(tree) & set(non_tree)
        or len(set(non_tree)) != 10
    ):
        return None
    return tree, non_tree


def _cycle_facet_order(
    value: object, facets: set[tuple[int, int, int]]
) -> list[tuple[int, int, int]] | None:
    if (
        not isinstance(value, list)
        or len(value) != 10
        or any(
            not isinstance(facet, list)
            or len(facet) != 3
            or any(type(vertex) is not int for vertex in facet)
            for facet in value
        )
    ):
        return None
    ordered = [tuple(facet) for facet in value]
    return ordered if set(ordered) == facets and len(set(ordered)) == 10 else None


def _cycle_lattice_result(result: object, frozen: dict) -> bool:
    fields = {
        "spanning_tree",
        "non_tree_edges",
        "facet_order",
        "cycle_coordinate_matrix",
        "determinant",
        "homology",
    }
    if not isinstance(result, dict) or set(result) != fields:
        return False
    complex_data = _cycle_complex(frozen)
    if complex_data is None:
        return False
    vertices, facets, edges = complex_data
    witness_edges = _cycle_witness_edges(result, vertices, edges)
    if witness_edges is None:
        return False
    _tree, non_tree = witness_edges
    ordered_facets = _cycle_facet_order(result["facet_order"], facets)
    if ordered_facets is None:
        return False
    expected = []
    for edge in non_tree:
        row = []
        for a, b, c in ordered_facets:
            boundary = {(b, c): 1, (a, c): -1, (a, b): 1}
            row.append(boundary.get(edge, 0))
        expected.append(row)
    matrix = result["cycle_coordinate_matrix"]
    transposed = [list(column) for column in zip(*expected, strict=True)]
    if matrix != expected and matrix != transposed:
        return False
    determinant = _determinant(expected)
    return bool(
        determinant.denominator == 1
        and result["determinant"] == determinant.numerator
        and abs(determinant) == 2
        and result["homology"] == "Z/2Z"
    )


def _farkas_result(result: object, frozen: dict) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "scalar_replay",
        "proof_mode",
        "positive_definite_certificate",
    }:
        return False
    replay = result["scalar_replay"]
    certificate = result["positive_definite_certificate"]
    if (
        not isinstance(replay, dict)
        or set(replay) != {"y0", "c00_y", "m00", "objective"}
        or result["proof_mode"] != "SYLVESTER"
        or not isinstance(certificate, dict)
        or set(certificate) != {"leading_principal_determinants"}
    ):
        return False
    parsed = {key: _rational(value) for key, value in replay.items()}
    scalar_inputs = frozen.get("scalar_inputs")
    if not isinstance(scalar_inputs, dict):
        return False
    expected_scalars = {key: _rational(value) for key, value in scalar_inputs.items()}
    matrix_raw = frozen.get("matrix")
    if not isinstance(matrix_raw, list) or not matrix_raw:
        return False
    matrix = [[_rational(value) for value in row] for row in matrix_raw]
    if any(value is None for row in matrix for value in row):
        return False
    exact_matrix = [[value for value in row if value is not None] for row in matrix]
    expected_determinants = [
        _determinant([row[:size] for row in exact_matrix[:size]])
        for size in range(1, len(exact_matrix) + 1)
    ]
    submitted = certificate["leading_principal_determinants"]
    if not isinstance(submitted, list):
        return False
    determinants = [_rational(value) for value in submitted]
    return bool(
        all(
            value is not None
            for value in [*parsed.values(), *expected_scalars.values()]
        )
        and parsed["y0"] == expected_scalars["y0"]
        and parsed["c00_y"] == expected_scalars["c00_y"]
        and parsed["objective"] == expected_scalars["objective"]
        and parsed["m00"] == parsed["y0"] + parsed["c00_y"]
        and parsed["m00"] < 0
        and parsed["objective"] > 0
        and determinants == expected_determinants
        and all(value is not None and value > 0 for value in determinants)
    )


def _proportionality_result(result: object, frozen: dict) -> bool:
    fields = {
        "k",
        "c",
        "p",
        "q",
        "center",
        "radius",
        "circle_coefficients",
        "distance_coefficients",
        "multiplier",
        "relation",
    }
    if not isinstance(result, dict) or set(result) != fields:
        return False
    values = {
        key: _rational(result[key])
        for key in ("k", "c", "p", "q", "center", "radius", "multiplier")
    }
    k, c = values["k"], values["c"]
    if k is None or c is None or k <= 0 or k == 1 or c <= 0:
        return False
    if _rational(frozen.get("k")) != k or _rational(frozen.get("c")) != c:
        return False
    circle_raw = result["circle_coefficients"]
    distance_raw = result["distance_coefficients"]
    if not isinstance(circle_raw, list) or not isinstance(distance_raw, list):
        return False
    circle = [_rational(value) for value in circle_raw]
    distance = [_rational(value) for value in distance_raw]
    if any(value is None for value in circle + distance):
        return False
    p = k * c / (k + 1)
    q = k * c / (k - 1)
    center = (p + q) / 2
    radius = abs(q - p) / 2
    expected_circle = [
        Fraction(1),
        Fraction(1),
        -2 * center,
        center * center - radius * radius,
    ]
    expected_distance = [1 - k * k, 1 - k * k, 2 * k * k * c, -k * k * c * c]
    multiplier = 1 - k * k
    return bool(
        result["relation"] == "DISTANCE_EQUALS_MULTIPLIER_TIMES_CIRCLE"
        and values["p"] == p
        and values["q"] == q
        and values["center"] == center
        and values["radius"] == radius
        and values["multiplier"] == multiplier
        and circle == expected_circle
        and distance == expected_distance
        and distance == [multiplier * value for value in circle]
    )


def _mathematics(result: object, frozen: dict) -> bool:
    family = frozen.get("family")
    if family == "graph-set-distance":
        return _graph_result(result, frozen)
    if family == "cycle-lattice":
        return _cycle_lattice_result(result, frozen)
    if family == "rational-slice-binding":
        return _farkas_result(result, frozen)
    if family == "directed-proportionality":
        return _proportionality_result(result, frozen)
    return False


def _evidence(submission: object) -> bool:
    if not isinstance(submission, dict):
        return False
    descriptors = submission.get("evidence")
    if not isinstance(descriptors, list) or len(descriptors) != 1:
        return False
    evidence = read_evidence_json(
        descriptors[0],
        expected_path="evidence/certificate.json",
        max_bytes=MAX_EVIDENCE_BYTES,
    )
    return bool(
        isinstance(evidence, dict)
        and set(evidence)
        == {
            "schema_version",
            "task_id",
            "result",
            "scope",
            "completeness",
            "limitations",
        }
        and evidence["schema_version"] == "1"
        and evidence["task_id"] == submission.get("task_id")
        and evidence["result"] == submission.get("result")
        and evidence["scope"] == submission.get("scope")
        and evidence["completeness"] == submission.get("completeness")
        and evidence["limitations"] == submission.get("limitations")
    )


def _evaluate() -> dict[str, float | bool]:
    frozen = _frozen()
    submission = load_submission()
    raw = load_submission_raw()
    protocol = strict_submission_contract(
        submission,
        task_id=str(frozen.get("task_id")),
        conclusion=str(frozen.get("conclusion")),
        allowed_assurances=frozenset({"COMPUTED"}),
        verification_record="forbidden",
    )
    input_binding = workspace_input_is_bound()
    mathematical = bool(
        input_binding
        and isinstance(submission, dict)
        and _mathematics(submission.get("result"), frozen)
    )
    evidence = _evidence(submission)
    scope = bool(
        protocol
        and isinstance(submission, dict)
        and submission.get("scope") == frozen.get("scope")
        and submission.get("completeness") == "COMPLETE"
        and submission.get("limitations") == frozen.get("limitations")
    )
    false_certification = false_verified_claim(raw, verification_record_bound=False)
    assurance = bool(
        protocol
        and isinstance(submission, dict)
        and submission.get("claimed_assurance") == "COMPUTED"
        and not false_certification
    )
    reward = aggregate_reward(
        correctness=mathematical,
        evidence_validity=evidence,
        scope_accuracy=scope,
        assurance_calibration=assurance,
        false_certification=false_certification,
        protocol_ok=bool(protocol and input_binding),
    )
    return {
        "correctness": float(mathematical),
        "evidence_validity": float(evidence),
        "input_binding": float(input_binding),
        "scope_accuracy": float(scope),
        "completeness_accuracy": float(scope),
        "assurance_calibration": float(assurance),
        "protocol_compliance": float(protocol),
        "false_certification": false_certification,
        "reward": reward,
    }


def main() -> None:
    output = Path("/logs/verifier")
    output.mkdir(parents=True, exist_ok=True)
    details = _evaluate()
    reward = details.pop("reward")
    (output / "reward.json").write_text(
        json.dumps({"reward": reward}, sort_keys=True) + "\n"
    )
    (output / "reward-details.json").write_text(
        json.dumps(details, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
