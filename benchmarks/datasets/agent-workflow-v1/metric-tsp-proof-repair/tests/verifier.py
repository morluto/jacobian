import json
import re
from collections import Counter
from itertools import combinations, pairwise, permutations
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

W = Path("/app")
E = Path("/tests")
MAX_EVIDENCE_BYTES = 1_048_576


def _load_frozen_input():
    try:
        workspace = W / "input.json"
        frozen = E / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        frozen_bytes = frozen.read_bytes()
        if workspace.read_bytes() != frozen_bytes:
            return {}
        value = json.loads(frozen_bytes)
    except (OSError, ValueError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def evidence_matches_result(evidence, result):
    if not evidence_list_is_bound(evidence):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        if target.stat().st_size > MAX_EVIDENCE_BYTES:
            return False
        lines = target.read_text().splitlines()
        marker = next(
            line.removeprefix("RESULT_JSON:").strip()
            for line in lines
            if line.startswith("RESULT_JSON:")
        )
        return json.loads(marker) == result and any(
            line.strip() and not line.startswith("RESULT_JSON:") for line in lines
        )
    except (OSError, StopIteration, UnicodeError, ValueError):
        return False


def edge_key(left, right):
    return tuple(sorted((left, right)))


def _is_two_approximation_claim(value):
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"[_-]+", " ", " ".join(value.casefold().split()))
    match = re.search(
        r"\b(?:factor\s*(?:of\s*)?2|two|2)\s+approximation\b",
        normalized,
    )
    if match is None:
        return False
    surrounding = normalized[max(0, match.start() - 80) : match.end() + 80]
    return (
        re.search(
            r"\b(?:not|never|cannot|can\s+not|false|invalid|reject|unsupported|"
            r"unproven|does\s+not|doesn\'t|isn\'t)\b",
            surrounding,
        )
        is None
    )


def is_tree(vertices, edges):
    if len(edges) != len(vertices) - 1:
        return False
    parent = {vertex: vertex for vertex in vertices}

    def root(vertex):
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for left, right in edges:
        if left == right or left not in parent or right not in parent:
            return False
        left_root, right_root = root(left), root(right)
        if left_root == right_root:
            return False
        parent[left_root] = right_root
    return len({root(vertex) for vertex in vertices}) == 1


def main():
    submission = load_submission()
    input_data = _load_frozen_input()
    expected = json.loads((E / "expected.json").read_text())
    result = submission.get("result") if isinstance(submission, dict) else None
    result = result if isinstance(result, dict) else {}
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    math_contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="optional",
    )

    vertices = input_data.get("vertices") or []
    matrix = input_data.get("distance_matrix")
    input_contract = (
        input_data.get("task_id") == expected["task_id"]
        and vertices == ["A", "B", "C", "D", "E", "F"]
        and isinstance(matrix, list)
        and len(matrix) == len(vertices)
        and all(isinstance(row, list) and len(row) == len(vertices) for row in matrix)
    )
    index = {vertex: position for position, vertex in enumerate(vertices or [])}

    def weight(left, right):
        return matrix[index[left]][index[right]]

    metric = bool(input_contract)
    if metric:
        metric = all(
            matrix[i][i] == 0
            and all(
                isinstance(matrix[i][j], int)
                and (i == j or matrix[i][j] > 0)
                and matrix[i][j] == matrix[j][i]
                for j in range(len(vertices))
            )
            for i in range(len(vertices))
        ) and all(
            matrix[i][j] <= matrix[i][k] + matrix[k][j]
            for i in range(len(vertices))
            for j in range(len(vertices))
            for k in range(len(vertices))
        )

    raw_edges = result.get("mst_edges")
    edges = []
    edge_shape = isinstance(raw_edges, list)
    if edge_shape:
        for edge in raw_edges:
            if (
                not isinstance(edge, list)
                or len(edge) != 2
                or any(
                    type(vertex) is not str or vertex not in vertices for vertex in edge
                )
            ):
                edge_shape = False
                break
            edges.append(edge_key(edge[0], edge[1]))
    unique_edges = edge_shape and len(set(edges)) == len(edges)
    tree = bool(unique_edges and is_tree(vertices, edges))
    tree_weight = sum(weight(*edge) for edge in edges) if tree and metric else -1

    all_edges = [edge_key(*edge) for edge in combinations(vertices, 2)]
    minimum_tree_weight = (
        min(
            sum(weight(*edge) for edge in candidate)
            for candidate in combinations(all_edges, len(vertices) - 1)
            if is_tree(vertices, candidate)
        )
        if metric
        else -1
    )

    euler = result.get("euler_walk")
    expected_counts = Counter(dict.fromkeys(edges, 2))
    euler_shape = (
        isinstance(euler, list)
        and len(euler) == 11
        and all(type(vertex) is str and vertex in vertices for vertex in euler)
    )
    actual_counts = (
        Counter(edge_key(left, right) for left, right in pairwise(euler))
        if euler_shape
        else Counter()
    )
    euler_valid = bool(
        tree
        and euler_shape
        and euler[0] == euler[-1]
        and actual_counts == expected_counts
    )
    euler_weight = (
        sum(weight(left, right) for left, right in pairwise(euler))
        if euler_valid
        else -1
    )

    shortcut = result.get("shortcut_tour")
    first_visits = []
    if euler_valid:
        for vertex in euler:
            if vertex not in first_visits:
                first_visits.append(vertex)
        first_visits.append(first_visits[0])
    shortcut_valid = (
        isinstance(shortcut, list)
        and shortcut == first_visits
        and len(shortcut) == len(vertices) + 1
    )
    shortcut_weight = (
        sum(weight(left, right) for left, right in pairwise(shortcut))
        if shortcut_valid
        else -1
    )

    optimal_tour = result.get("optimal_tour")
    optimal_valid = bool(
        isinstance(optimal_tour, list)
        and len(optimal_tour) == len(vertices) + 1
        and all(type(vertex) is str and vertex in vertices for vertex in optimal_tour)
        and optimal_tour[0] == optimal_tour[-1]
        and len(set(optimal_tour[:-1])) == len(vertices)
        and set(optimal_tour[:-1]) == set(vertices)
    )
    submitted_optimal_weight = (
        sum(weight(left, right) for left, right in pairwise(optimal_tour))
        if optimal_valid
        else -1
    )
    start = vertices[0] if vertices else None
    exact_optimal_weight = (
        min(
            sum(weight(left, right) for left, right in pairwise(cycle))
            for middle in permutations(vertices[1:])
            for cycle in [(start, *middle, start)]
        )
        if metric
        else -1
    )

    reported = result.get("weights")
    valid = bool(
        math_contract
        and set(result)
        == {
            "flaw_location",
            "invalid_inference",
            "corrected_claim",
            "mst_edges",
            "euler_walk",
            "shortcut_tour",
            "optimal_tour",
            "weights",
        }
        and isinstance(reported, dict)
        and set(reported) == {"mst", "euler", "shortcut", "optimal"}
        and all(type(value) is int for value in reported.values())
        and metric
        and result.get("flaw_location") == "STEP_4"
        and result.get("invalid_inference") == "SHORTCUTTING_PRESERVES_EXACT_COST"
        and _is_two_approximation_claim(result.get("corrected_claim"))
        and tree
        and tree_weight == minimum_tree_weight
        and euler_valid
        and euler_weight == 2 * tree_weight
        and shortcut_valid
        and optimal_valid
        and submitted_optimal_weight == exact_optimal_weight
        and shortcut_weight > exact_optimal_weight
        and shortcut_weight <= euler_weight
        and tree_weight <= exact_optimal_weight
        and shortcut_weight <= 2 * exact_optimal_weight
        and reported
        == {
            "mst": tree_weight,
            "euler": euler_weight,
            "shortcut": shortcut_weight,
            "optimal": exact_optimal_weight,
        }
    )

    math_correct = bool(valid)
    scope = bool(
        contract and submission["scope"] == " ".join(expected["required_scope_terms"])
    )
    correct = bool(contract and math_correct and scope)
    good = bool(
        contract
        and math_correct
        and evidence_matches_result(submission["evidence"], result)
    )
    assurance = bool(
        contract and submission["claimed_assurance"] == expected["maximum_assurance"]
    )
    false = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "VERIFIED"
    )
    reward = (
        0
        if not correct or false
        else 0.7 * correct + 0.1 * good + 0.1 * scope + 0.1 * assurance
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(good),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false,
            }
        )
    )


if __name__ == "__main__":
    main()
