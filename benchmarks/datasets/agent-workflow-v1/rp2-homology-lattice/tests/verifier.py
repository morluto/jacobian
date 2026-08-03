from __future__ import annotations

import json
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
TASK_ID = "jacobian/rp2-homology-lattice"
CONCLUSION = "INTEGRAL_H1_CERTIFIED"
SCOPE = "the frozen six-vertex ten-facet simplicial complex"
LIMITATION = (
    "This does not prove in a proof assistant that the geometric realization is "
    "the real projective plane."
)


def _input() -> dict:
    return json.loads((TESTS / "input.json").read_text())


def _edge(value: object) -> tuple[int, int] | None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int for item in value)
        or value[0] >= value[1]
    ):
        return None
    return (value[0], value[1])


def _determinant(values: list[list[int]]) -> int:
    matrix = [[Fraction(item) for item in row] for row in values]
    result = Fraction(1)
    for column in range(len(matrix)):
        pivot = next(
            (row for row in range(column, len(matrix)) if matrix[row][column]),
            None,
        )
        if pivot is None:
            return 0
        if pivot != column:
            matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
            result = -result
        value = matrix[column][column]
        result *= value
        for row in range(column + 1, len(matrix)):
            multiplier = matrix[row][column] / value
            for index in range(column, len(matrix)):
                matrix[row][index] -= multiplier * matrix[column][index]
    return int(result) if result.denominator == 1 else 0


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


def _result(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "spanning_tree",
        "non_tree_edges",
        "facet_order",
        "cycle_coordinate_matrix",
        "determinant",
        "homology",
    }:
        return False
    if not all(
        isinstance(value[field], list)
        for field in (
            "spanning_tree",
            "non_tree_edges",
            "facet_order",
            "cycle_coordinate_matrix",
        )
    ):
        return False
    data = _input()
    vertices = set(data["vertices"])
    facets = {tuple(facet) for facet in data["facets"]}
    edges = {edge for facet in facets for edge in combinations(tuple(sorted(facet)), 2)}
    tree_raw = [_edge(item) for item in value["spanning_tree"]]
    non_tree_raw = [_edge(item) for item in value["non_tree_edges"]]
    if any(item is None for item in tree_raw + non_tree_raw):
        return False
    tree = [item for item in tree_raw if item is not None]
    non_tree = [item for item in non_tree_raw if item is not None]
    if (
        not _tree_is_valid(tree, vertices)
        or set(tree) | set(non_tree) != edges
        or set(tree) & set(non_tree)
        or len(set(non_tree)) != 10
    ):
        return False
    ordered_facets: list[tuple[int, int, int]] = []
    for item in value["facet_order"]:
        if (
            not isinstance(item, list)
            or len(item) != 3
            or any(type(entry) is not int for entry in item)
        ):
            return False
        ordered_facets.append(tuple(item))
    if set(ordered_facets) != facets or len(set(ordered_facets)) != 10:
        return False
    expected: list[list[int]] = []
    for edge in non_tree:
        row = []
        for a, b, c in ordered_facets:
            boundary = {(b, c): 1, (a, c): -1, (a, b): 1}
            row.append(boundary.get(edge, 0))
        expected.append(row)
    matrix = value["cycle_coordinate_matrix"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != 10
        or any(
            not isinstance(row, list)
            or len(row) != 10
            or any(type(item) is not int for item in row)
            for row in matrix
        )
        or matrix != expected
    ):
        return False
    determinant = _determinant(expected)
    return bool(
        value["determinant"] == determinant
        and abs(determinant) == 2
        and value["homology"] == "Z/2Z"
    )


def _source_is_bound() -> bool:
    try:
        hidden = (TESTS / "input.json").read_bytes()
        data = json.loads(hidden)
        return bool(
            (WORKSPACE / "input.json").read_bytes() == hidden
            and data["source"]["revision"] == "8ecee59e510093bf96360177c52825b8e0603e59"
        )
    except (OSError, ValueError, KeyError):
        return False


def _evidence(value: object, result: object) -> bool:
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not evidence_list_is_bound(value)
    ):
        return False
    path = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    try:
        text = path.read_text()
        marker = [
            line[12:] for line in text.splitlines() if line.startswith("RESULT_JSON:")
        ]
        bound = json.loads(marker[0]) if len(marker) == 1 else None
    except (OSError, UnicodeError, ValueError, RecursionError):
        return False
    return bound == result and all(
        word in text.casefold() for word in ("cycle", "determinant", "index", "z/2z")
    )


def _evaluate(submission: object) -> dict[str, float | bool]:
    protocol = strict_submission_contract(
        submission,
        task_id=TASK_ID,
        conclusion=CONCLUSION,
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        protocol
        and isinstance(submission, dict)
        and _source_is_bound()
        and submission.get("scope") == SCOPE
        and submission.get("completeness") == "COMPLETE"
        and submission.get("claimed_assurance") == "COMPUTED"
        and _result(submission.get("result"))
        and _evidence(submission.get("evidence"), submission.get("result"))
        and submission.get("limitations") == [LIMITATION]
    )
    score = 1.0 if correct and not false_certification else 0.0
    return {
        "correctness": score,
        "evidence_validity": score,
        "scope_accuracy": score,
        "assurance_calibration": score,
        "reward": score,
        "false_certification": false_certification,
    }


def main() -> None:
    destination = Path("/logs/verifier/reward.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_evaluate(load_submission()), sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
