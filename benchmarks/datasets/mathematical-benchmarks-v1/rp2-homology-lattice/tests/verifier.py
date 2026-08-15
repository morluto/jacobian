from __future__ import annotations

import json
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")


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


def _graph_edges(
    value: dict,
    vertices: set[int],
    edges: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    tree_raw = [_edge(item) for item in value["spanning_tree"]]
    non_tree_raw = [_edge(item) for item in value["non_tree_edges"]]
    if any(item is None for item in tree_raw + non_tree_raw):
        return None
    tree = [item for item in tree_raw if item is not None]
    non_tree = [item for item in non_tree_raw if item is not None]
    if any(endpoint not in vertices for edge in tree + non_tree for endpoint in edge):
        return None
    if (
        not _tree_is_valid(tree, vertices)
        or set(tree) | set(non_tree) != edges
        or set(tree) & set(non_tree)
        or len(set(non_tree)) != 10
    ):
        return None
    return tree, non_tree


def _ordered_facets(
    value: dict,
    facets: set[tuple[int, int, int]],
) -> list[tuple[int, int, int]] | None:
    ordered: list[tuple[int, int, int]] = []
    for item in value["facet_order"]:
        if (
            not isinstance(item, list)
            or len(item) != 3
            or any(type(entry) is not int for entry in item)
        ):
            return None
        ordered.append(tuple(item))
    if set(ordered) != facets or len(set(ordered)) != 10:
        return None
    return ordered


def _coordinate_matrix(
    value: dict,
    non_tree: list[tuple[int, int]],
    ordered_facets: list[tuple[int, int, int]],
) -> list[list[int]] | None:
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
    ):
        return None
    transposed = [list(column) for column in zip(*expected, strict=True)]
    if matrix != expected and matrix != transposed:
        return None
    return expected


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
    graph = _graph_edges(value, vertices, edges)
    if graph is None:
        return False
    _tree, non_tree = graph
    ordered = _ordered_facets(value, facets)
    if ordered is None:
        return False
    expected = _coordinate_matrix(value, non_tree, ordered)
    if expected is None:
        return False
    determinant = _determinant(expected)
    return bool(
        value["determinant"] == determinant
        and abs(determinant) == 2
        and value["homology"] == "Z/2Z"
    )


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    input_bound = workspace_input_is_bound()
    result = data.get("result")
    math_correct = bool(
        isinstance(submission, dict) and input_bound and _result(result)
    )
    correct = math_correct
    destination = Path("/logs/verifier/reward.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": float(correct),
            }
        )
        + "\n"
    )
    normalize_reward_file(destination)


if __name__ == "__main__":
    main()
