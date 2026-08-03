from __future__ import annotations

import json
import re
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
MAX_EVIDENCE_BYTES = 64 * 1024
_AFFIRMATIVE_PROOF_ASSISTANT = re.compile(
    r"\b(?:verif(?:y|ied|ies)|formaliz(?:e|ed|es)|prove(?:s|d)|certif(?:y|ied|ies))\b"
    r"[^.;\n]{0,80}\b(?:proof assistant|lean)\b|"
    r"\b(?:proof assistant|lean)\b[^.;\n]{0,80}"
    r"\b(?:verif(?:y|ied|ies)|formaliz(?:e|ed|es)|prove(?:s|d)|certif(?:y|ied|ies))\b",
    re.I,
)
_NEGATED_PROOF_ASSISTANT = re.compile(
    r"\b(?:not|no|without|doesn['']?t|cannot|never)\b[^.;\n]{0,30}"
    r"\b(?:proof assistant|lean)\b",
    re.I,
)


def _evidence_denies_proof_assistant(text: str) -> bool:
    """Reject evidence that affirmatively claims proof-assistant verification.

    A clause with an affirmative proof-assistant claim is only safe when the
    negation appears within the same clause and within a small window of the
    proof-assistant phrase, so an unrelated negation elsewhere in the clause
    does not rescue a prohibited certification assertion.
    """

    folded = text.casefold()
    if "proof assistant" not in folded and "lean" not in folded:
        return True
    for clause in re.split(r"[.;\n]", folded):
        if _AFFIRMATIVE_PROOF_ASSISTANT.search(clause) and not (
            _NEGATED_PROOF_ASSISTANT.search(clause)
        ):
            return False
    return True


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
        if path.stat().st_size > MAX_EVIDENCE_BYTES:
            return False
        text = path.read_text()
        marker = [
            line[12:] for line in text.splitlines() if line.startswith("RESULT_JSON:")
        ]
        bound = json.loads(marker[0]) if len(marker) == 1 else None
    except (OSError, UnicodeError, ValueError, RecursionError):
        return False
    return (
        bound == result
        and all(word in text.casefold() for word in ("cycle", "determinant", "z/2z"))
        and _evidence_denies_proof_assistant(text)
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
    envelope = isinstance(submission, dict)
    mathematical = bool(
        envelope and _source_is_bound() and _result(submission.get("result"))
    )
    evidence_valid = bool(
        envelope and _evidence(submission.get("evidence"), submission.get("result"))
    )
    scope_accurate = bool(
        protocol
        and envelope
        and submission.get("scope") == SCOPE
        and submission.get("completeness") == "COMPLETE"
        and submission.get("limitations") == [LIMITATION]
    )
    assurance_calibrated = bool(
        envelope
        and submission.get("claimed_assurance") == "COMPUTED"
        and not false_certification
    )
    correct_score = 1.0 if mathematical else 0.0
    evidence_score = 1.0 if evidence_valid else 0.0
    scope_score = 1.0 if scope_accurate else 0.0
    assurance_score = 1.0 if assurance_calibrated else 0.0
    reward = (
        min(correct_score, evidence_score, scope_score, assurance_score)
        if protocol
        else 0.0
    )
    return {
        "correctness": correct_score,
        "evidence_validity": evidence_score,
        "scope_accuracy": scope_score,
        "assurance_calibration": assurance_score,
        "reward": reward,
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
