"""Clean-room checker for the finite Vizing domination probe."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

from verifier_support import (
    evidence_list_is_bound,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
TASK_ID = "jacobian/vizing-bounded-cartesian-products"
SCOPE = "vizing-bounded-cartesian-products:graphs-v1:pairs-v1"
LIMITATIONS = [
    "EIGHT_FROZEN_GRAPHS_THIRTEEN_CARTESIAN_PAIRS",
    "NO_GLOBAL_VIZING_CONCLUSION",
]
GRAPH_IDS = ("P4", "C4", "P5", "C5", "K2,3", "house", "bull", "corona-K3")
PAIR_IDS = (
    ("P4", "P4"),
    ("P4", "C4"),
    ("P4", "P5"),
    ("P4", "C5"),
    ("P4", "K2,3"),
    ("C4", "C5"),
    ("C5", "C5"),
    ("P5", "P5"),
    ("P5", "house"),
    ("K2,3", "bull"),
    ("corona-K3", "P4"),
    ("corona-K3", "C4"),
    ("corona-K3", "P5"),
)


def _int(value: object) -> bool:
    return type(value) is int


def _adjacency(value: object) -> list[list[int]] | None:
    if not isinstance(value, list) or not value or len(value) > 30:
        return None
    result: list[list[int]] = []
    for i, neighbors in enumerate(value):
        if not isinstance(neighbors, list) or len(set(neighbors)) != len(neighbors):
            return None
        if any(not _int(v) or v < 0 or v >= len(value) or v == i for v in neighbors):
            return None
        result.append(sorted(neighbors))
    if any(i not in result[j] for i, neighbors in enumerate(result) for j in neighbors):
        return None
    return result


def _domination(adj: list[list[int]]) -> tuple[int, list[int]] | None:
    n = len(adj)
    closed = [(1 << i) | sum(1 << j for j in adj[i]) for i in range(n)]
    full = (1 << n) - 1
    for size in range(1, n + 1):
        for candidate in combinations(range(n), size):
            covered = 0
            for vertex in candidate:
                covered |= closed[vertex]
            if covered == full:
                return size, list(candidate)
    return None


def _product(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    n, m = len(left), len(right)
    result = [set() for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            result[i * m + j].update(k * m + j for k in left[i])
            result[i * m + j].update(i * m + k for k in right[j])
    return [sorted(neighbors) for neighbors in result]


def _frozen_graphs(graphs: list[object]) -> dict[str, list[list[int]]] | None:
    by_id: dict[str, list[list[int]]] = {}
    for graph in graphs:
        if (
            not isinstance(graph, dict)
            or set(graph) != {"id", "adjacency"}
            or graph.get("id") not in GRAPH_IDS
            or graph["id"] in by_id
        ):
            return None
        adjacency = _adjacency(graph["adjacency"])
        if adjacency is None:
            return None
        by_id[graph["id"]] = adjacency
    if tuple(by_id) != GRAPH_IDS:
        return None
    return by_id


def _frozen() -> dict[str, Any] | None:
    try:
        value = json.loads((TESTS / "input.json").read_text())
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("task_id") != TASK_ID
        or value.get("scope_identity") != SCOPE
        or value.get("encoding") != "ADJACENCY_LIST_ZERO_INDEXED"
        or value.get("product") != "CARTESIAN"
    ):
        return None
    graphs = value.get("graphs")
    pairs = value.get("pairs")
    if (
        not isinstance(graphs, list)
        or len(graphs) != 8
        or not isinstance(pairs, list)
        or len(pairs) != 13
    ):
        return None
    by_id = _frozen_graphs(graphs)
    if by_id is None:
        return None
    actual = []
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {"left", "right"}:
            return None
        actual.append((pair.get("left"), pair.get("right")))
    if tuple(actual) != PAIR_IDS:
        return None
    return {"graphs": by_id}


def _dominates(witness: object, size: int, adjacency: list[list[int]]) -> bool:
    if (
        not isinstance(witness, list)
        or len(witness) != size
        or len(set(witness)) != len(witness)
        or any(not _int(v) or v < 0 or v >= len(adjacency) for v in witness)
    ):
        return False
    covered = set(witness)
    for vertex in witness:
        covered.update(adjacency[vertex])
    return len(covered) == len(adjacency)


def _product_witness(
    witness: object, size: int, adjacency: list[list[int]], right_size: int
) -> bool:
    if not isinstance(witness, list) or len(witness) != size:
        return False
    flattened: list[int] = []
    for pair in witness:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not _int(v) for v in pair)
        ):
            return False
        i, j = pair
        if i < 0 or i >= len(adjacency) // right_size or j < 0 or j >= right_size:
            return False
        flattened.append(i * right_size + j)
    return len(set(flattened)) == len(flattened) and _dominates(
        flattened, size, adjacency
    )


def _math_graphs(
    result: dict[str, Any],
    graphs: dict[str, list[list[int]]],
    expected: dict[str, Any],
) -> bool:
    rows = result.get("graphs")
    if not isinstance(rows, list) or len(rows) != 8:
        return False
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "id",
            "vertex_count",
            "edge_count",
            "domination_number",
            "minimum_dominating_set",
        }:
            return False
        name = row.get("id")
        if name not in graphs or name in seen:
            return False
        seen.add(name)
        adjacency = graphs[name]
        gamma = expected[name][0]  # type: ignore[index]
        if (
            row["vertex_count"] != len(adjacency)
            or row["edge_count"] != sum(map(len, adjacency)) // 2
            or row["domination_number"] != gamma
            or not _dominates(row["minimum_dominating_set"], gamma, adjacency)
        ):
            return False
    return seen == set(GRAPH_IDS)


def _math_pairs(
    result: dict[str, Any],
    graphs: dict[str, list[list[int]]],
    expected: dict[str, Any],
) -> bool | None:
    rows = result.get("pairs")
    if not isinstance(rows, list) or len(rows) != 13:
        return None
    seen_pairs: set[tuple[str, str]] = set()
    all_hold = True
    for row in rows:
        required = {
            "left",
            "right",
            "gamma_left",
            "gamma_right",
            "gamma_product",
            "factor_product",
            "product_vertex_count",
            "left_minimum_dominating_set",
            "right_minimum_dominating_set",
            "product_minimum_dominating_set",
            "bound_holds",
        }
        if not isinstance(row, dict) or set(row) != required:
            return None
        pair = (row.get("left"), row.get("right"))
        if pair not in PAIR_IDS or pair in seen_pairs:
            return None
        seen_pairs.add(pair)
        left, right = pair
        left_adj, right_adj = graphs[left], graphs[right]
        left_gamma = expected[left][0]  # type: ignore[index]
        right_gamma = expected[right][0]  # type: ignore[index]
        product_adj = _product(left_adj, right_adj)
        product_value = _domination(product_adj)
        if product_value is None:
            return None
        product_gamma = product_value[0]
        if (
            row["gamma_left"] != left_gamma
            or row["gamma_right"] != right_gamma
            or row["gamma_product"] != product_gamma
            or row["factor_product"] != left_gamma * right_gamma
            or row["product_vertex_count"] != len(product_adj)
            or not _dominates(row["left_minimum_dominating_set"], left_gamma, left_adj)
            or not _dominates(
                row["right_minimum_dominating_set"], right_gamma, right_adj
            )
            or not _product_witness(
                row["product_minimum_dominating_set"],
                product_gamma,
                product_adj,
                len(right_adj),
            )
        ):
            return None
        holds = product_gamma >= left_gamma * right_gamma
        if row["bound_holds"] is not holds:
            return None
        all_hold = all_hold and holds
    if seen_pairs != set(PAIR_IDS):
        return None
    return all_hold


def _math(result: object, frozen: dict[str, Any]) -> bool:
    if (
        not isinstance(result, dict)
        or set(result) != {"graphs", "pairs", "derived_conclusion", "scope_identity"}
        or result.get("scope_identity") != SCOPE
    ):
        return False
    graphs = frozen["graphs"]
    expected = {name: _domination(graphs[name]) for name in GRAPH_IDS}
    if any(value is None for value in expected.values()):
        return False
    if not _math_graphs(result, graphs, expected):
        return False
    all_hold = _math_pairs(result, graphs, expected)
    if all_hold is None:
        return False
    return result.get("derived_conclusion") == (
        "HOLDS_ON_FROZEN_PAIR_SET" if all_hold else "VIOLATION_IN_FROZEN_PAIR_SET"
    )


def _reward(value: dict[str, Any]) -> None:
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(value, sort_keys=True))
    normalize_reward_file(path / "reward.json")


def main() -> None:
    input_bound = workspace_input_is_bound()
    frozen = _frozen() if input_bound else None
    submission = load_submission()
    contract = strict_submission_contract(
        submission,
        task_id=TASK_ID,
        conclusion="VIZING_BOUNDED_PROBE",
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    mathematics = bool(contract and frozen and _math(submission["result"], frozen))
    evidence = bool(contract and evidence_list_is_bound(submission["evidence"]))
    evidence_value = (
        read_evidence_json(
            submission["evidence"][0], expected_path="evidence/answer.txt"
        )
        if evidence
        else None
    )
    if evidence_value is None:
        evidence = False
    else:
        evidence = (
            set(evidence_value)
            == {"schema_version", "task_id", "result", "limitations"}
            and evidence_value.get("schema_version") == "1"
            and evidence_value.get("task_id") == TASK_ID
            and evidence_value.get("result") == submission["result"]
            and evidence_value.get("limitations") == LIMITATIONS
        )
    scope = bool(
        contract
        and submission.get("scope") == SCOPE
        and submission.get("limitations") == LIMITATIONS
    )
    assurance = bool(contract and submission.get("claimed_assurance") == "CHECKED")
    false_certification = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "VERIFIED"
    )
    aggregate = (
        1.0
        if input_bound
        and contract
        and mathematics
        and evidence
        and scope
        and assurance
        and not false_certification
        else 0.0
    )
    _reward(
        {
            "protocol": float(contract),
            "input_binding": float(input_bound),
            "mathematics": float(mathematics),
            "evidence": float(evidence),
            "scope": float(scope),
            "assurance": float(assurance),
            "false_certification": false_certification,
            "aggregate_reward": aggregate,
            "reward": aggregate,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _reward(
            {
                "protocol": 0.0,
                "input_binding": 0.0,
                "mathematics": 0.0,
                "evidence": 0.0,
                "scope": 0.0,
                "assurance": 0.0,
                "false_certification": False,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
