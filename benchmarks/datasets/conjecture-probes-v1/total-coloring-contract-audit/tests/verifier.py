from __future__ import annotations

import json
from pathlib import Path

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    evidence_list_is_bound,
    is_regular_bounded_file,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/total-coloring-contract-audit"
SCOPE = "petersen-total-coloring-contract-v1"
LIMITATIONS = [
    "ONE_PETERSEN_GRAPH_INSTANCE",
    "PROJECTED_VALIDATOR_DEFECT_ONLY",
    "TOTAL_COLORING_CONJECTURE_NOT_ASSESSED",
]
VERTICES = list(range(10))
EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 4),
    (5, 7),
    (7, 9),
    (6, 9),
    (6, 8),
    (5, 8),
    (0, 5),
    (1, 6),
    (2, 7),
    (3, 8),
    (4, 9),
]


def _proper_vertices(colors: object) -> bool:
    return (
        isinstance(colors, list)
        and len(colors) == 10
        and all(type(c) is int and 0 <= c < 4 for c in colors)
        and all(colors[u] != colors[v] for u, v in EDGES)
    )


def _proper_edges(colors: object) -> bool:
    if (
        not isinstance(colors, list)
        or len(colors) != 15
        or not all(type(c) is int and 0 <= c < 4 for c in colors)
    ):
        return False
    return all(
        colors[i] != colors[j]
        for i, e in enumerate(EDGES)
        for j, f in enumerate(EDGES)
        if i < j and set(e) & set(f)
    )


def _collisions(vertices: list[int], edges: list[int]) -> list[dict[str, int]]:
    return [
        {"vertex": v, "edge_index": i}
        for i, (u, w) in enumerate(EDGES)
        for v in (u, w)
        if vertices[v] == edges[i]
    ]


def _collision_set(value: object) -> set[tuple[int, int]] | None:
    if not isinstance(value, list):
        return None
    rows: set[tuple[int, int]] = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {"vertex", "edge_index"}:
            return None
        vertex = row["vertex"]
        edge_index = row["edge_index"]
        if (
            type(vertex) is not int
            or not 0 <= vertex < len(VERTICES)
            or type(edge_index) is not int
            or not 0 <= edge_index < len(EDGES)
        ):
            return None
        rows.add((vertex, edge_index))
    return rows if len(rows) == len(value) else None


def _json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _assignment(value: object, *, require_total: bool) -> bool:
    if not isinstance(value, dict) or set(value) != {"vertex_colors", "edge_colors"}:
        return False
    vertices, edges = value["vertex_colors"], value["edge_colors"]
    if not _proper_vertices(vertices) or not _proper_edges(edges):
        return False
    return not require_total or not _collisions(vertices, edges)


def mathematics(result: object) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "flawed_pass",
        "incidence_collisions",
        "repair",
    }:
        return False
    flawed = result["flawed_pass"]
    repair = result["repair"]
    if not _assignment(flawed, require_total=False) or not _assignment(
        repair, require_total=True
    ):
        return False
    expected = {
        (row["vertex"], row["edge_index"])
        for row in _collisions(flawed["vertex_colors"], flawed["edge_colors"])
    }
    supplied = _collision_set(result["incidence_collisions"])
    return bool(expected and supplied == expected)


def _raw() -> dict | None:
    path = Path("/app/submission.json")
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _write(values: dict) -> None:
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(values, sort_keys=True))


def main() -> None:
    raw = _raw()
    bound = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    contract = strict_submission_contract(
        submission,
        task_id=TASK_ID,
        conclusion="PROJECTED_VALIDATOR_IS_UNSOUND_AND_REPAIRED",
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    math_ok = bool(isinstance(raw, dict) and mathematics(raw.get("result")))
    evidence_ok = bool(
        isinstance(raw, dict)
        and evidence_list_is_bound(raw.get("evidence"), max_bytes=None)
    )
    payload = (
        read_evidence_json(
            raw["evidence"][0], expected_path="evidence/answer.json", max_bytes=None
        )
        if evidence_ok
        else None
    )
    evidence_ok = bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result", "limitations"}
        and payload.get("schema_version") == "1"
        and payload.get("task_id") == TASK_ID
        and _json_equal(payload.get("result"), raw.get("result"))
        and payload.get("limitations") == LIMITATIONS
    )
    scope_ok = bool(
        isinstance(raw, dict)
        and raw.get("scope") == SCOPE
        and raw.get("completeness") == "COMPLETE"
        and raw.get("limitations") == LIMITATIONS
    )
    assurance_ok = bool(
        contract
        and isinstance(raw, dict)
        and raw.get("claimed_assurance") in {"UNVERIFIED", "COMPUTED", "CHECKED"}
    )
    values = {
        "input_binding": float(bound),
        "protocol": float(bool(contract)),
        "mathematics": float(math_ok),
        "correctness": float(math_ok),
        "evidence": float(evidence_ok),
        "evidence_validity": float(evidence_ok),
        "scope": float(scope_ok),
        "scope_accuracy": float(scope_ok),
        "assurance": float(assurance_ok),
        "assurance_calibration": float(assurance_ok),
    }
    reward = float(all(values.values()))
    values.update(
        {
            "aggregate_reward": reward,
            "reward": reward,
            "false_certification": bool(
                isinstance(raw, dict) and raw.get("claimed_assurance") == "VERIFIED"
            ),
        }
    )
    _write(values)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _write(
            {
                "protocol": 0.0,
                "input_binding": 0.0,
                "mathematics": 0.0,
                "correctness": 0.0,
                "evidence": 0.0,
                "evidence_validity": 0.0,
                "scope": 0.0,
                "scope_accuracy": 0.0,
                "assurance": 0.0,
                "assurance_calibration": 0.0,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "false_certification": False,
                "error": type(exc).__name__,
            }
        )
