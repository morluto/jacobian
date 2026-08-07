"""Exact coloring and clique-minor verifier for one Hadwiger instance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    evidence_list_is_bound,
    is_regular_bounded_file,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/hadwiger-triangle-free-minor-certificate"
SCOPE = "triangle-free-11-vertex-hadwiger-instance-v1"
LIMITATIONS = [
    "ONE_TRIANGLE_FREE_11_VERTEX_GRAPH",
    "EXHAUSTIVE_THREE_COLOR_REJECTION",
    "NO_GLOBAL_HADWIGER_CONCLUSION",
]
RESULT_KEYS = {
    "edges",
    "four_coloring",
    "branch_sets",
    "chromatic_number",
    "minor_order",
}


def three_colorable(adj):
    colors = [-1] * 11

    def visit(done):
        if done == 11:
            return True
        uncolored = [v for v in range(11) if colors[v] < 0]
        v = max(uncolored, key=lambda x: sum(colors[y] < 0 for y in adj[x]))
        used = {colors[y] for y in adj[v] if colors[y] >= 0}
        for color in range(3):
            if color not in used:
                colors[v] = color
                if visit(done + 1):
                    return True
        colors[v] = -1
        return False

    return visit(0)


def connected(vertices, adj):
    seen = {next(iter(vertices))}
    stack = list(seen)
    while stack:
        for w in adj[stack.pop()] & vertices:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return seen == vertices


def _result_parts(r: Any):
    if not isinstance(r, dict) or set(r) != RESULT_KEYS:
        return None
    edges = r.get("edges")
    colors = r.get("four_coloring")
    branches = r.get("branch_sets")
    if (
        not isinstance(edges, list)
        or len(edges) != 20
        or not isinstance(colors, list)
        or len(colors) != 11
        or not isinstance(branches, list)
        or len(branches) != 4
    ):
        return None
    return edges, colors, branches


def _normalized_edges(edges):
    normalized = []
    for e in edges:
        if (
            not isinstance(e, list)
            or len(e) != 2
            or any(type(v) is not int or not 0 <= v < 11 for v in e)
        ):
            return None
        normalized.append(tuple(sorted(e)))
    if len(set(normalized)) != 20:
        return None
    return normalized


def _adjacency(normalized):
    adj = [set() for _ in range(11)]
    for a, b in normalized:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _has_triangle(adj) -> bool:
    return any(
        b in adj[a] and c in adj[a] and c in adj[b]
        for a in range(11)
        for b in range(a + 1, 11)
        for c in range(b + 1, 11)
    )


def _graph_is_candidate(adj) -> bool:
    return (
        min(map(len, adj)) >= 3
        and connected(set(range(11)), adj)
        and not _has_triangle(adj)
    )


def _coloring_certifies_four_chromatic(colors, normalized, adj) -> bool:
    return not (
        any(type(c) is not int or not 0 <= c < 4 for c in colors)
        or any(colors[a] == colors[b] for a, b in normalized)
        or three_colorable(adj)
    )


def _connected_branch_sets(branches, adj):
    sets = []
    for branch in branches:
        if (
            not isinstance(branch, list)
            or not branch
            or not all(type(v) is int and 0 <= v < 11 for v in branch)
            or len(branch) != len(set(branch))
        ):
            return None
        s = set(branch)
        if not connected(s, adj):
            return None
        sets.append(s)
    return sets


def _branch_sets_form_k4_minor(sets, adj) -> bool:
    if any(sets[i] & sets[j] for i in range(4) for j in range(i)):
        return False
    return all(
        any(v in adj[u] for u in sets[i] for v in sets[j])
        for i in range(4)
        for j in range(i)
    )


def mathematics(r: Any) -> bool:
    parts = _result_parts(r)
    if parts is None:
        return False
    edges, colors, branches = parts
    normalized = _normalized_edges(edges)
    if normalized is None:
        return False
    adj = _adjacency(normalized)
    if not _graph_is_candidate(adj):
        return False
    if not _coloring_certifies_four_chromatic(colors, normalized, adj):
        return False
    sets = _connected_branch_sets(branches, adj)
    if sets is None or not _branch_sets_form_k4_minor(sets, adj):
        return False
    return (
        type(r.get("chromatic_number")) is int
        and r["chromatic_number"] == 4
        and type(r.get("minor_order")) is int
        and r["minor_order"] == 4
    )


def _raw_submission() -> dict[str, Any] | None:
    path = Path("/app/submission.json")
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def reward(v):
    p = Path("/logs/verifier")
    p.mkdir(parents=True, exist_ok=True)
    (p / "reward.json").write_text(json.dumps(v, sort_keys=True))


def main():
    ib = workspace_input_is_bound()
    s = load_submission(require_input_binding=False)
    c = strict_submission_contract(
        s,
        task_id=TASK_ID,
        conclusion="FINITE_HADWIGER_K4_CERTIFICATE",
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    raw = _raw_submission()
    m = bool(isinstance(raw, dict) and mathematics(raw.get("result")))
    e = bool(
        isinstance(raw, dict)
        and evidence_list_is_bound(raw.get("evidence"), max_bytes=None)
    )
    payload = (
        read_evidence_json(
            raw["evidence"][0],
            expected_path="evidence/answer.txt",
            max_bytes=None,
        )
        if e
        else None
    )
    e = bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result", "limitations"}
        and payload.get("schema_version") == "1"
        and payload.get("task_id") == TASK_ID
        and payload.get("result") == raw.get("result")
        and payload.get("limitations") == LIMITATIONS
    )
    sc = bool(
        isinstance(raw, dict)
        and raw.get("scope") == SCOPE
        and raw.get("limitations") == LIMITATIONS
    )
    scoreable_assurances = frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"})
    a = bool(
        isinstance(raw, dict) and raw.get("claimed_assurance") in scoreable_assurances
    )
    f = bool(isinstance(raw, dict) and raw.get("claimed_assurance") == "VERIFIED")
    agg = 1.0 if all((ib, c, m, e, sc, a)) and not f else 0.0
    reward(
        {
            "protocol": 1.0 if c else 0.0,
            "input_binding": 1.0 if ib else 0.0,
            "mathematics": 1.0 if m else 0.0,
            "evidence": 1.0 if e else 0.0,
            "scope": 1.0 if sc else 0.0,
            "assurance": 1.0 if a else 0.0,
            "false_certification": f,
            "aggregate_reward": agg,
            "reward": agg,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        reward(
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
