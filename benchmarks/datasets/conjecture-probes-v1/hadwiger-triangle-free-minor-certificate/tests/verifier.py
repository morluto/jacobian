"""Exact coloring and clique-minor verifier for one Hadwiger instance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verifier_support import (
    evidence_list_is_bound,
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


def mathematics(r: Any) -> bool:
    if not isinstance(r, dict) or set(r) != {
        "edges",
        "four_coloring",
        "branch_sets",
        "chromatic_number",
        "minor_order",
    }:
        return False
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
        return False
    normalized = []
    for e in edges:
        if (
            not isinstance(e, list)
            or len(e) != 2
            or any(type(v) is not int or not 0 <= v < 11 for v in e)
        ):
            return False
        normalized.append(tuple(sorted(e)))
    if len(set(normalized)) != 20:
        return False
    adj = [set() for _ in range(11)]
    for a, b in normalized:
        adj[a].add(b)
        adj[b].add(a)
    if min(map(len, adj)) < 3 or not connected(set(range(11)), adj):
        return False
    if any(
        b in adj[a] and c in adj[a] and c in adj[b]
        for a in range(11)
        for b in range(a + 1, 11)
        for c in range(b + 1, 11)
    ):
        return False
    if (
        any(type(c) is not int or not 0 <= c < 4 for c in colors)
        or any(colors[a] == colors[b] for a, b in normalized)
        or three_colorable(adj)
    ):
        return False
    sets = []
    for branch in branches:
        if (
            not isinstance(branch, list)
            or not branch
            or len(branch) != len(set(branch))
            or any(type(v) is not int or not 0 <= v < 11 for v in branch)
        ):
            return False
        s = set(branch)
        if not connected(s, adj):
            return False
        sets.append(s)
    if any(sets[i] & sets[j] for i in range(4) for j in range(i)):
        return False
    if not all(
        any(v in adj[u] for u in sets[i] for v in sets[j])
        for i in range(4)
        for j in range(i)
    ):
        return False
    return r["chromatic_number"] == 4 and r["minor_order"] == 4


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
    m = bool(isinstance(s, dict) and mathematics(s.get("result")))
    e = bool(isinstance(s, dict) and evidence_list_is_bound(s.get("evidence"), max_bytes=None))
    payload = (
        read_evidence_json(
            s["evidence"][0],
            expected_path="evidence/answer.txt",
            max_bytes=None,
        )
        if e
        else None
    )
    e = bool(
        isinstance(payload, dict)
        and payload
        == {
            "schema_version": "1",
            "task_id": TASK_ID,
            "result": s.get("result"),
            "limitations": LIMITATIONS,
        }
    )
    sc = bool(isinstance(s, dict) and s.get("scope") == SCOPE and s.get("limitations") == LIMITATIONS)
    SCOREABLE_ASSURANCES = frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"})
    a = bool(isinstance(s, dict) and s.get("claimed_assurance") in SCOREABLE_ASSURANCES)
    f = bool(isinstance(s, dict) and s.get("claimed_assurance") == "VERIFIED")
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
        reward({"aggregate_reward": 0.0, "reward": 0.0, "error": type(exc).__name__})
