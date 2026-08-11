import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    evidence_list_is_bound,
    normalize_reward_file,
)
from verifier_support import (
    load_submission as load_strict_submission,
)

W = Path("/app")
E = Path("/tests")


def load_submission():
    return load_strict_submission()


def contract(s, expected):
    required = {
        "task_id",
        "conclusion",
        "result",
        "claimed_assurance",
        "scope",
        "completeness",
        "evidence",
        "limitations",
    }
    return (
        isinstance(s, dict)
        and set(s) == required
        and s["task_id"] == expected["task_id"]
        and s["conclusion"] == expected["conclusion"]
        and isinstance(s["result"], dict)
        and isinstance(s["claimed_assurance"], str)
        and isinstance(s["claimed_assurance"], str)
        and s["claimed_assurance"] in {"UNVERIFIED", "COMPUTED", "CHECKED", "VERIFIED"}
        and isinstance(s["scope"], str)
        and s["completeness"] == "COMPLETE"
        and isinstance(s["evidence"], list)
        and isinstance(s["limitations"], list)
    )


def evidence(s):
    return bool(s and evidence_list_is_bound(s.get("evidence")))


def _graph_arrays(result):
    raw_vertices = result.get("vertices")
    edges = result.get("edges")
    if (
        set(result) == {"vertices", "edges"}
        and isinstance(raw_vertices, list)
        and isinstance(edges, list)
    ):
        return raw_vertices, edges
    return [], []


def _build_adjacency(vertices, edges):
    adj = {v: set() for v in vertices}
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            return None
        a, b = map(str, edge)
        if a == b or {a, b} - vertices:
            return None
        a, b = sorted((a, b))
        if b in adj[a]:
            return None
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _is_connected(vertices, adj):
    seen, todo = set(), ["0"]
    while todo:
        v = todo.pop()
        if v in seen:
            continue
        seen.add(v)
        todo.extend(adj[v] - seen)
    return seen == vertices


def _has_triangle(vertices, adj):
    return any(c in adj[a] for a in vertices for b in adj[a] for c in adj[b] if c != a)


def _is_non_bipartite(vertices, adj):
    colors = {}
    for start in vertices:
        if start in colors:
            continue
        colors[start] = 0
        todo = [start]
        while todo:
            v = todo.pop()
            for n in adj[v]:
                if n in colors and colors[n] == colors[v]:
                    return True
                if n not in colors:
                    colors[n] = 1 - colors[v]
                    todo.append(n)
    return False


def graph_ok(result):
    raw_vertices, edges = _graph_arrays(result)
    vertices = {str(x) for x in raw_vertices}
    if vertices != {str(i) for i in range(6)}:
        return False
    adj = _build_adjacency(vertices, edges)
    if adj is None:
        return False
    if min(map(len, adj.values())) < 2:
        return False
    if not _is_connected(vertices, adj):
        return False
    if _has_triangle(vertices, adj):
        return False
    return _is_non_bipartite(vertices, adj)


def main():
    s = load_submission()
    expected = json.loads((E / "expected.json").read_text())
    valid = contract(s, expected)
    scope = bool(valid and s["scope"] == " ".join(expected["required_scope_terms"]))
    assurance = bool(valid and s["claimed_assurance"] == expected["maximum_assurance"])
    ev = evidence(s) if valid else False
    claimed_verified = isinstance(s, dict) and s.get("claimed_assurance") == "VERIFIED"
    math_correct = bool(valid and graph_ok(s["result"]))
    correct = bool(math_correct and not claimed_verified)
    false_cert = claimed_verified
    reward = aggregate_reward(
        correctness=correct,
        evidence_validity=ev,
        scope_accuracy=scope,
        assurance_calibration=assurance,
        false_certification=false_cert,
        soft_assurance=True,
    )
    (Path("/logs/verifier")).mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(ev),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false_cert,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
