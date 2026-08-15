import json
from pathlib import Path

from verifier_support import (
    load_submission as load_strict_submission,
)
from verifier_support import (
    normalize_reward_file,
    witness_list_is_bound,
)

W = Path("/app")
E = Path("/tests")


def load_submission():
    return load_strict_submission()


def evidence(s):
    return bool(s and witness_list_is_bound(s.get("witness")))


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
    valid = isinstance(s, dict) and isinstance(s.get("result"), dict)
    ev = evidence(s) if valid else False
    math_correct = bool(valid and graph_ok(s["result"]))
    reward = float(math_correct and ev)
    (Path("/logs/verifier")).mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "witness_validity": float(ev),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
