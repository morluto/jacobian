from __future__ import annotations

import json
from pathlib import Path

from verifier_support import (
    json_value_equal,
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

_json_equal = json_value_equal

TASK_ID = "jacobian/total-coloring-contract-audit"
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
        and all((colors[u] != colors[v] for u, v in EDGES))
    )


def _proper_edges(colors: object) -> bool:
    if (
        not isinstance(colors, list)
        or len(colors) != 15
        or (not all(type(c) is int and 0 <= c < 4 for c in colors))
    ):
        return False
    return all(
        (
            colors[i] != colors[j]
            for i, e in enumerate(EDGES)
            for j, f in enumerate(EDGES)
            if i < j and set(e) & set(f)
        )
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
            or (not 0 <= edge_index < len(EDGES))
        ):
            return None
        rows.add((vertex, edge_index))
    return rows if len(rows) == len(value) else None


def _assignment(value: object, *, require_total: bool) -> bool:
    if not isinstance(value, dict) or set(value) != {"vertex_colors", "edge_colors"}:
        return False
    vertices, edges = (value["vertex_colors"], value["edge_colors"])
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


def _write(values: dict) -> None:
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(values, sort_keys=True))
    normalize_reward_file(path / "reward.json")


def main() -> None:
    input_bound = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    protocol_ok = submission is not None
    math_ok = bool(protocol_ok and mathematics(submission.get("result")))
    reward = float(math_ok)
    _write(
        {
            "protocol_compliance": float(protocol_ok),
            "input_binding": float(input_bound),
            "correctness": float(math_ok),
            "reward": reward,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _write(
            {
                "protocol_compliance": 0.0,
                "input_binding": 0.0,
                "correctness": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
