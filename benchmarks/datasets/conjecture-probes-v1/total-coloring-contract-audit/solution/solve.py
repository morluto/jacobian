from __future__ import annotations

import json
from pathlib import Path

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


def collisions(vertices: list[int], edges: list[int]) -> list[dict[str, int]]:
    return [
        {"vertex": v, "edge_index": i}
        for i, (u, w) in enumerate(EDGES)
        for v in (u, w)
        if vertices[v] == edges[i]
    ]


def main() -> None:
    root = Path("/app")
    vertex_colors = [0, 1, 0, 1, 2, 1, 2, 3, 3, 1]
    repaired_edges = [2, 3, 2, 3, 1, 0, 2, 3, 1, 2, 3, 0, 1, 0, 0]
    flawed_edges = [(c + 1) % 4 for c in repaired_edges]
    result = {
        "flawed_pass": {"vertex_colors": vertex_colors, "edge_colors": flawed_edges},
        "incidence_collisions": collisions(vertex_colors, flawed_edges),
        "repair": {"vertex_colors": vertex_colors, "edge_colors": repaired_edges},
    }
    submission = {"result": result}
    (root / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
