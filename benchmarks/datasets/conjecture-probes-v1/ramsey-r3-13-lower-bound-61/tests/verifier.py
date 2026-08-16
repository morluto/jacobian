"""Fail-closed verifier for the R(3,13) lower-bound graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

VERTEX_COUNT = 60
TARGET = 13
ALL_VERTICES = (1 << VERTEX_COUNT) - 1


def _graph(result: Any) -> list[int] | None:
    if not isinstance(result, dict) or set(result) != {"edges"}:
        return None
    edges = result["edges"]
    if not isinstance(edges, list) or not 1 <= len(edges) <= 1770:
        return None
    normalized: set[tuple[int, int]] = set()
    adjacency = [0] * VERTEX_COUNT
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(type(vertex) is not int for vertex in edge)
            or any(not 0 <= vertex < VERTEX_COUNT for vertex in edge)
            or edge[0] == edge[1]
        ):
            return None
        left, right = sorted(edge)
        if (left, right) in normalized:
            return None
        normalized.add((left, right))
        adjacency[left] |= 1 << right
        adjacency[right] |= 1 << left
    return adjacency


def _triangle_free(adjacency: list[int]) -> bool:
    return all(
        not (adjacency[left] & adjacency[right])
        for left in range(VERTEX_COUNT)
        for right in range(left + 1, VERTEX_COUNT)
        if adjacency[left] & (1 << right)
    )


def _color_order(candidates: int, neighbors: list[int]) -> tuple[list[int], list[int]]:
    order: list[int] = []
    bounds: list[int] = []
    color = 0
    remaining = candidates
    while remaining:
        color += 1
        available = remaining
        while available:
            bit = available & -available
            vertex = bit.bit_length() - 1
            order.append(vertex)
            bounds.append(color)
            remaining ^= bit
            available ^= bit
            available &= ~neighbors[vertex]
    return order, bounds


def _has_clique_of_size(neighbors: list[int], target: int) -> bool:
    def expand(candidates: int, size: int) -> bool:
        if size >= target:
            return True
        order, bounds = _color_order(candidates, neighbors)
        for index in range(len(order) - 1, -1, -1):
            if size + bounds[index] < target:
                return False
            vertex = order[index]
            bit = 1 << vertex
            if candidates & bit and expand(candidates & neighbors[vertex], size + 1):
                return True
            candidates &= ~bit
        return False

    return expand(ALL_VERTICES, 0)


def _mathematics(result: Any) -> bool:
    adjacency = _graph(result)
    if adjacency is None or not _triangle_free(adjacency):
        return False
    complement = [ALL_VERTICES & ~(adjacency[v] | (1 << v)) for v in range(60)]
    return not _has_clique_of_size(complement, TARGET)


def _reward(payload: dict[str, Any]) -> None:
    path = Path("/logs/verifier/reward.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))
    normalize_reward_file(path)


def main() -> None:
    input_binding = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    protocol = submission is not None
    mathematics = bool(protocol and _mathematics(submission.get("result")))
    reward = float(input_binding and protocol and mathematics)
    _reward(
        {
            "input_binding": float(input_binding),
            "protocol_compliance": float(protocol),
            "correctness": float(mathematics),
            "reward": reward,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _reward(
            {
                "input_binding": 0.0,
                "protocol_compliance": 0.0,
                "correctness": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
