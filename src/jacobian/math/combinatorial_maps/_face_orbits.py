"""Pure facial-orbit kernel shared by value admission and map operations."""

from __future__ import annotations

from jacobian.math.combinatorial_maps.values import FiniteCombinatorialMap


def face_orbit_data(
    map_: FiniteCombinatorialMap,
) -> tuple[list[list[int]], dict[int, int], list[int]]:
    """Return facial walks, their per-dart assignment, and successor map.

    This is deliberately independent of operation adapters so value admission
    can prove that dual construction remains inside its canonical envelope.
    """

    n = len(map_.darts)
    successor: list[int] = [0] * n
    for dart in range(n):
        tail = map_.darts[dart][0]
        row = map_.rotations[tail]
        next_around = row[(row.index(dart) + 1) % len(row)]
        successor[dart] = map_.darts[next_around][2]

    visited = [False] * n
    walks: list[list[int]] = []
    face_of_dart: dict[int, int] = {}
    for start in range(n):
        if visited[start]:
            continue
        walk: list[int] = []
        current = start
        while not visited[current]:
            visited[current] = True
            face_of_dart[current] = len(walks)
            walk.append(current)
            current = successor[current]
        walks.append(walk)
    return walks, face_of_dart, successor


__all__ = ["face_orbit_data"]
