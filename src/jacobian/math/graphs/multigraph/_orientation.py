"""Orientation resolution shared by multigraph flow execution helpers."""

from typing import Protocol


class _EdgeEndpoints(Protocol):
    left: int
    right: int


def oriented_endpoints(edge: _EdgeEndpoints, orientation: str) -> tuple[int, int]:
    """Return the tail and head selected by one declared orientation."""

    if orientation == "left_to_right":
        return edge.left, edge.right
    return edge.right, edge.left


__all__ = ["oriented_endpoints"]
